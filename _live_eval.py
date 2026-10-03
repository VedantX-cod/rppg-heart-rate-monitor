import csv
import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, sosfiltfilt, detrend

FS = 30
WINDOW_SEC = 12
MIN_SEC = 8
LOW, HIGH = 0.7, 4.0

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)
sos = butter(3, [LOW, HIGH], btype="bandpass", fs=FS, output="sos")


# ---------- ROI helpers ----------
def pt(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)


def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def get_rois(lm, w, h):
    fx1, _ = pt(lm, 104, w, h)
    fx2, _ = pt(lm, 333, w, h)
    _, fy1 = pt(lm, 10, w, h)
    _, fy2 = pt(lm, 9, w, h)
    forehead = clamp_box(fx1, fy1, fx2, fy2, w, h)
    size = int(abs(fx2 - fx1) * 0.35)
    lcx, lcy = pt(lm, 50, w, h)
    rcx, rcy = pt(lm, 280, w, h)
    left = clamp_box(lcx - size, lcy - size, lcx + size, lcy + size, w, h)
    right = clamp_box(rcx - size, rcy - size, rcx + size, rcy + size, w, h)
    return [forehead, left, right]


def mean_rgb(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    b, g, r = roi.reshape(-1, 3).mean(axis=0)
    return np.array([r, g, b])


# ---------- rPPG methods (input C: N x 3, columns R, G, B) ----------
def green_method(C):
    return detrend(C[:, 1])


def chrom_method(C):
    Cn = C / C.mean(axis=0)
    R, G, B = Cn[:, 0], Cn[:, 1], Cn[:, 2]
    Xs = 3 * R - 2 * G
    Ys = 1.5 * R + G - 1.5 * B
    return Xs - (Xs.std() / (Ys.std() + 1e-8)) * Ys


def pos_method(C, win_sec=1.6):
    l = int(win_sec * FS)
    H = np.zeros(len(C))
    for n in range(l - 1, len(C)):
        m = n - l + 1
        seg = C[m:n + 1]
        Cn = seg / seg.mean(axis=0)
        S1 = Cn[:, 1] - Cn[:, 2]
        S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
        h = S1 + (S1.std() / (S2.std() + 1e-8)) * S2
        H[m:n + 1] += h - h.mean()
    return H


METHODS = {"Green": green_method, "CHROM": chrom_method, "POS": pos_method}
NAMES = list(METHODS)


def bpm_from(sig):
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    filt = sosfiltfilt(sos, sig)
    nfft = 2 ** 15
    spec = np.abs(np.fft.rfft(filt * np.hanning(len(filt)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1 / FS)
    mask = (freqs >= LOW) & (freqs <= HIGH)
    return freqs[mask][np.argmax(spec[mask])] * 60, filt


def estimate_all(times, values):
    t = np.array(times)
    v = np.array(values)
    t_uni = np.arange(t[0], t[-1], 1 / FS)
    C = np.column_stack([np.interp(t_uni, t, v[:, i]) for i in range(3)])
    out = {}
    for name, fn in METHODS.items():
        out[name] = bpm_from(fn(C))
    return out


# ---------- main loop ----------
buf_t, buf_v = deque(), deque()
history = {n: deque(maxlen=5) for n in NAMES}
current = {n: None for n in NAMES}
wave = np.zeros(1)
selected = "POS"
samples = []  # each: {"Green": x, "CHROM": y, "POS": z}
last_calc = 0
status = "Measuring..."
t0 = time.time()

print("Keys: 1=Green  2=CHROM  3=POS  r=save sample  q=quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    now = time.time() - t0
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        rois = get_rois(lm, w, h)
        vals = [mean_rgb(frame, b) for b in rois]

        if all(v is not None for v in vals):
            buf_t.append(now)
            buf_v.append(np.mean(vals, axis=0))
            while buf_t and now - buf_t[0] > WINDOW_SEC:
                buf_t.popleft()
                buf_v.popleft()

        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        have = (buf_t[-1] - buf_t[0]) if len(buf_t) > 1 else 0
        if have >= MIN_SEC and now - last_calc >= 1:
            est = estimate_all(buf_t, buf_v)
            for n in NAMES:
                history[n].append(est[n][0])
                current[n] = float(np.median(history[n]))
            wave = est[selected][1]
            status = f"{selected}: {current[selected]:.0f} BPM"
            last_calc = now
        elif have < MIN_SEC:
            status = f"Measuring... {have:.0f}/{MIN_SEC}s"
    else:
        buf_t.clear()
        buf_v.clear()
        for n in NAMES:
            history[n].clear()
            current[n] = None
        status = "No face"

    cv2.putText(frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3)
    cv2.putText(frame, f"Samples: {len(samples)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if len(wave) > 2:
        gh = 80
        pts = wave[-FS * 8:]
        xs = np.linspace(10, w - 10, len(pts)).astype(int)
        ys = (h - 10 - gh / 2 - np.clip(pts, -3, 3) / 3 * (gh / 2)).astype(int)
        cv2.polylines(frame, [np.column_stack((xs, ys))], False, (0, 200, 255), 2)

    cv2.imshow("rPPG Live + Eval", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key in (ord("1"), ord("2"), ord("3")):
        selected = NAMES[key - ord("1")]
    elif key == ord("r") and all(current[n] is not None for n in NAMES):
        samples.append(dict(current))
        print(f"Sample {len(samples)} saved: " +
              ", ".join(f"{n} {current[n]:.0f}" for n in NAMES) +
              "  -> ab apni watch ka BPM note kar lo")

cap.release()
cv2.destroyAllWindows()

# ---------- evaluation ----------
if samples:
    print("\nAb har sample ke liye reference (watch) BPM daalo. Skip ke liye Enter.")
    rows = []
    for i, s in enumerate(samples, 1):
        raw = input(f"Sample {i} reference BPM: ").strip()
        if not raw:
            continue
        try:
            ref = float(raw)
        except ValueError:
            continue
        rows.append({"sample": i, "reference": ref, **s})

    if rows:
        with open("eval_log.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["sample", "reference"] + NAMES)
            wr.writeheader()
            wr.writerows(rows)

        print(f"\n{len(rows)} samples. Results:")
        print(f"{'Method':8} {'MAE (BPM)':>10} {'RMSE':>8}")
        for n in NAMES:
            err = np.array([r[n] - r["reference"] for r in rows])
            print(f"{n:8} {np.abs(err).mean():10.2f} {np.sqrt((err ** 2).mean()):8.2f}")
        print("Saved to eval_log.csv")