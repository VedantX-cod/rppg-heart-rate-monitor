import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import butter, sosfiltfilt, detrend

FS = 30            # resampling rate (Hz)
WINDOW_SEC = 12    # sliding window length
MIN_SEC = 8        # wait this long before first reading
LOW, HIGH = 0.7, 4.0

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

sos = butter(3, [LOW, HIGH], btype="bandpass", fs=FS, output="sos")


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


def mean_green(frame, box):
    x1, y1, x2, y2 = box
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi[:, :, 1].mean()  # OpenCV is BGR, so index 1 = green


def estimate_bpm(times, values):
    t = np.array(times)
    v = np.array(values)
    t_uni = np.arange(t[0], t[-1], 1 / FS)
    sig = np.interp(t_uni, t, v)
    sig = detrend(sig)
    sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    filt = sosfiltfilt(sos, sig)
    nfft = 2 ** 15
    spec = np.abs(np.fft.rfft(filt * np.hanning(len(filt)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1 / FS)
    mask = (freqs >= LOW) & (freqs <= HIGH)
    return freqs[mask][np.argmax(spec[mask])] * 60, filt


buf_t, buf_v = deque(), deque()
recent_bpm = deque(maxlen=5)
last_calc = 0
bpm_text = "Measuring..."
wave = np.zeros(1)
t0 = time.time()

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
        vals = [mean_green(frame, b) for b in rois]

        if all(v is not None for v in vals):
            buf_t.append(now)
            buf_v.append(np.mean(vals))
            while buf_t and now - buf_t[0] > WINDOW_SEC:
                buf_t.popleft()
                buf_v.popleft()

        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Recalculate once per second
        have = (buf_t[-1] - buf_t[0]) if len(buf_t) > 1 else 0
        if have >= MIN_SEC and now - last_calc >= 1:
            bpm, wave = estimate_bpm(buf_t, buf_v)
            recent_bpm.append(bpm)
            smooth = float(np.median(recent_bpm))
            bpm_text = f"{smooth:.0f} BPM"
            last_calc = now
        elif have < MIN_SEC:
            bpm_text = f"Measuring... {have:.0f}/{MIN_SEC}s"
    else:
        buf_t.clear()
        buf_v.clear()
        recent_bpm.clear()
        bpm_text = "No face"

    cv2.putText(frame, bpm_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.1, (0, 255, 0), 3)

    # Small waveform at the bottom
    if len(wave) > 2:
        gh = 80
        pts = wave[-FS * 8:]
        xs = np.linspace(10, w - 10, len(pts)).astype(int)
        ys = (h - 10 - gh / 2 - np.clip(pts, -3, 3) / 3 * (gh / 2)).astype(int)
        cv2.polylines(frame, [np.column_stack((xs, ys))], False, (0, 200, 255), 2)

    cv2.imshow("rPPG Live Heart Rate (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()