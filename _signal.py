import time
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

RECORD_SECONDS = 30

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)


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
    return r, g, b


times, signal = [], []
start = None
print("Press 's' to start recording (30 sec). Sit still, face the light.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    status = "Press 's' to start"

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark
        rois = get_rois(lm, w, h)

        if start is not None:
            vals = [mean_rgb(frame, b) for b in rois]
            if all(v is not None for v in vals):
                avg = np.mean(vals, axis=0)  # average of 3 ROIs -> (R, G, B)
                times.append(time.time() - start)
                signal.append(avg)
            elapsed = time.time() - start
            status = f"Recording: {elapsed:.0f}/{RECORD_SECONDS}s"
            if elapsed >= RECORD_SECONDS:
                break

        for (x1, y1, x2, y2) in rois:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        status = "No face"

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    cv2.imshow("rPPG Step 3 - Record", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s") and start is None:
        start = time.time()

cap.release()
cv2.destroyAllWindows()

if len(signal) < 30:
    print("Bahut kam data mila. Dobara try karo.")
else:
    t = np.array(times)
    sig = np.array(signal)  # shape (N, 3) -> R, G, B
    fps = len(t) / (t[-1] - t[0])
    print(f"Frames: {len(t)}, actual FPS: {fps:.1f}")

    np.save("rppg_times.npy", t)
    np.save("rppg_rgb.npy", sig)

    plt.figure(figsize=(10, 4))
    plt.plot(t, sig[:, 1], color="green")
    plt.title("Raw green channel (mean of forehead + cheeks)")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean intensity")
    plt.tight_layout()
    plt.show()