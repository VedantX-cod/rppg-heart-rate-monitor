import cv2
import mediapipe as mp

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True)

cap = cv2.VideoCapture(0)


def pt(lm, idx, w, h):
    return int(lm[idx].x * w), int(lm[idx].y * h)


def clamp_box(x1, y1, x2, y2, w, h):
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        # Forehead: between brows (9) and hairline (10), width from 104 to 333
        fx1, _ = pt(lm, 104, w, h)
        fx2, _ = pt(lm, 333, w, h)
        _, fy1 = pt(lm, 10, w, h)
        _, fy2 = pt(lm, 9, w, h)
        forehead = clamp_box(fx1, fy1, fx2, fy2, w, h)

        # Cheeks: small boxes around landmarks 50 (left) and 280 (right)
        size = int(abs(fx2 - fx1) * 0.35)
        lcx, lcy = pt(lm, 50, w, h)
        rcx, rcy = pt(lm, 280, w, h)
        left_cheek = clamp_box(lcx - size, lcy - size, lcx + size, lcy + size, w, h)
        right_cheek = clamp_box(rcx - size, rcy - size, rcx + size, rcy + size, w, h)

        for (x1, y1, x2, y2), color in [
            (forehead, (0, 255, 0)),
            (left_cheek, (255, 200, 0)),
            (right_cheek, (255, 200, 0)),
        ]:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, "Face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("rPPG Step 2 - ROI", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()