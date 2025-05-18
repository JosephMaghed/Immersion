import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark indices
LEFT_EYE_IDX = [33, 133]
RIGHT_EYE_IDX = [362, 263]
LEFT_EYE_CORNER_IDX = [130, 133]  # Inner and outer corners
RIGHT_EYE_CORNER_IDX = [362, 359]
LEFT_EYELID_IDX = [159, 145]
RIGHT_EYELID_IDX = [386, 374]
NOSE_TIP_IDX = 1
FOREHEAD_CENTER_IDX = 10
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454
CHIN_IDX = 152
LEFT_NOSTRIL_IDX = 98
RIGHT_NOSTRIL_IDX = 327
NOSE_TOP_IDX = 6
MOUTH_TOP_IDX = 13
MOUTH_BOTTOM_IDX = 14
MOUTH_LEFT_IDX = 61
MOUTH_RIGHT_IDX = 291

# Initialize camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    skin_color_text = "Skin RGB: N/A"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # ----- NOSE TIP -----
            nose_tip = landmarks[NOSE_TIP_IDX]
            x_nose = int(nose_tip.x * w)
            y_nose = int(nose_tip.y * h)
            cv2.circle(frame, (x_nose, y_nose), 6, (0, 255, 0), -1)
            cv2.putText(frame, "Nose Tip", (x_nose - 30, y_nose - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # ----- NOSTRILS & NOSE TOP -----
            left_nostril = landmarks[LEFT_NOSTRIL_IDX]
            right_nostril = landmarks[RIGHT_NOSTRIL_IDX]
            nose_top = landmarks[NOSE_TOP_IDX]

            x_ln, y_ln = int(left_nostril.x * w), int(left_nostril.y * h)
            x_rn, y_rn = int(right_nostril.x * w), int(right_nostril.y * h)
            x_nt, y_nt = int(nose_top.x * w), int(nose_top.y * h)

            cv2.circle(frame, (x_ln, y_ln), 4, (255, 100, 100), -1)
            cv2.putText(frame, "L-Nostril", (x_ln - 30, y_ln - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

            cv2.circle(frame, (x_rn, y_rn), 4, (255, 100, 100), -1)
            cv2.putText(frame, "R-Nostril", (x_rn - 30, y_rn - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 100), 1)

            cv2.circle(frame, (x_nt, y_nt), 4, (200, 255, 200), -1)
            cv2.putText(frame, "Top Nose", (x_nt - 30, y_nt - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 255, 200), 1)

            # ----- EYE ELLIPSES & CORNER POINTS -----
            for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                x1_eye, y1_eye = int(landmarks[eye_idx[0]].x * w), int(landmarks[eye_idx[0]].y * h)
                x2_eye, y2_eye = int(landmarks[eye_idx[1]].x * w), int(landmarks[eye_idx[1]].y * h)
                center = ((x1_eye + x2_eye) // 2, (y1_eye + y2_eye) // 2)
                axis_length = (abs(x2_eye - x1_eye) // 2, abs(y2_eye - y1_eye) // 2 + 5)
                cv2.ellipse(frame, center, axis_length, 0, 0, 360, (0, 0, 255), 2)

            for eye_corners in [LEFT_EYE_CORNER_IDX, RIGHT_EYE_CORNER_IDX]:
                for idx in eye_corners:
                    x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (150, 0, 255), -1)

            # ----- EYELID POINTS -----
            for eyelid_idx in [LEFT_EYELID_IDX, RIGHT_EYELID_IDX]:
                upper = landmarks[eyelid_idx[0]]
                lower = landmarks[eyelid_idx[1]]
                x_upper, y_upper = int(upper.x * w), int(upper.y * h)
                x_lower, y_lower = int(lower.x * w), int(lower.y * h)
                cv2.circle(frame, (x_upper, y_upper), 3, (255, 255, 0), -1)
                cv2.circle(frame, (x_lower, y_lower), 3, (0, 255, 255), -1)

            # ----- FOREHEAD TOP POINT -----
            forehead = landmarks[FOREHEAD_CENTER_IDX]
            x_fh, y_fh = int(forehead.x * w), int(forehead.y * h) - 20  # Higher under hair
            cv2.circle(frame, (x_fh, y_fh), 5, (255, 0, 255), -1)
            cv2.putText(frame, "Forehead Top", (x_fh - 40, y_fh - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # ----- LEFT & RIGHT CHEEK -----
            left_cheek = landmarks[LEFT_CHEEK_IDX]
            right_cheek = landmarks[RIGHT_CHEEK_IDX]
            x_lc, y_lc = int(left_cheek.x * w), int(left_cheek.y * h)
            x_rc, y_rc = int(right_cheek.x * w), int(right_cheek.y * h)
            cv2.circle(frame, (x_lc, y_lc), 5, (0, 128, 255), -1)
            cv2.putText(frame, "Left Cheek", (x_lc - 40, y_lc - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)
            cv2.circle(frame, (x_rc, y_rc), 5, (0, 128, 255), -1)
            cv2.putText(frame, "Right Cheek", (x_rc - 40, y_rc - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

            # ----- CHIN -----
            chin = landmarks[CHIN_IDX]
            x_chin, y_chin = int(chin.x * w), int(chin.y * h)
            cv2.circle(frame, (x_chin, y_chin), 5, (0, 255, 128), -1)
            cv2.putText(frame, "Chin", (x_chin - 20, y_chin + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)

            # ----- MOUTH POINTS -----
            # ----- MOUTH POINTS with dynamic label positioning based on lip thickness -----
            # ----- MOUTH POINTS with corrected label position (above/below lips) -----
            # ----- MOUTH POINTS with fixed vertical offset for labels -----
            # ----- MOUTH POINTS with visible offset and debug lines -----
            for idx, label, color in zip(
                    [MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX],
                    ["Mouth Top", "Mouth Bottom", "Mouth Left", "Mouth Right"],
                    [(255, 200, 0), (200, 150, 50), (100, 255, 100), (100, 255, 100)]):

                x, y = int(landmarks[idx].x * w), int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 4, color, -1)
                # Offset label position far from mouth center
                if label == "Mouth Top":
                    label_pos = (x - 30, y - 40)
                elif label == "Mouth Bottom":
                    label_pos = (x - 30, y + 40)
                elif label == "Mouth Left":
                    label_pos = (x - 60, y)
                elif label == "Mouth Right":
                    label_pos = (x + 10, y)
                else:
                    label_pos = (x - 30, y - 10)

                # Optional: draw a line from point to label
                cv2.line(frame, (x, y), label_pos, color, 1)

                cv2.putText(frame, label, label_pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # ----- LIP THICKNESS MEASUREMENT -----
            top_lip = landmarks[MOUTH_TOP_IDX]
            bottom_lip = landmarks[MOUTH_BOTTOM_IDX]

            # Convert normalized coordinates to pixel values
            x_top, y_top = int(top_lip.x * w), int(top_lip.y * h)
            x_bot, y_bot = int(bottom_lip.x * w), int(bottom_lip.y * h)



            # ----- Skin color detection (average RGB on left cheek patch) -----
            patch_size = 20
            x_start = max(x_lc - patch_size // 2, 0)
            y_start = max(y_lc - patch_size // 2, 0)
            x_end = min(x_start + patch_size, w)
            y_end = min(y_start + patch_size, h)

            patch = frame[y_start:y_end, x_start:x_end]

            if patch.size != 0:
                b_mean = int(np.mean(patch[:, :, 0]))
                g_mean = int(np.mean(patch[:, :, 1]))
                r_mean = int(np.mean(patch[:, :, 2]))

                skin_color_text = f"Skin RGB: R={r_mean} G={g_mean} B={b_mean}"

            # Draw a small rectangle showing the sampled patch
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 255), 1)

    # Display skin color text at the bottom left
    cv2.putText(frame, skin_color_text, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Mesh Key Points + Skin RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
