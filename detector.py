import math
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize globally to avoid loading the model on every request
mp_face_mesh = mp.solutions.face_mesh
# static_image_mode=True is safer for API calls (2 FPS) to avoid tracking drift
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Note: In a true multi-user production app, you'd key this history by user/session ID.
# Kept as a global deque here to mirror your local script's behavior.
history = deque(maxlen=15)

def distance(p1, p2):
    return math.dist(p1, p2)

def get_point(landmarks, i, w, h):
    return (landmarks[i].x * w, landmarks[i].y * h)

def get_jaw_angle(p_ear, p_jaw, p_chin):
    a = distance(p_jaw, p_chin)
    b = distance(p_ear, p_jaw)
    c = distance(p_ear, p_chin)
    try:
        angle = math.degrees(math.acos((a**2 + b**2 - c**2) / (2*a*b)))
        return angle
    except:
        return 110

def detect_face_shape(image: np.ndarray) -> dict:
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {
            "status": "No face detected",
            "shape": None,
            "angle": None,
            "ratios": None
        }

    for face_landmarks in results.multi_face_landmarks:
        lm = face_landmarks.landmark

        chin = get_point(lm, 152, w, h)
        forehead_top = get_point(lm, 10, w, h)
        
        face_mid_height = distance(forehead_top, chin)
        hairline = (forehead_top[0], forehead_top[1] - (face_mid_height * 0.25))

        f_l, f_r = get_point(lm, 103, w, h), get_point(lm, 332, w, h)
        c_l, c_r = get_point(lm, 234, w, h), get_point(lm, 454, w, h)
        j_l, j_r = get_point(lm, 132, w, h), get_point(lm, 361, w, h)

        ear_l = get_point(lm, 234, w, h)
        jaw_corner = get_point(lm, 172, w, h)
        l_eye, r_eye = get_point(lm, 33, w, h), get_point(lm, 263, w, h)

        FL = distance(hairline, chin)
        FW = distance(f_l, f_r)
        CW = distance(c_l, c_r)
        JW = distance(j_l, j_r)
        
        ratio_FL_CW = FL / CW if CW != 0 else 0
        angle = get_jaw_angle(ear_l, jaw_corner, chin)

        # Tilt Filter
        if abs(l_eye[1] - r_eye[1]) > 20:
            return {
                "status": "Keep head straight",
                "shape": None,
                "angle": round(angle, 1),
                "ratios": {"FL_CW": round(ratio_FL_CW, 2)}
            }

        scores = {k: 0 for k in ["Oval", "Square", "Round", "Rectangle", "Heart", "Diamond", "Triangle"]}

        if ratio_FL_CW > 1.55:
            scores["Rectangle"] += 3
        elif 1.35 < ratio_FL_CW <= 1.55:
            scores["Oval"] += 3
        
        if ratio_FL_CW <= 1.35:
            if angle < 105:
                scores["Square"] += 2
            else:
                scores["Round"] += 2

        if FW > CW and CW > JW:
            scores["Heart"] += 3
        if CW > FW and CW > JW:
            scores["Diamond"] += 3
        if JW > CW:
            scores["Triangle"] += 4

        current_pred = max(scores, key=scores.get)
        history.append(current_pred)
        smoothed_shape = max(set(history), key=history.count)

        return {
            "status": "Success",
            "shape": smoothed_shape,
            "angle": round(angle, 1),
            "ratios": {
                "FL_CW": round(ratio_FL_CW, 2)
            }
        }