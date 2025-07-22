# facial_structure_measurements.py

import cv2
import mediapipe as mp
import math

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark pairs for key measurements (left ↔ right or top ↔ bottom)
LANDMARKS = {
    'Eye Distance (cm)': (33, 263),  # left eye outer ↔ right eye outer
    'Nose Length (cm)': (1, 2),      # top of nose ↔ tip of nose
    'Lip Width (cm)': (61, 291),    # left ↔ right mouth
    'Jaw Width (cm)': (234, 454),   # jaw left ↔ jaw right
    'Forehead to Chin (cm)': (10, 152),  # top center ↔ chin center
}

# Euclidean distance between two points
def euclidean(pt1, pt2):
    return math.sqrt((pt1.x - pt2.x)**2 + (pt1.y - pt2.y)**2)

# Capture video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Real-world reference: interpupillary distance (IPD) ~6.3 cm
REFERENCE_POINTS = (33, 263)
REFERENCE_CM = 6.3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]

        # Get all necessary coordinates once
        landmarks = {i: face_landmarks.landmark[i] for i in range(478)}

        # Compute reference pixels for scale
        ref_dist = euclidean(landmarks[REFERENCE_POINTS[0]], landmarks[REFERENCE_POINTS[1]])
        pixels_per_cm = ref_dist / REFERENCE_CM if ref_dist != 0 else 1

        # Draw mesh and calculate distances
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        y_offset = 30
        for label, (id1, id2) in LANDMARKS.items():
            p1 = landmarks[id1]
            p2 = landmarks[id2]
            dist_px = euclidean(p1, p2)
            dist_cm = dist_px / pixels_per_cm

            text = f"{label}: {dist_cm:.2f}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25

    cv2.imshow('Facial Measurements (cm)', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()