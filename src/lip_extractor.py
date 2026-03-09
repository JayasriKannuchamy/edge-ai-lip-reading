import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Lip landmark indices (MediaPipe)
lip_indices = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
    95, 185, 40, 39, 37, 0, 267, 269, 270, 409
]

def get_lip_coordinates(landmarks, image_shape):
    """
    Extract 2D coordinates of lip landmarks from MediaPipe landmarks.
    """
    h, w, _ = image_shape
    coords = []

    for idx in lip_indices:
        lm = landmarks[idx]
        x = int(lm.x * w)
        y = int(lm.y * h)
        coords.append((x, y))

    return coords


def crop_lip_region(image, coords):
    """
    Crop lip region using bounding box of lip landmarks.
    """
    if not coords:
        return None

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    padding = 10

    x_min = max(0, x_min - padding)
    x_max = min(image.shape[1], x_max + padding)
    y_min = max(0, y_min - padding)
    y_max = min(image.shape[0], y_max + padding)

    lip = image[y_min:y_max, x_min:x_max]

    if lip.size != 0:
        lip = cv2.resize(lip, (64, 64))

    return lip


def draw_lip_contour(image, coords):
    """
    Draw lip contour for visualization.
    """
    if not coords:
        return

    for i in range(len(coords) - 1):
        cv2.line(image, coords[i], coords[i + 1], (0, 255, 0), 2)

    cv2.line(image, coords[-1], coords[0], (0, 255, 0), 2)


try:
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                coords = get_lip_coordinates(face_landmarks.landmark, frame.shape)

                cropped_lip = crop_lip_region(frame, coords)

                # Draw contour
                draw_lip_contour(frame, coords)

                # Show face frame
                cv2.imshow("Face with Lip Contour", frame)

                # Show cropped lips
                if cropped_lip is not None:
                    cv2.imshow("Cropped Lip Region", cropped_lip)

        # Exit when pressing q
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
