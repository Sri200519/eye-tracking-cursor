import cv2
import numpy as np
import pyautogui

# Structure: [left_corner, top_mid_1, top_mid_2, right_corner, bottom_mid_1, bottom_mid_2]
LEFT_EYE_EAR_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDXS = [362, 385, 387, 263, 373, 380]


def get_eye_bbox(landmarks, frame_w, frame_h, eye_side="left"):
    """
    Calculates a bounding box around a specified eye based on MediaPipe landmarks.
    """

    if eye_side == "left":
        bbox_idxs = [
            33,
            7,
            163,
            144,
            145,
            153,
            154,
            155,
            157,
            158,
            159,
            160,
            161,
            173,
            246,  # Main eye landmarks
            130,
            247,
            23,
            137,
            215,
            27,
            28,
            29,
            30,
            31,
            32,  # Eyebrow points
        ]
    else:  # right eye
        bbox_idxs = [
            362,
            249,
            390,
            373,
            374,
            380,
            381,
            382,
            384,
            385,
            386,
            387,
            388,
            466,  # Main eye landmarks
            359,
            467,
            253,
            366,
            435,
            257,
            258,
            259,
            260,
            261,
            262,  # Eyebrow points
        ]

    xs = [
        landmarks[i].x * frame_w for i in bbox_idxs if i < len(landmarks)
    ]  # Ensure index is valid
    ys = [
        landmarks[i].y * frame_h for i in bbox_idxs if i < len(landmarks)
    ]  # Ensure index is valid

    if not xs or not ys:
        return 0, 0, 0, 0

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # For glasses users
    pad_x = int((x_max - x_min) * 0.8)
    pad_y = int((y_max - y_min) * 1.0)

    x_min = max(0, x_min - pad_x)
    x_max = min(frame_w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(frame_h, y_max + pad_y)

    if x_min >= x_max:
        x_max = x_min + 20
    if y_min >= y_max:
        y_max = y_min + 20

    return x_min, y_min, x_max, y_max


def crop_and_preprocess_eye(frame, bbox, target_size=(64, 64)):
    """
    Crops an eye region from the frame, resizes it, converts to grayscale,
    applies CLAHE for contrast, and normalizes pixel values.
    """
    x_min, y_min, x_max, y_max = bbox
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)

    if x_max <= x_min or y_max <= y_min:
        return np.zeros((1, target_size[0], target_size[1]), dtype=np.float32)

    eye_img = frame[y_min:y_max, x_min:x_max]

    eye_img = cv2.resize(eye_img, target_size, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    clahe_img = clahe_img.astype(np.float32) / 255.0
    clahe_img = np.expand_dims(clahe_img, axis=0)
    return clahe_img


def generate_grid(rows=5, cols=7, screen_w=1920, screen_h=1080):
    """
    Generates a grid of calibration points evenly distributed across the screen.
    """
    x_step = screen_w / (cols + 1)
    y_step = screen_h / (rows + 1)

    return [
        (int(x_step * (c + 1)), int(y_step * (r + 1)))
        for r in range(rows)
        for c in range(cols)
    ]


def is_face_stable(history, thresh=5.0):
    """
    Checks if the face position has been stable over a recent history of iris coordinates.
    Used to prevent capturing calibration samples or moving the cursor if the user's head is moving too much.
    """
    if len(history) < 10:
        return True

    diffs = [
        np.linalg.norm(np.array(history[i]) - np.array(history[i - 1]))
        for i in range(1, len(history))
    ]

    avg_diff = sum(diffs) / len(diffs)

    return avg_diff < thresh


def draw_calibration_dot(frame, x, y, screen_w, screen_h):
    """
    Draws a visual calibration dot on the camera feed frame.
    """
    frame_h, frame_w, _ = frame.shape

    fx = int(x * frame_w / screen_w)
    fy = int(y * frame_h / screen_h)

    cv2.circle(frame, (fx, fy), 20, (0, 0, 255), -1)  # Red
    cv2.circle(frame, (fx, fy), 25, (255, 255, 255), 2)  # White


def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) for a given eye.
    EAR is a scalar value that indicates if an eye is open or closed.
    """
    if not all(idx < len(landmarks) for idx in eye_indices):
        return 0.0  # Return 0 if any landmark index is out of bounds

    p = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in eye_indices])

    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])

    C = np.linalg.norm(p[0] - p[3])

    # Calculate the Eye Aspect Ratio (EAR)
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear


def find_working_camera():
    """
    Attempts to find and return the index of a working camera.
    """
    for i in range(5):  # Check typical camera indices (0, 1, 2, 3, 4)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.shape[0] > 0:
                print(f"Found working camera at index: {i}")
                return i
    print("No working camera found. Defaulting to index 0 (may not work).")
    return 0
