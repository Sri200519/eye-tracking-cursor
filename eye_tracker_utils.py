import cv2
import numpy as np
import pyautogui # Required for screen size in generate_grid and mouse control
# No need for json here as it's used in raycast_eye_tracker_app and main.py directly.

# Global constants for eye landmarks used by MediaPipe Face Mesh.
# These indices are typically fixed for MediaPipe's output.
# Used for Eye Aspect Ratio (EAR) calculation.
# Structure: [left_corner, top_mid_1, top_mid_2, right_corner, bottom_mid_1, bottom_mid_2]
LEFT_EYE_EAR_IDXS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_IDXS = [362, 385, 387, 263, 373, 380]

def get_eye_bbox(landmarks, frame_w, frame_h, eye_side='left'):
    """
    Calculates a bounding box around a specified eye based on MediaPipe landmarks.

    Args:
        landmarks (list): List of MediaPipe face landmarks.
        frame_w (int): Width of the camera frame.
        frame_h (int): Height of the camera frame.
        eye_side (str): 'left' or 'right' to specify which eye.

    Returns:
        tuple: (x_min, y_min, x_max, y_max) coordinates of the bounding box.
               Returns (0,0,0,0) if no valid landmarks are found.
    """
    # Use a wider range of landmarks for bounding box to better capture the entire eye region,
    # including eyelids and some surrounding facial context.
    # These indices are chosen from the MediaPipe face mesh for a general eye area.
    if eye_side == 'left':
        # More comprehensive set of landmarks for left eye area (adjust as needed for specific models)
        bbox_idxs = [
            33, 7, 163, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 173, 246, # Main eye landmarks
            130, 247, 23, 137, 215, # Additional points around left eye for wider coverage
            27, 28, 29, 30, 31, 32 # Eyebrow points
        ]
    else: # right eye
        # More comprehensive set of landmarks for right eye area
        bbox_idxs = [
            362, 249, 390, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 466, # Main eye landmarks
            359, 467, 253, 366, 435, # Additional points around right eye
            257, 258, 259, 260, 261, 262 # Eyebrow points
        ]

    xs = [landmarks[i].x * frame_w for i in bbox_idxs if i < len(landmarks)] # Ensure index is valid
    ys = [landmarks[i].y * frame_h for i in bbox_idxs if i < len(landmarks)] # Ensure index is valid

    if not xs or not ys: # Handle cases where no valid landmarks are found for the chosen indices
        return 0, 0, 0, 0

    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))

    # Add padding to the bounding box. This is crucial for robust detection and
    # to provide sufficient context for the CNN, especially for users with glasses.
    # These padding factors can be tuned based on experimental results.
    pad_x = int((x_max - x_min) * 0.8) # 80% of current width added horizontally
    pad_y = int((y_max - y_min) * 1.0) # 100% of current height added vertically

    x_min = max(0, x_min - pad_x) # Ensure bbox stays within frame boundaries
    x_max = min(frame_w, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(frame_h, y_max + pad_y)

    # Ensure min < max and provide a minimum size to prevent zero-area crops
    # which can cause errors with cv2.resize or model input
    if x_min >= x_max: x_max = x_min + 20 # Ensure at least 20 pixels width
    if y_min >= y_max: y_max = y_min + 20 # Ensure at least 20 pixels height

    return x_min, y_min, x_max, y_max

def crop_and_preprocess_eye(frame, bbox, target_size=(64, 64)):
    """
    Crops an eye region from the frame, resizes it, converts to grayscale,
    applies CLAHE for contrast, and normalizes pixel values.

    Args:
        frame (np.array): The original camera frame (BGR).
        bbox (tuple): (x_min, y_min, x_max, y_max) bounding box coordinates.
        target_size (tuple): Desired (width, height) for the output image.

    Returns:
        np.array: Preprocessed eye image as a NumPy array (1, target_size[0], target_size[1]),
                  normalized to [0, 1]. Returns a black image if bbox is invalid.
    """
    x_min, y_min, x_max, y_max = bbox
    # Ensure bbox coordinates are within frame dimensions to prevent out-of-bounds access
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)

    if x_max <= x_min or y_max <= y_min:
        # Return a black image of target size if bbox is invalid (e.g., eye not fully detected or too small)
        return np.zeros((1, target_size[0], target_size[1]), dtype=np.float32)

    eye_img = frame[y_min:y_max, x_min:x_max]
    
    # Resize the cropped eye image to the target size. INTER_AREA is good for shrinking.
    eye_img = cv2.resize(eye_img, target_size, interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # This helps improve contrast in different lighting conditions, especially useful for eye features.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(gray)
    
    clahe_img = clahe_img.astype(np.float32) / 255.0 # Normalize pixel values to [0, 1]
    clahe_img = np.expand_dims(clahe_img, axis=0)  # Add channel dimension (1, H, W) for model input
    return clahe_img

def generate_grid(rows=5, cols=7, screen_w=1920, screen_h=1080):
    """
    Generates a grid of calibration points evenly distributed across the screen.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        screen_w (int): Width of the screen in pixels.
        screen_h (int): Height of the screen in pixels.

    Returns:
        list: A list of (x, y) tuples, representing calibration points.
    """
    # Calculate step sizes for X and Y, adding 1 to rows/cols for padding from screen edges
    x_step = screen_w / (cols + 1)
    y_step = screen_h / (rows + 1)
    
    # Generate points by iterating through rows and columns.
    # (c + 1) and (r + 1) ensure points are not at the extreme edges (e.g., starts at 1*step, not 0*step)
    return [(int(x_step * (c + 1)), int(y_step * (r + 1)))
            for r in range(rows) for c in range(cols)]

def is_face_stable(history, thresh=5.0):
    """
    Checks if the face position has been stable over a recent history of iris coordinates.
    Used to prevent capturing calibration samples or moving the cursor if the user's head is moving too much.

    Args:
        history (list): A list of (x, y) tuples representing recent iris/face center positions.
        thresh (float): Maximum allowed average pixel displacement between frames to be considered stable.

    Returns:
        bool: True if face is stable, False otherwise.
    """
    if len(history) < 10: # Need at least 10 frames of history for a meaningful average
        return True # Consider stable if not enough history
    
    # Calculate Euclidean distance (displacement) between consecutive points in history
    diffs = [np.linalg.norm(np.array(history[i]) - np.array(history[i-1])) for i in range(1, len(history))]
    
    # Calculate the average displacement over the recent history
    avg_diff = sum(diffs) / len(diffs)
    
    return avg_diff < thresh # Return True if average displacement is below threshold

def draw_calibration_dot(frame, x, y, screen_w, screen_h):
    """
    Draws a visual calibration dot on the camera feed frame.

    Args:
        frame (np.array): The current camera frame.
        x (int): X-coordinate of the calibration point on the screen.
        y (int): Y-coordinate of the calibration point on the screen.
        screen_w (int): Width of the screen in pixels.
        screen_h (int): Height of the screen in pixels.
    """
    frame_h, frame_w, _ = frame.shape
    
    # Map the screen coordinates (x,y) to the corresponding pixel coordinates on the camera frame.
    # This ensures the dot appears at the correct relative position on the video feed.
    fx = int(x * frame_w / screen_w)
    fy = int(y * frame_h / screen_h)
    
    # Draw a large red filled circle as the primary calibration target
    cv2.circle(frame, (fx, fy), 20, (0, 0, 255), -1) # Red (BGR format), -1 for filled
    # Add a white outline to the circle for better visibility against various backgrounds
    cv2.circle(frame, (fx, fy), 25, (255, 255, 255), 2) # White, 2 pixels thick

def eye_aspect_ratio(landmarks, eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) for a given eye.
    EAR is a scalar value that indicates if an eye is open or closed.

    Args:
        landmarks (list): List of MediaPipe face landmarks.
        eye_indices (list): A list of 6 specific landmark indices defining the eye (e.g., LEFT_EYE_EAR_IDXS).

    Returns:
        float: The calculated Eye Aspect Ratio. Returns 0.0 if landmarks are invalid or missing.
    """
    # Check if all required landmarks exist within the provided list
    if not all(idx < len(landmarks) for idx in eye_indices):
        return 0.0 # Return 0 if any landmark index is out of bounds

    # Extract (x,y) coordinates for the 6 specified eye landmarks
    p = np.array([[landmarks[idx].x, landmarks[idx].y] for idx in eye_indices])

    # Calculate vertical Euclidean distances between the top and bottom eyelid landmarks
    # A: distance between point 2 and point 6 (e.g., 160-144 for left eye)
    # B: distance between point 3 and point 5 (e.g., 158-153 for left eye)
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    
    # Calculate horizontal Euclidean distance across the eye (inner to outer corner)
    # C: distance between point 1 and point 4 (e.g., 33-133 for left eye)
    C = np.linalg.norm(p[0] - p[3])
    
    # Calculate the Eye Aspect Ratio (EAR)
    # Add a small epsilon (1e-6) to the denominator to prevent division by zero if C is 0
    ear = (A + B) / (2.0 * C + 1e-6)
    return ear

def find_working_camera():
    """
    Attempts to find and return the index of a working camera.

    Returns:
        int: The index of the first working camera found (typically 0).
             Returns 0 as a fallback if no camera is explicitly found.
    """
    for i in range(5): # Check typical camera indices (0, 1, 2, 3, 4)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Attempt to read a frame to confirm the camera is truly functional
            ret, frame = cap.read()
            cap.release() # Release the camera immediately after checking
            if ret and frame is not None and frame.shape[0] > 0:
                print(f"Found working camera at index: {i}")
                return i
    print("No working camera found. Defaulting to index 0 (may not work).")
    return 0 # Fallback if no camera is explicitly found