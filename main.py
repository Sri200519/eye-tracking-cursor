import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import pyautogui
import mediapipe as mp
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QFont, QColor, QPalette
from PyQt5.QtCore import QObject, pyqtSignal, QThread 

from PIL import Image, ImageDraw


from gui_eye_tracker_app import RaycastEyeTrackerApp, CALIBRATION_COMPLETE_DATA, TRACKING_ACTIVE
from eye_tracker_models import KalmanFilter2D, EyeTransformer
from eye_tracker_utils import (
    get_eye_bbox, crop_and_preprocess_eye, is_face_stable, eye_aspect_ratio,
    find_working_camera, LEFT_EYE_EAR_IDXS, RIGHT_EYE_EAR_IDXS
)

_current_tracking_data = None


class TrackingWorker(QObject):
    finished = pyqtSignal()
    
    def __init__(self, data):
        super().__init__()
        self.tracking_data = data
        self._is_running = True 

    def run(self):
        """This method contains the main real-time tracking loop."""
        print("TrackingWorker: Starting background tracking loop.")
        
        tracking_data = self.tracking_data
        gesture_click_map = tracking_data["gesture_click_map"]
        x_min_calib = tracking_data["x_min_calib"]
        y_min_calib = tracking_data["y_min_calib"]
        x_max_calib = tracking_data["x_max_calib"]
        y_max_calib = tracking_data["y_max_calib"]
        model_path = tracking_data["model_path"]
        cam_index = tracking_data["cam_index"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"TrackingWorker: Using device: {device}")

        screen_w, screen_h = pyautogui.size()
        print(f"TrackingWorker: Screen resolution: {screen_w}x{screen_h}")

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

        model = EyeTransformer().to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
            print("TrackingWorker: Loaded trained model.")
        except FileNotFoundError:
            print(f"TrackingWorker Error: Model file '{model_path}' not found. Cannot start tracking.")
            self._is_running = False
        except Exception as e:
            print(f"TrackingWorker Error: Error loading model: {e}. Cannot start tracking.")
            self._is_running = False

        if not self._is_running: 
            self.finished.emit()
            return

        model.eval()
        cap = cv2.VideoCapture(cam_index) 
        if not cap.isOpened():
            print(f"TrackingWorker Error: Could not open camera at index {cam_index}.")
            self._is_running = False
        
        if not self._is_running: # Exit if camera failed
            self.finished.emit()
            return

        time.sleep(1)
        kf = KalmanFilter2D()
        iris_history = []

        EAR_THRESH = 0.20
        LONG_BLINK_FRAMES = 12
        BLINK_FRAMES = 3
        left_eye_closed_frames = 0
        right_eye_closed_frames = 0
        both_eyes_closed_frames = 0
        left_wink_triggered = False
        right_wink_triggered = False
        long_blink_triggered = False

        print("TrackingWorker: Real-time loop in background. Press 'q' on OpenCV window or use GUI to quit.")

        while self._is_running: # Loop based on internal flag
            ret, frame = cap.read()
            if not ret:
                print("TrackingWorker: Failed to grab frame. Exiting tracking loop.")
                break
            
            frame = cv2.flip(frame, 1)
            frame_h, frame_w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_EAR_IDXS)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_IDXS)

                left_eye_closed = left_ear < EAR_THRESH
                right_eye_closed = right_ear < EAR_THRESH

                if left_eye_closed: left_eye_closed_frames += 1
                else: left_eye_closed_frames = 0
                if right_eye_closed: right_eye_closed_frames += 1
                else: right_eye_closed_frames = 0
                if left_eye_closed and right_eye_closed: both_eyes_closed_frames += 1
                else: both_eyes_closed_frames = 0

                if gesture_click_map.get("Long Blink") and both_eyes_closed_frames == LONG_BLINK_FRAMES and not long_blink_triggered:
                    pyautogui.click(button=gesture_click_map["Long Blink"])
                    long_blink_triggered = True
                    cv2.putText(frame, f"{gesture_click_map['Long Blink'].capitalize()} Click (Long Blink)", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                if both_eyes_closed_frames == 0: long_blink_triggered = False

                if gesture_click_map.get("Left Wink") and left_eye_closed_frames == BLINK_FRAMES and right_eye_closed_frames == 0 and not left_wink_triggered:
                    pyautogui.click(button=gesture_click_map["Left Wink"])
                    left_wink_triggered = True
                    cv2.putText(frame, f"{gesture_click_map['Left Wink'].capitalize()} Click (Left Wink)", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                if left_eye_closed_frames == 0 or right_eye_closed_frames > 0: left_wink_triggered = False

                if gesture_click_map.get("Right Wink") and right_eye_closed_frames == BLINK_FRAMES and left_eye_closed_frames == 0 and not right_wink_triggered:
                    pyautogui.click(button=gesture_click_map["Right Wink"])
                    right_wink_triggered = True
                    cv2.putText(frame, f"{gesture_click_map['Right Wink'].capitalize()} Click (Right Wink)", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                if right_eye_closed_frames == 0 or left_eye_closed_frames > 0: right_wink_triggered = False

                if len(landmarks) > 33 and len(landmarks) > 263:
                    center_x = (landmarks[33].x + landmarks[263].x) / 2 * frame_w
                    center_y = (landmarks[33].y + landmarks[263].y) / 2 * frame_h
                    iris_history.append((center_x, center_y))
                    if len(iris_history) > 10: iris_history.pop(0)

                stable = is_face_stable(iris_history)
                if not stable:
                    cv2.putText(frame, "Face unstable - skipping frame", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.imshow("Eye Tracker", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    continue

                bbox_left = get_eye_bbox(landmarks, frame_w, frame_h, 'left')
                bbox_right = get_eye_bbox(landmarks, frame_w, frame_h, 'right')
                left_eye_img = crop_and_preprocess_eye(frame, bbox_left)
                right_eye_img = crop_and_preprocess_eye(frame, bbox_right)
                eye_input = np.concatenate([left_eye_img, right_eye_img], axis=0)
                input_tensor = torch.tensor(eye_input, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    model.eval()
                    pred = model(input_tensor).cpu().numpy()[0]

                pred[0] = np.clip(pred[0], x_min_calib, x_max_calib)
                pred[1] = np.clip(pred[1], y_min_calib, y_max_calib)

                scale_x = screen_w / (x_max_calib - x_min_calib + 1e-6)
                scale_y = screen_h / (y_max_calib - y_min_calib + 1e-6)
                scaled_x = (pred[0] - x_min_calib) * scale_x
                scaled_y = (pred[1] - y_min_calib) * scale_y

                smooth_x, smooth_y = kf.update(np.array([scaled_x, scaled_y]))

                smooth_x = int(np.clip(smooth_x, 0, screen_w - 1))
                smooth_y = int(np.clip(smooth_y, 0, screen_h - 1))

                pyautogui.moveTo(smooth_x, smooth_y, duration=0.01)

                cv2.rectangle(frame, (bbox_left[0], bbox_left[1]), (bbox_left[2], bbox_left[3]), (0,255,0), 2)
                cv2.rectangle(frame, (bbox_right[0], bbox_right[1]), (bbox_right[2], bbox_right[3]), (0,255,0), 2)

                frame_pred_x = int(smooth_x * frame_w / screen_w)
                frame_pred_y = int(smooth_y * frame_h / screen_h)
                cv2.circle(frame, (frame_pred_x, frame_pred_y), 10, (255,0,0), 2)
                cv2.putText(frame, "Tracking...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
                cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            else:
                cv2.putText(frame, "Face not detected", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            cv2.imshow("Eye Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("TrackingWorker: 'q' pressed on OpenCV window. Stopping tracking.")
                self._is_running = False 
                break
            
            time.sleep(0.001)

            if not TRACKING_ACTIVE: 
                print("TrackingWorker: Received external stop signal.")
                self._is_running = False

        cap.release()
        cv2.destroyAllWindows()
        print("TrackingWorker: Background tracking finished and resources released.")
        self.finished.emit() 

    def stop(self):
        """Method to externally stop the tracking worker."""
        self._is_running = False


_tracking_thread = None # QThread object
_tracking_worker = None # TrackingWorker object

def _start_tracking_handler(tracking_data):
    """Starts the real-time tracking in a new QThread."""
    global _tracking_thread, _tracking_worker, TRACKING_ACTIVE
    
    if _tracking_thread and _tracking_thread.isRunning():
        print("Main: Existing tracking thread found. Stopping it first...")
        _stop_tracking_handler() # Stop the old one first
        _tracking_thread.wait(1000) # Give it some time to stop

    TRACKING_ACTIVE = True
    print("\nMain: Starting new tracking thread.")
    _tracking_thread = QThread()
    _tracking_worker = TrackingWorker(tracking_data)
    
    _tracking_worker.moveToThread(_tracking_thread)
    _tracking_thread.started.connect(_tracking_worker.run)
    _tracking_worker.finished.connect(_tracking_thread.quit)
    _tracking_worker.finished.connect(_tracking_worker.deleteLater)
    _tracking_thread.finished.connect(_tracking_thread.deleteLater)
    
    _tracking_thread.start()

def _stop_tracking_handler():
    """Stops the real-time tracking thread."""
    global _tracking_worker, _tracking_thread, TRACKING_ACTIVE
    if _tracking_worker:
        _tracking_worker.stop() 
    TRACKING_ACTIVE = False 
    if _tracking_thread and _tracking_thread.isRunning():
        print("Main: Requesting tracking thread to terminate...")
        _tracking_thread.quit()
        _tracking_thread.wait(2000) 
        if _tracking_thread.isRunning():
            print("Main: Warning: Tracking thread did not terminate gracefully. Terminating forcefully.")
            _tracking_thread.terminate() # Force terminate if it doesn't quit
            _tracking_thread.wait(1000)
    _tracking_worker = None
    _tracking_thread = None
    print("Main: Tracking thread reference cleared.")


if __name__ == "__main__":
    try:
        with open("arrow_down.png", "rb") as f:
            pass
    except FileNotFoundError:
        img = Image.new('RGBA', (28, 28), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.polygon([(7, 10), (21, 10), (14, 18)], fill=(243, 243, 247, 255))
        img.save("arrow_down.png")
        print("Generated dummy arrow_down.png for QComboBox. You can replace this with a custom icon.")

    screen_width, screen_height = pyautogui.size()

    app = QApplication(sys.argv)
    app.setApplicationName("Eye Tracker")

    font = QFont("Inter")
    app.setFont(font)

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor("#18181A"))
    palette.setColor(QPalette.WindowText, QColor("#F3F3F7"))
    palette.setColor(QPalette.Base, QColor("#23232B"))
    palette.setColor(QPalette.Text, QColor("#F3F3F7"))
    palette.setColor(QPalette.Button, QColor("#23232B"))
    palette.setColor(QPalette.ButtonText, QColor("#F3F3F7"))
    app.setPalette(palette)

    gui = RaycastEyeTrackerApp(screen_width, screen_height)

    # --- Connect GUI signals to main loop functions ---
    gui.startTrackingSignal.connect(_start_tracking_handler)
    gui.stopTrackingSignal.connect(_stop_tracking_handler)


    # Start the PyQt event loop.
    sys.exit(app.exec_())

    # --- After GUI closes (only if GUI is completely exited, not minimized) ---
    if _tracking_thread and _tracking_thread.isRunning():
        print("\nMain: GUI closed. Stopping background tracking thread.")
        _stop_tracking_handler()
        _tracking_thread.wait(5000)