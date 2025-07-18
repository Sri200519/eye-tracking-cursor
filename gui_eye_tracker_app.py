import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.optim as optim
import mediapipe as mp
import pyautogui
import os
import shutil 

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QGraphicsDropShadowEffect,
    QSizePolicy, QSpacerItem, QStackedWidget, QInputDialog, QMessageBox,
    QLineEdit, QProgressBar, QScrollArea, QListWidget, QListWidgetItem 
)
from PyQt5.QtCore import Qt, QSize, QTimer, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QPixmap, QImage, QMouseEvent, QCursor

from eye_tracker_models import EyeTransformer, KalmanFilter2D
from eye_tracker_utils import (
    get_eye_bbox, crop_and_preprocess_eye, generate_grid,
    is_face_stable, draw_calibration_dot, eye_aspect_ratio,
    find_working_camera, LEFT_EYE_EAR_IDXS, RIGHT_EYE_EAR_IDXS
)

CALIBRATION_COMPLETE_DATA = None
TRACKING_ACTIVE = False # Managed by main.py based on GUI signals

class RaycastEyeTrackerApp(QWidget):
    startTrackingSignal = pyqtSignal(dict)
    stopTrackingSignal = pyqtSignal()

    _resize_border_width = 8
    _title_bar_height = 40
    _profiles_base_dir = "profiles"

    def __init__(self, screen_w, screen_h):
        super().__init__()
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.setWindowTitle("Eye Tracker Calibration & Setup")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        self.setup_ui()
        self.apply_stylesheet()

        self.showMaximized()
        self.original_window_size = self.size()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
        self.model = None
        self.criterion = None
        self.optimizer = None
        self.model_path = None
        self.calibration_data_path = None

        self.cam_index = find_working_camera()
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera at index {self.cam_index}. Make sure it's not in use.")


        self.setup_video_timer = QTimer(self)
        self.setup_video_timer.timeout.connect(self._update_setup_video_feed)

        self.calibration_video_timer = QTimer(self)
        self.calibration_video_timer.timeout.connect(self._update_calibration_video_feed)


        self.calibration_points = generate_grid(rows=5, cols=7, screen_w=self.screen_w, screen_h=self.screen_h)
        self.samples_per_point = 10
        self.calibration_data = []
        self.calibration_labels = []

        self.calibration_state = {
            "capturing": False,
            "current_point_idx": 0,
            "recorded_samples": 0,
            "total_points": len(self.calibration_points)
        }

        self.gesture_click_map = {}
        self.x_min_calib, self.y_min_calib, self.x_max_calib, self.y_max_calib = 0.0, 0.0, 0.0, 0.0

        self.iris_history = []

        self._dragging_window = False
        self._resizing_window = False
        self._drag_position = QPoint()
        self._resize_direction = Qt.ArrowCursor

        self._show_welcome_screen() # Start on welcome screen

    # --- Window Dragging and Resizing Logic ---
    def mousePressEvent(self, event: QMouseEvent):
        if self.windowState() != Qt.WindowMaximized:
            pos = event.pos()
            rect = self.rect()

            on_left = pos.x() <= self._resize_border_width
            on_right = pos.x() >= rect.width() - self._resize_border_width
            on_top = pos.y() <= self._resize_border_width
            on_bottom = pos.y() >= rect.height() - self._resize_border_width

            if on_top and on_left: self._resize_direction = Qt.SizeFDiagCursor
            elif on_top and on_right: self._resize_direction = Qt.SizeBDiagCursor
            elif on_bottom and on_left: self._resize_direction = Qt.SizeBDiagCursor
            elif on_bottom and on_right: self._resize_direction = Qt.SizeFDiagCursor
            elif on_left: self._resize_direction = Qt.SizeHorCursor
            elif on_right: self._resize_direction = Qt.SizeHorCursor
            elif on_top: self._resize_direction = Qt.SizeVerCursor
            elif on_bottom: self._resize_direction = Qt.SizeVerCursor
            elif pos.y() <= self._title_bar_height: self._resize_direction = Qt.SizeAllCursor
            else: self._resize_direction = Qt.ArrowCursor

            if self._resize_direction != Qt.ArrowCursor and self._resize_direction != Qt.SizeAllCursor:
                self._resizing_window = True
                self._drag_position = event.globalPos()
                event.accept()
            elif event.button() == Qt.LeftButton and pos.y() <= self._title_bar_height:
                self._dragging_window = True
                self._drag_position = event.globalPos() - self.pos()
                event.accept()
            else:
                super().mousePressEvent(event)
        else: 
            if event.button() == Qt.LeftButton and event.pos().y() <= self._title_bar_height:
                 self.showNormal()
                 self.move(event.globalPos() - self.rect().center())
                 self._dragging_window = True
                 self._drag_position = event.globalPos() - self.pos()
                 event.accept()
            else:
                 super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging_window = False
        self._resizing_window = False
        self._resize_direction = Qt.ArrowCursor
        QApplication.restoreOverrideCursor()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.windowState() != Qt.WindowMaximized:
            pos = event.pos()
            rect = self.rect()

            on_left = pos.x() <= self._resize_border_width
            on_right = pos.x() >= rect.width() - self._resize_border_width
            on_top = pos.y() <= self._resize_border_width
            on_bottom = pos.y() >= rect.height() - self._resize_border_width

            if on_top and on_left: self.setCursor(Qt.SizeFDiagCursor)
            elif on_top and on_right: self.setCursor(Qt.SizeBDiagCursor)
            elif on_bottom and on_left: self.setCursor(Qt.SizeBDiagCursor)
            elif on_bottom and on_right: self.setCursor(Qt.SizeFDiagCursor)
            elif on_left or on_right: self.setCursor(Qt.SizeHorCursor)
            elif on_top or on_bottom: self.setCursor(Qt.SizeVerCursor)
            elif pos.y() <= self._title_bar_height: self.setCursor(Qt.SizeAllCursor)
            else: self.setCursor(Qt.ArrowCursor)

        if self._dragging_window and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self._drag_position)
        elif self._resizing_window and event.buttons() == Qt.LeftButton:
            current_geom = self.geometry()
            delta_x = event.globalX() - self._drag_position.x()
            delta_y = event.globalY() - self._drag_position.y()
            new_x, new_y, new_w, new_h = current_geom.x(), current_geom.y(), current_geom.width(), current_geom.height()

            if self._resize_direction == Qt.SizeHorCursor:
                if pos.x() <= self._resize_border_width: new_x += delta_x; new_w -= delta_x
                else: new_w += delta_x
            elif self._resize_direction == Qt.SizeVerCursor:
                if pos.y() <= self._resize_border_width: new_y += delta_y; new_h -= delta_y
                else: new_h += delta_y
            elif self._resize_direction == Qt.SizeFDiagCursor:
                if pos.x() <= self._resize_border_width and pos.y() <= self._resize_border_width: new_x += delta_x; new_w -= delta_x; new_y += delta_y; new_h -= delta_y
                else: new_w += delta_x; new_h += delta_y
            elif self._resize_direction == Qt.SizeBDiagCursor:
                if pos.x() >= rect.width() - self._resize_border_width and pos.y() <= self._resize_border_width: new_w += delta_x; new_y += delta_y; new_h -= delta_y
                else: new_x += delta_x; new_w -= delta_x; new_h += delta_y

            min_width = self.minimumSizeHint().width()
            min_height = self.minimumSizeHint().height()
            if new_w < min_width: new_w = min_width
            if new_h < min_height: new_h = min_height
            self.setGeometry(new_x, new_y, new_w, new_h)
            self._drag_position = event.globalPos()
        else:
            super().mouseMoveEvent(event)
            
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and event.pos().y() <= self._title_bar_height:
            self._toggle_maximize()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
    # --- End Window Dragging and Resizing Logic ---


    # --- Window State Management ---
    def _toggle_maximize(self):
        """Toggles the window state between maximized and normal."""
        if self.windowState() == Qt.WindowMaximized:
            self.setWindowState(Qt.WindowNoState)
            self.showNormal()
            self.setFixedSize(self.original_window_size)
        else:
            self.original_window_size = self.size()
            self.setWindowState(Qt.WindowMaximized)
            self.showMaximized()

    def _show_welcome_screen(self):
        """Switches the stacked widget to the welcome/profile selection screen."""
        self.stacked_widget.setCurrentWidget(self.welcome_screen)
        # Stop calibration video if it was running
        if self.calibration_video_timer.isActive():
            self.calibration_video_timer.stop()
        
        # Start the video feed for the welcome screen
        if self.cap and self.cap.isOpened():
            self.setup_video_timer.start(30)
            self.welcome_status_label.setText("Choose a profile or create a new one to begin.")
            self.create_profile_btn.setEnabled(True)
            self.load_profile_btn.setEnabled(True)
            self.manage_profiles_btn.setEnabled(True)
        else:
            self.welcome_status_label.setText(f"Camera at index {self.cam_index} not found. Please check connection.\nCannot create or load profiles without camera.")
            self.create_profile_btn.setEnabled(False)
            self.load_profile_btn.setEnabled(False)
            self.manage_profiles_btn.setEnabled(False)


    def _show_setup_mode(self):
        """Switches the stacked widget to the calibration setup/gesture mapping screen."""
        self.stacked_widget.setCurrentWidget(self.setup_widgets)
        # Stop calibration video timer if it was running
        if self.calibration_video_timer.isActive():
            self.calibration_video_timer.stop()
        # Ensure setup video timer is running
        if self.cap and self.cap.isOpened() and not self.setup_video_timer.isActive():
            self.setup_video_timer.start(30)
        
        self.status_label_setup_mode.setText(
            f"Profile: {self.profile_name if self.profile_name else '[None Selected]'}\n"
            "Please do calibration in FULL SCREEN mode for best accuracy.\n"
            "Press 'Start Calibration' to begin."
        )

        self.start_calib_btn.setEnabled(self.cap and self.cap.isOpened())


    def _on_create_new_profile(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Camera Not Ready", "Cannot create a new profile without a connected camera. Please ensure your camera is working.")
            return

        while True:
            profile_name, ok = QInputDialog.getText(self, "Create New Profile", "Enter a unique profile name:", QLineEdit.Normal, "")
            if not ok:
                return
            
            if not profile_name.strip():
                QMessageBox.warning(self, "Invalid Name", "Profile name cannot be empty.")
                continue

            sanitized_name = profile_name.strip().replace(' ', '_').lower()
            if not all(c.isalnum() or c == '_' for c in sanitized_name):
                 QMessageBox.warning(self, "Invalid Name", "Profile name must be alphanumeric with optional spaces or underscores.")
                 continue

            self.profile_dir = os.path.join(self._profiles_base_dir, sanitized_name)

            if os.path.exists(self.profile_dir):
                QMessageBox.warning(self, "Profile Exists", f"Profile '{profile_name}' already exists. Please choose a different name or load the existing profile.")
            else:
                try:
                    os.makedirs(self.profile_dir)
                    break
                except OSError as e:
                    QMessageBox.critical(self, "Directory Error", f"Could not create profile directory: {e}")
                    return

        self.profile_name = profile_name
        self.model_path = os.path.join(self.profile_dir, "eye_transformer_model.pt")
        self.calibration_data_path = os.path.join(self.profile_dir, "calibration_data.json")

        self.model = EyeTransformer().to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

        # Reset gesture mappings to default for new profile
        self._set_default_gesture_mappings()

        QMessageBox.information(self, "Profile Created", f"Profile '{self.profile_name}' created. Proceeding to calibration setup.")
        self._show_setup_mode()
        self.start_calib_btn.setText("Start Calibration")
        self.finish_setup_btn.setText("Go Back")
        self.finish_setup_btn.clicked.disconnect()
        self.finish_setup_btn.clicked.connect(self._show_welcome_screen)
        self.save_mappings_btn.hide()


    def _on_load_profile(self):
        if not os.path.exists(self._profiles_base_dir):
            QMessageBox.information(self, "No Profiles", "No existing profiles found. Please create a new profile.")
            return

        available_profiles = [d for d in os.listdir(self._profiles_base_dir) if os.path.isdir(os.path.join(self._profiles_base_dir, d))]
        if not available_profiles:
            QMessageBox.information(self, "No Profiles", "No existing profiles found. Please create a new profile.")
            return

        profile_name, ok = QInputDialog.getItem(self, "Load Profile", "Select a profile:", available_profiles, 0, False)
        if not ok or not profile_name:
            return

        self.profile_name = profile_name
        sanitized_name = profile_name.replace(' ', '_').lower()
        self.profile_dir = os.path.join(self._profiles_base_dir, sanitized_name)
        self.model_path = os.path.join(self.profile_dir, "eye_transformer_model.pt")
        self.calibration_data_path = os.path.join(self.profile_dir, "calibration_data.json")
        self.mappings_path = os.path.join(self.profile_dir, "gesture_mappings.json") # Path for mappings

        try:
            self.model = EyeTransformer().to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device)['model_state_dict'])
            self.model.eval()
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0005)

            with open(self.calibration_data_path, 'r') as f:
                calib_data = json.load(f)
                self.x_min_calib = calib_data['x_min_calib']
                self.y_min_calib = calib_data['y_min_calib']
                self.x_max_calib = calib_data['x_max_calib']
                self.y_max_calib = calib_data['y_max_calib']
            
            # Load gesture mappings
            if os.path.exists(self.mappings_path):
                with open(self.mappings_path, 'r') as f:
                    loaded_mappings = json.load(f)
                    self._set_gesture_mappings_from_dict(loaded_mappings)
            else:
                self._set_default_gesture_mappings() # Set defaults if no saved mappings
                QMessageBox.information(self, "Mappings", "No custom mappings found for this profile. Default mappings loaded.")


            QMessageBox.information(self, "Profile Loaded", f"Profile '{self.profile_name}' loaded successfully. You can now start tracking or re-calibrate.")
            self._show_setup_mode()
            
            self.start_calib_btn.setText("Re-calibrate Profile")
            self.finish_setup_btn.setText("Start Tracking")
            self.finish_setup_btn.clicked.disconnect()
            self.finish_setup_btn.clicked.connect(self._start_tracking_from_setup)
            self.save_mappings_btn.show() 

        except FileNotFoundError:
            QMessageBox.critical(self, "Load Error", "Profile data (model.pt or calibration.json) not found for this profile. It might be incomplete.")
            self.profile_name = None
            self.profile_dir = None
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading profile: {e}")
            self.profile_name = None
            self.profile_dir = None


    def _set_default_gesture_mappings(self):
        """Sets gesture mapping comboboxes to default values."""
        self.gesture_comboboxes["Long Blink"].setCurrentText("Right Click")
        self.gesture_comboboxes["Left Wink"].setCurrentText("Left Click")
        self.gesture_comboboxes["Right Wink"].setCurrentText("None")

    def _set_gesture_mappings_from_dict(self, mappings_dict):
        """Sets gesture mapping comboboxes from a dictionary."""
        for gesture, action in mappings_dict.items():
            if action is None:
                self.gesture_comboboxes[gesture].setCurrentText("None")
            elif action == "left":
                self.gesture_comboboxes[gesture].setCurrentText("Left Click")
            elif action == "right":
                self.gesture_comboboxes[gesture].setCurrentText("Right Click")


    def _on_save_mappings(self):
        """Saves current gesture mappings to the profile directory."""
        if not self.profile_name:
            QMessageBox.warning(self, "No Profile", "Please create or load a profile before saving mappings.")
            return

        current_mappings = self.gesture_comboboxes_to_map()
        self.mappings_path = os.path.join(self.profile_dir, "gesture_mappings.json")
        try:
            with open(self.mappings_path, 'w') as f:
                json.dump(current_mappings, f)
            QMessageBox.information(self, "Mappings Saved", f"Gesture mappings saved for profile '{self.profile_name}'.")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save mappings: {e}")


    def _on_manage_profiles(self):
        """Switches to the profile management screen."""
        self.stacked_widget.setCurrentWidget(self.profile_management_screen)
        self._populate_profile_list() 


    def _populate_profile_list(self):
        """Populates the list widget with available profiles."""
        self.profile_list_widget.clear()
        if not os.path.exists(self._profiles_base_dir):
            os.makedirs(self._profiles_base_dir, exist_ok=True) # Create if not exists
        
        available_profiles = [d for d in os.listdir(self._profiles_base_dir) if os.path.isdir(os.path.join(self._profiles_base_dir, d))]
        
        if not available_profiles:
            item = QListWidgetItem("No profiles found.")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.profile_list_widget.addItem(item)
            self.delete_selected_profile_btn.setEnabled(False) # Disable delete button
            return

        for profile_dir_name in available_profiles:
            item_text = profile_dir_name.replace('_', ' ').title() 
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, profile_dir_name) 
            self.profile_list_widget.addItem(item)
        
        self.delete_selected_profile_btn.setEnabled(False) 

    def _on_profile_list_selection_changed(self):
        """Enables delete button when a profile is selected."""
        if self.profile_list_widget.selectedItems():
            self.delete_selected_profile_btn.setEnabled(True)
        else:
            self.delete_selected_profile_btn.setEnabled(False)


    def _on_delete_selected_profile(self):
        """Deletes the selected profile from disk."""
        selected_items = self.profile_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Selection", "Please select a profile to delete.")
            return

        selected_item = selected_items[0]
        profile_dir_name_to_delete = selected_item.data(Qt.UserRole) # Get the actual directory name
        readable_name = selected_item.text()

        reply = QMessageBox.question(self, "Confirm Delete",
                                     f"Are you sure you want to delete profile '{readable_name}'?\nThis action cannot be undone.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            profile_path = os.path.join(self._profiles_base_dir, profile_dir_name_to_delete)
            try:
                shutil.rmtree(profile_path)
                QMessageBox.information(self, "Profile Deleted", f"Profile '{readable_name}' has been deleted.")
                self._populate_profile_list() # Refresh the list
                # If the deleted profile was the currently loaded one, clear it
                if self.profile_name and self.profile_name.replace(' ', '_').lower() == profile_dir_name_to_delete:
                    self.profile_name = None
                    self.profile_dir = None
            except OSError as e:
                QMessageBox.critical(self, "Delete Error", f"Could not delete profile '{readable_name}': {e}")


    def _start_tracking_from_setup(self):
        if not self.profile_name or not self.model or not (self.x_max_calib and self.y_max_calib):
            QMessageBox.warning(self, "No Profile/Data", "Please ensure a profile is loaded with valid calibration data before starting tracking.")
            return
        
        if not self.cap or not self.cap.isOpened():
             QMessageBox.critical(self, "Camera Not Ready", "Camera is not open. Please ensure your camera is connected and working before starting tracking.")
             return

        if self.setup_video_timer.isActive():
            self.setup_video_timer.stop()

        # Prepare tracking data dictionary
        tracking_data = {
            "gesture_click_map": self.gesture_comboboxes_to_map(),
            "x_min_calib": self.x_min_calib,
            "y_min_calib": self.y_min_calib,
            "x_max_calib": self.x_max_calib,
            "y_max_calib": self.y_max_calib,
            "model_path": self.model_path,
            "cam_index": self.cam_index
        }
        
        # Emit signal to main loop to start tracking
        self.startTrackingSignal.emit(tracking_data)

        # Auto-minimize the GUI
        self.showMinimized()
        self.setCursor(Qt.ArrowCursor)


    def gesture_comboboxes_to_map(self):
        mapping = {}
        gesture_options = ["Long Blink", "Left Wink", "Right Wink"]
        for gesture in gesture_options:
            val = self.gesture_comboboxes[gesture].currentText()
            if val == "Left Click": mapping[gesture] = "left"
            elif val == "Right Click": mapping[gesture] = "right"
            else: mapping[gesture] = None
        return mapping

    def setup_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        window_controls_container = QWidget()
        window_controls_container.setObjectName("windowControlsContainer")
        window_controls_layout = QHBoxLayout(window_controls_container)
        window_controls_layout.setSpacing(7)
        window_controls_layout.setContentsMargins(10, 8, 10, 8)
        
        self.min_close_button = QPushButton()
        self.min_close_button.setFixedSize(12, 12)
        self.min_close_button.setObjectName("closeWindowButton")
        self.min_close_button.clicked.connect(self.close)
        window_controls_layout.addWidget(self.min_close_button)

        self.min_minimize_button = QPushButton()
        self.min_minimize_button.setFixedSize(12, 12)
        self.min_minimize_button.setObjectName("minimizeWindowButton")
        self.min_minimize_button.clicked.connect(self.showMinimized)
        window_controls_layout.addWidget(self.min_minimize_button)

        self.min_maximize_button = QPushButton()
        self.min_maximize_button.setFixedSize(12, 12)
        self.min_maximize_button.setObjectName("maximizeWindowButton")
        self.min_maximize_button.clicked.connect(self._toggle_maximize)
        window_controls_layout.addWidget(self.min_maximize_button)

        window_controls_layout.addStretch(1)

        self.main_layout.addWidget(window_controls_container, alignment=Qt.AlignLeft | Qt.AlignTop)


        self.stacked_widget = QStackedWidget(self)
        self.main_layout.addWidget(self.stacked_widget)
        self.main_layout.setStretchFactor(self.stacked_widget, 1)


        # --- Welcome Screen (Profile Selection) ---
        self.welcome_screen = QWidget()
        self.welcome_screen.setObjectName("welcomeScreen")
        welcome_layout = QVBoxLayout(self.welcome_screen)
        welcome_layout.setContentsMargins(50, 50, 50, 50)
        welcome_layout.setSpacing(20)
        welcome_layout.addStretch(1)

        welcome_header_label = QLabel("Welcome to Eye Tracker")
        welcome_header_label.setObjectName("welcomeHeader")
        welcome_header_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(welcome_header_label)

        self.welcome_status_label = QLabel("Choose a profile or create a new one to begin.")
        self.welcome_status_label.setObjectName("statusLabel")
        self.welcome_status_label.setAlignment(Qt.AlignCenter)
        welcome_layout.addWidget(self.welcome_status_label)

        welcome_layout.addSpacing(50)

        button_container_layout = QHBoxLayout()
        button_container_layout.addStretch(1)

        self.create_profile_btn = QPushButton("Create New Profile")
        self.create_profile_btn.setObjectName("accentButton")
        self.create_profile_btn.clicked.connect(self._on_create_new_profile)
        button_container_layout.addWidget(self.create_profile_btn)

        self.load_profile_btn = QPushButton("Load Existing Profile")
        self.load_profile_btn.setObjectName("controlButton")
        self.load_profile_btn.clicked.connect(self._on_load_profile)
        button_container_layout.addWidget(self.load_profile_btn)

        button_container_layout.addStretch(1)
        welcome_layout.addLayout(button_container_layout)
        
        manage_profiles_button_layout = QHBoxLayout()
        manage_profiles_button_layout.addStretch(1)
        self.manage_profiles_btn = QPushButton("Manage Profiles")
        self.manage_profiles_btn.setObjectName("controlButton")
        self.manage_profiles_btn.clicked.connect(self._on_manage_profiles)
        manage_profiles_button_layout.addWidget(self.manage_profiles_btn)
        manage_profiles_button_layout.addStretch(1)
        welcome_layout.addLayout(manage_profiles_button_layout)


        welcome_layout.addStretch(1)
        
        self.stacked_widget.addWidget(self.welcome_screen)


        # --- Setup Mode Widgets (Calibration Setup and Gesture Mapping) ---
        self.setup_widgets = QWidget(self)
        self.setup_widgets.setObjectName("setupWidgets")
        setup_layout_content = QVBoxLayout(self.setup_widgets)
        setup_layout_content.setContentsMargins(30, 30, 30, 30)
        setup_layout_content.setSpacing(20)

        self.video_label_setup_mode = QLabel("Camera Feed (Loading...)")
        self.video_label_setup_mode.setAlignment(Qt.AlignCenter)
        self.video_label_setup_mode.setFixedSize(500, 320)
        self.video_label_setup_mode.setObjectName("videoFeed")
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(0, 0, 0, 100))
        shadow.setOffset(0, 5)
        self.video_label_setup_mode.setGraphicsEffect(shadow)
        setup_layout_content.addWidget(self.video_label_setup_mode, alignment=Qt.AlignCenter)

        self.status_label_setup_mode = QLabel("Profile: [None Selected]\nPlease do calibration in FULL SCREEN mode for best accuracy.\nPress 'Start Calibration' to begin.")
        self.status_label_setup_mode.setAlignment(Qt.AlignCenter)
        self.status_label_setup_mode.setObjectName("statusLabel")
        setup_layout_content.addWidget(self.status_label_setup_mode)

        setup_layout_content.addSpacing(20)
        gesture_title_label = QLabel("Gesture to Mouse Action Mapping")
        gesture_title_label.setObjectName("sectionTitle")
        gesture_title_label.setAlignment(Qt.AlignCenter)
        setup_layout_content.addWidget(gesture_title_label)

        gesture_mapping_layout = QHBoxLayout()
        gesture_mapping_layout.addStretch()

        mapping_grid_layout = QVBoxLayout()
        mapping_grid_layout.setSpacing(10)

        gesture_options = ["Long Blink", "Left Wink", "Right Wink"]
        click_options = ["Left Click", "Right Click", "None"]
        self.gesture_comboboxes = {}

        for gesture in gesture_options:
            row_layout = QHBoxLayout()
            gesture_label = QLabel(gesture)
            gesture_label.setObjectName("gestureLabel")
            row_layout.addWidget(gesture_label)

            action_combobox = QComboBox()
            action_combobox.addItems(click_options)
            action_combobox.setObjectName("actionCombobox")
            self.gesture_comboboxes[gesture] = action_combobox
            row_layout.addWidget(action_combobox)
            mapping_grid_layout.addLayout(row_layout)

        gesture_mapping_layout.addLayout(mapping_grid_layout)
        gesture_mapping_layout.addStretch()
        setup_layout_content.addLayout(gesture_mapping_layout)
        setup_layout_content.addStretch(1)

        setup_button_layout = QHBoxLayout()
        setup_button_layout.setSpacing(15)
        setup_button_layout.addStretch()

        self.start_calib_btn = QPushButton("Start Calibration")
        self.start_calib_btn.setObjectName("controlButton")
        self.start_calib_btn.clicked.connect(self.start_calibration)
        setup_button_layout.addWidget(self.start_calib_btn)

        self.finish_setup_btn = QPushButton("Go Back")
        self.finish_setup_btn.setObjectName("controlButton")
        self.finish_setup_btn.clicked.connect(self._show_welcome_screen)
        setup_button_layout.addWidget(self.finish_setup_btn)

        self.save_mappings_btn = QPushButton("Save Mappings")
        self.save_mappings_btn.setObjectName("controlButton")
        self.save_mappings_btn.clicked.connect(self._on_save_mappings)
        self.save_mappings_btn.hide()
        setup_button_layout.addWidget(self.save_mappings_btn)


        setup_button_layout.addStretch()
        setup_layout_content.addLayout(setup_button_layout)
        setup_layout_content.addSpacing(20)

        self.stacked_widget.addWidget(self.setup_widgets)


        # --- Profile Management Screen ---
        self.profile_management_screen = QWidget()
        self.profile_management_screen.setObjectName("profileManagementScreen")
        profile_manage_layout = QVBoxLayout(self.profile_management_screen)
        profile_manage_layout.setContentsMargins(50, 50, 50, 50)
        profile_manage_layout.setSpacing(20)

        profile_manage_header = QLabel("Manage Profiles")
        profile_manage_header.setObjectName("welcomeHeader")
        profile_manage_header.setAlignment(Qt.AlignCenter)
        profile_manage_layout.addWidget(profile_manage_header)

        self.profile_list_widget = QListWidget()
        self.profile_list_widget.setObjectName("profileListWidget")
        self.profile_list_widget.setSelectionMode(QListWidget.SingleSelection)
        self.profile_list_widget.itemSelectionChanged.connect(self._on_profile_list_selection_changed)
        profile_manage_layout.addWidget(self.profile_list_widget)

        profile_manage_buttons_layout = QHBoxLayout()
        profile_manage_buttons_layout.addStretch(1)

        self.delete_selected_profile_btn = QPushButton("Delete Selected Profile")
        self.delete_selected_profile_btn.setObjectName("stopButton")
        self.delete_selected_profile_btn.setEnabled(False)
        self.delete_selected_profile_btn.clicked.connect(self._on_delete_selected_profile)
        profile_manage_buttons_layout.addWidget(self.delete_selected_profile_btn)

        profile_manage_buttons_layout.addStretch(1)
        profile_manage_layout.addLayout(profile_manage_buttons_layout)

        profile_manage_back_btn = QPushButton("Back to Welcome")
        profile_manage_back_btn.setObjectName("controlButton")
        profile_manage_back_btn.clicked.connect(self._show_welcome_screen)
        profile_manage_layout.addWidget(profile_manage_back_btn, alignment=Qt.AlignCenter)

        self.stacked_widget.addWidget(self.profile_management_screen)


        # --- Calibration Mode Widgets (Fullscreen View) ---
        self.calibration_view = QWidget(self)
        self.calibration_view.setObjectName("calibrationView")

        calibration_layout = QVBoxLayout(self.calibration_view)
        calibration_layout.setContentsMargins(0, 0, 0, 0)
        calibration_layout.setSpacing(0)
        
        self.fullscreen_video_label = QLabel("Camera Feed (Calibrating...)")
        self.fullscreen_video_label.setAlignment(Qt.AlignCenter)
        self.fullscreen_video_label.setObjectName("fullscreenVideoFeed")
        self.fullscreen_video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.fullscreen_video_label.setMinimumSize(QSize(1, 1))
        calibration_layout.addWidget(self.fullscreen_video_label)

        self.training_progress_bar = QProgressBar(self.calibration_view)
        self.training_progress_bar.setObjectName("trainingProgressBar")
        self.training_progress_bar.setTextVisible(True)
        self.training_progress_bar.setFormat("Training: %p%")
        self.training_progress_bar.hide()
        
        self.capture_btn_overlay = QPushButton("Capture Sample")
        self.capture_btn_overlay.setObjectName("controlButton")
        self.capture_btn_overlay.clicked.connect(self.capture_sample)
        self.capture_btn_overlay.setParent(self.calibration_view)
        self.capture_btn_overlay.hide()

        self.finish_btn_overlay = QPushButton("Finish & Start Tracking")
        self.finish_btn_overlay.setObjectName("accentButton")
        self.finish_btn_overlay.clicked.connect(self.finish_and_start_tracking)
        self.finish_btn_overlay.setParent(self.calibration_view)
        self.finish_btn_overlay.hide()

        self.stop_btn_overlay = QPushButton("Stop & Exit")
        self.stop_btn_overlay.setObjectName("stopButton")
        self.stop_btn_overlay.clicked.connect(self._force_quit_application)
        self.stop_btn_overlay.setParent(self.calibration_view)
        self.stop_btn_overlay.hide()

        self.calibration_status_overlay = QLabel("")
        self.calibration_status_overlay.setObjectName("calibrationStatusOverlay")
        self.calibration_status_overlay.setAlignment(Qt.AlignCenter)
        self.calibration_status_overlay.setParent(self.calibration_view)
        self.calibration_status_overlay.hide()

        self.stacked_widget.addWidget(self.calibration_view)


    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.stacked_widget.currentWidget() == self.calibration_view:
            current_width = self.calibration_view.width()
            current_height = self.calibration_view.height()

            padding = 30
            btn_width = 250
            btn_height = 60
            progress_bar_height = 25
            progress_bar_width = 400

            self.capture_btn_overlay.setGeometry(padding, current_height - btn_height - padding, btn_width, btn_height)
            self.capture_btn_overlay.raise_()

            stop_btn_x = padding + btn_width + 15
            self.stop_btn_overlay.setGeometry(stop_btn_x, current_height - btn_height - padding, btn_width, btn_height)
            self.stop_btn_overlay.raise_()

            self.finish_btn_overlay.setGeometry(current_width - btn_width - padding, current_height - btn_height - padding, btn_width, btn_height)
            self.finish_btn_overlay.raise_()

            status_label_width = 600
            status_label_height = 80
            y_offset_for_top_controls = self._title_bar_height + 10
            self.calibration_status_overlay.setGeometry(
                (current_width - status_label_width) // 2,
                y_offset_for_top_controls,
                status_label_width,
                status_label_height
            )
            self.calibration_status_overlay.raise_()

            self.training_progress_bar.setGeometry(
                (current_width - progress_bar_width) // 2,
                current_height - btn_height - padding - 15 - progress_bar_height,
                progress_bar_width,
                progress_bar_height
            )
            self.training_progress_bar.raise_()
        
        elif self.stacked_widget.currentWidget() == self.profile_management_screen:
            pass 


    def apply_stylesheet(self):
        stylesheet = """
        QWidget {
            background-color: #18181A;
            color: #F3F3F7;
            font-family: 'Inter', sans-serif;
            font-size: 15px;
            border-radius: 12px; /* Applies to main window */
        }

        #welcomeScreen, #setupWidgets, #profileManagementScreen {
            background-color: #18181A;
            border-radius: 0px; /* Pages inside stacked widget won't have rounded corners */
        }

        #welcomeHeader {
            font-size: 36px;
            font-weight: bold;
            color: #F3F3F7;
            margin-bottom: 20px;
        }

        #calibrationView {
            background-color: black;
            border-radius: 0px;
        }

        #fullscreenVideoFeed {
            background-color: black;
            color: white;
            font-size: 24px;
        }

        #calibrationStatusOverlay {
            background-color: rgba(0, 0, 0, 150);
            color: #F3F3F7;
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
        }

        QProgressBar {
            background-color: #23232B;
            color: #F3F3F7;
            border-radius: 8px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #6366F1;
            border-radius: 8px;
        }

        #windowControlsContainer {
            background-color: rgba(0,0,0,0);
            padding-left: 10px;
            padding-top: 8px;
            max-width: 100px;
        }
        #closeWindowButton { background-color: #FF5F56; border-radius: 6px; border: none; }
        #closeWindowButton:hover { background-color: #FF2D2D; }

        #minimizeWindowButton { background-color: #FFBD2E; border-radius: 6px; border: none; }
        #minimizeWindowButton:hover { background-color: #FFA500; }

        #maximizeWindowButton { background-color: #27C93F; border-radius: 6px; border: none; }
        #maximizeWindowButton:hover { background-color: #1AA02B; }

        #videoFeed {
            background-color: #23232B; border-radius: 10px; border: 1px solid #39394B;
            padding: 5px; color: #AAAAAA;
        }
        #statusLabel {
            font-size: 17px; font-weight: bold; color: #C0C0C0; padding: 10px;
        }
        #sectionTitle {
            font-size: 20px; font-weight: bold; color: #F3F3F7;
            padding-bottom: 5px; margin-top: 15px;
        }
        #gestureLabel {
            font-size: 16px; padding-right: 15px; min-width: 120px; text-align: right;
        }
        QComboBox {
            background-color: #23232B; border: 1px solid #39394B; border-radius: 8px;
            padding: 8px 15px; color: #F3F3F7; selection-background-color: #6366F1;
            selection-color: #F3F3F7; min-width: 150px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding; subcontrol-position: top right; width: 25px;
            border-left-width: 1px; border-left-color: #39394B; border-left-style: solid;
            border-top-right-radius: 8px; border-bottom-right-radius: 8px;
        }
        QComboBox::down-arrow { image: url(arrow_down.png); width: 14px; height: 14px; }
        QComboBox QAbstractItemView {
            background-color: #23232B; border: 1px solid #39394B; border-radius: 8px;
            selection-background-color: #6366F1; color: #F3F3F7; padding: 5px;
        }
        QComboBox QAbstractItemView::item { padding: 5px 10px; }

        #controlButton {
            background-color: #23232B; border: none; border-radius: 8px;
            color: #F3F3F7; padding: 12px 25px; font-weight: bold; min-width: 180px;
        }
        #controlButton:hover { background-color: #39394B; }
        #controlButton:pressed { background-color: #505060; }
        #controlButton:disabled { background-color: #1A1A1C; color: #666666; }

        #accentButton {
            background-color: #6366F1; border: none; border-radius: 8px;
            color: #F3F3F7; padding: 12px 25px; font-weight: bold; min-width: 180px;
        }
        #accentButton:hover { background-color: #7B7EF7; }
        #accentButton:pressed { background-color: #5052C8; }
        #accentButton:disabled { background-color: #3C3E96; color: #999999; }

        #stopButton {
            background-color: #DC2626; border: none; border-radius: 8px;
            color: #F3F3F7; padding: 12px 25px; font-weight: bold; min-width: 180px;
        }
        #stopButton:hover { background-color: #EF4444; }
        #stopButton:pressed { background-color: #B91C1C; }

        /* NEW: Profile List Widget Styling */
        #profileListWidget {
            background-color: #23232B;
            border: 1px solid #39394B;
            border-radius: 8px;
            padding: 10px;
            outline: none; /* Remove focus rectangle */
        }
        #profileListWidget::item {
            padding: 8px 10px;
            color: #F3F3F7;
        }
        #profileListWidget::item:selected {
            background-color: #6366F1; /* Accent color on selection */
            color: #F3F3F7;
        }
        #profileListWidget::item:hover {
            background-color: #39394B; /* Darker hover */
        }
        """
        self.setStyleSheet(stylesheet)

        font = QFont("Inter")
        font.setPointSize(15)
        QApplication.instance().setFont(font)


    def _update_video_feed_common(self, target_label):
        if not self.cap or not self.cap.isOpened():
            target_label.setText("No Camera Detected or Failed to Open.")
            return None

        ret, frame = self.cap.read()
        if not ret:
            target_label.setText("Failed to capture frame.")
            return None

        frame = cv2.flip(frame, 1)

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        label_size = target_label.size()
        
        if label_size.width() <= 0 or label_size.height() <= 0:
            return frame

        p = convert_to_qt_format.scaled(label_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        target_label.setPixmap(QPixmap.fromImage(p))
        return frame


    def _update_setup_video_feed(self):
        self._update_video_feed_common(self.video_label_setup_mode)


    def _update_calibration_video_feed(self):
        frame = self._update_video_feed_common(self.fullscreen_video_label)
        
        if frame is not None:
            if self.calibration_state["capturing"] and self.calibration_state["current_point_idx"] < self.calibration_state["total_points"]:
                cx, cy = self.calibration_points[self.calibration_state["current_point_idx"]]
                draw_calibration_dot(frame, cx, cy, self.screen_w, self.screen_h)
                
                rgb_image_with_dot = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image_with_dot.shape
                bytes_per_line = ch * w
                convert_to_qt_format = QImage(rgb_image_with_dot.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                label_size = self.fullscreen_video_label.size()
                if label_size.width() <= 0 or label_size.height() <= 0:
                     return

                p = convert_to_qt_format.scaled(label_size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
                self.fullscreen_video_label.setPixmap(QPixmap.fromImage(p))


    def start_calibration(self):
        if not self.profile_name:
            QMessageBox.warning(self, "No Profile Selected", "Please select or create a profile before starting calibration.")
            return

        self.setup_video_timer.stop()
        
        if not self.cap or not self.cap.isOpened():
            QMessageBox.critical(self, "Camera Error", "Camera is not open. Please check connection and permissions, then restart the application.")
            return

        self.setWindowState(Qt.WindowMaximized)
        self.stacked_widget.setCurrentWidget(self.calibration_view)
        
        self.resizeEvent(None)
        self.capture_btn_overlay.show(); self.capture_btn_overlay.raise_()
        self.finish_btn_overlay.show(); self.finish_btn_overlay.raise_()
        self.stop_btn_overlay.show(); self.stop_btn_overlay.raise_()
        self.calibration_status_overlay.show(); self.calibration_status_overlay.raise_()
        self.training_progress_bar.hide()

        self.capture_btn_overlay.setEnabled(True)
        self.finish_btn_overlay.setEnabled(False)

        self.calibration_state["capturing"] = True
        self.calibration_state["current_point_idx"] = 0
        self.calibration_state["recorded_samples"] = 0
        self.calibration_data = []
        self.calibration_labels = []
        self.iris_history = []

        self.calibration_video_timer.start(30)

        self.update_calibration_status_text()


    def update_calibration_status_text(self):
        self.calibration_status_overlay.setText(
            f"Profile: {self.profile_name}\n"
            f"Calibration in progress: Point {self.calibration_state['current_point_idx']+1}/"
            f"{self.calibration_state['total_points']}, Sample {self.calibration_state['recorded_samples']}/"
            f"{self.samples_per_point}\nLook at the red dot."
        )


    def capture_sample(self):
        if not self.calibration_state["capturing"]:
            self.calibration_status_overlay.setText("Please start calibration first.")
            return

        ret, frame = self.cap.read()
        if not ret:
            self.calibration_status_overlay.setText("Failed to capture frame for sample. Try again.")
            return
        
        frame = cv2.flip(frame, 1)
        frame_h, frame_w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            if len(landmarks) > 33 and len(landmarks) > 263:
                center_x = (landmarks[33].x + landmarks[263].x) / 2 * frame_w
                center_y = (landmarks[33].y + landmarks[263].y) / 2 * frame_h
                self.iris_history.append((center_x, center_y))
                if len(self.iris_history) > 10:
                    self.iris_history.pop(0)

            if not is_face_stable(self.iris_history):
                self.calibration_status_overlay.setText("Face unstable. Keep your head still and try capturing again.")
                return

            bbox_left = get_eye_bbox(landmarks, frame_w, frame_h, 'left')
            bbox_right = get_eye_bbox(landmarks, frame_w, frame_h, 'right')
            
            left_eye_img = crop_and_preprocess_eye(frame, bbox_left)
            right_eye_img = crop_and_preprocess_eye(frame, bbox_right)
            
            eye_input = np.concatenate([left_eye_img, right_eye_img], axis=0)
            
            cx, cy = self.calibration_points[self.calibration_state["current_point_idx"]]
            
            self.calibration_data.append(eye_input)
            self.calibration_labels.append([cx, cy])
            
            self.calibration_state["recorded_samples"] += 1

            if self.calibration_state["recorded_samples"] >= self.samples_per_point:
                self.calibration_state["current_point_idx"] += 1
                self.calibration_state["recorded_samples"] = 0

            self.update_calibration_status_text()

            if self.calibration_state["current_point_idx"] >= self.calibration_state["total_points"]:
                self.calibration_state["capturing"] = False
                self.capture_btn_overlay.setEnabled(False)
                self.finish_btn_overlay.setEnabled(True)
                self.stop_btn_overlay.hide()
                self.calibration_status_overlay.setText("Calibration data collected. Click 'Finish & Start Tracking'.")
                self.calibration_video_timer.stop()
                self.cap.release()
                
        else:
            self.calibration_status_overlay.setText("Face not detected. Adjust camera or your position and try again.")


    def finish_and_start_tracking(self):
        if len(self.calibration_data) < self.samples_per_point * self.calibration_state["total_points"]:
            QMessageBox.warning(self, "Incomplete Calibration", "Not enough calibration data. Please capture all samples for all points before finishing.")
            return

        self.calibration_status_overlay.setText(f"Profile: {self.profile_name}\nInitiating model training... Please wait.")
        self.training_progress_bar.show()
        self.training_progress_bar.setRange(0, 800)
        self.training_progress_bar.setValue(0)
        QApplication.processEvents()

        print("Initiating model training...")
        X = torch.tensor(np.array(self.calibration_data), dtype=torch.float32).to(self.device)
        y = torch.tensor(np.array(self.calibration_labels), dtype=torch.float32).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
        
        self.model.train()
        epochs = 800
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            self.training_progress_bar.setValue(epoch + 1)
            QApplication.processEvents()
            
            if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} loss: {total_loss / len(loader):.6f}")

        self.training_progress_bar.hide()
        QApplication.processEvents()

        torch.save({'model_state_dict': self.model.state_dict()}, self.model_path)
        print(f"Trained model saved to {self.model_path}")

        calib_array = np.array(self.calibration_labels)
        self.x_min_calib = float(calib_array[:, 0].min())
        self.y_min_calib = float(calib_array[:, 1].min())
        self.x_max_calib = float(calib_array[:, 0].max())
        self.y_max_calib = float(calib_array[:, 1].max())

        with open(self.calibration_data_path, 'w') as f:
            json.dump({
                'x_min_calib': self.x_min_calib,
                'y_min_calib': self.y_min_calib,
                'x_max_calib': self.x_max_calib,
                'y_max_calib': self.y_max_calib
            }, f)
        print(f"Calibration data bounds saved to {self.calibration_data_path}")

        gesture_options = ["Long Blink", "Left Wink", "Right Wink"]
        for gesture in gesture_options:
            val = self.gesture_comboboxes[gesture].currentText()
            if val == "Left Click":
                self.gesture_click_map[gesture] = "left"
            elif val == "Right Click":
                self.gesture_click_map[gesture] = "right"
            else:
                self.gesture_click_map[gesture] = None
        
        print(f"Final Gesture Mapping: {self.gesture_click_map}")

        self.calibration_status_overlay.setText("Setup complete. Starting eye tracking!")
        
        # --- After training, switch to preview mode and auto-minimize ---
        self.setWindowState(Qt.WindowNoState)
        self.showNormal()
        self.setFixedSize(self.original_window_size)
        self.setCursor(Qt.ArrowCursor)

        self.calibration_video_timer.stop()

        self._show_setup_mode()
        self.status_label_setup_mode.setText(f"Profile: {self.profile_name}\nCalibration and Training Complete.\nReady for Tracking!")
        
        self.start_calib_btn.setText("Start Tracking")
        self.start_calib_btn.clicked.disconnect()
        self.start_calib_btn.clicked.connect(self._start_tracking_from_setup)
        
        self.finish_setup_btn.setText("New/Load Profile")
        self.finish_setup_btn.clicked.disconnect()
        self.finish_setup_btn.clicked.connect(self._show_welcome_screen)

        self.save_mappings_btn.show()

        # --- Auto-minimize the application ---
        self.showMinimized()

        global CALIBRATION_COMPLETE_DATA
        CALIBRATION_COMPLETE_DATA = {
            "gesture_click_map": self.gesture_comboboxes_to_map(),
            "x_min_calib": self.x_min_calib,
            "y_min_calib": self.y_min_calib,
            "x_max_calib": self.x_max_calib,
            "y_max_calib": self.y_max_calib,
            "model_path": self.model_path,
            "cam_index": self.cam_index
        }
        self.startTrackingSignal.emit(CALIBRATION_COMPLETE_DATA)


    def _force_quit_application(self):
        print("Stop button pressed. Force quitting application.")
        
        if self.setup_video_timer.isActive():
            self.setup_video_timer.stop()
        if self.calibration_video_timer.isActive():
            self.calibration_video_timer.stop()
        
        if self.cap and self.cap.isOpened():
            self.cap.release()
        
        global TRACKING_ACTIVE
        if TRACKING_ACTIVE:
            self.stopTrackingSignal.emit()
        
        QApplication.instance().quit()
        sys.exit(0)


    def closeEvent(self, event):
        global TRACKING_ACTIVE
        if TRACKING_ACTIVE:
            self.stopTrackingSignal.emit()
        
        if self.setup_video_timer.isActive():
            self.setup_video_timer.stop()
        if self.calibration_video_timer.isActive():
            self.calibration_video_timer.stop()
        if self.cap and self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)