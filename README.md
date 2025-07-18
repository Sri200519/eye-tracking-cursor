# Eye Tracking Cursor

## Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Custom GUI & Logic](#custom-gui--logic)
- [Current Status](#current-status)
- [Running Locally](#running-locally)
- [Development](#development)
- [Roadmap & Future Recommendations](#roadmap--future-recommendations)

## Overview
This project enables hands-free control of your computer cursor using only your eyes. It leverages computer vision and deep learning to track your gaze via webcam and move the mouse pointer accordingly. The system is designed for real-time performance and accuracy, using a combination of PyQt5 for the GUI, MediaPipe for facial landmark detection, and custom neural models for gaze estimation.

## How It Works
- **Face and Eye Detection:** Uses MediaPipe to detect your face and extract eye regions from webcam frames.
- **Gaze Estimation:** Custom neural models (see `eye_tracker_models.py`) predict the direction of your gaze.
- **Cursor Control:** The predicted gaze coordinates are mapped to your screen using PyAutoGUI, enabling cursor movement.
- **Calibration:** The app includes a calibration step to adapt to individual users and screen setups.
- **Threaded Processing:** Real-time tracking runs in a background thread for smooth interaction.

## Custom GUI & Logic

The GUI is built with PyQt5 and provides a streamlined, fullscreen experience for calibration and gaze control. It features:
- Frameless, resizable window with custom drag/resize/maximize logic
- Profile management (create, load, manage)
- Calibration workflow with live video and sample collection
- Gesture mapping (blinks, holds to mouse actions)
- State transitions and error handling

See `gui_eye_tracker_app.py` for implementation details.

## ML, Deep Learning & Filtering Logic

The core of the system is a real-time gaze estimation pipeline combining classical computer vision, deep learning, and signal filtering:

### 1. Preprocessing & Eye Extraction
- **Face & Landmark Detection:** MediaPipe detects the user’s face and facial landmarks from the webcam stream.
- **Eye Bounding Box:** Custom logic (`get_eye_bbox`) finds the bounding box for each eye using key landmark indices.
- **Cropping & Enhancement:** The eye region is cropped, resized to 64x64, converted to grayscale, and enhanced using CLAHE for contrast (`crop_and_preprocess_eye`).

### 2. Gaze Estimation Model
- **CNN Backbone:** A custom convolutional network extracts features from the preprocessed eye images.
- **Patch Embedding:** The CNN output is split into patches and embedded for transformer input.
- **Transformer Encoder:** Several layers of self-attention (transformer encoder) model spatial relationships in the eye region, improving robustness to noise and variation.
- **MLP Head:** The transformer output is decoded to predict normalized gaze coordinates (x, y) on the screen.
- **Training:** The model is trained per-user using calibration data collected in the app (supervised regression).

### 3. Filtering & Smoothing
- **Kalman Filter:** A 2D Kalman filter (`KalmanFilter2D`) smooths the predicted gaze coordinates frame-to-frame, reducing jitter and noise for stable cursor movement.
- **Face Stability Check:** The system uses a rolling buffer of iris positions to ensure the face is stable before capturing calibration samples or updating the cursor.

### Pseudocode: Main GUI Logic
```python
# Pseudocode for main event flow
if app starts:
    show welcome/profile screen
    if camera available:
        enable profile actions
    else:
        show error

if create profile:
    validate name
    create directory and model
    switch to setup mode

if start calibration:
    show calibration grid
    collect samples for each point
    show progress
    save calibration data

if drag/resize window:
    update window geometry based on mouse events
```

**For more details, see:**
- `gui_eye_tracker_app.py` (main GUI logic)
- `main.py` (application entry point)
- `eye_tracker_models.py`, `eye_tracker_utils.py` (core model/utils)

## Current Status
- **Supported OS:** Only works on **macOS 15.5 (Sonoma) ARM64** (Apple Silicon) at this time.
- **Python Version:** Python 3.10+ is recommended.
- **Dependencies:** See `requirements.txt`. The dependency tree is complex and tightly coupled to the current macOS/architecture. 

## Running Locally
### 1. Clone the Repository
```bash
git clone https://github.com/Sri200519/eye-tracking-cursor.git
cd eye-tracking-cursor
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv eye_tracker_env
source eye_tracker_env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** If you encounter errors related to `pyobjc` or `mediapipe`, ensure you are on macOS 15.5 ARM64. Other OS/arch are not supported yet.

### 4. Run the Application
```bash
python main.py
```

- The GUI will guide you through calibration and usage.
- Make sure your webcam is connected and accessible.

## Development
- Main entry point: `main.py`
- Core logic: `raycast_eye_tracker_app.py`, `eye_tracker_models.py`, `eye_tracker_utils.py`
- Test scripts and profiles are in their respective folders.

## Roadmap & Future Recommendations
- **Cross-Platform Support:**
    - The biggest challenge is the dependency tree (`pyobjc`, `mediapipe`, and others). Help is welcome to make this work on Intel Macs, Linux, and Windows!
    - If you have experience with dependency management or cross-platform Python packaging, please open an issue or PR.
- **App Packaging:**
    - Package the app as a downloadable `.app` bundle for Mac, and eventually as an `.exe` for Windows and a Linux AppImage.
    - Explore tools like PyInstaller, Briefcase, or PyOxidizer.
- **Performance Improvements:**
    - Optimize model inference and threading for lower latency.
    - Support for external cameras and multi-monitor setups.
- **Accessibility:**
    - Add more gesture support and accessibility features.
    - Improve calibration UX for users with glasses or varied lighting.

## Contributing & Help Needed
- **Dependency Hell:** The current setup is highly specific to macOS 15.5 ARM64. If you can help untangle the dependency tree and make the project more portable, your help would be invaluable!
- **Feature Requests:** Open an issue for any feature ideas or compatibility fixes.

## Contact
If you have questions, suggestions, or want to collaborate, please open an issue or reach out directly.
