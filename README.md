# Eye Tracking Cursor

## Overview
This project enables hands-free control of your computer cursor using only your eyes. It leverages computer vision and deep learning to track your gaze via webcam and move the mouse pointer accordingly. The system is designed for real-time performance and accuracy, using a combination of PyQt5 for the GUI, MediaPipe for facial landmark detection, and custom neural models for gaze estimation.

## How It Works
- **Face and Eye Detection:** Uses MediaPipe to detect your face and extract eye regions from webcam frames.
- **Gaze Estimation:** Custom neural models (see `eye_tracker_models.py`) predict the direction of your gaze.
- **Cursor Control:** The predicted gaze coordinates are mapped to your screen using PyAutoGUI, enabling cursor movement.
- **Calibration:** The app includes a calibration step to adapt to individual users and screen setups.
- **Threaded Processing:** Real-time tracking runs in a background thread for smooth interaction.

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
