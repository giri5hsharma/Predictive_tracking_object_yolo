# Real-Time UGV Pan-Tilt Camera Tracking System

## Overview
Developed for **PuneSymbiHackathon Startupcon 5.0**, this project directly addresses the problem statement: *Design and implement a real-time camera tracking + gimbal control system for a moving UGV.*

When a Unmanned Ground Vehicle (UGV) navigates rough terrain, the camera experiences severe vibrations, sudden jolts, and temporary visual occlusions. To solve this, we engineered a closed-loop teleoperation system featuring Human-AI Shared Control. It utilizes computer vision for target acquisition, predictive filtering to track through occlusions, and a heavily damped PID control loop to act as a digital shock absorber for the physical gimbal.

The repository includes the full Python tracking stack, 3D mechanical renders for the pan-tilt mechanism, and a web-based system simulation.

## Key Features
* **Human-AI Shared Control (Teleoperation):** Seamlessly blend autonomous AI tracking with smooth, continuous manual keyboard override. If the UGV encounters complex environments, an operator can manually guide the camera, then release the controls to let the AI resume tracking.
* **Rough-Terrain Predictive Tracking:** Integrates YOLOv8n (Eyes) with a 4-state Kalman Filter (Brain). If the UGV hits a bump and YOLO loses the target for a few frames, the Kalman filter predicts the trajectory, preventing the gimbal from stuttering.
* **Vibration-Resistant PID Control:** Features a custom-tuned Proportional-Integral-Derivative controller designed to glide smoothly to the target, preventing derivative explosions or jitter caused by UGV chassis vibrations.
* **Hardware-Ready Safety Constraints:** Includes physical limit clamping (+/- 45 degrees) and absolute center-zeroing to protect physical servo gears from stripping.
* **Auto-Recovery Protocol:** Implements a 5-second memory timer that automatically returns the gimbal to the center origin if a target is permanently lost.
* **Cross-Platform Support:** Automatically detects the host operating system (Windows, macOS, Linux) and selects the optimal OpenCV camera backend.
* **Startup GUI:** Includes a Tkinter interface to select specific target classes (e.g., Bottle, Apple, Cell Phone) before initializing the camera feed.

## Repository Structure
* `/src`: Contains the main Python tracking script (`gimbal_tracker.py`).
* `/cad_renders`: Includes high-resolution 3D renders of the mechanical design (Base, Pan-Tilt Mechanism, and Camera Mount).
* `/simulation`: Contains the HTML/JS web-based simulation of the tracking environment.

## Requirements & Installation
The software is written in Python and optimized for standard laptops without requiring a dedicated GPU.

1. Clone the repository:
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name

2. Install the required dependencies:
   pip install opencv-python ultralytics numpy

## Usage
Run the main Python script from your terminal. The startup GUI will appear, allowing you to select your target object.

python gimbal_tracker.py

### Command Line Arguments
* `--source 0`: Use the default webcam (Default).
* `--source video.mp4`: Run the tracker on a pre-recorded video file.
* `--demo`: Run a purely synthetic, background simulation without needing a camera.

### Manual Controls
* **W / A / S / D** or **Arrow Keys**: Manually drive the camera gimbal.
* **R**: Reset the PID controller and release manual control.
* **Q / ESC**: Quit the application.

## System Architecture

### 1. The Setpoint (YOLOv8)
The system uses YOLOv8 nano to process frames and identify the bounding boxes of selected targets. This provides the raw pixel coordinates representing the current location of the object.

### 2. The Predictive Model (Kalman Filter)
A 4-state Kalman filter calculates the velocity and trajectory of the target. This is crucial for UGV applications, as it provides a stable tracking vector even when the camera feed is compromised by rapid movement or obstacles.

### 3. The Actuator Simulation (PID Controller)
The system calculates the error between the camera's true Cartesian center and the predicted target location. The PID controller continuously outputs smooth positional adjustments to "drag" the pan-tilt mechanism to the target without overshooting.

## Hardware Integration (Future Scope)
The current software accurately calculates the required Pan and Tilt degrees. The next phase of development involves formatting these values into a serial string (e.g., `<P:15.2,T:-10.5>`) to be transmitted via USB/UART to an embedded microcontroller (like an Arduino) to drive physical PWM servo motors attached to the 3D-printed CAD mounts.
