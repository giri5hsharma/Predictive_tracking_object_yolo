# Real-Time AI Pan-Tilt Camera Tracking System

## Overview
This repository contains a complete software and hardware-simulation package for a Real-Time Pan-Tilt Camera Tracking System. Developed as a hackathon prototype, this project demonstrates a closed-loop teleoperation system featuring human-AI shared control. 

It utilizes computer vision to detect targets, predictive filtering to track them through occlusions, and a tuned control loop to simulate physical gimbal movements. The repository also includes 3D mechanical renders for the physical pan-tilt mechanism and a web-based simulation of the system.

## Key Features
* **Human-AI Shared Control:** Seamlessly blend autonomous AI tracking with smooth, continuous manual keyboard override.
* **Advanced Target Tracking:** Integrates YOLOv8n for object detection (Eyes) and a 4-state Kalman Filter (Brain) to predict trajectories even when the target is temporarily hidden.
* **Closed-Loop PID Control:** Features a tuned Proportional-Integral-Derivative controller for aggressive, smooth positional tracking without derivative explosion or jitter.
* **Hardware-Ready Safety Constraints:** Includes physical limit clamping (+/- 45 degrees) and absolute zeroing to protect physical servo gears.
* **Auto-Recovery Protocol:** Implements a 5-second memory timer that automatically returns the gimbal to the center origin if a target is lost.
* **Cross-Platform Support:** Automatically detects the host operating system (Windows, macOS, Linux) and selects the optimal OpenCV camera backend to prevent freezes.
* **Startup GUI:** Includes a Tkinter interface to select specific target classes (e.g., Bottle, Apple, Cell Phone) before initializing the camera.

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
A 4-state Kalman filter calculates the velocity and trajectory of the target. If YOLO temporarily loses sight of the object (e.g., it passes behind an obstacle), the Kalman filter continues to output estimated coordinates, preventing the system from freezing.

### 3. The Actuator Simulation (PID Controller)
The blue crosshair represents the physical aim of the camera. The PID controller continuously calculates the error between the camera's current aim and the predictive model, outputting smooth positional adjustments to "drag" the pan-tilt mechanism to the target. 

## Hardware Integration (Future Scope)
The current software accurately calculates the required Pan and Tilt degrees relative to a true Cartesian center. The next phase of development involves formatting these values into a serial string (e.g., `<P:15.2,T:-10.5>`) to be transmitted via USB/UART to an embedded microcontroller (like an Arduino) to drive physical PWM servo motors attached to the 3D-printed CAD mounts.
