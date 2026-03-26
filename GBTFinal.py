"""
=============================================================
 REAL-TIME CAMERA TRACKING SYSTEM — Hackathon Prototype
 Pan-Tilt Gimbal Simulator with Human-AI Shared Control
=============================================================
 Features:
   - YOLOv8 object detection (person + car)
   - Brightness-based target selection
   - Kalman Filter prediction
   - Dead zone + PID controller
   - Human-AI blended control (keyboard override)
   - Target re-acquisition
   - Full simulation (no hardware needed)

 Requirements:
   pip install opencv-python ultralytics numpy
=============================================================
"""

import cv2
import numpy as np
import time
import argparse
import sys
from collections import deque

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CFG = {
    "frame_width": 640,
    "frame_height": 480,
    "detect_interval": 3,
    "confidence_threshold": 0.40,
    "target_classes": [39, 47, 67],  # 39=bottle, 47=apple, 67=cell phone
    "dead_zone_px": 5,
    "pid_kp": 0.15,        # Lowered so it smoothly glides to the target
    "pid_ki": 0.000,       # Turned off to prevent mathematical "windup"
    "pid_kd": 0.005,       # Drastically lowered to prevent the explosion!
    "alpha_decay": 0.05,
    "alpha_rise": 0.6,
    "manual_speed": 8,
    "max_lost_frames": 45,
    "brightness_weight": 0.35,
    "kalman_process_noise": 1e-4,
    "kalman_measurement_noise": 1e-2,
}

# Add this right below CFG so the text labels match the objects!
CLASS_NAMES = {
    39: "bottle",
    47: "apple",
    67: "cell phone"
}

# ─────────────────────────────────────────────────────────────
# KALMAN FILTER
# ─────────────────────────────────────────────────────────────
class KalmanTracker:
    def __init__(self, frame_w, frame_h):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        q = CFG["kalman_process_noise"]
        r = CFG["kalman_measurement_noise"]
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * q
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        cx, cy = frame_w // 2, frame_h // 2
        self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
        self.initialized = False

    def predict(self):
        pred = self.kf.predict()
        px, py = int(pred[0][0]), int(pred[1][0])

        # Prevent the prediction from flying off into infinity when lost
        px = max(0, min(CFG["frame_width"], px))
        py = max(0, min(CFG["frame_height"], py))

        return px, py

    def correct(self, x, y):
        measurement = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measurement)
        self.kf.correct(measurement)
        self.initialized = True

# ─────────────────────────────────────────────────────────────
# PID CONTROLLER
# ─────────────────────────────────────────────────────────────
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(2, dtype=np.float64)
        self.prev_error = np.zeros(2, dtype=np.float64)
        self.last_time = time.time()

    def update(self, error: np.ndarray) -> np.ndarray:
        now = time.time()
        dt = max(now - self.last_time, 1e-3)
        self.last_time = now
        self.integral += error * dt
        self.integral = np.clip(self.integral, -200, 200)
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error.copy()
        return output

    def reset(self):
        self.integral[:] = 0
        self.prev_error[:] = 0

# ─────────────────────────────────────────────────────────────
# TARGET SELECTOR
# ─────────────────────────────────────────────────────────────
def select_target(detections, gray_frame):
    if not detections:
        return None

    scores = []
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        area = max((x2 - x1) * (y2 - y1), 1)
        roi = gray_frame[max(0, y1):y2, max(0, x1):x2]
        brightness = float(np.mean(roi)) / 255.0 if roi.size > 0 else 0.0
        scores.append((area, brightness, conf, det))

    max_area = max(s[0] for s in scores) or 1
    bw = CFG["brightness_weight"]
    best_score = -1
    best_det = None

    for area, brightness, conf, det in scores:
        area_norm = area / max_area
        score = (1 - bw) * area_norm + bw * brightness
        score *= conf
        if score > best_score:
            best_score = score
            best_det = det

    return best_det

# ─────────────────────────────────────────────────────────────
# MAIN TRACKING SYSTEM
# ─────────────────────────────────────────────────────────────
class GimbalTracker:
    def __init__(self, source, use_yolo=True):
        self.W = CFG["frame_width"]
        self.H = CFG["frame_height"]
        self.cx = self.W // 2
        self.cy = self.H // 2

        self.source = source
        self.cap = None
        self.use_yolo = use_yolo
        self.model = None

        if use_yolo:
            self._load_yolo()

        self.kalman = KalmanTracker(self.W, self.H)
        self.pid = PIDController(CFG["pid_kp"], CFG["pid_ki"], CFG["pid_kd"])
        self.gimbal_pos = np.array([self.cx, self.cy], dtype=np.float64)

        # State
        self.frame_count = 0
        self.lost_frames = 0
        self.last_bbox = None
        self.mode = "SEARCH"

        # NEW: Timer to track how long the object has been lost
        self.last_seen_time = time.time()

        self.alpha = 0.0
        self.manual_delta = np.zeros(2, dtype=np.float64)

        # NEW: Memory bank for smooth continuous WASD movement
        self.key_states = {'w': 0, 'a': 0, 's': 0, 'd': 0}

        self.pred_trail = deque(maxlen=40)
        self.actual_trail = deque(maxlen=40)
        self.pan = 0.0
        self.tilt = 0.0
        self.fps_deque = deque(maxlen=30)

    def _load_yolo(self):
        try:
            from ultralytics import YOLO
            print("[INFO] Loading YOLOv8n...")
            self.model = YOLO("yolov8n.pt")
            print("[INFO] YOLO ready.")
        except ImportError:
            print("[WARN] ultralytics not found. Running in DEMO mode.")
            self.use_yolo = False

    def _open_camera(self):
        if self.source == "demo":
            return True
        try:
            self.cap = cv2.VideoCapture(int(self.source))
        except ValueError:
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.source}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.H)
        return True

    def _detect(self, frame):
        if not self.use_yolo or self.model is None:
            return []
        results = self.model(frame, verbose=False, conf=CFG["confidence_threshold"])
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in CFG["target_classes"]:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append([x1, y1, x2, y2, conf, cls_id])
        return detections

    def _generate_demo_object(self, t):
        cx = int(self.W // 2 + 150 * np.sin(t * 0.4))
        cy = int(self.H // 2 + 80 * np.cos(t * 0.6))
        w, h = 90, 130
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        if int(t * 2) % 14 == 0:
            return []
        return [[x1, y1, x2, y2, 0.88, 0]]

    def _handle_keys(self, key):
        now = time.time()

        # 1. Update the timestamp for whichever key is currently pressed
        if key == ord('a') or key == 81: self.key_states['a'] = now
        if key == ord('d') or key == 83: self.key_states['d'] = now
        if key == ord('w') or key == 82: self.key_states['w'] = now
        if key == ord('s') or key == 84: self.key_states['s'] = now

        if key == ord('r'):
            self.pid.reset()
            self.alpha = 0.0
            print("[INFO] Reset.")

        # 2. Calculate smooth movement based on recent keys
        manual_speed = CFG["manual_speed"]
        delta = np.zeros(2, dtype=np.float64)
        moved = False

        # Grace period: 50ms (0.05) bridges the gap between OS keyboard signals
        timeout = 0.01

        if now - self.key_states['a'] < timeout: delta[0] -= manual_speed; moved = True
        if now - self.key_states['d'] < timeout: delta[0] += manual_speed; moved = True
        if now - self.key_states['w'] < timeout: delta[1] -= manual_speed; moved = True
        if now - self.key_states['s'] < timeout: delta[1] += manual_speed; moved = True

        # 3. Apply the movement
        if moved:
            self.manual_delta = delta
            self.alpha = min(1.0, self.alpha + CFG["alpha_rise"])
        else:
            self.manual_delta = np.zeros(2)
            self.alpha = max(0.0, self.alpha - CFG["alpha_decay"])

    def _control_step(self, pred_x, pred_y, confidence):
        # FIX: Calculate error relative to the blue crosshair's current position,
        # NOT the static screen center. This makes it actively follow the target!
        error = np.array([pred_x - self.gimbal_pos[0], pred_y - self.gimbal_pos[1]], dtype=np.float64)

        # If the target is within the dead zone of the crosshair, stop moving
        if np.linalg.norm(error) < CFG["dead_zone_px"]:
            error = np.zeros(2)

        auto_output = self.pid.update(error)
        # FIX: Remove the confidence multiplier so the gimbal faithfully follows
        # the yellow Kalman prediction even when the object is hidden!
        ai_weight = 1.0 - self.alpha
        final = self.alpha * self.manual_delta + ai_weight * auto_output

        # Update position
        self.gimbal_pos += final

        # Constrain the blue crosshair to never leave the screen
        self.gimbal_pos[0] = np.clip(self.gimbal_pos[0], 0, self.W)
        self.gimbal_pos[1] = np.clip(self.gimbal_pos[1], 0, self.H)

        # Calculate Pan and Tilt (Center is 0, Y is inverted so UP = positive)
        raw_pan = (self.gimbal_pos[0] - self.cx) / self.cx * 45.0
        raw_tilt = -(self.gimbal_pos[1] - self.cy) / self.cy * 45.0

        # HARD LIMIT: Clamp the angles to exactly stop at ±45.0 degrees
        self.pan = float(np.clip(raw_pan, -45.0, 45.0))
        self.tilt = float(np.clip(raw_tilt, -45.0, 45.0))

        # HARD CODE: Force exact 0.0 if the crosshair is dead center
        if int(self.gimbal_pos[0]) == self.cx:
            self.pan = 0.0
        if int(self.gimbal_pos[1]) == self.cy:
            self.tilt = 0.0

    def run(self):
        if not self._open_camera():
            return

        window_name = "Gimbal Tracker"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.W + 220, self.H)

        demo_t = 0.0
        last_confidence = 0.0

        # Initialize the lost timer right before we start tracking
        self.last_seen_time = time.time()

        while True:
            t_start = time.time()

            if self.source == "demo":
                frame = self._make_demo_frame(demo_t)
                demo_t += 0.05
                detections_raw = self._generate_demo_object(demo_t)
            else:
                ret, frame = self.cap.read()
                if not ret: break
                frame = cv2.resize(frame, (self.W, self.H))
                detections_raw = []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.frame_count % CFG["detect_interval"] == 0:
                if self.source != "demo":
                    detections_raw = self._detect(frame)

                best = select_target(detections_raw, gray)

                if best is not None:
                    x1, y1, x2, y2, conf, cls_id = best
                    self.kalman.correct((x1 + x2) // 2, (y1 + y2) // 2)
                    self.last_bbox = (x1, y1, x2, y2, conf, cls_id)
                    last_confidence = conf
                    self.lost_frames = 0

                    # Target found! Reset the stopwatch.
                    self.last_seen_time = time.time()

                    self.mode = "MANUAL" if self.alpha > 0.3 else "AUTO"
                else:
                    self.lost_frames += 1
                    last_confidence = max(0.0, last_confidence - 0.05)

                    # Calculate how long the object has been missing
                    time_lost = time.time() - self.last_seen_time

                    if time_lost > 5.0:
                        # 5 SECONDS PASSED: Return to center
                        self.mode = "SEARCH"
                        self.last_bbox = None

                        # Snap the predictive yellow dot back to the absolute center (0,0)
                        # The PID controller will automatically drag the blue crosshair back to it!
                        self.kalman.kf.statePost = np.array([[self.cx], [self.cy], [0], [0]], dtype=np.float32)

                    elif self.lost_frames > 5:
                        self.mode = "RE-ACQUIRE"

            pred_x, pred_y = self.kalman.predict()
            self.pred_trail.append((pred_x, pred_y))

            if self.last_bbox:
                bx1, by1, bx2, by2, _, _ = self.last_bbox
                self.actual_trail.append(((bx1 + bx2) // 2, (by1 + by2) // 2))

            self._control_step(pred_x, pred_y, last_confidence)
            vis = self._draw(frame, pred_x, pred_y, last_confidence)
            sidebar = self._make_sidebar(last_confidence)

            combined = np.hstack([vis, sidebar])
            cv2.imshow(window_name, combined)

            dt = time.time() - t_start
            self.fps_deque.append(1.0 / max(dt, 1e-3))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            self._handle_keys(key)
            self.frame_count += 1

        if self.cap: self.cap.release()
        cv2.destroyAllWindows()

    def _make_demo_frame(self, t):
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        for y in range(self.H):
            val = int(20 + 15 * np.sin(y / self.H * np.pi + t * 0.3))
            frame[y, :] = [val, val + 5, val + 10]
        px = int(self.W // 2 + 150 * np.sin(t * 0.4))
        py = int(self.H // 2 + 80 * np.cos(t * 0.6))
        cv2.rectangle(frame, (px - 45, py - 65), (px + 45, py + 65), (60, 160, 220), -1)
        cv2.circle(frame, (px, py - 75), 22, (200, 160, 100), -1)
        return frame

    def _draw(self, frame, pred_x, pred_y, confidence):
        vis = frame.copy()

        # ── Grid ──
        for i in range(0, self.W, 80): cv2.line(vis, (i, 0), (i, self.H), (20, 20, 20), 1)
        for j in range(0, self.H, 80): cv2.line(vis, (0, j), (self.W, j), (20, 20, 20), 1)

        cv2.putText(vis, "REAL-TIME AI TRACKING SYSTEM", (200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 2)

        # ── Trails ──
        for i in range(1, len(self.pred_trail)):
            cv2.line(vis, self.pred_trail[i-1], self.pred_trail[i], (0,255,255), 1)
        for i in range(1, len(self.actual_trail)):
            cv2.line(vis, self.actual_trail[i-1], self.actual_trail[i], (0,255,120), 1)

        # ── Bounding Box ──
        if self.last_bbox:
            x1, y1, x2, y2, conf, cls_id = self.last_bbox
            color = (0, 255, 120)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), color, 1)

            # Look up the correct name using the ID, default to "object" if missing
            obj_name = CLASS_NAMES.get(cls_id, "object")
            label = f"{obj_name} {conf:.2f}"

            cv2.putText(vis, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(vis, "LOCKED", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── Prediction Pulse ──
        pulse = int(10 + 5*np.sin(time.time()*4))
        cv2.circle(vis, (pred_x, pred_y), 6, (0,255,255), -1)
        cv2.circle(vis, (pred_x, pred_y), pulse, (0,255,255), 1)

        # ── Center Crosshair (True 0,0) ──
        cv2.line(vis, (self.cx-15, self.cy), (self.cx+15, self.cy), (0,255,255), 2)
        cv2.line(vis, (self.cx, self.cy-15), (self.cx, self.cy+15), (0,255,255), 2)

        # ── Gimbal Position & Cartesian Text ──
        gx, gy = int(self.gimbal_pos[0]), int(self.gimbal_pos[1])
        cv2.drawMarker(vis, (gx, gy), (255,120,0), cv2.MARKER_CROSS, 30, 2)

        # Calculate Cartesian coordinates relative to center (0,0)
        cartesian_x = gx - self.cx
        cartesian_y = -(gy - self.cy)  # Negative because OpenCV Y goes down

        # Draw the Cartesian coordinates next to the marker
        cv2.putText(vis, f"C({cartesian_x}, {cartesian_y})",
                (gx + 10, gy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,255), 2)

        # ── Mode Box ──
        cv2.rectangle(vis, (8, 8), (210, 50), (20,20,20), -1)
        cv2.rectangle(vis, (8, 8), (210, 50), (0,255,120), 2)
        cv2.putText(vis, f"MODE: {self.mode}", (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,120), 2)

        return vis
    def _make_sidebar(self, confidence):
        sb = np.zeros((self.H, 220, 3), dtype=np.uint8)
        sb[:] = (12, 12, 16)

        def put(text, row, color=(200, 200, 200), scale=0.5, thick=1):
            cv2.putText(sb, text, (12, row), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)

        # Title
        put("AI TRACKING SYSTEM", 40, (0, 255, 120), 0.7, 2)

        # Pan/Tilt
        put("PAN", 90, (120, 120, 120))
        put(f"{self.pan:+.1f} deg", 115, (0, 200, 255), 0.8, 2)

        put("TILT", 160, (120, 120, 120))
        put(f"{self.tilt:+.1f} deg", 185, (0, 200, 255), 0.8, 2)

        # Confidence Bar
        put("CONFIDENCE", 230, (120, 120, 120))
        bar = int(confidence * 190)
        cv2.rectangle(sb, (12, 240), (200, 255), (40, 40, 40), -1)
        cv2.rectangle(sb, (12, 240), (12 + bar, 255), (0, 255, 120), -1)
        put(f"{confidence:.2f}", 285, (0, 255, 120), 0.6)

        # True Cartesian Position (0,0 at center, Y goes up)
        gx, gy = int(self.gimbal_pos[0]), int(self.gimbal_pos[1])
        cart_x = gx - self.cx
        cart_y = -(gy - self.cy)

        put("POSITION", 330, (120, 120, 120))
        put(f"X:{cart_x} Y:{cart_y}", 355, (200, 200, 255))

        # Status & Telemetry
        status = "TRACKING" if self.lost_frames < 5 else "SEARCHING"
        color = (0, 255, 120) if self.lost_frames < 5 else (255, 150, 0)

        put(f"STATUS: {status}", 390, color, 0.6, 2)
        put(f"LOST: {self.lost_frames}", 420, (255, 120, 80))
        put(f"CONTROL a: {self.alpha:.2f}", 450, (0, 200, 255))

        # FPS Counter
        fps = np.mean(self.fps_deque) if self.fps_deque else 0
        put(f"FPS: {fps:.1f}", 480, (200, 200, 200))

        return sb

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    source = "demo" if args.demo else args.source
    use_yolo = not args.no_yolo and source != "demo"
    tracker = GimbalTracker(source=source, use_yolo=use_yolo)
    tracker.run()