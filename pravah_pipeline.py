# ============================================================
#  PRaVAH — AI-Powered Real-Time Traffic Intelligence
#  YOLOv8 + ByteTrack | Anomaly Detection | Signal Optimization
#
#  Run locally:
#      python pravah_pipeline.py
#
#  Requirements:
#      pip install -r requirements.txt
# ============================================================

import cv2
import numpy as np
import json
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ultralytics import YOLO


# ── Configuration ────────────────────────────────────────────

VIDEO_PATH  = "traffic1.mp4"
OUTPUT_PATH = "pravah_output.mp4"


@dataclass
class PRaVAHConfig:
    # I/O
    video_path: str = VIDEO_PATH
    output_path: str = OUTPUT_PATH

    # Model — yolov8n (nano) for speed; swap to yolov8s/m for accuracy
    model_weights: str = "yolov8n.pt"

    # Detection settings
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.5

    # COCO class IDs: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck
    vehicle_classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    pedestrian_class: int = 0

    # Speed / stall thresholds (pixels per frame)
    stall_speed_px_per_frame: float = 0.0
    stall_frame_count: int = 25

    # Density thresholds (vehicles per lane)
    density_high: int = 8
    density_medium: int = 4

    # Emergency vehicle: add custom model class IDs if fine-tuned.
    # Leave empty to use color heuristic fallback.
    emergency_class_ids: List[int] = field(default_factory=list)

    # Lane ROIs — auto-computed from video resolution if not set.
    # Format: {"Lane_1": np.array([[x,y], ...]), ...}
    lane_polygons: Dict = field(default_factory=dict)

    # Number of auto-divided lanes (used if lane_polygons not set)
    auto_lanes: int = 2


# ── Lane ROI Helpers ─────────────────────────────────────────

def auto_lane_rois(frame_shape: Tuple, n_lanes: int = 2) -> Dict:
    """
    Divide the bottom 2/3 of the frame into n equal vertical strips.
    Works for most front-facing or overhead traffic camera footage.
    For real deployments, draw ROIs manually using CVAT or LabelMe.
    """
    h, w = frame_shape[:2]
    y_start = h // 3
    lane_w = w // n_lanes
    polygons = {}
    for i in range(n_lanes):
        x0 = i * lane_w
        x1 = (i + 1) * lane_w
        polygons[f"Lane_{i + 1}"] = np.array(
            [[x0, y_start], [x1, y_start], [x1, h], [x0, h]], dtype=np.int32
        )
    return polygons


def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(
        polygon.astype(np.float32), (float(point[0]), float(point[1])), False
    ) >= 0


# ── Layer 3: Track State ─────────────────────────────────────

class TrackState:
    """
    Maintains per-track history across frames.
    After YOLOv8 + ByteTrack assigns a consistent track_id,
    we accumulate center positions and estimate speed from displacement.
    """

    def __init__(self, history_len: int = 30):
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_len))
        self.speeds: Dict[int, Optional[float]] = {}
        self.classes: Dict[int, int] = {}
        self.active_ids: set = set()

    def update(self, track_id: int, center: Tuple[int, int], class_id: int):
        self.history[track_id].append(center)
        self.classes[track_id] = class_id
        self.active_ids.add(track_id)
        self._compute_speed(track_id)

    def _compute_speed(self, track_id: int, lookback: int = 5):
        hist = self.history[track_id]
        if len(hist) >= lookback:
            dx = hist[-1][0] - hist[-lookback][0]
            dy = hist[-1][1] - hist[-lookback][1]
            self.speeds[track_id] = np.sqrt(dx**2 + dy**2) / lookback
        else:
            self.speeds[track_id] = None

    def velocity_angle(self, track_id: int, lookback: int = 5) -> Optional[float]:
        """Direction of motion in degrees (0° = right, 90° = down)."""
        hist = self.history[track_id]
        if len(hist) < lookback:
            return None
        dx = hist[-1][0] - hist[-lookback][0]
        dy = hist[-1][1] - hist[-lookback][1]
        return float(np.degrees(np.arctan2(dy, dx)))

    def clear_stale(self, active_ids: set):
        self.active_ids = active_ids


# ── Layer 4A: Traffic State Estimator ────────────────────────

class TrafficStateEstimator:
    """
    Counts vehicles per lane, estimates queue length and density level.
    Inputs  : list of (track_id, class_id, center) from Layer 3
    Outputs : per-lane dict with count, queue_length, density, avg_speed
    """

    def __init__(self, lane_polygons: Dict, config: PRaVAHConfig):
        self.lane_polygons = lane_polygons
        self.config = config
        self._count_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

    def estimate(self, track_state: TrackState) -> Dict:
        lane_stats = {}

        for lane_name, polygon in self.lane_polygons.items():
            vehicles_in_lane = [
                tid for tid in track_state.active_ids
                if track_state.classes.get(tid) in self.config.vehicle_classes
                and track_state.history[tid]
                and point_in_polygon(track_state.history[tid][-1], polygon)
            ]

            count = len(vehicles_in_lane)
            speeds = [
                track_state.speeds[t]
                for t in vehicles_in_lane
                if track_state.speeds.get(t) is not None
            ]
            avg_speed = float(np.mean(speeds)) if speeds else 0.0

            queue_length = sum(
                1 for t in vehicles_in_lane
                if (track_state.speeds.get(t) or 999.0) < self.config.stall_speed_px_per_frame
            )

            if count >= self.config.density_high:
                density = "HIGH"
            elif count >= self.config.density_medium:
                density = "MEDIUM"
            else:
                density = "LOW"

            self._count_history[lane_name].append(count)

            lane_stats[lane_name] = {
                "count": count,
                "queue_length": queue_length,
                "density": density,
                "avg_speed_px_per_frame": round(avg_speed, 2),
                "vehicle_ids": vehicles_in_lane,
            }

        return lane_stats

    def predict_queue_growth(self, lane_name: str, horizon_frames: int = 30) -> Optional[float]:
        """Linear extrapolation of vehicle count N frames into the future."""
        hist = list(self._count_history[lane_name])
        if len(hist) < 6:
            return None
        recent = hist[-6:]
        trend_per_frame = (recent[-1] - recent[0]) / 5.0
        predicted = recent[-1] + trend_per_frame * horizon_frames
        return max(0.0, round(predicted, 1))


# ── Layer 4B: Anomaly Detector ───────────────────────────────

class AnomalyDetector:
    """
    Rule-based anomaly detection on tracked object states.

    Detects:
    - STALLED_VEHICLE       : near-zero speed for N frames
    - WRONG_WAY             : vehicle moving against expected lane direction
    - CONGESTION_CRITICAL   : sudden high density with long queue
    - PEDESTRIAN_INTRUSION  : pedestrian inside vehicle-only lane ROI
    """

    def __init__(self, config: PRaVAHConfig, lane_polygons: Dict):
        self.config = config
        self.lane_polygons = lane_polygons
        self._stall_counters: Dict[int, int] = defaultdict(int)
        # Optional: set expected direction per lane in degrees
        # e.g. {"Lane_1": 90.0}  (90° = downward flow in top-view camera)
        self.lane_direction_deg: Dict[str, float] = {}

    def detect(self, track_state: TrackState, lane_stats: Dict) -> Dict:
        anomalies: Dict = {}

        for tid in track_state.active_ids:
            cls   = track_state.classes.get(tid)
            speed = track_state.speeds.get(tid)

            # Stalled vehicle
            if cls in self.config.vehicle_classes and speed is not None:
                if speed < self.config.stall_speed_px_per_frame:
                    self._stall_counters[tid] += 1
                else:
                    self._stall_counters[tid] = 0

                if self._stall_counters[tid] >= self.config.stall_frame_count:
                    pos = track_state.history[tid][-1] if track_state.history[tid] else (0, 0)
                    anomalies[f"stall_{tid}"] = {
                        "type": "STALLED_VEHICLE",
                        "severity": "HIGH",
                        "track_id": tid,
                        "stall_frames": self._stall_counters[tid],
                        "position": pos,
                    }

            # Wrong-way detection (only if lane direction is configured)
            for lane_name, expected_deg in self.lane_direction_deg.items():
                polygon = self.lane_polygons.get(lane_name)
                if polygon is None:
                    continue
                if (cls in self.config.vehicle_classes
                        and track_state.history[tid]
                        and point_in_polygon(track_state.history[tid][-1], polygon)):
                    angle = track_state.velocity_angle(tid)
                    if angle is not None:
                        diff = abs(angle - expected_deg)
                        diff = min(diff, 360 - diff)
                        if diff > 120:
                            anomalies[f"wrongway_{tid}"] = {
                                "type": "WRONG_WAY",
                                "severity": "HIGH",
                                "track_id": tid,
                                "lane": lane_name,
                            }

            # Pedestrian intrusion into vehicle lane
            if cls == self.config.pedestrian_class and track_state.history[tid]:
                pos = track_state.history[tid][-1]
                for lane_name, polygon in self.lane_polygons.items():
                    if point_in_polygon(pos, polygon):
                        anomalies[f"ped_intrusion_{tid}"] = {
                            "type": "PEDESTRIAN_INTRUSION",
                            "severity": "MEDIUM",
                            "track_id": tid,
                            "lane": lane_name,
                        }
                        break

        # Lane-level congestion anomaly
        for lane_name, stats in lane_stats.items():
            if stats["density"] == "HIGH" and stats["queue_length"] >= 5:
                anomalies[f"congestion_{lane_name}"] = {
                    "type": "CONGESTION_CRITICAL",
                    "severity": "MEDIUM",
                    "lane": lane_name,
                    "queue_length": stats["queue_length"],
                }

        # Cleanup stale stall counters
        for tid in list(self._stall_counters):
            if tid not in track_state.active_ids:
                del self._stall_counters[tid]

        return anomalies


# ── Layer 4C: Emergency Vehicle Detector ─────────────────────

class EmergencyDetector:
    """
    Two-stage emergency vehicle detection:

    Stage 1 (primary): class_id match if custom model is loaded.
        Train YOLOv8 on: https://universe.roboflow.com/roboflow-100/emergency-vehicles-gkhso
        and add class IDs to config.emergency_class_ids.

    Stage 2 (fallback): dominant red/blue pixel heuristic in the top
        25% of the bounding box — proxy for flashing siren lights.
    """

    def __init__(self, config: PRaVAHConfig):
        self.config = config

    def detect(
        self,
        boxes: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        confidences: np.ndarray,
        frame: np.ndarray,
    ) -> Dict:
        emergency: Dict = {}
        if len(boxes) == 0:
            return emergency

        for box, cls, tid, conf in zip(boxes, class_ids, track_ids, confidences):
            if int(cls) in self.config.emergency_class_ids:
                emergency[int(tid)] = {
                    "method": "model_class",
                    "confidence": float(conf),
                    "box": box.tolist(),
                }
                continue

            if int(cls) in self.config.vehicle_classes:
                x1, y1, x2, y2 = map(int, box)
                height = y2 - y1
                siren_roi = frame[
                    max(0, y1) : max(0, y1 + height // 4),
                    max(0, x1) : max(0, x2),
                ]
                if siren_roi.size > 50 and self._siren_colors_present(siren_roi):
                    emergency[int(tid)] = {
                        "method": "color_heuristic",
                        "confidence": 0.45,
                        "box": box.tolist(),
                    }

        return emergency

    @staticmethod
    def _siren_colors_present(roi: np.ndarray, threshold: float = 0.30) -> bool:
        """Return True if red or blue dominates more than threshold of pixels."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total = roi.shape[0] * roi.shape[1]

        red_lo = cv2.inRange(hsv, np.array([0,   160, 160]), np.array([10,  255, 255]))
        red_hi = cv2.inRange(hsv, np.array([170, 160, 160]), np.array([180, 255, 255]))
        blue   = cv2.inRange(hsv, np.array([100, 130, 100]), np.array([130, 255, 255]))

        red_ratio  = (cv2.countNonZero(red_lo) + cv2.countNonZero(red_hi)) / total
        blue_ratio = cv2.countNonZero(blue) / total
        return red_ratio > threshold or blue_ratio > threshold


# ── Signal Optimizer ──────────────────────────────────────────

class SignalOptimizer:
    """
    Converts Layer 4 outputs into a concrete signal recommendation.

    Priority order:
      EMERGENCY_CORRIDOR > ANOMALY_RESPONSE > ADAPTIVE_GREEN > DEFAULT

    Green time formula: base=10s + 3s × vehicle_count, clamped to [10, 60]
    """

    def recommend(self, lane_stats: Dict, anomalies: Dict, emergency: Dict) -> Dict:

        if emergency:
            tid  = next(iter(emergency))
            info = emergency[tid]
            return {
                "action": "EMERGENCY_CORRIDOR",
                "description": "Clear all — emergency vehicle detected",
                "priority": "CRITICAL",
                "target_lane": None,
                "green_time": None,
                "details": info,
            }

        high_severity = [
            v for v in anomalies.values()
            if isinstance(v, dict) and v.get("severity") == "HIGH"
        ]
        if high_severity:
            return {
                "action": "ANOMALY_ALERT",
                "description": high_severity[0]["type"].replace("_", " ").title(),
                "priority": "HIGH",
                "target_lane": high_severity[0].get("lane"),
                "green_time": 30,
                "details": high_severity[0],
            }

        critical_lanes = [
            v["lane"] for v in anomalies.values()
            if isinstance(v, dict) and v.get("type") == "CONGESTION_CRITICAL"
        ]
        if critical_lanes:
            lane = critical_lanes[0]
            count = lane_stats[lane]["count"]
            green_time = min(10 + count * 3, 60)
            return {
                "action": "EXTEND_GREEN",
                "description": f"Extend green → {lane}",
                "priority": "HIGH",
                "target_lane": lane,
                "green_time": green_time,
            }

        if lane_stats:
            priority_lane = max(lane_stats, key=lambda l: lane_stats[l]["count"])
            count = lane_stats[priority_lane]["count"]
            green_time = min(10 + count * 3, 60)
            return {
                "action": "ADAPTIVE_GREEN",
                "description": f"Green → {priority_lane} ({count} vehicles)",
                "priority": "NORMAL",
                "target_lane": priority_lane,
                "green_time": green_time,
            }

        return {
            "action": "DEFAULT_CYCLE",
            "description": "No significant traffic detected",
            "priority": "LOW",
            "target_lane": None,
            "green_time": 20,
        }


# ── Visualizer ────────────────────────────────────────────────

COCO_NAMES = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

CLASS_COLORS = {
    "car":        (50,  205,  80),
    "motorcycle": (30,  200, 220),
    "bus":        (30,  120, 220),
    "truck":      (20,   70, 200),
    "person":     (210, 100,  30),
}

DENSITY_COLORS = {
    "HIGH":   (0,   0, 120),
    "MEDIUM": (0,  90, 180),
    "LOW":    (0, 130,  40),
}

PRIORITY_COLORS = {
    "CRITICAL": (0,   0, 255),
    "HIGH":     (0, 100, 255),
    "NORMAL":   (0, 200,  80),
    "LOW":      (160, 160, 160),
}


def draw_detections(frame, boxes, class_ids, track_ids,
                    track_state, anomaly_track_ids, emergency_track_ids):
    for box, cls, tid in zip(boxes, class_ids, track_ids):
        x1, y1, x2, y2 = map(int, box)
        label = COCO_NAMES.get(int(cls), str(cls))
        speed = track_state.speeds.get(int(tid))

        if int(tid) in emergency_track_ids:
            color, thickness = (0, 0, 255), 3
            label = "EMERGENCY"
        elif int(tid) in anomaly_track_ids:
            color, thickness = (0, 60, 230), 2
        else:
            color, thickness = CLASS_COLORS.get(label, (180, 180, 180)), 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        spd = f"{speed:.1f}px/f" if speed is not None else "—"
        tag = f"#{tid} {label} {spd}"
        cv2.putText(frame, tag, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

        hist = list(track_state.history[int(tid)])
        for i in range(1, len(hist)):
            cv2.line(frame, hist[i - 1], hist[i], color, 1, cv2.LINE_AA)

    return frame


def draw_lane_overlays(frame, lane_polygons, lane_stats):
    overlay = frame.copy()
    for lane_name, polygon in lane_polygons.items():
        stats = lane_stats.get(lane_name, {})
        color = DENSITY_COLORS.get(stats.get("density", "LOW"), (80, 80, 80))
        cv2.fillPoly(overlay, [polygon], color)
    frame = cv2.addWeighted(overlay, 0.18, frame, 0.82, 0)

    for polygon in lane_polygons.values():
        cv2.polylines(frame, [polygon], True, (180, 180, 180), 1, cv2.LINE_AA)

    for lane_name, polygon in lane_polygons.items():
        stats = lane_stats.get(lane_name, {})
        cx = int(polygon[:, 0].mean())
        cy = int(polygon[:, 1].mean()) - 20
        cv2.putText(frame, lane_name,
                    (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Cnt: {stats.get('count', 0)} | Q: {stats.get('queue_length', 0)}",
                    (cx - 40, cy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 80), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{stats.get('density', '')}",
                    (cx - 20, cy + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 240, 180), 1, cv2.LINE_AA)

    return frame


def draw_dashboard(frame, signal_rec, anomalies, emergency, frame_idx):
    h, w = frame.shape[:2]
    px, py, pw, ph = w - 270, 8, 262, 130
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (18, 18, 18), -1)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (70, 70, 70), 1)

    action   = signal_rec.get("action", "")
    priority = signal_rec.get("priority", "NORMAL")
    desc     = signal_rec.get("description", "")[:36]
    p_color  = PRIORITY_COLORS.get(priority, (160, 160, 160))
    gt       = signal_rec.get("green_time")

    cv2.putText(frame, "PRaVAH DASHBOARD",
                (px + 6, py + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.putText(frame, action,
                (px + 6, py + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, p_color, 2, cv2.LINE_AA)
    cv2.putText(frame, desc,
                (px + 6, py + 53), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    anom_count = len(anomalies)
    anom_color = (0, 80, 255) if anom_count else (80, 220, 80)
    cv2.putText(frame, f"Anomalies detected : {anom_count}",
                (px + 6, py + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.38, anom_color, 1)

    emerg_flag  = bool(emergency)
    emerg_color = (0, 0, 255) if emerg_flag else (80, 220, 80)
    cv2.putText(frame, f"Emergency vehicle  : {'YES' if emerg_flag else 'NO'}",
                (px + 6, py + 89), cv2.FONT_HERSHEY_SIMPLEX, 0.38, emerg_color, 1)

    if gt is not None:
        cv2.putText(frame, f"Recommended green  : {gt}s",
                    (px + 6, py + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 220, 200), 1)

    cv2.putText(frame, f"Frame: {frame_idx}",
                (px + 6, py + 123), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 100), 1)

    return frame


# ── Main Pipeline ─────────────────────────────────────────────

def run_pravah(config: PRaVAHConfig):
    print("Loading YOLOv8 model ...")
    model = YOLO(config.model_weights)

    cap = cv2.VideoCapture(config.video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {config.video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}×{H} @ {fps:.1f} fps | {total_frames} frames")

    out = cv2.VideoWriter(
        config.output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (W, H)
    )

    if not config.lane_polygons:
        config.lane_polygons = auto_lane_rois((H, W), config.auto_lanes)
        print(f"Auto-computed {config.auto_lanes} lane ROIs")

    track_state   = TrackState(history_len=30)
    traffic_est   = TrafficStateEstimator(config.lane_polygons, config)
    anomaly_det   = AnomalyDetector(config, config.lane_polygons)
    emergency_det = EmergencyDetector(config)
    signal_opt    = SignalOptimizer()

    frame_log = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Layer 3: Detection + Tracking
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=config.vehicle_classes + [config.pedestrian_class],
                conf=config.confidence_threshold,
                iou=config.iou_threshold,
                verbose=False,
            )

            boxes     = np.empty((0, 4))
            class_ids = np.array([], dtype=int)
            track_ids = np.array([], dtype=int)
            confs     = np.array([])

            r = results[0]
            if r.boxes is not None and r.boxes.id is not None:
                boxes     = r.boxes.xyxy.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                track_ids = r.boxes.id.cpu().numpy().astype(int)
                confs     = r.boxes.conf.cpu().numpy()

                new_active = set()
                for box, cls, tid in zip(boxes, class_ids, track_ids):
                    cx = int((box[0] + box[2]) / 2)
                    cy = int((box[1] + box[3]) / 2)
                    track_state.update(int(tid), (cx, cy), int(cls))
                    new_active.add(int(tid))
                track_state.clear_stale(new_active)

            # Layer 4A: Traffic state
            lane_stats = traffic_est.estimate(track_state)

            # Layer 4B: Anomaly detection
            anomalies = anomaly_det.detect(track_state, lane_stats)
            anomaly_track_ids = {
                v["track_id"] for v in anomalies.values()
                if isinstance(v, dict) and "track_id" in v
            }

            # Layer 4C: Emergency detection
            emergency = emergency_det.detect(boxes, class_ids, track_ids, confs, frame)

            # Signal recommendation
            signal_rec = signal_opt.recommend(lane_stats, anomalies, emergency)

            # Visualize and write frame
            frame = draw_lane_overlays(frame, config.lane_polygons, lane_stats)
            frame = draw_detections(frame, boxes, class_ids, track_ids,
                                    track_state, anomaly_track_ids, set(emergency.keys()))
            frame = draw_dashboard(frame, signal_rec, anomalies, emergency, frame_idx)
            out.write(frame)

            # Log this frame
            frame_log.append({
                "frame": frame_idx,
                "lane_stats": {
                    k: {kk: vv for kk, vv in v.items() if kk != "vehicle_ids"}
                    for k, v in lane_stats.items()
                },
                "anomaly_count": len(anomalies),
                "anomaly_types": list({v["type"] for v in anomalies.values()
                                       if isinstance(v, dict) and "type" in v}),
                "emergency_detected": bool(emergency),
                "signal_action": signal_rec["action"],
                "signal_priority": signal_rec["priority"],
                "recommended_green_time": signal_rec.get("green_time"),
            })

            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"  Frame {frame_idx}/{total_frames} ({pct:.0f}%)")

    finally:
        cap.release()
        out.release()

    log_path = config.output_path.replace(".mp4", "_log.json")
    with open(log_path, "w") as f:
        json.dump(frame_log, f, indent=2)

    print(f"\n✓ Done. {frame_idx} frames processed.")
    print(f"  Output video : {config.output_path}")
    print(f"  Frame log    : {log_path}")
    _print_summary(frame_log)
    return frame_log


def _print_summary(log: list):
    if not log:
        return
    emergency_frames = sum(1 for f in log if f["emergency_detected"])
    anomaly_frames   = sum(1 for f in log if f["anomaly_count"] > 0)
    all_types = set()
    for f in log:
        all_types.update(f.get("anomaly_types", []))
    actions = {}
    for f in log:
        a = f["signal_action"]
        actions[a] = actions.get(a, 0) + 1

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Total frames          : {len(log)}")
    print(f"  Frames with emergency : {emergency_frames}")
    print(f"  Frames with anomaly   : {anomaly_frames}")
    print(f"  Anomaly types seen    : {all_types or 'None'}")
    print(f"  Signal action dist    : {actions}")
    print("─────────────────────────────────────────────────")


# ── Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    config = PRaVAHConfig(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        model_weights="yolov8n.pt",   # or "yolov8s.pt", "yolov8m.pt"
        auto_lanes=2,
        # Uncomment for 3 lanes:
        # auto_lanes=3,
        #
        # Uncomment if using a fine-tuned emergency vehicle model:
        # emergency_class_ids=[0],
    )
    run_pravah(config)
