# PRaVAH — AI-Powered Real-Time Traffic Intelligence

> **P**redictive **R**eal-time **A**daptive **V**ehicle & **A**nomaly **H**andler

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/PRaVAH/blob/main/notebooks/pravah_colab.ipynb)
![Python](https://img.shields.io/badge/python-3.8+-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Overview

PRaVAH is a real-time traffic intelligence pipeline built on **YOLOv8 + ByteTrack** that processes any traffic camera video and outputs:

- Annotated video with bounding boxes, track trails, and lane overlays
- Per-frame JSON log of all traffic states and events
- Live signal recommendations based on detected conditions

---

## Architecture

```
Video Input
    └── Layer 3 : YOLOv8 Detection + ByteTrack Tracking
            └── Layer 4A : Traffic State Estimator  (density, queue, avg speed)
            └── Layer 4B : Anomaly Detector          (stall, wrong-way, pedestrian intrusion)
            └── Layer 4C : Emergency Vehicle Detector (model class + siren color heuristic)
                    └── Signal Optimizer → Recommended green time
                            └── Annotated Output Video + JSON Log
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-class detection** | Cars, motorcycles, buses, trucks, pedestrians |
| **Persistent tracking** | ByteTrack assigns consistent IDs across frames |
| **Lane-level analytics** | Per-lane density (LOW / MEDIUM / HIGH), queue length, avg speed |
| **Anomaly detection** | Stalled vehicles, wrong-way driving, pedestrian intrusion, critical congestion |
| **Emergency detection** | Color heuristic (red/blue siren lights) + optional fine-tuned model |
| **Signal optimization** | Rule-based green time recommendation with 4 priority levels |
| **JSON logging** | Per-frame structured log for downstream analytics |

---

## Quick Start

### Option A — Google Colab (recommended, no setup needed)

Click the **Open in Colab** badge above, then:
1. Go to **Runtime → Change runtime type → T4 GPU**
2. Run all cells top to bottom
3. Upload your traffic video when prompted in Cell 3
4. Download the annotated output from Cell 14

### Option B — Run locally

```bash
git clone https://github.com/YOUR_USERNAME/PRaVAH.git
cd PRaVAH
pip install -r requirements.txt
python src/pravah_pipeline.py
```

Change `VIDEO_PATH` at the top of `pravah_pipeline.py` to point to your video.

---

## Configuration

Key parameters in `PRaVAHConfig`:

```python
config = PRaVAHConfig(
    video_path       = "traffic1.mp4",
    model_weights    = "yolov8n.pt",   # nano=fast, yolov8s/m=accurate
    auto_lanes       = 2,              # number of lane ROIs to auto-compute
    confidence_threshold = 0.35,
    stall_frame_count    = 25,         # frames before declaring STALLED
    density_high         = 8,          # vehicles/lane for HIGH density
    density_medium       = 4,
)
```

---

## Output

The pipeline produces two files:

- `pravah_output.mp4` — annotated video with overlays and dashboard
- `pravah_output_log.json` — per-frame structured log:

```json
{
  "frame": 120,
  "lane_stats": {
    "Lane_1": {"count": 6, "queue_length": 2, "density": "MEDIUM", "avg_speed_px_per_frame": 3.4}
  },
  "anomaly_count": 1,
  "anomaly_types": ["STALLED_VEHICLE"],
  "emergency_detected": false,
  "signal_action": "EXTEND_GREEN",
  "recommended_green_time": 28
}
```

---

## Signal Priority Logic

| Priority | Action | Trigger |
|----------|--------|---------|
| CRITICAL | `EMERGENCY_CORRIDOR` | Emergency vehicle detected |
| HIGH | `ANOMALY_ALERT` | Stalled vehicle or wrong-way driver |
| HIGH | `EXTEND_GREEN` | Critical congestion (density=HIGH, queue≥5) |
| NORMAL | `ADAPTIVE_GREEN` | Busiest lane gets priority |
| LOW | `DEFAULT_CYCLE` | No significant traffic |

---

## CAD Design
<img width="323" height="196" alt="image" src="https://github.com/user-attachments/assets/19e8e495-6154-48ad-a51c-a415db0a002b" />

<img width="200" height="238" alt="image" src="https://github.com/user-attachments/assets/0c96e7fc-1975-4ef3-b259-034cf5ee3064" />

---

| Priority | Action | Trigger |
|----------|--------|---------|
| CRITICAL | `EMERGENCY_CORRIDOR` | Emergency vehicle detected |
| HIGH | `ANOMALY_ALERT` | Stalled vehicle or wrong-way driver |
| HIGH | `EXTEND_GREEN` | Critical congestion (density=HIGH, queue≥5) |
| NORMAL | `ADAPTIVE_GREEN` | Busiest lane gets priority |
| LOW | `DEFAULT_CYCLE` | No significant traffic |

---

## Repository Structure

```
PRaVAH/
├── src/
│   └── pravah_pipeline.py       ← standalone script (local use)
├── notebooks/
│   └── pravah_colab.ipynb       ← Colab notebook (cell-by-cell)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Extending PRaVAH

- **Custom lane ROIs** — replace `auto_lane_rois()` with manually drawn polygons using [CVAT](https://cvat.ai) or [LabelMe](https://github.com/labelmeai/labelme)
- **Emergency vehicle model** — fine-tune YOLOv8 on [this dataset](https://universe.roboflow.com/roboflow-100/emergency-vehicles-gkhso) and set `emergency_class_ids`
- **Wrong-way detection** — set `anomaly_det.lane_direction_deg = {"Lane_1": 90.0}` after initialization
- **Better accuracy** — swap `yolov8n.pt` → `yolov8s.pt` or `yolov8m.pt`

---

## License

MIT License — free to use, modify, and distribute.
