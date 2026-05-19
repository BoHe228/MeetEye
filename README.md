# MeetEye

Real-time person localization system based on fisheye panoramic cameras. Detects, tracks, and estimates the azimuth, elevation, and distance of multiple persons in a wide-angle scene, broadcasting results as a structured JSON stream.

---

## System Overview

```
Fisheye Camera
      │  MJPEG over WebSocket
      ▼
┌─────────────────────────────────────────────┐
│              GPU Inference Server            │
│                                             │
│  Fisheye Unwarping (GPU)                    │
│      → Panorama Slicing                     │
│      → Batch YOLO-Pose Detection (GPU)      │
│      → Slice Merge + Deduplication          │
│      → OSNet ReID Feature Extraction (GPU)  │
│      → BoT-SORT Multi-target Tracking       │
│      → Azimuth / Elevation Calculation      │
│      → Distance Estimation (eye keypoints)  │
│                                             │
│  FastAPI  ┌──/ws/inference  (JSON stream)   │
│  WebUI    ├──/video/infer   (MJPEG preview) │
│           └──/              (dashboard)      │
└─────────────────────────────────────────────┘
      │  JSON over WebSocket
      ▼
 Angle Visualizer
  ├── 3D hemisphere  (azimuth + elevation)
  └── 2D radar view  (azimuth + distance, 0–5 m)
```

---

## Key Features

- **GPU-accelerated pipeline** — fisheye unwarping, YOLO inference, and ReID feature extraction all run on GPU; end-to-end latency is typically under 50 ms per frame
- **Panoramic slice detection** — the full 360° panorama is split into overlapping slices for higher detection accuracy; cross-slice and wrap-around duplicates are removed automatically
- **Robust multi-target tracking** — BoT-SORT with IoU + ReID fusion; boundary-crossing matching keeps IDs consistent when persons move across the left/right wrap edge
- **Angle & distance output** — per-target azimuth (°), elevation (°), and estimated distance (m) derived from the inter-eye pixel span using a calibrated formula
- **Real-time WebUI** — annotated MJPEG stream and live JSON inference feed accessible from any browser on the LAN
- **Standalone visualizer** — a separate Matplotlib window subscribes to `/ws/inference` and renders targets on a 3D hemisphere and a 2D top-down radar

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the inference server

```bash
cd mytest
python main_GPU_webui.py \
    --model-path ../yolo26n-pose.engine \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz
```

Open the printed URL in a browser to view the live dashboard.

### 3. Connect a camera client

```bash
python camera_client.py ws://<SERVER_IP>:<PORT>/ws/camera
```

The camera client captures frames from a local fisheye camera (or video file) and streams them to the server via WebSocket.

### 4. Launch the angle visualizer (optional)

```bash
python angle_visualizer.py ws://<SERVER_IP>:<PORT>

# Test with synthetic data (no camera required)
python angle_visualizer.py --test
```

---

## Output JSON Format

Each inference result is broadcast on `/ws/inference`:

```json
{
  "timestamp": 1747612800.123,
  "frame_id": 42,
  "targets": {
    "1": {
      "id": 1,
      "azimuth":        12.5,
      "elevation":       3.1,
      "eye_pixel_dist": 18.4,
      "distance":        2.1,
      "features":       [...]
    }
  }
}
```

| Field            | Unit | Description                              |
|------------------|------|------------------------------------------|
| `azimuth`        | °    | Horizontal angle; 0° = front, CW+       |
| `elevation`      | °    | Vertical angle; 0° = horizontal plane   |
| `eye_pixel_dist` | px   | Inter-eye keypoint pixel distance        |
| `distance`       | m    | Estimated range (0–5 m typical)          |

---

## Project Structure

```
MeetEye/
├── mytest/
│   ├── main_GPU_webui.py      # Inference server entry point
│   ├── camera_client.py       # Camera streaming client
│   ├── angle_visualizer.py    # Real-time angle/distance visualizer
│   ├── config.py              # CLI arguments and defaults
│   ├── core/                  # Detection, tracking, angle, panorama
│   ├── utils/                 # Display, feature extractor, helpers
│   └── webui/                 # FastAPI routes, state, GPU monitor
├── maps/                      # Pre-computed fisheye unwarp maps
├── requirements.txt
└── export_trt.py              # ONNX → TensorRT engine export
```

---

## Experimental Results

> *Results and demo materials will be added here.*

---

## Dependencies

Core: `PyTorch`, `Ultralytics YOLO`, `FastAPI`, `OpenCV`, `aiortc`, `torchreid`

See [`requirements.txt`](requirements.txt) for the full list.
