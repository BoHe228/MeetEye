# MeetEye

Real-time person localization system based on fisheye panoramic cameras. Detects, tracks, and estimates the azimuth, elevation, and distance of multiple persons in a wide-angle scene. Two entry points are provided: a standalone local mode and a distributed GPU WebUI server mode.

---

## System Overview

### Mode A — Local Standalone (`main.py`)

Suitable for single-machine use with a directly attached camera, video file, or image folder. Results are displayed in local OpenCV windows.

```
Fisheye Camera / Video / Image Folder
      │
      ▼
 Fisheye Unwarping (CPU)
      → Panorama Slicing
      → YOLO-Pose Detection
      → Slice Merge + Deduplication
      → OSNet ReID Feature Extraction
      → BoT-SORT Multi-target Tracking
      → Azimuth / Elevation Calculation
      → Distance Estimation (eye keypoints)
      │
      ▼
 Local OpenCV Display Window
  ├── Annotated panorama (bounding boxes, keypoints, angles)
  └── Optional: YOLO-only window, saved video / frames / crops
```

### Mode B — GPU WebUI Server (`main_GPU_webui.py`)

Separates the camera and the inference server. Any camera on the LAN can stream to the server; results are accessible in a browser or via WebSocket.

```
Fisheye Camera  ──(MJPEG/WebSocket)──▶  GPU Inference Server
                                          │
                                          │  Fisheye Unwarping (GPU)
                                          │  → Panorama Slicing
                                          │  → Batch YOLO-Pose (GPU)
                                          │  → Slice Merge + Deduplication
                                          │  → OSNet ReID (GPU)
                                          │  → BoT-SORT Tracking
                                          │  → Angle & Distance Calculation
                                          │
                                     FastAPI WebUI
                                      ├── /              (dashboard)
                                      ├── /video/infer   (MJPEG preview)
                                      └── /ws/inference  (JSON stream)
                                               │
                                               ▼
                                        Angle Visualizer
                                         ├── 3D hemisphere (azimuth + elevation)
                                         └── 2D radar view (azimuth + distance, 0–5 m)
```

---

## Key Features

- **Two run modes** — local standalone for quick testing; WebUI server for distributed deployment
- **GPU-accelerated pipeline** — fisheye unwarping, YOLO inference, and ReID feature extraction all run on GPU in server mode; end-to-end latency is typically under 50 ms per frame
- **Panoramic slice detection** — the full 360° panorama is split into overlapping slices for higher detection accuracy; cross-slice and wrap-around duplicates are removed automatically
- **Robust multi-target tracking** — BoT-SORT with IoU + ReID fusion; boundary-crossing matching keeps IDs consistent when persons move across the left/right wrap edge
- **Angle & distance output** — per-target azimuth (°), elevation (°), and estimated distance (m) derived from the inter-eye pixel span using a calibrated formula
- **Real-time WebUI** — annotated MJPEG stream and live JSON inference feed accessible from any browser on the LAN
- **Standalone visualizer** — a separate Matplotlib window subscribes to `/ws/inference` and renders targets on a 3D hemisphere and a 2D top-down radar

---

## Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Mode A — Local Standalone

```bash
cd mytest

# From a connected fisheye camera
python main.py --model-path ../yolo26n-pose.pt \
               --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz

# From a video file, save result video
python main.py --video-path /path/to/video.mp4 \
               --model-path ../yolo26n-pose.pt \
               --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz \
               --save-video

# From an image folder (batch processing)
python main.py --folder-path /path/to/images/ \
               --model-path ../yolo26n-pose.pt \
               --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz
```

**Runtime keyboard shortcuts:** `q` quit · `s` save frame · `i` toggle confidence threshold · `o` toggle IOU threshold · `a` cycle angle display mode

---

### Mode B — GPU WebUI Server

**Step 1 — Start the inference server** (on the GPU machine):

```bash
cd mytest
python main_GPU_webui.py \
    --model-path ../yolo26n-pose.engine \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz
```

Open the printed URL in a browser to view the live dashboard.

**Step 2 — Connect a camera client** (on the camera machine):

```bash
python camera_client.py ws://<SERVER_IP>:<PORT>/ws/camera
```

**Step 3 — Launch the angle visualizer** (optional, any machine):

```bash
python angle_visualizer.py ws://<SERVER_IP>:<PORT>

# Test with synthetic data (no camera required)
python angle_visualizer.py --test
```

---

## Output JSON Format

Each inference result is broadcast on `/ws/inference` (Mode B):

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

| Field            | Unit | Description                             |
|------------------|------|-----------------------------------------|
| `azimuth`        | °    | Horizontal angle; 0° = front, CW+      |
| `elevation`      | °    | Vertical angle; 0° = horizontal plane  |
| `eye_pixel_dist` | px   | Inter-eye keypoint pixel distance       |
| `distance`       | m    | Estimated range (0–5 m typical)         |

---

## Project Structure

```
MeetEye/
├── mytest/
│   ├── main.py                # Local standalone entry point
│   ├── main_GPU_webui.py      # GPU WebUI server entry point
│   ├── camera_client.py       # Camera streaming client (Mode B)
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
