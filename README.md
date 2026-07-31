# MeetEye

**Real-time multi-person localization for fisheye panoramic and wide-angle cameras.**
MeetEye detects faces/persons, keeps stable IDs, estimates azimuth/elevation/distance, and streams annotated video plus JSON from either a GPU workstation or an RK3588 edge board.

[中文文档 →](README_zh.md)

---

## Demo

> **Large conference room · multi-person meeting** — recall boost + 8-sector aggregation + track coasting  
> HybridSORT tracking · 3-slice panorama · 1280 wide · first 180 s

<!-- To embed: open this README in the GitHub web editor → Edit → drag your local
     docs/demo_coast.mp4 into the edit box → GitHub generates a
     https://github.com/user-attachments/assets/xxxx URL → replace the placeholder
     line below with that URL → save to show the inline player -->
https://github.com/user-attachments/assets/141b880a-fcd8-4ae8-ba29-a69648f7ba5c

<br>

> **↳ Same scene · sector angle visualizer window** — live radar + hemisphere view, each sector's azimuth/elevation rendered in real time (the local `angle_visualizer.py` window paired with the result video above)

<!-- To embed: open this README in the GitHub web editor → drag your local docs/demo_coast_angle.mp4 into the edit box → replace the placeholder line below with the generated user-attachments URL -->
https://github.com/user-attachments/assets/d406b9bf-8586-4573-85f3-2c9957d9e83b

<br>

> **Small conference room · 4 persons · blackboard discussion** — HybridSORT tracking · OSNet ReID · 3-slice panorama · 960 × 630

https://github.com/user-attachments/assets/10e2b8d3-aa76-4ed0-9236-3f568cd06181

*To compress your own result video for GitHub, run ffmpeg directly:*
```bash
ffmpeg -y -i your_result.mp4 -t 180 -vf scale=1280:-2 -c:v libx264 -crf 18 -preset medium -movflags +faststart demo.mp4
```

---

## What It Does

```
Fisheye camera                    Wide-angle camera
      │                                  │
      ▼                                  ▼
Panorama unwrap + 3 slices        Undistort / full-frame letterbox
      │                                  │
      └─────────────── YOLO face/person detection ───────────────┐
                                                                 │
       Cross-slice / same-frame / boundary deduplication          │
                                                                 │
       Native HybridSORT tracking + raw/public/display IDs        │
                                                                 │
       Optional AdaFace RKNN face recognition                     │
       IR18 / IR50 · five-point alignment · dynamic FaceID library│
                                                                 │
       Angle / distance calculation                               │
                                                                 │
       Output: annotated WebUI │ JSON/WebSocket │ JSONL/profile   │
```

---

## Key Features

| Feature | Detail |
|---------|--------|
| **Fisheye and wide-angle input** | Fisheye mode unwraps to a 2560/3840 panorama and uses overlapping slices. Wide-angle mode uses a calibrated full-frame path and disables fisheye-only wrap recovery. |
| **RK3588 C++ edge runtime** | `face_rc/board_cpp` is the current board path: libjpeg-turbo decode, OpenCL remap/letterbox, RKNN inference, native HybridSORT, WebUI, JSONL, profiling, and camera/image-list input. |
| **Tracking stability** | Native tracker includes small residual-box protection, tracker-input deduplication, lost-track reconnect, wrap-aware distance checks, raw duplicate retirement, and output-level coasting. |
| **Display ID slots** | `--display-id-max-count` limits the public-facing IDs. Existing track/display bindings are kept first; new targets replace old slots only after confirmation, primarily by track hit age and bbox area. |
| **Face recognition** | Optional AdaFace IR18/IR50 RKNN recognition with five-point 112x112 alignment, dynamic FaceID library, raw-track binding, tentative carry, relink, and async inference. |
| **WebUI** | Browser dashboard shows annotated video, hardware load, JSON, display slots, FaceID thumbnails, and supports downscaled WebUI JPEG output via `--webui-frame-scale`. |
| **Pipeline execution** | Image/camera decode prefetch queue, fisheye staging pipeline, wide-angle dual YOLO worker mode, async AdaFace worker, and async WebUI JPEG publishing reduce blocking in the main loop. |
| **Angle and distance output** | Per-target azimuth, elevation, distance, eye distance, FaceID, public ID, raw track ID, and display ID are emitted in JSON. |
| **Sector aggregation** | `--sector-output` switches output to sector-level largest-target reporting for downstream control systems that do not need identity continuity. |
| **GPU prototype and tools** | `mytest/` keeps the GPU/WebUI prototype, face-recognition observer, wide-angle testing path, and visualization tools. |
| **YOLO fine-tuning workspace** | Dataset conversion, pose/face fine-tuning, CVAT review, calibration slice export, and TensorRT/RKNN checks live in [`fine-tune/`](fine-tune/). |

---

## Quick Start

### 1 · Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torchreid` must be installed manually if you want OSNet ReID features:
> ```bash
> pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
> ```

---

### 2 · Local mode (`main.py`)

```bash
cd mytest

# Live fisheye camera
python main.py \
    --model-path ../yolo26n-pose.pt \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz

# Video file → save annotated video
python main.py \
    --video-path /path/to/video.mp4 \
    --model-path ../yolo26n-pose.pt \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz \
    --save-video --video-name result.mp4

# Image folder (batch, no display)
python main.py \
    --folder-path /path/to/images/ \
    --model-path  ../yolo26n-pose.pt \
    --map-file    ../maps/3840_fisheye_maps_2026.5.18.npz
```

**Runtime keyboard shortcuts**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save current frame (3 images) |
| `i` | Toggle confidence threshold 0.3 ↔ 0.5 |
| `o` | Toggle IoU threshold 0.3 ↔ 0.45 |
| `a` | Cycle angle display: detail → overview → off |

---

### 3 · WebUI mode

**Server** (GPU machine):
```bash
cd mytest
python main_GPU_webui.py \
    --model-path ../yolo26n-pose.engine \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz
# Open the printed URL in any browser on the LAN
```

**Camera client** (camera machine):
```bash
python camera_client.py ws://<SERVER_IP>:<PORT>/ws/camera
```

**Angle visualizer** (optional, any machine):
```bash
python angle_visualizer.py ws://<SERVER_IP>:<PORT>
# or with synthetic test data
python angle_visualizer.py --test
```

---

### 4 · RK3588 edge deployment

The RK3588 deployment path is maintained under [`face_rc/`](face_rc/). The recommended board runtime is the C++ path in [`face_rc/board_cpp`](face_rc/board_cpp/). It supports fisheye panorama mode and calibrated wide-angle full-frame mode.

Fisheye image-list example:

```bash
cd face_rc/board_cpp

sudo taskset -c 0-6 bash run_smoke.sh \
  --image-list test_frames/meeting_6_4_multi_10min_all_frames.txt \
  --map-dir "maps/meeting_6_4_multi_10min_cpp" \
  --webui \
  --face-rec \
  --face-rec-dynamic-library \
  --face-rec-model-preset ir50_webface4m \
  --face-rec-align-mode five-point \
  --face-rec-async \
  --display-id-max-count 8 \
  --webui-frame-scale 0.5 \
  --no-output-jsonl \
  --no-stdout-json
```

Wide-angle camera example:

```bash
cd face_rc/board_cpp

sudo taskset -c 0-6 bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 2560 \
  --camera-height 1440 \
  --camera-fps 30 \
  --map-dir maps/wide_angle_2k_full_cpp \
  --projection-mode wide-angle \
  --webui \
  --face-rec \
  --face-rec-dynamic-library \
  --face-rec-model-preset ir50_webface4m \
  --face-rec-align-mode five-point \
  --dynamic-face-library-dir face_library_dynamic_wide \
  --no-output-jsonl \
  --no-stdout-json
```

For board setup, map generation, model conversion, lock-frequency checks, profiling reports, and troubleshooting, see [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md).

---

### 5 · YOLO fine-tuning and calibration

Training and calibration utilities are maintained under [`fine-tune/`](fine-tune/). This covers YOLO pose fine-tuning, data conversion, CVAT-assisted correction, meeting-video auto-labeling, calibration-slice generation, TensorRT INT8 export, and JSONL comparison against FP16 baselines.

For the current fine-tuning and calibration flow, see [`fine-tune/README.md`](fine-tune/README.md).

---

## Configuration Reference

### Tracker selection

```bash
--tracker hybridsort   # default; best for dense/crossing scenarios
--tracker botsort      # alternative; better for sparse scenes with stable ReID
--tracker none         # detection only, no tracking
```

### Tracker tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--use-reid` / `--no-use-reid` | `True` | Enable OSNet ReID features in HybridSORT (auto-disabled under `--no-use-osnet` — no features, no appearance association) |
| `--reid-emb-weight-high` | `0.1` | Weight of ReID embedding in HybridSORT first-round cost |
| `--botsort-match-thresh` | `0.3` | BoT-SORT first-stage association threshold |
| `--appearance-thresh` | `0.2` | BoT-SORT ReID gate threshold |
| `--smooth-bbox` / `--no-smooth-bbox` | `True` | EMA smoothing on output bounding boxes |
| `--smooth-bbox-alpha` | `0.5` | EMA coefficient (0 = no smoothing, 1 = frozen) |
| `--coast-frames` | `0` | Keep lost tracks alive with Kalman-predicted box for up to N frames (>0 to enable; thin cyan box, normal boxes untouched) |
| `--kalman-bbox` | off | Output Kalman state box instead of YOLO raw box, and keep showing predicted boxes for lost targets |

### Detection & panorama

| Flag | Default | Description |
|------|---------|-------------|
| `--model-path` | `yolo26n-pose.engine` | YOLO model (`.pt` or `.engine`) |
| `--conf-threshold` | `0.1` | YOLO confidence threshold |
| `--num-slices` | `3` | Panorama sub-images per frame (2–7) |
| `--slice-overlap` | `0.1` | Overlap ratio between adjacent slices |
| `--crop-divisor` | `3` | Crop top `1/N` of panorama (removes fisheye artifacts) |
| `--osnet-model` | `osnet_ain_x1_0` | ReID backbone (`osnet_x1_0`, `osnet_ain_x1_0`, …) |
| `--no-use-osnet` | — | Skip OSNet feature extraction entirely (faster; ReID association auto-disabled) |
| `--kpt-track` | off | Use keypoint-derived box instead of YOLO raw box for tracking (reduces large-box overlap mismatches) |
| `--kpt-display` | off | Use keypoint-derived box for rendering only (does not affect tracking) |

### Recall boost

Run a second detection-only model to recover occluded / back-facing persons the pose model misses, fused in as keypoint-less boxes.

| Flag | Default | Description |
|------|---------|-------------|
| `--recall-boost` | off | Enable recall boost |
| `--recall-model` | `./yolo_model/yolo26n.engine` | Recall detector (use a nano detection model; must differ from the main model) |
| `--recall-conf-threshold` | `0.4` | Recall model's own confidence threshold (independent of `--conf-threshold`) |
| `--recall-match-iou` | `0.3` | Drop a recall box if its IoU with any pose box ≥ this (already covered) |
| `--recall-head-ratio` | `0.12` | For keypoint-less boxes, synthesize the nose point at top + ratio×height |

### Sector aggregation (WebUI mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--sector-output` | off | Switch JSON to sector-aggregated format; pick the largest target per sector and highlight it with a red box |
| `--num-sectors` | `8` | Number of equal sectors over the 360° horizon (e.g. 16) |

### Face ID / speaking detection (optional)

| Flag | Default | Description |
|------|---------|-------------|
| `--use-face-rec` / `--no-use-face-rec` | `False` | Enable AdaFace face recognition in the Python/GPU path; labels identities separately from track IDs |
| `--face-library-dir` | `face_library` | Face gallery dir (one `.npy` per person, filename = name) |
| `--talking-detection` / `--no-talking-detection` | `False` | Enable MediaPipe FaceLandmarker mouth-aspect-ratio (MAR) speaking detection |
| `--talking-mar-threshold` | `0.06` | MAR threshold above which a target is marked speaking |

In `face_rc/board_cpp`, use `--face-rec` instead. The board runtime supports `--face-rec-model-preset ir18_webface4m` and `--face-rec-model-preset ir50_webface4m`; IR18 is the fallback path, while IR50 WebFace4M is the current preferred model for better low-pixel discrimination. Face recognition can be run asynchronously with `--face-rec-async`.

### Board C++ ID/output controls

| Flag | Default | Description |
|------|---------|-------------|
| `--display-id-max-count` | `8` | Maximum public-facing display slots; `0` means unlimited |
| `--display-id-replace-confirm-frames` | `3` | Consecutive frames required before a new target can replace an occupied display slot |
| `--target-output-limit` | `8` | Maximum non-sector JSON targets; `0` means unlimited |
| `--target-output-compact-ids` / `--no-target-output-compact-ids` | compact on | Controls whether JSON target keys are `1..N` or actual display/public IDs |
| `--webui-frame-scale` | `1.0` | Downscale only the WebUI JPEG stream before sending; does not change inference coordinates |

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `--output-dir` | `yolo_pose_output` | Directory for saved videos / frames |
| `--save-video` | off | Save annotated output as `.mp4` (both Local and WebUI; WebUI records to `--video-name` from start, faststart on exit) |
| `--video-name` | auto | Output video filename |
| `--save-frames` | off | Save every frame as JPEG |
| `--save-crops` | off | Save per-person crop images |
| `--save-json` | off | Append every frame's inference result to a JSONL file (WebUI mode only) |
| `--use-dual-windows` | off | Show YOLO-only and tracking windows side by side (Local mode) |

---

## Output JSON Format

Every frame result is broadcast on `/ws/inference` (WebUI mode) and optionally written to a JSON file:

```json
{
  "timestamp": 1747612800.123,
  "frame_id": 42,
  "targets": {
    "1": {
      "id": 1,
      "display_id": 1,
      "public_id": 5,
      "raw_track_id": 7,
      "azimuth": 12.5,
      "elevation": 3.1,
      "eye_pixel_dist": 18.4,
      "distance": 2.1,
      "face_id": "face1",
      "face_recognition": {
        "matched": true,
        "face_id": "face1",
        "score": 0.713,
        "dynamic": true
      }
    }
  }
}
```

| Field | Unit | Description |
|-------|------|-------------|
| `azimuth` | ° | Horizontal angle from camera front; clockwise positive |
| `elevation` | ° | Vertical angle; 0° = horizontal plane, positive upward |
| `eye_pixel_dist` | px | Left–right eye keypoint distance in panorama pixels |
| `distance` | m | Estimated range (calibrated polynomial, 0–5 m typical) |
| `id` / `display_id` | — | Public-facing reusable display slot; bounded by `--display-id-max-count` when enabled |
| `public_id` | — | Long-running output track ID maintained above the native raw tracker |
| `raw_track_id` | — | Native HybridSORT raw track ID; FaceID is primarily bound to this value in `board_cpp` |
| `face_id` | — | AdaFace dynamic/static identity label, or `null` when not available |
| `features` | — | 512-dim L2-normalised OSNet ReID feature vector in the Python/GPU path |

### Sector-aggregated format (`--sector-output`)

When enabled, output is keyed by sector (ID-agnostic) instead of track_id; one entry per sector, with `has_target` indicating whether the sector holds a target this frame:

```json
{
  "timestamp": 1747612800.123,
  "frame_id": 42,
  "num_sectors": 8,
  "sectors": {
    "0": { "has_target": true,  "azimuth": 12.5, "elevation": 3.1 },
    "1": { "has_target": false, "azimuth": null, "elevation": null }
  }
}
```

---

## Project Structure

```
MeetEye/
├── mytest/
│   ├── main.py                  # ① Local standalone entry point
│   ├── config.py                # CLI argument definitions and defaults
│   ├── core/
│   │   ├── panorama.py          # GPU fisheye unwarping (grid_sample)
│   │   ├── detector.py          # YOLOv8 / YOLO26 pose detection wrapper
│   │   ├── slicer.py            # Panorama slicing, cross-slice NMS + ReID merge, recall fusion
│   │   ├── tracker.py           # BoT-SORT and HybridSortTracker wrappers (incl. coasting)
│   │   ├── angle_calculator.py  # Azimuth / elevation / distance estimation
│   │   ├── camera.py            # Camera / video / image-folder input
│   │   └── boundary_matcher.py  # Wrap-around boundary re-ID
│   ├── utils/
│   │   ├── feature_extractor.py # OSNet torchreid wrapper (GPU crop path)
│   │   ├── visualizer.py        # Box / keypoint / angle drawing
│   │   ├── sector.py            # Sector aggregation (shared by WebUI JSON + red-box highlight)
│   │   ├── distance_estimator.py# Head-pose-corrected distance estimation
│   │   ├── talking_detector.py  # MediaPipe mouth-aspect-ratio speaking detection
│   │   └── display.py           # OpenCV display / layout helpers
│   ├── face_rec/                # AdaFace recognition, clustering/debug tools, observer support
│   ├── models/                  # MediaPipe model weights (face_landmarker.task)
│   ├── main_GPU_webui.py        # ② WebUI mode entry point (FastAPI)
│   ├── main_GPU_face_rc_webui.py# GPU WebUI path aligned with board_cpp output logic
│   ├── local_window/            # Camera client and local JSON visualizer
│   └── webui/                   # Inference processor, FastAPI routes, WebSocket, GPU monitor
├── face_rc/
│   ├── board_cpp/               # RK3588 C++ runtime, WebUI, native tracker, RKNN/AdaFace tools
│   │   ├── src/                 # Split C++ modules included by meeteye_cpp_smoke.cpp
│   │   └── tools/               # OpenCL/RKNN/native tracker/AdaFace build and conversion tools
│   └── board/                   # Python board runtime and legacy board-side experiments
├── fine-tune/                   # YOLO fine-tuning, dataset conversion, and calibration utilities
├── HybridSORT/                  # Hybrid-SORT tracker source
│   └── trackers/hybrid_sort_tracker/
│       ├── hybrid_sort.py       # Core tracker + velocity-magnitude gate (patched)
│       ├── hybrid_sort_reid.py  # ReID variant (same patch)
│       └── association.py       # IoU / VDC / TCM association functions
├── maps/                        # Pre-computed fisheye unwarp maps (.npz)
├── export_trt.py                # YOLO ONNX → TensorRT engine export
└── requirements.txt
```

---

## Tracking: Design Decisions & Bug Fixes

### HybridSORT — Velocity-Magnitude Gate

The original HybridSORT VDC (Velocity Direction Consistency) assumes monotonic motion. When a near-stationary person performs a **brief oscillatory movement** (e.g., head lean toward a neighbour and back), the tracker accumulates a stale velocity in the lean direction. On the return movement, VDC penalises the correct match and rewards the wrong one, causing an ID swap.

**Fix** (`hybrid_sort.py`, `hybrid_sort_reid.py`): before updating the velocity vectors `velocity_lt/rt/lb/rb`, measure the centre-to-centre displacement from the oldest reference observation to the current detection, normalised by average bounding-box height.

- Displacement **≥ 5 % of body height** → update velocity normally (continuous motion detected).
- Displacement **< 5 %** → **decay** the existing velocity by × 0.5 per frame instead of overwriting it. After 3–4 such frames the magnitude approaches zero, VDC contribution drops to near zero, and assignment falls back to pure IoU. Dense-crowd tracking is unaffected because dancer-scale motion always exceeds the threshold.

### BoT-SORT — Pre-Assignment Overlap Detection

The original BoT-SORT runs its overlap check on *matched* detection pairs **after** `linear_assignment`, too late to influence the assignment itself. Two additional issues existed:

1. **`fuse_score` confidence bias** — the cost matrix was modulated by detection confidence, so a high-confidence overlapping detection received unfairly low cost for *all* tracks, directly triggering ID swaps.
2. **Contaminated ReID in assignment** — when two bounding boxes overlapped, OSNet crops included the neighbour's body, yet contaminated embeddings were still used to compute the embedding distance that feeds `linear_assignment`.

**Fix** (`tracker.py`): overlap detection (IoU between detection pairs > 0.1 / > 0.3) is now performed **before** cost-matrix construction. Results are used in three places:
- Overlapping detections use `score = 1.0` in the fuse-score step (removes confidence bias).
- Overlapping detection columns in `emb_dists` are forced to 1.0 (excludes contaminated ReID from assignment).
- `freeze_feat` / `near_other` flags for the Kalman and feature-update steps are derived from the same pre-computed sets.

---

## Hardware & Performance

| Setup | Typical latency | FPS |
|-------|-----------------|-----|
| RTX 3080 · YOLO `.engine` · 3 slices · HybridSORT | 30–45 ms | 22–30 |
| RTX 3080 · YOLO `.pt` · 3 slices · HybridSORT | 55–80 ms | 12–18 |
| RK3588 · C++ fisheye · RKNN INT8 · OpenCL direct-slice | See [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md) | Board-dependent |
| RK3588 · C++ wide-angle · dual YOLO workers | See [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md) | Board-dependent |
| CPU only (no GPU) | 300–600 ms | 1–3 |

> Latency breakdown (30-frame average): ①CPU→GPU 2 ms  ②Fisheye unwarping 3 ms  ③GPU→CPU 1 ms  ④Slicing 2 ms  ⑤YOLO 18 ms  ⑥Merge+ReID 8 ms  ⑦Tracking 2 ms  ⑧Angle calc 1 ms

---

## TensorRT Export

```bash
python export_trt.py \
    --model yolo26n-pose.pt \
    --imgsz 1280 \
    --device 0
```

The exported `.engine` file is bound to the GPU it was created on.

---

## Dependencies

| Package | Role |
|---------|------|
| `torch` + `torchvision` | GPU inference, grid_sample unwarping |
| `ultralytics` | YOLOv8 / YOLO26 detection |
| `torchreid` | OSNet ReID feature extraction |
| `opencv-python` | Video I/O, annotation |
| `fastapi` + `uvicorn` | WebUI server |
| `numpy` | Array operations |
| `lap` | Hungarian algorithm for BoT-SORT (optional) |

Full list: [`requirements.txt`](requirements.txt)

---

## License

This project is released for research and educational use.  
HybridSORT source included under its original license (see `HybridSORT/`).
