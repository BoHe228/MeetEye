# MeetEye board_cpp minimal runtime

This directory is a minimal C++ smoke runtime for the RK3588 board path:

1. read one JPEG/MJPEG frame with libjpeg-turbo
2. decode it to BGR in CPU memory
3. run existing `libdirect_slice_opencl_fused.so` to produce RGB NHWC RKNN inputs
4. run existing `librknn_capi_parallel.so`
5. run existing `libhybrid_sort_native.so` for HybridSORT tracking
6. run C++ HybridSORT post-tracking cleanup
7. print Python-compatible `targets` inference JSON

It intentionally does not use OpenCV C++, torch, ultralytics, scipy, or Python in
the board runtime path.

## Runtime Module Chain

```text
JPEG/MJPEG frame or V4L2 camera
      |
      v
libjpeg-turbo decoder
      |
      v
meeteye_cpp_smoke.cpp
      |
      +--> libdirect_slice_opencl_fused.so
      |        fisheye/base-map lookup + three direct slices + RKNN tensor layout
      |
      +--> librknn_capi_parallel.so
      |        RKNN C API inference on RK3588 NPU
      |
      +--> C++ decode/merge path
      |        raw outputs -> slice merge -> final boundary duplicate suppression
      |
      +--> libhybrid_sort_native.so
      |        native HybridSORT association and track state
      |
      v
targets JSON / sector JSON / JSONL / WebSocket / profile summary
```

The runtime does not link `libmerge_fast.so` or `libtracker_assoc_fast.so`
directly in this C++ path. Those libraries are still useful for the Python board
runtime and for older native acceleration experiments.

## Prepare Map On Firefly Or Another Normal Linux/Aarch64 Machine

Preferred path: generate the direct-slice map from a camera or video and export
the C++ map files in the same step.

From `MeetEye`:

```bash
python3 face_rc/tools/generate_yolo_slice_maps.py \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-format mjpeg \
  --output-map-file face_rc/board/maps/6.22_2560_yolo_slices_640.npz \
  --cpp-output-dir face_rc/board_cpp/map_export
```

For a video source:

```bash
python3 face_rc/tools/generate_yolo_slice_maps.py \
  --video-path data/大会议室_6.4_多人开会_40秒.mp4 \
  --output-map-file face_rc/board/maps/6.22_2560_yolo_slices_640.npz \
  --cpp-output-dir face_rc/board_cpp/map_export
```

Compatibility path: if the `.npz` map already exists, export only the C++ files:

```bash
cd MeetEye/face_rc/board_cpp

python3 export_direct_slice_map.py \
  --input ../board/maps/6.22_2560_yolo_slices_640.npz \
  --output map_export
```

This repository copy already includes a packaged map at:

```text
maps/6.22_2560_yolo_slices_640_cpp/
```

## Build

From `MeetEye/face_rc/board_cpp`:

```bash
bash build.sh
```

If the system only has `libturbojpeg.so.0` and no development symlink, the build
script links directly to `/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0`.

## Run

```bash
bash run_smoke.sh --image frame.jpg
```

Run multiple JPEG frames with one RKNN/OpenCL initialization. `--output-jsonl`
now writes the same high-level inference shape as `board/main.py`:
`timestamp`, `frame_id`, and `targets`.

```bash
bash run_smoke.sh \
  --image-list test_frames/frames.txt \
  --track-buffer 120 \
  --output-jsonl board_cpp_results.jsonl \
  --no-stdout-json
```

Write the full debug detection/tracker JSONL at the same time:

```bash
bash run_smoke.sh \
  --image-list test_frames/frames.txt \
  --track-buffer 120 \
  --decode-prefetch \
  --output-jsonl board_cpp_targets.jsonl \
  --debug-jsonl board_cpp_debug.jsonl \
  --print-profile-summary \
  --no-stdout-json
```

Debug JSONL includes runtime FPS fields under both `fps` and `timings_ms`:
`fps.instant`, `fps.average`, and `fps.frame_ms`.
Add `--print-profile-summary` to print per-stage average and max timings to the
terminal on stderr; `--profile-interval N` controls running summaries, and the
final summary is always printed when the flag is enabled.
For image-list profiling, `--decode-prefetch` runs JPEG file read/decode in a
worker thread so decode can overlap RKNN/OpenCL work. In that mode,
`jpeg_decode` is the worker's real decode time and `decode_wait` is how long the
main inference thread waited for a decoded frame.

Run a V4L2 MJPEG camera directly:

```bash
bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --track-buffer 120 \
  --max-frames 30 \
  --output-jsonl camera_results.jsonl \
  --no-stdout-json
```

By default the C++ runtime also starts a lightweight JSON WebSocket server that
matches the Python board path:

```text
ws://<board-ip>:8001/ws/inference
```

Use these flags to change or disable it:

```bash
--ws-host 0.0.0.0
--ws-port 8001
--no-ws-json
```

The server sends the same high-level `targets` payload as stdout/JSONL. Frames
are sent as WebSocket binary messages, matching Python's `send_bytes()` behavior.
For offline image-list comparisons, add `--no-ws-json` if you do not need a live
subscriber or if port `8001` is already occupied.

Python-compatible sector aggregate output is also supported. With
`--sector-output`, stdout, JSONL, and WebSocket payloads switch from `targets` to
the Python board shape:

```json
{"timestamp": 0.0, "frame_id": 1, "num_sectors": 8, "sectors": {"0": {"has_target": false, "azimuth": null, "elevation": null}}}
```

Each sector represents `360 / num_sectors` degrees. If multiple targets fall in
the same sector, the target with the largest bbox area is used, matching
`board/utils/sector.py`.

```bash
bash run_smoke.sh \
  --image-list test_frames/frames_3s.txt \
  --track-buffer 120 \
  --sector-output \
  --num-sectors 8 \
  --ws-host 0.0.0.0 \
  --ws-port 8001 \
  --no-output-jsonl \
  --no-stdout-json
```

Native HybridSORT tracking is enabled by default. Use `--no-tracker` only when
you want to isolate JPEG decode, OpenCL preprocessing, and RKNN inference.

The C++ tracker output also runs a conservative final spatial/boundary duplicate
suppression pass by default. It mirrors the Python board runtime's old spatial
final-dedup rule: suppress boxes only when the smaller box is mostly covered, or
when center distance, x/y overlap, and area ratio all indicate the same target.
Use `--no-final-boundary-dedup` to disable this pass while debugging. The tuning
flags are:

```bash
--final-boundary-dedup-iou 0.70
--final-boundary-dedup-center 0.80
--final-boundary-dedup-size 0.25
```

Do not lower these aggressively just to match total target counts: the current
3-second comparison shows that many remaining C++-only rows are low-score short
tracks rather than overlapping duplicate boxes, so a loose boundary-neighbor
rule can remove real adjacent people.

The C++ runtime accepts the main Python board CLI flags used by the current
headless command, including `--camera-device`, `--profile-interval`,
`--profile-system-load`, `--system-load-interval-ms`, `--angle-vectorized`,
`--force-build`, and `--track-buffer`. `--force-build` is intentionally a no-op
here because all native libraries are prebuilt before deploying to Buildroot.
`--profile-system-load` is currently parsed for command compatibility; the C++
runtime does not yet write Python's separate `*_system_profile.jsonl`.

Two realtime acceleration paths are enabled by default:

- Camera mode uses a worker thread to read V4L2 MJPEG frames and decode JPEG
  ahead of the main OpenCL/RKNN/tracker thread. Disable with
  `--no-camera-prefetch` when isolating camera or decoder issues.
- RKNN bound input is attempted before the older CPU input-buffer path. OpenCL
  imports the RKNN input DMABUF fds and writes slice tensors directly into the
  bound input buffers. If OpenCL DMABUF import is unavailable, the program
  falls back to a bound-copy path that reads OpenCL output into RKNN bound input
  virtual addresses. If RKNN bound input itself is unavailable, it prints one
  warning and falls back to the older CPU input-buffer path. Disable explicitly
  with `--no-bound-input`.

Double-buffered RKNN bound-input pipelining is not enabled in the main runtime
yet. The standalone probe in `../board/tools/probe_rknn_double_bound_input.cpp`
has verified that two RKNN input memories can be switched and that OpenCL can
write the next input buffer while a non-blocking RKNN run is in flight. The main
runtime still keeps one bound-input set per RKNN context until per-context
double-buffer state is wired into the production path.

Equivalent C++ camera command for the current Python smoke path:

```bash
LD_LIBRARY_PATH=$PWD/lib:$PWD/lib/lib:$LD_LIBRARY_PATH \
taskset -c 0-6 \
bash run_smoke.sh \
  --camera-device /dev/video0 \
  --profile-system-load \
  --system-load-interval-ms 200 \
  --angle-vectorized \
  --force-build \
  --track-buffer 120 \
  --max-frames 1000000 \
  --ws-host 0.0.0.0 \
  --ws-port 8001 \
  --print-profile-summary \
  --profile-interval 30 \
  --output-jsonl board_output/board_cpp_camera.jsonl \
  --debug-jsonl board_output/board_cpp_camera_debug.jsonl \
  --no-stdout-json
```

Angle output defaults to the same current Python behavior: `fit_degree=4`, so
it uses the `fit_4` coefficients. It does not follow
`fisheye_calib.yaml`'s `default_fit: "5_constrained"` unless the code is changed
explicitly.

The JPEG frame must match the source resolution used when generating the map.
If `meta.txt` contains `img_width` and `img_height`, the program checks this
before touching OpenCL/RKNN.

## Files needed on Buildroot

Minimum runtime files:

- `board_cpp/bin/meeteye_cpp_smoke`
- `board_cpp/run_smoke.sh`
- `board_cpp/maps/6.22_2560_yolo_slices_640_cpp/map_x.bin`
- `board_cpp/maps/6.22_2560_yolo_slices_640_cpp/map_y.bin`
- `board_cpp/maps/6.22_2560_yolo_slices_640_cpp/base_map_x.bin`
- `board_cpp/maps/6.22_2560_yolo_slices_640_cpp/base_map_y.bin`
- `board_cpp/maps/6.22_2560_yolo_slices_640_cpp/meta.txt`
- `board_cpp/models/yolov8n-face-640-b1-int8-hybrid-split-kptconf-rk3588.rknn`
- `board_cpp/lib/libdirect_slice_opencl_fused.so`
- `board_cpp/lib/librknn_capi_parallel.so`
- `board_cpp/lib/libhybrid_sort_native.so`
- `board_cpp/lib/lib/librknnrt.so`
- `libturbojpeg.so.0`, either in target system library path or copied to `board_cpp/lib`
- target system `libOpenCL.so.1`

This is the C++ checkpoint for JPEG decode, V4L2 MJPEG camera input, OpenCL
preprocessing, RKNN inference, native merge, native HybridSORT tracking,
Python-compatible target JSON, and fit_4 angle/distance fields. If
`base_map_x.bin` and `base_map_y.bin` are present, C++ elevation uses the same
panorama-to-fisheye lookup path as the Python board runtime.
