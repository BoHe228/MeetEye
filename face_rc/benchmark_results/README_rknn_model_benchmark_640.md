# RK3588 Model Benchmark Plan

This folder is for board-side RKNN timing results.

Timing scope:
- C API `rknn_run()` only
- warmup loops excluded
- zero input buffer set once before timing
- input size fixed to batch=1, 640x640
- RK3588 target, INT8 RKNN unless noted

Conversion policy:
- Official YOLOv8-pose keeps the Rockchip/rknn_model_zoo hybrid quantization recipe.
- YOLO26 models follow the local `MeetEye/ultralytics` RKNN export semantics: export ONNX with `end2end=False`, then convert that ONNX with RKNN Toolkit INT8 quantization.
- Do not benchmark the old YOLO26 `*-plain/` RKNN files. Those include the end-to-end postprocess branch and can crash at `rknn_run()` on RK3588.

Converted models are in:

```text
face_rc/yolo_model/RK3588/benchmark_640/
```

Converted model list:

| model | family | task | input | quantization | RKNN |
|---|---|---|---:|---|---|
| yolov8n-face | YOLOv8 | face pose | 1x640x640 | plain INT8 | `benchmark_640/yolov8n-face-640-i8-plain/yolov8n-face-640-b1-rk3588-i8.rknn` |
| yolov8n-pose-official | YOLOv8 | person pose | 1x640x640 | Rockchip YOLOv8-pose hybrid INT8 | `benchmark_640/yolov8n-pose-official-640-i8-hybrid/yolov8n-pose-official-640-b1-rk3588-i8.rknn` |
| yolo26n | YOLO26 | detect | 1x640x640 | plain INT8, end2end disabled | `benchmark_640/yolo26n-640-i8-plain_noe2e/yolo26n-640-b1-rk3588-i8.rknn` |
| yolo26n-pose | YOLO26 | person pose | 1x640x640 | plain INT8, end2end disabled | `benchmark_640/yolo26n-pose-640-i8-plain_noe2e/yolo26n-pose-640-b1-rk3588-i8.rknn` |
| yolo26s-pose | YOLO26 | person pose | 1x640x640 | plain INT8, end2end disabled | `benchmark_640/yolo26s-pose-640-i8-plain_noe2e/yolo26s-pose-640-b1-rk3588-i8.rknn` |
| yolo26n-seg | YOLO26 | segment | 1x640x640 | plain INT8, end2end disabled | `benchmark_640/yolo26n-seg-640-i8-plain_noe2e/yolo26n-seg-640-b1-rk3588-i8.rknn` |
| yolo26x-pose | YOLO26 | person pose | 1x640x640 | plain INT8, end2end disabled | `benchmark_640/yolo26x-pose-640-i8-plain_noe2e/yolo26x-pose-640-b1-rk3588-i8.rknn` |

Board command:

```bash
cd ~/MeetEye
bash face_rc/tools/build_rknn_run_benchmark.sh
python face_rc/tools/benchmark_rknn_models.py \
  --manifest face_rc/yolo_model/RK3588/benchmark_640/manifest.json \
  --loops 300 \
  --warmup 30 \
  --core-masks 1,7
```

The script writes:

```text
face_rc/benchmark_results/rknn_run_benchmark_YYYYMMDD_HHMMSS.csv
face_rc/benchmark_results/rknn_run_benchmark_YYYYMMDD_HHMMSS.md
```

Reference-only command for the current production face model:

```bash
LD_LIBRARY_PATH=face_rc/tools/bin/lib \
face_rc/tools/bin/rknn_run_benchmark \
  face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model/yolov8n-face-608-b1-int8-split-rk3588.rknn \
  300 30 7
```

That model is 608x608, so it should not be mixed into the 640x640 comparison table without marking it as a separate reference.
