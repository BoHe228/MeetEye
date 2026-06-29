# MeetEye YOLO Fine-Tune Workspace

`fine-tune/` 是服务器侧 YOLO 微调和数据准备工作区。当前主要覆盖两条线：

- **YOLO pose 微调**：OmniLab、PoseFES、COCO-Pose、会议视频自动标注数据，用于训练/评估 `yolo26n-pose`。
- **YOLOv8n-face 部署校准**：导出与运行时一致的全景切片，用于 TensorRT INT8 或后续 RKNN 对齐分析。

板端 `face_rc/` 当前正式部署使用 `yolov8n-face` RKNN 模型；服务器侧微调和校准必须保证输入切片、展开矩阵、裁剪和切片参数与运行时一致，否则 INT8 和 FP16/RKNN 的召回对比会失真。

## 目录结构

```text
fine-tune/
  train_yolo26_pose.py                 # 通用 YOLO pose 微调入口
  train_yolo26_pose_mixed_640.py       # mixed_pose_640 默认训练入口
  scripts/
    prepare_omnilab_zhankai.py         # OmniLab 鱼眼展开 + 切片 + YOLO pose 标签
    prepare_posefes_zhankai.py         # PoseFES 鱼眼展开 + 切片 + YOLO pose 标签
    prepare_meeting_video_autolabel.py # 会议视频抽帧、展开、切片、自动标注
    cvat_coco_keypoints_to_yolo_pose.py# CVAT COCO keypoints 回转 YOLO pose
    yolo_pose_to_cvat_coco.py          # YOLO pose 转 CVAT 检查包
    dedup_yolo_pose_labels.py          # 自动标注去重
    check_fisheye_pose_label_quality.py# 标签质量抽查
    monitor_yolo_checkpoints.py        # 训练时额外保存 best/recall/周期 checkpoint
  datasets/
    omnilab_zhankai/
    posefes_zhankai/
    mixed_pose_640/
    small_meeting_xiding_autolabel/
    small_meeting_xiding_cvat*/
  runs/
```

相关的服务器 INT8 校准脚本在仓库根目录 `tools/`：

```text
tools/export_yolov8_face_slice_calib.py
tools/build_yolov8_face_int8_engine.py
tools/compare_face_jsonl.py
```

## 环境

在服务器侧使用 Python 3.10/3.11 + CUDA 环境，优先使用本目录内置的 `fine-tune/ultralytics`，避免全局 Ultralytics 版本漂移。

训练前确认：

```bash
python3 - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
PY
```

TensorRT INT8 构建需要可用的 CUDA 和 TensorRT Python 包。

## 数据集现状

已准备的数据：

- `datasets/omnilab_zhankai/`：OmniLab 天花板鱼眼数据展开成全景切片。
- `datasets/posefes_zhankai/`：PoseFES 天花板鱼眼数据展开成全景切片。
- `datasets/coco-pose/`：COCO-Pose 子集。
- `datasets/mixed_pose_640/`：OmniLab + PoseFES + COCO-Pose 混合 640 训练列表。
- `datasets/small_meeting_xiding_autolabel/`：小会议室吸顶视频自动标注数据。
- `datasets/small_meeting_xiding_cvat*/`：CVAT 复核/导出后的数据。

OmniLab/PoseFES 预处理约定：

- 展开：`view_type=bottom`
- 输出：`3840x1080`
- 顶部裁剪：默认不裁剪或按脚本参数控制
- 切片：3 slices，`slice_overlap=0.1`
- 标签：YOLO pose，COCO 17 keypoints
- 空切片：保留空 `.txt`，用于负样本

`mixed_pose_640` 当前规模：

- train：12760 fisheye + 12760 COCO = 25520 images
- test/val：3189 fisheye + 3189 COCO = 6378 images
- COCO val 贡献 2346 test images，其余 843 COCO test images 从 COCO train hold out

## Pose 微调

从仓库根目录运行。

OmniLab 单数据集：

```bash
python3 fine-tune/train_yolo26_pose.py \
  --device 0 \
  --epochs 50 \
  --imgsz 1536 \
  --batch 4
```

混合 640 数据集：

```bash
python3 fine-tune/train_yolo26_pose_mixed_640.py \
  --device 0 \
  --epochs 50
```

默认参数：

- `--imgsz 640`
- `--batch 16`
- `--optimizer AdamW`
- `--lr0 0.0005`
- `--mosaic 0.3`
- `--scale 0.3`

输出目录：

```text
fine-tune/runs/pose/yolo26n-pose-mixed-640/
```

显存不足时优先降低 `--batch`，其次降低 `--imgsz`。继续训练加 `--resume`。

## 训练 checkpoint 监控

Ultralytics 默认保存 `weights/best.pt` 和 `weights/last.pt`。训练过程中额外保留 pose mAP、pose recall 和周期 checkpoint：

```bash
python3 fine-tune/scripts/monitor_yolo_checkpoints.py
```

只执行一次：

```bash
python3 fine-tune/scripts/monitor_yolo_checkpoints.py --once
```

## 重建公开鱼眼数据

OmniLab：

```bash
python3 fine-tune/scripts/prepare_omnilab_zhankai.py \
  --src-dir fine-tune/datasets/omnilab_raw \
  --out-dir fine-tune/datasets/omnilab_zhankai \
  --view-type bottom \
  --eval-ratio 0.2 \
  --eval-split test \
  --no-debug
```

PoseFES：

```bash
python3 fine-tune/scripts/prepare_posefes_zhankai.py \
  --src-dir data/fine-tune_dataset/PoseFES \
  --out-dir fine-tune/datasets/posefes_zhankai \
  --overwrite
```

COCO 子集和 mixed 640 列表：

```bash
python3 fine-tune/scripts/download_coco_pose_subset.py
python3 fine-tune/scripts/build_mixed_pose_dataset.py
```

## 会议视频自动标注

用于把真实会议视频转成可复核的 YOLO pose 切片数据：

```bash
python3 fine-tune/scripts/prepare_meeting_video_autolabel.py \
  --videos data/小会议室_吸顶_真实会议_3min短视频.mp4 \
  --map-file maps/xiding_maps_bottom.npz \
  --model fine-tune/models/yolo26n-pose.pt \
  --out-dir fine-tune/datasets/small_meeting_xiding_autolabel \
  --fps 1.0 \
  --num-slices 3 \
  --slice-overlap 0.1 \
  --imgsz 640 \
  --conf 0.25 \
  --overwrite
```

只展开/切片，不自动标注：

```bash
python3 fine-tune/scripts/prepare_meeting_video_autolabel.py \
  --videos data/小会议室_吸顶_真实会议_3min短视频.mp4 \
  --map-file maps/xiding_maps_bottom.npz \
  --out-dir fine-tune/datasets/small_meeting_xiding_autolabel \
  --no-autolabel \
  --overwrite
```

自动标注数据必须人工抽查。重点检查：

- 展开矩阵是否与视频匹配
- 切片边界处是否重复标注或漏标
- bbox 是否覆盖全身/头肩区域
- 关键点可见性是否合理
- 空切片是否确实无目标

## CVAT 回流

常用流程：

1. 自动标注生成切片数据和 preview。
2. 转成 CVAT COCO keypoints 包。
3. CVAT 人工修正。
4. 导出 COCO keypoints。
5. 回转 YOLO pose。
6. 重新训练或做测试集评估。

相关脚本：

```bash
python3 fine-tune/scripts/yolo_pose_to_cvat_coco.py ...
python3 fine-tune/scripts/cvat_coco_keypoints_to_yolo_pose.py ...
python3 fine-tune/scripts/dedup_yolo_pose_labels.py ...
python3 fine-tune/scripts/check_fisheye_pose_label_quality.py ...
```

具体参数以脚本 `--help` 为准。

## YOLOv8n-face INT8 校准切片

服务器 TensorRT INT8 校准必须使用**运行时进入 YOLO 前的切片图像**，不能直接用原始鱼眼帧。切片必须和视频对应的展开矩阵一致。

例如 `data/大会议室_6.4_多人开会_40秒.mp4` 必须使用：

```text
maps/3840_fisheye_maps_6.4.npz
```

不要混用其他日期或其他相机位的 map，否则会出现校准切片展开错位，INT8 引擎召回异常。

导出 864 输入的运行时切片校准集：

```bash
python3 tools/export_yolov8_face_slice_calib.py \
  --source data/大会议室_6.4_多人开会_40秒.mp4 \
  --source-map maps/3840_fisheye_maps_6.4.npz \
  --output-dir yolo_model/int8_calib_slices_864 \
  --output-width 3840 \
  --output-height 1080 \
  --crop-divisor 3 \
  --num-slices 3 \
  --slice-overlap 0.1 \
  --total-frames 300 \
  --batch 3
```

导出后检查：

```bash
find yolo_model/int8_calib_slices_864/images -type f | head
head yolo_model/int8_calib_slices_864/dataset.txt
```

如果图像看起来没有正确展开，优先检查 `--source-map` 是否和视频匹配。

## 构建 TensorRT INT8 engine

使用切片校准集构建：

```bash
python3 tools/build_yolov8_face_int8_engine.py \
  --onnx yolo_model/yolov8n-face.onnx \
  --engine yolo_model/yolov8n-face_int8_slices_6p4.engine \
  --cache yolo_model/yolov8n-face_int8_slices_6p4.calib \
  --metadata-source yolo_model/yolov8n-face.engine \
  --calib-dataset yolo_model/int8_calib_slices_864/dataset.txt \
  --calib-images 0 \
  --force-recalibrate
```

TensorRT 日志出现类似下面内容，说明 engine 已构建完成：

```text
Engine generation completed
[build] wrote: yolo_model/yolov8n-face_int8_slices_6p4.engine
```

## INT8/FP16 JSONL 对比

运行服务器侧推理后，用 `tools/compare_face_jsonl.py` 对比检测数量和角度匹配：

```bash
python3 tools/compare_face_jsonl.py \
  --fp16 yolo_pose_output/yolov8n-face_fp16_大会议室_6.4_多人开会_40秒.jsonl \
  --int8 yolo_pose_output/yolov8n-face_int8_大会议室_6.4_多人开会_40秒.jsonl \
  --output face_rc/benchmark_results/yolov8n-face_fp16_vs_int8_jsonl_compare.txt
```

当前结论：

- 板端 RKNN INT8 和服务器 TensorRT INT8 不是天然同一种量化策略。
- 服务器 INT8 要尽量贴近板端效果，必须先保证校准切片、展开矩阵、裁剪、切片 overlap 与运行时完全一致。
- 6.4 视频使用 `maps/3840_fisheye_maps_6.4.npz` 是硬性要求。

## 与板端部署的关系

`fine-tune/` 负责训练、数据和服务器侧校准；`face_rc/` 负责 RK3588 实时部署。同步模型到板端前需要确认：

- 模型输入尺寸与 `--imgsz` 一致。
- direct-slice map 与输入尺寸一致。
- RKNN split-output 模型目录结构和 `face_rc/core/detector.py` 期望一致。
- JSONL 召回、扇区输出和角度范围在服务器侧已经做过抽查。

板端运行说明见：

```text
face_rc/README.md
```
