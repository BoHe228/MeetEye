# Fish-Eye RK3588 Edge Runtime

`face_rc/` 是 MeetEye 在 RK3588 板端的精简部署路径，负责：

- 鱼眼输入：视频文件或本地 V4L2 摄像头
- YOLOv8n-face RKNN 推理：默认使用 608 INT8 split-output 模型
- 3 切片并行：`--rknn-parallel-slices` 创建 core0/core1/core2 三个推理实例
- 直接切片展开：headless 下通过 `--direct-slice-remap` 跳过完整全景图、切片和 letterbox
- OpenCL remap：`--direct-slice-remap-backend opencl` 使用 OpenCV UMat/OpenCL stacked remap
- C++ merge：`libmerge_fast.so` 将 RKNN raw output decode、NMS、slice merge 轻量化
- HybridSort 跟踪：默认启用，输出稳定 `track_id`
- 扇区 JSON：`--sector-output` 将 360 度按 `--num-sectors` 聚合，每个扇区取最大目标
- WebUI：可选浏览器预览、保存标注视频、输出 JSONL

当前推荐的高性能链路是 headless + direct-slice + OpenCL + RKNN 三核并行 + pipeline overlap + C++ merge。

## 目录和关键文件

```text
face_rc/
  main.py                         # 板端入口，WebUI/headless 共用
  config.py                       # 命令行参数
  processor.py                    # 鱼眼展开、检测、跟踪、角度、扇区
  core/
    detector.py                   # YOLO/RKNN 推理封装
    rknn_capi.py                  # RKNN C API 并行切片封装
    merge_fast.py                 # Python ctypes wrapper
    slicer.py                     # 全景切片与 Python merge fallback
    angle_calculator.py           # 方位/俯仰角计算
  tools/
    build_rknn_capi_parallel.sh   # 构建 librknn_capi_parallel.so
    build_merge_fast.sh           # 构建 libmerge_fast.so
  maps/
    6.22_2560_yolo_slices_608.npz # headless direct-slice 608 映射
  yolo_model/RK3588/
    yolov8n-face_608_b1_int8_split_rknn_model/
```

## 板端依赖

在 RK3588 环境中使用 Python 3.8 的 `rknn-lite` 环境。运行前需要确保 OpenCV、RKNN Runtime、RKNNLite、`face_rc/tools/bin/lib` 下的 native 库可用。

构建 native 库：

```bash
bash face_rc/tools/build_rknn_capi_parallel.sh
bash face_rc/tools/build_merge_fast.sh
```

运行时带上库路径：

```bash
LD_LIBRARY_PATH=$PWD/face_rc/tools/bin/lib python face_rc/main.py ...
```

如果日志出现：

```text
[merge-fast] 不可用，回退 Python merge
```

说明 `libmerge_fast.so` 没有被加载到。此时功能仍可运行，但 slice merge 会回到 Python 路径，延迟会增加。

## 性能建议

建议先锁定 GPU/NPU 频率，避免 OpenCL remap 和 RKNN 推理抖动：

```bash
sudo sh -c 'echo performance > /sys/class/devfreq/fb000000.gpu/governor'
sudo sh -c 'echo 1000000000 > /sys/class/devfreq/fb000000.gpu/min_freq'
sudo sh -c 'echo 1000000000 > /sys/class/devfreq/fb000000.gpu/max_freq'

sudo sh -c 'echo performance > /sys/class/devfreq/fdab0000.npu/governor'
sudo sh -c 'echo 1000000000 > /sys/class/devfreq/fdab0000.npu/min_freq'
sudo sh -c 'echo 1000000000 > /sys/class/devfreq/fdab0000.npu/max_freq'
```

确认：

```bash
cat /sys/class/devfreq/fb000000.gpu/governor
cat /sys/class/devfreq/fb000000.gpu/cur_freq
cat /sys/class/devfreq/fdab0000.npu/governor
cat /sys/class/devfreq/fdab0000.npu/cur_freq
```

本轮优化后的参考结果：40 秒 1080p 会议视频，`direct-slice + OpenCL + RKNN 三核并行 + pipeline queue=2 + libmerge_fast`，平均约 23 FPS。检测输出相对优化前不应变化，主要收益来自 merge 轻量化和流水线重叠。

`--headless-remap-prefetch` 目前是实验项，实测会把 direct-slice remap 的统计 wall time 拉高，不建议作为默认运行参数。

## Headless 视频输入

推荐用于离线 benchmark 或文件转 JSONL。

```bash
LD_LIBRARY_PATH=$PWD/face_rc/tools/bin/lib \
python face_rc/main.py \
  --video-path face_rc/data/大会议室_6.4_多人开会_40秒.mp4 \
  --output-jsonl face_rc/benchmark_results/pipeline_parallel_queue2.jsonl \
  --model-path face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model \
  --imgsz 608 \
  --rknn-parallel-slices \
  --direct-slice-remap \
  --direct-slice-remap-backend opencl \
  --headless-pipeline-parallel \
  --headless-pipeline-queue-size 2 \
  --sector-output \
  --profile-interval 30
```

说明：

- `--headless-pipeline-queue-size 2` 是当前推荐值；更大的队列通常不会提升吞吐。
- `--rknn-parallel-slices` 对 split RKNN 模型很关键，会把 3 个切片分配到 NPU 三核。
- `--direct-slice-remap` 只在 headless 快路径启用；WebUI 会为了显示完整全景图而忽略它。

## Headless 本地摄像头输入

不启 WebUI，直接从板端摄像头输出 JSONL。持续运行：

```bash
LD_LIBRARY_PATH=$PWD/face_rc/tools/bin/lib \
python face_rc/main.py \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --camera-format mjpeg \
  --output-jsonl face_rc_output/rknn_608/camera_sector.jsonl \
  --model-path face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model \
  --imgsz 608 \
  --rknn-parallel-slices \
  --direct-slice-remap \
  --direct-slice-remap-backend opencl \
  --headless-pipeline-parallel \
  --headless-pipeline-queue-size 2 \
  --sector-output \
  --profile-interval 30
```

快速测试 300 帧：

```bash
LD_LIBRARY_PATH=$PWD/face_rc/tools/bin/lib \
python face_rc/main.py \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --camera-format mjpeg \
  --max-frames 300 \
  --output-jsonl face_rc_output/rknn_608/camera_sector_test.jsonl \
  --model-path face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model \
  --imgsz 608 \
  --rknn-parallel-slices \
  --direct-slice-remap \
  --direct-slice-remap-backend opencl \
  --headless-pipeline-parallel \
  --headless-pipeline-queue-size 2 \
  --sector-output \
  --profile-interval 30
```

如果 `/dev/video0` 不对，先查看：

```bash
ls /dev/video*
```

如果摄像头不支持 MJPEG，把 `--camera-format mjpeg` 改成 `--camera-format yuyv`。

## WebUI 模式

WebUI 主要用于查看画面、调试扇区和保存标注视频。注意：WebUI 需要完整全景图显示，因此会忽略 `--direct-slice-remap`，速度明显低于 headless 快路径。

```bash
LD_LIBRARY_PATH=$PWD/face_rc/tools/bin/lib \
python face_rc/main.py \
  --webui \
  --video-path face_rc/data/大会议室_6.4_多人开会_40秒.mp4 \
  --model-path face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model \
  --output-jsonl face_rc_output/rknn_608/6.29_FPS23.jsonl \
  --imgsz 608 \
  --rknn-parallel-slices \
  --sector-output \
  --show-sectors \
  --save-video \
  --video-name 6.29_PFS23 \
  --profile-interval 30
```

保存行为：

- `--save-video` 默认只保存标注后的视频。
- 原始视频默认不保存。
- 需要同时保存原始视频时，显式加 `--save-original-video`。

## 扇区 JSON 输出

打开 `--sector-output` 后，每行 JSON 固定输出所有扇区：

```json
{
  "timestamp": 1782713083.072,
  "frame_id": 2,
  "num_sectors": 8,
  "sectors": {
    "0": {"has_target": true, "azimuth": 21.926, "elevation": 10.162},
    "1": {"has_target": false, "azimuth": null, "elevation": null}
  }
}
```

空扇区输出 `has_target=false` 和 `null` 是正常的。检查有效扇区：

```bash
rg -n '"has_target": true' face_rc_output/rknn_608/camera_sector.jsonl | head
```

角度计算现在有 bbox 兜底：当 face keypoints 无效或 tracker 没有可用 keypoints 时，会使用检测框中心偏上的点计算方位角，避免出现“检测正常但扇区全 null”。

## 常见问题

### WebUI 没有 direct-slice 速度

这是预期行为。WebUI 为了显示全景画面，会走完整 OpenCV 鱼眼展开路径。性能测试和正式 JSONL 输出优先用 headless。

### JSONL 全是 null

先确认打开了本次命令的 `--output-jsonl` 文件，而不是旧文件。然后检查：

```bash
rg -n '"has_target": true' <output>.jsonl | head
```

如果仍然没有，查看日志中的 `det=` 是否为 0。`det` 有值但扇区为空时，检查是否已同步包含 bbox 兜底的代码。

### OpenCL remap 有 18-20 ms 抖动

优先确认 GPU/NPU 是否锁频。OpenCL remap profile 中偶发 `get` 阻塞通常是 GPU/内存调度抖动，锁频后会明显稳定。

### queue size 改大没有提升

`--headless-pipeline-queue-size` 是 detection result 队列深度，不是 RKNN batch。当前瓶颈主要在 YOLO/NPU、tracker 和 remap，推荐保持 2。

### 使用新的 RKNN 模型

模型目录需要包含 RKNN split-output 文件，并与 `--imgsz`、direct-slice map 对齐。608 模型默认使用：

```text
face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model
face_rc/maps/6.22_2560_yolo_slices_608.npz
```
