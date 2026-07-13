# MeetEye board_cpp 最小 C++ 运行链路

`board_cpp` 是面向 RK3588 板端的最小 C++ 运行闭环，目标是把原来
`face_rc/board/main.py` 里的核心推理链路迁移到不依赖 Python 虚拟环境的
C++ 程序中。

当前链路做的事情：

1. 读取 JPEG/MJPEG 图片，或直接读取 V4L2 MJPEG 摄像头。
2. 使用 `libjpeg-turbo` 在 CPU 内存中解码成 BGR。
3. 调用已有的 `libdirect_slice_opencl_fused.so` 做鱼眼展开、切片和 RKNN 输入排布。
4. 调用已有的 `librknn_capi_parallel.so` 在 RK3588 NPU 上执行 RKNN 推理。
5. 调用已有的 `libhybrid_sort_native.so` 做 HybridSORT 跟踪。
6. 执行 C++ 版跟踪后处理、边界去重、短暂丢失保留、扇区输出等逻辑。
7. 输出与 Python 板端兼容的 `targets` JSON、扇区 JSON、JSONL、WebSocket 数据和性能统计。

这个运行链路不依赖 OpenCV C++、torch、ultralytics、scipy，也不依赖 Python
运行时或 Python 虚拟环境。

## 运行模块链路

```text
JPEG/MJPEG 图片或 V4L2 摄像头
      |
      v
libjpeg-turbo 解码
      |
      v
meeteye_cpp_smoke.cpp
      |
      +--> libdirect_slice_opencl_fused.so
      |        鱼眼/base-map 查表 + 三路直接切片 + RKNN tensor 排布
      |
      +--> librknn_capi_parallel.so
      |        RKNN C API，使用 RK3588 NPU 推理
      |
      +--> C++ decode/merge
      |        RKNN raw output -> 切片合并 -> 最终边界去重
      |
      +--> libhybrid_sort_native.so
      |        native HybridSORT 匹配和 track 状态维护
      |
      v
targets JSON / sector JSON / JSONL / WebSocket / profile summary
```

当前 C++ 链路不会直接链接 `libmerge_fast.so` 或
`libtracker_assoc_fast.so`。这两个库仍然可以给 Python 板端链路和旧的 native
加速实验使用。

`board_cpp` 运行所需的 native C++ 源码和构建脚本已经收在
`board_cpp/tools/` 下。编译这些 `.so` 不依赖 Python，也不依赖虚拟环境包；
只需要板端或交叉编译环境提供 C++ 编译器、RKNN 头文件/runtime、OpenCL runtime
和系统 C/C++ 运行库。

## 准备展开映射

推荐做法是在板端 `board_cpp` 内直接生成 C++ 可读的二进制 map。这个流程不依赖
Python、OpenCV、torch 或虚拟环境包；编译只需要 C++ 编译器。实际读取 MJPEG
摄像头时，目标系统需要提供 `libturbojpeg.so.0`。

在 `MeetEye/face_rc/board_cpp` 目录下执行：

```bash
bash tools/build_generate_board_cpp_maps.sh

tools/bin/generate_board_cpp_maps_from_camera \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --camera-format mjpeg \
  --camera-warmup-frames 10 \
  --sample-frames 30 \
  --output-map-file ../board/maps/6.22_2560_yolo_slices_640.npz \
  --cpp-output-dir maps/6.22_2560_yolo_slices_640_cpp \
  --imgsz 640 \
  --output-width 2560 \
  --output-height 720 \
  --process-width 2560 \
  --crop-divisor 3 \
  --num-slices 3 \
  --slice-overlap 0.1 \
  --vertical-fov 100
```

输出目录包含：

```text
map_x.bin
map_y.bin
base_map_x.bin
base_map_y.bin
meta.txt
```

如果已有 Python 版检测得到的鱼眼参数，建议显式传入，避免 C++ 简化检测和
OpenCV 检测存在细小差异：

```bash
tools/bin/generate_board_cpp_maps_from_camera \
  --no-camera \
  --camera-width 1920 \
  --camera-height 1080 \
  --center-x 926 \
  --center-y 564 \
  --radius 488 \
  --cpp-output-dir maps/6.22_2560_yolo_slices_640_cpp
```

`--output-map-file` 在 C++ 工具里只写入 `meta.txt` 的 `source_npz` 字段，不会
生成 `.npz`。如果还需要给 Python 板端使用的 `.npz` 文件，可以继续使用原来的
Python 流程。

在 `MeetEye` 目录下执行：

```bash
python3 face_rc/tools/generate_yolo_slice_maps.py \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-format mjpeg \
  --output-map-file face_rc/board/maps/6.22_2560_yolo_slices_640.npz \
  --cpp-output-dir face_rc/board_cpp/map_export
```

如果使用视频源生成：

```bash
python3 face_rc/tools/generate_yolo_slice_maps.py \
  --video-path data/大会议室_6.4_多人开会_40秒.mp4 \
  --output-map-file face_rc/board/maps/6.22_2560_yolo_slices_640.npz \
  --cpp-output-dir face_rc/board_cpp/map_export
```

如果 `.npz` 映射文件已经存在，只需要导出 C++ 文件：

```bash
cd MeetEye/face_rc/board_cpp

python3 export_direct_slice_map.py \
  --input ../board/maps/6.22_2560_yolo_slices_640.npz \
  --output map_export
```

当前仓库里已经带了一份打包好的 C++ map：

```text
maps/6.22_2560_yolo_slices_640_cpp/
```

## 编译

在 `MeetEye/face_rc/board_cpp` 下执行：

```bash
bash build_libs.sh
bash build.sh
```

`build_libs.sh` 会从 `board_cpp/tools/*.cpp` 生成：

```text
lib/libdirect_slice_opencl_fused.so
lib/librknn_capi_parallel.so
lib/libhybrid_sort_native.so
lib/libmerge_fast.so
lib/libtracker_assoc_fast.so
```

如果系统里只有 `libturbojpeg.so.0`，没有开发环境常见的
`libturbojpeg.so` 软链接，`build.sh` 会直接链接
`/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0`。

## 运行图片

单张图片：

```bash
bash run_smoke.sh --image frame.jpg
```

多张 JPEG 图片共用一次 RKNN/OpenCL 初始化。`--output-jsonl` 写出的结构与
`board/main.py` 的高层推理结果一致，包含 `timestamp`、`frame_id` 和
`targets`。

```bash
bash run_smoke.sh \
  --image-list test_frames/frames.txt \
  --track-buffer 120 \
  --output-jsonl board_cpp_results.jsonl \
  --no-stdout-json
```

同时写出完整 debug JSONL：

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

debug JSONL 会在 `fps` 和 `timings_ms` 里写入运行帧率信息：
`fps.instant`、`fps.average`、`fps.frame_ms`。

参数说明：

- `--print-profile-summary`：在终端 stderr 打印各阶段平均耗时和最大耗时。
- `--profile-interval N`：每 N 帧打印一次运行中的 profile 汇总。
- `--profile-system-load`：启动后台硬件负载采样线程。
- `--system-load-interval-ms N`：硬件采样间隔，默认 200 ms，最小 50 ms。
- `--decode-prefetch`：图片列表模式下用 worker 线程提前读取和解码下一帧，
  让 JPEG 解码与 RKNN/OpenCL 推理重叠。
- `--track-buffer N`：跟踪器内部保留丢失 track 的最大帧数。
- `--lost-velocity-decay VALUE`：目标丢失后，每帧将 Kalman 预测速度和方向速度
  乘以该系数，默认 `0.85`；设为 `1.0` 等于保持原常速度预测。
- `--inherit-center-dist-thresh VALUE`：短暂遮挡 ID 继承的中心距离阈值，单位是
  平均框高，默认 `1.0`；设为 `0` 基本关闭短暂遮挡继承。
- `--inherit-size-ratio-thresh VALUE`：短暂遮挡 ID 继承的尺寸比例阈值，默认
  `0.5`；新框和旧观测框/Kalman 框大小差太多时不继承。
- `--inherit-ambiguity-margin VALUE`：短暂遮挡 ID 继承的唯一最佳候选阈值，默认
  `0.25`；最近候选和第二近候选距离差太小时不继承。
- `--smooth-bbox-alpha VALUE`：跟踪框 EMA 防抖系数，默认 `0.5`；检测框抖动明显
  时可先试 `0.7`，数值越大越稳但跟随移动越慢。
- `--no-smooth-bbox`：关闭跟踪框 EMA 平滑。
- `--kalman-bbox`：使用 native Kalman state 框输出，通常更平滑但可能有预测漂移
  或滞后，不建议作为首选防抖方案。

开启硬件采样后，会在 debug 或 output JSONL 旁边生成：

```text
*_system_profile.jsonl
*_system_profile_summary.json
```

## 生成耗时和硬件负载网页

根据 debug JSONL 生成独立 HTML 报告：

```bash
python3 render_debug_profile_html.py \
  --input board_output/board_cpp_camera_debug.jsonl \
  --output board_output/board_cpp_camera_profile.html
```

如果同目录下存在
`board_output/board_cpp_camera_debug_system_profile.jsonl`，渲染脚本会自动加载
硬件负载曲线。也可以用 `--system-profile PATH` 手动指定采样文件。

面向整个 `board_output` 目录的平均耗时报告：

```bash
python3 render_board_output_profile_html.py --board-output board_output
```

## 运行摄像头

V4L2 MJPEG 摄像头直接运行：

```bash
bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --track-buffer 120 \
  --lost-velocity-decay 0.85 \
  --inherit-center-dist-thresh 1.0 \
  --inherit-size-ratio-thresh 0.5 \
  --inherit-ambiguity-margin 0.25 \
  --smooth-bbox-alpha 0.7 \
  --output-jsonl camera_results.jsonl \
  --no-stdout-json
```

摄像头模式默认持续运行。只有需要固定帧数测试时，才加：

```bash
--max-frames N
```

## WebSocket 输出

C++ runtime 默认启动轻量级 JSON WebSocket 服务，路径与 Python 板端一致：

```text
ws://<board-ip>:8001/ws/inference
```

可用参数：

```bash
--ws-host 0.0.0.0
--ws-port 8001
--no-ws-json
```

WebSocket 发送的高层 payload 与 stdout/JSONL 一致。发送方式是 WebSocket
binary message，对齐 Python 版本里的 `send_bytes()` 行为。

离线图片列表对比时，如果不需要实时订阅，或者端口 `8001` 已被占用，可以加：

```bash
--no-ws-json
```

## C++ WebUI 可视化

如果需要像 `face_rc/main.py --webui` 一样在浏览器里看“画好之后的推理画面”，
可以打开 C++ WebUI：

```bash
bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --track-buffer 120 \
  --lost-velocity-decay 0.85 \
  --inherit-center-dist-thresh 1.0 \
  --inherit-size-ratio-thresh 0.5 \
  --inherit-ambiguity-margin 0.25 \
  --smooth-bbox-alpha 0.7 \
  --sector-output \
  --num-sectors 8 \
  --webui \
  --webui-port 8080 \
  --debug-jsonl board_output/board_cpp_camera_debug.jsonl \
  --print-profile-summary \
  --profile-interval 30 \
  --no-output-jsonl \
  --no-stdout-json
```

浏览器访问：

```text
http://<板端IP>:8080/
```

程序启动时会在终端打印实际监听地址和可访问地址，例如：

```text
[board_cpp-webui] listening: http://0.0.0.0:8080/
[board_cpp-webui] open: http://172.16.30.68:8080/
```

C++ WebUI 提供的核心端点：

```text
GET  /                  浏览器页面
GET  /video/infer       画框后的 MJPEG 流
GET  /inference/latest  最新推理 JSON
GET  /system/latest     最新 CPU/GPU/NPU/内存/温度采样 JSON
WS   /ws/inference      实时推理 JSON，binary message
WS   /ws/system         实时 CPU/GPU/NPU/内存/温度 JSON，binary message
WS   /ws/frame          画框后的 JPEG 帧，binary message
```

相关参数：

- `--webui`：打开 C++ WebUI；不开时不做画框和 JPEG 编码。
- `--webui-host 0.0.0.0`：WebUI 监听地址。
- `--webui-port 8080`：WebUI HTTP 端口。不要和 `--ws-port 8001` 使用同一个端口。
- `--webui-jpeg-quality 80`：画框后 JPEG 推流质量，取值 1 到 100。
- 开启 `--webui` 时会自动启动内存里的硬件负载采样，用于页面显示 CPU、每核
  CPU、GPU、NPU、内存、可用内存和温度。
- 如果还需要把硬件负载保存为 JSONL/summary，再额外加
  `--profile-system-load --system-load-interval-ms 200`。

实现说明：

- JSON WebSocket `/ws/inference` 仍保持 Python 板端兼容结构。
- 硬件负载 WebSocket `/ws/system` 不改变推理 JSON 结构，只服务 WebUI。
- 浏览器画面使用导出的 `base_map_x.bin/base_map_y.bin` 在 C++ 中重建全景图，
  再叠加 bbox、关键点、track id 和 FPS。
- WebUI 只在有浏览器订阅 `/video/infer` 或 `/ws/frame` 时才额外重建全景和编码
  JPEG，避免无人观看时消耗 CPU。

## 扇区输出

C++ 已支持 Python 兼容的扇区聚合输出。开启 `--sector-output` 后，stdout、
JSONL 和 WebSocket payload 会从 `targets` 切换为 Python 板端使用的扇区结构：

```json
{"timestamp": 0.0, "frame_id": 1, "num_sectors": 8, "sectors": {"0": {"has_target": false, "azimuth": null, "elevation": null}}}
```

每个扇区覆盖 `360 / num_sectors` 度。如果同一个扇区里有多个目标，会选择 bbox
面积最大的目标，与 `board/utils/sector.py` 的行为一致。

示例：

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

## 跟踪和去重

native HybridSORT 默认开启。只有在单独排查 JPEG 解码、OpenCL 预处理或 RKNN
推理时，才使用：

```bash
--no-tracker
```

C++ 跟踪输出默认还会执行一层保守的最终空间/边界去重。它对齐 Python 板端旧
空间去重规则：只有当小框大部分被覆盖，或者中心距离、x/y 重叠、面积比例都显示
为同一目标时，才会抑制重复框。

调试参数：

```bash
--no-final-boundary-dedup
--final-boundary-dedup-iou 0.70
--final-boundary-dedup-center 0.80
--final-boundary-dedup-size 0.25
```

不要为了强行匹配目标总数而过度降低这些阈值。当前 3 秒对比里，很多 C++ 多出的
行是低分短 track，不一定是重叠重复框；过松的边界邻居规则可能误删真实相邻的人。

## Python 命令兼容性

C++ runtime 接受当前 Python 板端无头命令里常用的参数，包括：

```text
--camera-device
--profile-interval
--profile-system-load
--system-load-interval-ms
--angle-vectorized
--force-build
--track-buffer
--lost-velocity-decay
--inherit-center-dist-thresh
--inherit-size-ratio-thresh
--inherit-ambiguity-margin
```

其中 `--force-build` 在 C++ 版本里是 no-op，因为部署到 Buildroot 前所有 native
库都应该已经预编译好。

## 默认加速路径

当前有三条实时加速路径默认开启：

1. 摄像头读取/解码预取
   摄像头模式使用 worker 线程提前读取 V4L2 MJPEG 帧并解码 JPEG，让摄像头读取、
   JPEG 解码与主线程 OpenCL/RKNN/tracker 工作重叠。排查摄像头或解码问题时可用：

   ```bash
   --no-camera-prefetch
   ```

2. RKNN bound input
   程序会优先尝试 RKNN bound input。OpenCL 会导入 RKNN input DMABUF fd，并尽量
   直接把切片 tensor 写入 bound input buffer。如果 OpenCL DMABUF import 不可用，
   会回退到 bound-copy 路径：OpenCL 输出读回到 RKNN bound input 虚拟地址。如果
   RKNN bound input 本身不可用，会打印一次 warning，并回退到旧的 CPU input-buffer
   路径。

   需要显式关闭时使用：

   ```bash
   --no-bound-input
   ```

3. staging pipeline
   默认重叠下一帧 OpenCL staging 和当前帧 blocking RKNN，减少主线程空等。
   这不是 OpenCL 直接写 RKNN input 的 zero-copy；当前路径仍是 OpenCL 写 staging
   buffer，RKNN 完成后再 staging -> CPU host -> RKNN input -> `rknn_mem_sync`。

   需要显式关闭时使用：

   ```bash
   --no-staging-pipeline
   ```

主运行链路没有启用双 buffer RKNN bound-input zero-copy 流水线。独立探针
`../board/tools/probe_rknn_double_bound_input.cpp` 已验证“两套 RKNN input
memory 可以切换，并且 OpenCL 可以在非阻塞 RKNN run 期间写下一套 buffer”。但生产
路径当前默认使用更稳的 staging pipeline。

## 板端更新和重新编译

更新代码到板端后，在 `board_cpp` 内部重新编译 native 库和 C++ 可执行文件：

```bash
cd MeetEye/face_rc/board_cpp
bash build_libs.sh
bash build.sh
```

## 当前推荐摄像头运行命令

这条命令对齐当前 Python 板端 smoke path，并开启扇区输出、WebSocket、debug
JSONL、硬件采样和终端 profile 汇总：

```bash
taskset -c 0-6 bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --track-buffer 120 \
  --lost-velocity-decay 0.85 \
  --inherit-center-dist-thresh 1.0 \
  --inherit-size-ratio-thresh 0.5 \
  --inherit-ambiguity-margin 0.25 \
  --smooth-bbox-alpha 0.7 \
  --sector-output \
  --num-sectors 8 \
  --debug-jsonl board_output/board_cpp_camera_debug.jsonl \
  --profile-system-load \
  --system-load-interval-ms 200 \
  --print-profile-summary \
  --profile-interval 30 \
  --no-output-jsonl \
  --no-stdout-json
```

命令说明：

- `taskset -c 0-6`：把进程绑定到 CPU 0 到 CPU 6，留一个 CPU 给系统和其他服务。
- `bash run_smoke.sh`：通过 wrapper 启动 C++ 程序，并自动设置随包库路径。
- `--camera-device /dev/video0`：从 `/dev/video0` 读取 V4L2 MJPEG 摄像头流。
  当前 USB 摄像头的 `/dev/video1` 是 metadata 节点，不是图像节点。
- `--camera-width 1920 --camera-height 1080 --camera-fps 30`：请求 1920x1080、
  30 FPS 的 MJPEG 采集。
- `--track-buffer 120`：跟踪器内部将丢失 track 保留最多 120 帧。
- `--lost-velocity-decay 0.85`：目标丢失后逐帧削弱预测速度，降低遮挡后轨迹继续
  漂移的幅度。
- `--inherit-center-dist-thresh 1.0`：短暂遮挡继承旧 ID 的中心距离门限，`1.0`
  表示新框与旧轨迹最后观测框或 Kalman 预测框的中心距离小于约 1 个平均框高。
- `--inherit-size-ratio-thresh 0.5`：短暂遮挡继承旧 ID 的尺寸门限，新框和旧框
  的宽高比例乘积低于 0.5 时不继承。
- `--inherit-ambiguity-margin 0.25`：短暂遮挡继承旧 ID 的唯一候选门限，最近旧轨迹
  需要比第二近旧轨迹至少近 0.25 个平均框高，否则不继承。
- `--smooth-bbox-alpha 0.7`：增强 bbox 防抖；默认是 0.5，调到 0.7 会更稳但有轻微
  跟随延迟。
- `--sector-output --num-sectors 8`：输出 Python 兼容的 8 扇区 JSON。
- `--debug-jsonl board_output/board_cpp_camera_debug.jsonl`：写完整逐帧 debug
  JSON，包括耗时、检测、跟踪、扇区和 FPS。
- `--profile-system-load --system-load-interval-ms 200`：每 200 ms 后台采样
  CPU/GPU/NPU/内存/温度，并写出 `*_system_profile.jsonl` 和 summary JSON。
- `--print-profile-summary --profile-interval 30`：每 30 帧打印一次各阶段平均
  耗时，程序退出时再打印最终汇总。
- `--no-output-jsonl`：不生成默认的 target-only `board_*.jsonl` 文件。
- `--no-stdout-json`：不在终端打印逐帧 JSON，避免影响实时运行和查看 profile。

摄像头模式默认一直运行。如需短测试，再加：

```bash
--max-frames N
```

## 角度输出

角度输出默认对齐当前 Python 行为：`fit_degree=4`，即使用 `fit_4` 系数。
它不会自动跟随 `fisheye_calib.yaml` 里的 `default_fit: "5_constrained"`，除非
显式改代码或命令参数。

输入 JPEG 帧的分辨率必须与生成 map 时的源分辨率一致。如果 `meta.txt` 里包含
`img_width` 和 `img_height`，程序会在进入 OpenCL/RKNN 前先检查分辨率。

如果 `base_map_x.bin` 和 `base_map_y.bin` 存在，C++ elevation 会使用与 Python
板端一致的 panorama-to-fisheye 查表路径。

## Buildroot 最小运行文件

部署到 Buildroot 设备时，至少需要这些文件：

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
- `libturbojpeg.so.0`，可以来自目标系统库路径，也可以复制到 `board_cpp/lib`
- 目标系统里的 `libOpenCL.so.1`

当前 checkpoint 已覆盖 JPEG 解码、V4L2 MJPEG 摄像头输入、OpenCL 预处理、RKNN
推理、native merge、native HybridSORT 跟踪、Python 兼容 target JSON、扇区
JSON、WebSocket 输出、fit_4 角度/距离字段和硬件 profile 采样。
