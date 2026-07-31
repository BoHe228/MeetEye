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
6. 可选调用 `libadaface_rknn.so` 做 AdaFace 人脸识别，并把 FaceID 绑定到
   `public_id/track_id`。
7. 执行 C++ 版跟踪后处理、边界去重、短暂丢失保留、扇区输出等逻辑。
8. 输出与 Python 板端兼容的 `targets` JSON、扇区 JSON、JSONL、WebSocket 数据和性能统计。

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
      +--> libadaface_rknn.so（可选）
      |        AdaFace RKNN embedding + TrackID/FaceID 绑定
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

AdaFace 人脸识别是可选功能，默认关闭。开启后只在高质量人脸框上做人脸识别：
人脸框最小边默认需要大于 `100px`，左右眼和鼻子关键点置信度默认需要大于
`0.6`。两眼距离 gate 默认关闭；如需重新限制，可以手动设置
`--face-rec-min-eye-dist`。识别结果主绑定内部 raw track id，不绑定可复用的
`display_id`，也不会反向修改跟踪结果。
WebUI 画面标签会显示 `ID:<display_id> F:<face_id>`；FaceID 绑定默认会在
raw track 暂时不输出后继续保留 `30` 秒。

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

### 广角相机整图 map

如果输入不是鱼眼全景，而是 `Wide-Angle_test` 里使用的广角相机正畸图，可以单独
生成 `wide_angle` map。这个通路不做三切片，正畸后的整图直接 letterbox 到
`640x640` 后送入 YOLO 模型；后续 tracker、FaceID、WebUI 和 JSON 输出继续复用
`board_cpp` 主链路。

在 `MeetEye` 目录下执行：

```bash
python3 face_rc/board_cpp/tools/generate_board_cpp_wide_angle_maps.py \
  --calib-yaml "Wide-Angle_test/2026.4.9_Calibrator_YoLoV8 (正畸+人脸识别+嘴部+角度)/calibration_yaml/2k_camera_calibration.yaml" \
  --cpp-output-dir face_rc/board_cpp/maps/wide_angle_2k_full_cpp \
  --imgsz 640 \
  --wide-undistort-alpha 0.5 \
  --wide-undistort-crop
```

输出目录同样包含：

```text
map_x.bin
map_y.bin
base_map_x.bin
base_map_y.bin
meta.txt
```

`meta.txt` 中会写入 `projection_mode=wide_angle` 和正畸后的
`wide_fx/wide_fy/wide_cx/wide_cy`。运行时读到这个字段后，会自动关闭鱼眼左右
环绕恢复、边界去重和三切片 staging pipeline，避免把广角图左右两侧误判为相邻
位置。

广角摄像头整图运行示例：

```bash
taskset -c 0-6 bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 2560 \
  --camera-height 1440 \
  --camera-fps 30 \
  --map-dir maps/wide_angle_2k_full_cpp \
  --projection-mode wide-angle \
  --webui \
  --webui-port 8080 \
  --debug-jsonl board_output/wide_angle_debug.jsonl \
  --no-output-jsonl \
  --no-stdout-json
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
lib/libadaface_rknn.so
lib/libhybrid_sort_native.so
lib/libmerge_fast.so
lib/libtracker_assoc_fast.so
```

如果系统里只有 `libturbojpeg.so.0`，没有开发环境常见的
`libturbojpeg.so` 软链接，`build.sh` 会直接链接
`/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0`。

## 可选：AdaFace 人脸识别

AdaFace 迁移分为离线模型准备和板端 C++ 运行两步。板端运行阶段不依赖 Python、
torch 或虚拟环境；只需要已经导出的 AdaFace RKNN 模型和可选的人脸特征库。
C++ runtime 会读取 RKNN 输入/输出 shape，只要模型输入仍是 `1x3x112x112`、
输出仍是 `512` 维 embedding，IR18 和 IR50 可以共用同一套 C++ 推理代码。

推荐使用 R50 WebFace4M。先在 PC/转换环境中用
`mytest/face_rec/AdaFace/net.py` 导出 ONNX：

```bash
cd MeetEye

python3 face_rc/board_cpp/tools/export_adaface_onnx.py \
  --model-name ir_50 \
  --checkpoint face_rc/board_cpp/models/adaface_ir50_webface4m.ckpt \
  --output face_rc/board_cpp/models/adaface_ir50_webface4m_112.onnx
```

在安装了 `rknn-toolkit2` 的转换环境中转 RKNN。第一版建议使用 FP 模式，避免人脸
embedding 被 INT8 量化后影响相似度：

```bash
cd MeetEye

python3 face_rc/board_cpp/tools/convert_adaface_onnx_to_rknn.py \
  --onnx face_rc/board_cpp/models/adaface_ir50_webface4m_112.onnx \
  --output face_rc/board_cpp/models/adaface_ir50_webface4m_112.rknn \
  --dtype fp \
  --target-platform rk3588
```

如果需要回退到 IR18，可以继续使用旧模型路径
`models/adaface_ir18_112.rknn`，或者把导出脚本的 `--model-name` 改成 `ir_18`，
checkpoint/output 路径同步改成 IR18 文件。

如果已有 Python 版静态人脸库，可以直接复制 `.npy` 特征文件到
`face_rc/board_cpp/face_library/`。C++ 版支持读取 1D `512` 或 2D `N x 512`
float32 `.npy` 特征，每个文件名作为 `face_id/name`。

启用 C++ AdaFace：

```bash
taskset -c 0-6 bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-fps 30 \
  --map-dir maps/7.10_2560_yolo_slices_640_cpp \
  --track-buffer-seconds 30 \
  --face-rec \
  --face-rec-model-preset ir50_webface4m \
  --face-library-dir face_library \
  --face-rec-dynamic-library \
  --dynamic-face-library-dir face_library_dynamic_cpp \
  --face-rec-min-box-size 100 \
  --face-rec-min-keypoint-conf 0.6 \
  --face-rec-threshold 0.65 \
  --face-rec-margin 0.08 \
  --face-rec-relink \
  --face-rec-relink-threshold 0.65 \
  --face-rec-relink-margin 0.08 \
  --face-rec-relink-confirm-frames 2 \
  --face-rec-relink-max-current-samples 2 \
  --face-rec-display-tentative \
  --face-rec-tentative-threshold 0.50 \
  --face-rec-tentative-fail-frames 2 \
  --face-rec-update-similarity 0.65 \
  --face-rec-pose-sample-similarity 0.55 \
  --face-rec-update-max-similarity 0.75 \
  --face-rec-update-min-box-size 100 \
  --face-rec-update-min-keypoint-conf 0.7 \
  --face-rec-update-min-score 0.5 \
  --face-rec-primary-min-box-size 110 \
  --face-rec-primary-min-keypoint-conf 0.8 \
  --face-rec-primary-min-score 0.6 \
  --face-rec-primary-max-yaw-deg 10 \
  --face-rec-primary-ema-alpha 0.1 \
  --face-rec-max-samples-per-id 5 \
  --face-rec-max-per-frame 3 \
  --face-rec-binding-ttl-seconds 30 \
  --sector-output \
  --webui \
  --webui-port 8080 \
  --no-output-jsonl \
  --no-stdout-json
```

输出 JSON 中 `id` 默认使用可复用的 `display_id`，同时保留 `public_id`；如果加
`--no-display-id-reuse`，`id` 会退回原 public ID 行为。人脸识别结果单独写入：

```json
"face_recognition": {
  "matched": true,
  "face_id": "face1",
  "name": "face1",
  "score": 0.713,
  "second_score": 0.421,
  "dynamic": true,
  "quality_passed": true,
  "updated": false,
  "event": "match"
}
```

debug JSONL 中的 `face_recognition.debug` 会记录本帧 FaceID 决策过程：
`best_name/best_score/second_score` 表示当前 embedding 最像哪个 FaceID；
`strong_identity_match` 表示是否通过 `--face-rec-threshold` 和 `--face-rec-margin`；
`best_reserved_by_other` 和 `used_by_other_this_frame` 表示该 FaceID 是否已被其他
`public_id/track_id` 占用；`blocked_by_owner` 表示是否因此没有分配 FaceID。
如果同一帧内多个目标都实际运行了 AdaFace，`nearest_frame_public_id` 和
`nearest_frame_feature_score` 会给出本帧最近目标及其 embedding dot 相似度。

注意：FaceID 绑定的是 `public_id/track_id`，不是 `display_id`。低质量帧不会触发
AdaFace RKNN，也不会清空已有 FaceID，只会沿用上一次绑定结果。目标短暂未输出时，
绑定会按 `--face-rec-binding-ttl-seconds` 保留；如果同一个 `public_id/track_id` 在保留窗口内
重新输出，会继续使用原 FaceID，不会重新分配。

`--face-rec-model-preset ir50_webface4m` 是快捷开关，等价于使用
`models/adaface_ir50_webface4m_112.rknn`；`ir18_webface4m` 等价于
`models/adaface_ir18_112.rknn`。如果要测试其他 AdaFace 权重或手动命名的 RKNN，
直接使用 `--face-rec-model PATH` 指定完整模型路径即可。

同一帧内两个不同 `public_id/track_id` 不会同时使用同一个 FaceID。当前帧开始时，
程序会先把已经绑定的 FaceID 预留给原 `public_id/track_id`；未绑定的新 track 即使
相似度达到阈值，也不能抢占仍在画面中的旧绑定 FaceID。这样优先保证已有
track 与 FaceID 的稳定绑定。只有强匹配到已占用 FaceID 的新 track 会被阻止；如果
相似度低于匹配阈值，说明它不是该 FaceID，动态库开启时仍允许新建自己的 FaceID。

FaceID 后验回并默认开启，用于修正“同一人回来后 public_id 变了、首次匹配没过阈值而
误建新 FaceID”的情况。添加前，已绑定目标每 `1` 秒抽检时只和自己当前 FaceID 计算
相似度，并更新自己的动态库；如果一开始误建了 `face7`，后续不会自动改回历史
`face4`。添加后，已绑定的年轻动态 FaceID 会在抽检时额外排除自己，再和历史库重新
比对；如果连续命中同一个更早的 FaceID，且满足阈值、margin、质量和占用保护，就把
当前 `public_id/track_id` 的 FaceID 切回旧 FaceID，并把误建的动态 FaceID 合并删除。

回并适用条件如下：

- 当前绑定必须是动态 FaceID，且样本数不超过 `--face-rec-relink-max-current-samples`
  默认 `2`，主要处理刚误建的 FaceID；
- 候选 FaceID 必须是静态库身份，或者编号更早的动态 FaceID，不能回并到更晚新建的
  FaceID；
- 当前抽检帧必须满足动态库更新质量门控，也就是
  `--face-rec-update-min-box-size`、`--face-rec-update-min-keypoint-conf`、
  `--face-rec-update-min-score`；
- 候选相似度必须达到 `--face-rec-relink-threshold` 默认 `0.65`，并且比第二候选至少高
  `--face-rec-relink-margin` 默认 `0.08`；
- 候选 FaceID 不能被其他当前目标占用，也不能在同一帧被其他目标使用；
- 同一候选需要连续 `--face-rec-relink-confirm-frames` 次抽检命中，默认 `2` 次，才正式
  回并。

display_id 临时 FaceID 继承默认开启，用于处理“同一个人短暂断开后 display_id 已经接回，
但 public_id/track_id 变了，导致 FaceID 重新注册”的情况。触发条件更严格：

- 新 `public_id/track_id` 通过 display ID 的空间复用拿到旧 `display_id`；或者当前
  `display_id` 在 FaceID 记忆窗口内仍对应一个旧 FaceID；
- 旧 `public_id/track_id` 的 FaceID 绑定仍在 `--face-rec-binding-ttl-seconds` 保留窗口内；
- 新框附近不能有多个当前目标，避免多人靠近时冒领 FaceID；
- 旧 FaceID 不能被其他当前活跃 `public_id/track_id` 占用。

触发后先输出旧 FaceID，但标记为 `tentative=true`，事件为
`tentative_face_carry`。小框、cooldown 或每帧 AdaFace 次数限制导致本帧不运行 AdaFace
时，只继续临时继承，不更新特征库。后续满足识别条件时，如果与旧 FaceID 的相似度
达到 `--face-rec-tentative-threshold` 默认 `0.50`，事件变为 `tentative_confirm`，正式
复用旧 FaceID；如果连续 `--face-rec-tentative-fail-frames` 次低于阈值，默认 `2` 次，
会释放临时继承并回到正常匹配/动态注册流程。

同一个 raw track 只要刚运行过 AdaFace，就会按 `--face-rec-cooldown-seconds`
间隔限制下一次识别尝试；默认每 `0.333333` 秒最多尝试一次，在 30 FPS 下等价于约 `10` 帧。
现在 cooldown 按 FaceID 状态分层：
未绑定目标使用 `--face-rec-cooldown-seconds` 默认 `0.333333` 秒，优先尽快注册 FaceID；
刚绑定目标使用 `--face-rec-newly-bound-cooldown-seconds` 默认 `1` 秒，连续成功抽检
`--face-rec-newly-bound-checks` 默认 `3` 次后进入稳定状态；稳定目标使用
`--face-rec-stable-cooldown-seconds` 默认 `5` 秒，减少对 NPU 的长期占用；raw track
短暂消失后重新出现时，会在 `--face-rec-reappeared-probe-seconds` 默认 `1` 秒内使用
`--face-rec-reappeared-cooldown-seconds` 默认 `0.333333` 秒做快速复核。低于质量阈值的
小框仍不运行 AdaFace，只沿用已有 FaceID 绑定。
每帧最多运行 `--face-rec-max-per-frame` 次 AdaFace，默认 `3` 次。同步模式下，鱼眼链路
最多创建 3 个 AdaFace RKNN context，分别固定到 NPU0/NPU1/NPU2；同一帧内最多 3 个
人脸任务会并发推理，全部完成后再按原目标顺序写回 FaceID，所以 FaceID 决策顺序仍与
原同步版本一致。广角模式和异步模式只使用 1 个 AdaFace context，避免和广角 YOLO worker
抢同一个 NPU 核。手动传大于 `3` 的值会被限制为 `3`；传 `0` 会按默认 `3` 处理。

`--face-rec-async` 会启用保守异步识别：主线程只裁剪并提交 AdaFace 任务，不等待
RKNN 识别完成；后台只使用 1 个 AdaFace worker，结果在后续帧回收并写回当前仍然活跃
的 raw track。默认不开启，关闭时仍完全使用原同步流程。异步结果超过
`--face-rec-async-max-age-seconds` 默认 `1` 秒，或者对应 raw track 已经不在当前帧输出，
会被丢弃，避免旧识别结果污染新目标。`--face-rec-async-max-pending` 默认 `8`，限制
队列中和正在运行的 AdaFace 任务总数；同一个 raw track 同时最多挂起 1 个识别任务。

动态 FaceID 的 `.npy` 文件第一行作为主特征，后续行作为姿态/距离样本；匹配时仍对
同一 FaceID 的所有向量取最大 cosine 相似度。低于识别触发阈值的小框只继承已有
FaceID 绑定，不重新运行 AdaFace，也不会写入特征库；低质量抽检只用于匹配，不会写入特征库。
新动态 FaceID 注册和样本追加需要满足 `--face-rec-update-min-box-size`、
`--face-rec-update-min-keypoint-conf`、`--face-rec-update-min-score`。未匹配到已有
FaceID 的未知目标不会单帧立即注册；默认需要累计 `3` 次满足 sample 门控的 AdaFace
尝试，并且这几次特征两两相似度都达到 `--face-rec-threshold`，才有资格创建新动态
FaceID。正式创建时还要求 pending 样本里至少有 `1` 张满足 primary 门控，并优先用这
张 primary 样本作为主特征；如果只有 sample_ok 但没有 primary_ok，会继续 pending，
不会直接建库。确认次数由 `--face-rec-dynamic-enroll-confirm-frames` 控制，设为 `1`
可恢复旧的单次注册行为。普通高质量姿态样本默认达到
`--face-rec-pose-sample-similarity=0.55` 即可追加入库；主特征更新仍要求达到
`--face-rec-update-similarity=0.65`，避免主向量被侧脸、低清晰度帧拉偏。高于
`--face-rec-update-max-similarity=0.75` 的样本认为信息增量有限，不再重复追加。主特征只在更高质量帧上
更新，门控由 `--face-rec-primary-min-box-size`、`--face-rec-primary-min-keypoint-conf`、
`--face-rec-primary-min-score` 和 `--face-rec-primary-max-yaw-deg` 控制；默认只允许
yaw 不超过 `10°` 的正脸帧更新主特征，更新时按 `--face-rec-primary-ema-alpha` 做保守
EMA。每个动态 FaceID 默认最多保留 `12` 条向量，由 `--face-rec-max-samples-per-id`
控制；裁剪样本时会保留第一行主特征。静态 `.npy` 人脸库不会在运行中被自动改写。

排查人脸相似度异常时，可以保存 AdaFace 实际输入的 `112x112` crop：

```bash
--face-rec-align-mode five-point \
--face-rec-five-point-scale 1.20 \
--face-rec-debug-crops \
--face-rec-debug-crop-dir board_output/face_rec_crops
```

`--face-rec-align-mode eye` 是旧的两眼中点裁剪方式，`five-point` 使用左右眼、鼻尖、
左右嘴角对齐到 ArcFace/AdaFace 常用的 `112x112` 五点模板。实际推理使用
`--face-rec-align-mode` 指定的模式。`--face-rec-five-point-scale` 只影响
`five-point` 模式，默认 `1.20`；设置为 `1.10-1.20` 会在保持五点对齐关系的同时
采样更大范围，保留更多下巴和脸型轮廓。debug crop 功能默认关闭，只会在本帧实际
运行 AdaFace RKNN 时保存 JPEG。

打开 `--face-rec-debug-crops` 后，同一帧会同时保存两种 crop：

- `frame_000003_eye_public_1_display_1_score_0.806_size_91x122.jpg`
- `frame_000003_five_point_public_1_display_1_score_0.806_size_91x122.jpg`

文件名包含 `frame/public/display/score/size`，用于和 debug JSONL 逐帧对应，检查模型
看到的人脸是否裁偏、过小、侧脸或受鱼眼展开影响。

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
  --input-fps 30 \
  --track-buffer-seconds 30 \
  --output-jsonl board_cpp_results.jsonl \
  --no-stdout-json
```

同时写出完整 debug JSONL：

```bash
bash run_smoke.sh \
  --image-list test_frames/frames.txt \
  --input-fps 30 \
  --track-buffer-seconds 30 \
  --decode-prefetch \
  --output-jsonl board_cpp_targets.jsonl \
  --debug-jsonl board_cpp_debug.jsonl \
  --print-profile-summary \
  --no-stdout-json
```

debug JSONL 会在 `fps` 和 `timings_ms` 里写入运行帧率信息：
`fps.instant`、`fps.average`、`fps.frame_ms`。

跟踪排查时，debug JSONL 还会写入 `tracker_debug`：

- `tracker_debug.snapshots`：底层 native tracker 当前保留的 raw track，包括
  `raw_id`、`public_id`、`time_since_update`、`last_observation`、Kalman `state`，
  以及是否还能作为 `lost_confirmed` 继承候选。
- `tracker_debug.short_inherit`：旧 raw track 仍在有效候选时的短暂遮挡继承判断，
  记录每个候选 old raw 的中心距离、尺寸分数和过滤原因。
- `tracker_debug.public_recover`：旧 raw track 已不在有效候选时，使用 public 历史
  位置兜底恢复的判断，记录 local single target、missing frames、strict/relaxed
  gate 是否通过和最终结果。

参数说明：

时间窗口参数优先使用秒单位，启动时按 `--input-fps` 换算成内部帧数；
如果没有设置 `--input-fps`，默认使用 `--camera-fps`，当前默认是 `30` FPS。
旧的帧数参数仍可使用，并会作为兼容覆盖项。默认换算关系如下：

- `--track-buffer-seconds 30`：丢失 track 保留 30 秒，30 FPS 下约 900 帧。
- `--coast-seconds 0.3`：目标短暂消失后继续输出最后位置 0.3 秒，30 FPS 下约 9 帧。
- `--public-recover-max-seconds 30`：public ID 兜底恢复回看 30 秒，30 FPS 下约 900 帧。
- `--display-id-reuse-max-seconds 30`：display ID 空间复用回看 30 秒，30 FPS 下约 900 帧。
- `--display-id-reuse-fallback-min-seconds 5`：display ID 无条件复用至少等待 5 秒，
  30 FPS 下约 150 帧。
- `--display-id-public-binding-ttl-seconds 2`：`public_id -> display_id` 独占保留 2 秒，
  30 FPS 下约 60 帧。
- `--face-rec-cooldown-seconds 0.333333`：同一个 raw track 每 0.333333 秒最多运行一次 AdaFace，
  作为未绑定目标的主动识别间隔；30 FPS 下约 10 帧。
- `--face-rec-newly-bound-cooldown-seconds 1`：刚绑定 FaceID 后每 1 秒抽检一次。
- `--face-rec-newly-bound-checks 3`：同一个 FaceID 成功确认 3 次后进入稳定状态。
- `--face-rec-stable-cooldown-seconds 5`：稳定 FaceID 每 5 秒低频抽检一次。
- `--face-rec-reappeared-cooldown-seconds 0.333333`：raw track 断续回来后快速抽检间隔。
- `--face-rec-reappeared-probe-seconds 1`：断续回来后的快速抽检窗口保留 1 秒。
- `--face-rec-async`：开启保守异步 AdaFace，主帧循环不等待人脸识别 RKNN 完成。
- `--face-rec-async-max-pending 8`：最多保留 8 个排队或正在运行的异步识别任务。
- `--face-rec-async-max-age-seconds 1`：异步结果超过 1 秒才返回时直接丢弃。
- `--face-rec-binding-ttl-seconds 30`：FaceID 绑定保留 30 秒，30 FPS 下约 900 帧。
- `--boundary-time-window-seconds 3`：左右环绕边界恢复历史窗口 3 秒，30 FPS 下约 90 帧。

- `--print-profile-summary`：在终端 stderr 打印各阶段平均耗时和最大耗时。
- `--profile-interval N`：每 N 帧打印一次运行中的 profile 汇总。
- `--profile-system-load`：启动后台硬件负载采样线程。RK3588 NPU 占用会优先读取
  `/sys/kernel/debug/rknpu/load`，按 `Core0/Core1/Core2` 分别输出；如果进程没有
  debugfs 读取权限，NPU 三核负载会显示为空，不再退回不可靠的 devfreq 单值。
- `--system-load-interval-ms N`：硬件采样间隔，默认 200 ms，最小 50 ms。
- `--wide-angle-yolo-workers N`：广角整图模式下的 YOLO NPU worker 数，默认 `2`，
  最大 `2`。图片列表和摄像头预取模式都会复用同一套 wide-angle worker 流水线；
  frame N 和 frame N+1 会分别在 NPU0/NPU1 上并行推理，tracker、FaceID、JSON 和
  WebUI 仍按原帧序输出。广角模式下 AdaFace 会固定使用 NPU2，避免和 YOLO 两路 worker
  抢同一个 NPU 核。
- `--decode-prefetch`：图片列表模式下用 worker 线程提前读取和解码后续帧，
  让 JPEG 解码与 RKNN/OpenCL 推理、跟踪、人脸识别和输出阶段重叠。当前默认开启。
- `--no-decode-prefetch`：关闭图片列表解码流水线，退回逐帧同步读取和解码。
  调试 profile 时注意：`jpeg_decode` 表示后台线程实际解码耗时，`decode_wait`
  才是主线程等待解码结果的阻塞时间。
- `--input-fps VALUE`：把秒单位窗口换算成帧数时使用的输入 FPS。默认使用
  `--camera-fps`，也就是默认 `30`。图片列表如果是隔 3 帧/隔 5 帧抽帧，应显式设置
  为原视频 FPS 除以抽帧间隔，例如 `--input-fps 10` 或 `--input-fps 6`。
- `--track-buffer-seconds VALUE`：跟踪器内部保留丢失 track 的时间，默认 `30` 秒；
  在 30 FPS 下等价于约 `900` 帧。`--track-buffer N` 仍保留为旧帧数覆盖参数。
- `--lost-velocity-decay VALUE`：目标丢失后，每帧将 Kalman 预测速度和方向速度
  乘以该系数，默认 `0.85`；设为 `1.0` 等于保持原常速度预测。
- `--tracker-new-thresh VALUE`：允许新建 raw track 的检测框置信度阈值，默认
  `0.7`。未匹配到已有轨迹的框只有高于该阈值，才会进入新轨迹创建流程。
- `--tracker-min-hits N`：新 raw track 连续命中多少帧后才确认并输出 ID，默认
  `3`。这可以过滤只出现 1-2 帧的偶发小假框。
- `--tracker-input-small-dedup`：默认开启，在检测框进入 native HybridSORT 前先去掉
  同一小区域内的相邻小残框，减少同一个真实目标被切成多条 raw track。
- `--no-tracker-input-small-dedup`：关闭 tracker 输入前小框去重，用于排查是否误删。
- `--close-small-dedup-max-side VALUE`：参与小残框去重的最大边长，默认 `30` px；
  大于该尺寸的框不再走小框去重，避免真实相邻小人脸被误删。
- `--close-small-dedup-center-px VALUE`、`--close-small-dedup-x-gap-px VALUE`、
  `--close-small-dedup-y-center-px VALUE`：小残框去重的绝对像素距离门槛，默认分别为
  `32`、`8`、`12`，不再主要依赖按框高归一化的距离。
- `--close-small-dedup-y-cover VALUE`：小残框去重的纵向重合比例，默认 `0.65`；
  纵向重合不足时，即使横向接近也不会删除。
- `--close-small-public-alias-*`：小残框去重后是否允许合并 public ID 的更严格门槛；
  默认只有中心距离 `20` px 内、横向间隔 `4` px 内、纵向重合 `0.85` 以上，或两条
  raw track 本来已经映射到同一 public ID 时，才允许合并 public。
- `--new-track-center-suppress VALUE`：native 新建 raw track 时的中心距离抑制阈值，
  单位为平均框高，默认 `1.0`；新小框贴着已有小轨迹时不会直接创建新 raw track。
- `--new-track-suppress-max-side VALUE`：参与新建轨迹中心抑制的小框最大边长，默认
  `80`；值为 `0` 等于关闭该抑制。
- `--tracker-match-thresh VALUE`：native tracker 检测框与已有轨迹的关联阈值，默认
  `0.12`；值越低越容易把检测框接到已有 raw track，但多人靠近时错接风险会增加。
- `--lost-track-reconnect-center-thresh VALUE`：native 在新建 raw track 前，尝试把高分
  未匹配检测框直接接回已丢失 raw track 的中心距离阈值，单位为平均框高，默认
  `1.6`。
- `--lost-track-reconnect-multi-target-center-thresh VALUE`：当当前检测框附近还有其他检测框
  时使用的更严格接回阈值，默认 `1.2`。
- `--lost-track-reconnect-size-ratio VALUE`：native lost-track 直接接回的尺寸比例阈值，
  默认 `0.5`；比 public ID 继承更严格。
- `--lost-track-reconnect-ambiguity-margin VALUE`：native lost-track 直接接回的唯一最佳候选
  margin，默认 `0.35`；多个旧轨迹候选距离接近时不直接接回。
- raw track 接续使用全景环形坐标：native 层会把全景左右边界视为相邻位置，主匹配
  IoU、BYTE/OCR 补匹配 IoU、lost-track 接回中心距离、新建 raw track 前的中心/重叠
  抑制、方向速度估计和 Kalman 更新输入都会按环形距离处理，避免目标跨左右边界时被
  当成远距离目标而新建 raw track。
- `--byte-residual-size-ratio VALUE`：native 残框可靠观测保护的尺寸比例阈值，默认
  `0.45`；高分或低分小框相对上一可靠观测尺寸掉得过多时，不更新旧轨迹
  `last_observation` 和速度。
- `--byte-residual-max-side VALUE`：参与残框可靠观测保护的小框最大边长，默认 `40`；
  值为 `0` 等于关闭该保护。参数名保留 `byte-residual` 是为了兼容旧命令。
- `--inherit-center-dist-thresh VALUE`：短暂遮挡 ID 继承的中心距离阈值，单位是
  平均框高，默认 `2.0`；设为 `0` 基本关闭短暂遮挡继承。
- `--inherit-multi-target-center-dist-thresh VALUE`：当新框附近还有其他当前目标时，
  短暂遮挡继承会自动使用这个更严格的中心距离阈值，默认 `1.2`，用于降低多人靠近
  时的冒领 ID 风险。
- `--inherit-size-ratio-thresh VALUE`：短暂遮挡 ID 继承的尺寸比例阈值，默认
  `0.4`；新框和旧观测框/Kalman 框大小差太多时不继承。
- `--inherit-ambiguity-margin VALUE`：短暂遮挡 ID 继承的唯一最佳候选阈值，默认
  `0.25`；最近候选和第二近候选距离差太小时不继承。
- `--public-recover-center-dist-thresh VALUE`：旧 raw track 已不在有效候选时，
  使用最近 public ID 记录做低歧义恢复的中心距离阈值，默认 `2.0`。只有当前新框
  附近没有第二个当前目标时才会使用这个放宽阈值。
- `--public-recover-max-seconds VALUE`：低歧义 public ID 恢复的最大丢失时间，默认
  `30` 秒；超过该窗口后不再用旧 public ID 兜底恢复。`--public-recover-max-frames N`
  仍保留为旧帧数覆盖参数。
- `--public-alias-confirm-frames N`：两个 raw track 因边界/小框去重被认为是同一目标时，
  需要连续确认多少帧后才真正合并到同一个 public ID，默认 `6`。当前帧仍会删除一个
  重复输出；如果后续两个 raw track 明显分离，pending alias 会取消，不会长期绑死。
- `--raw-duplicate-retire`：默认开启。两个 raw track 已经被边界/小框/普通重复框去重
  连续判定为同一个目标后，才会把被删除输出的弱 raw track 从 native tracker 中退役；
  这不是 Kalman 状态融合，而是保留一条主 raw track、删除重复 raw track。
- `--no-raw-duplicate-retire`：关闭 raw track 退役，只保留当前帧去重和 public alias
  兜底。
- `--raw-duplicate-retire-confirm-frames N`：raw track 退役需要连续确认的帧数，默认
  `5`，比 public alias 更严格。
- `--raw-duplicate-retire-center VALUE`、`--raw-duplicate-retire-iou VALUE`、
  `--raw-duplicate-retire-size-ratio VALUE`、`--raw-duplicate-retire-y-cover VALUE`：
  raw track 退役的严格几何条件，默认分别为中心距离 `0.45` 个平均框高、IoU
  `0.55`、尺寸比例 `0.65`、纵向覆盖 `0.75`。
- `--raw-duplicate-retire-local-dist VALUE`：第三目标歧义半径，默认 `1.5` 个平均框高；
  如果重复框附近还有第三个当前目标，不执行 raw track 退役。
- `--raw-duplicate-retire-strong-public-seen N`：两个不同 public ID 都已经稳定输出超过该
  帧数时，不允许把其中一个 raw 退役，默认 `10`，用于避免把两个真实目标误合并。
- `--target-output-limit N`：非扇区 `targets` JSON 最多输出多少个目标，默认 `8`；
  设为 `0` 表示不限制。超过上限时按检测置信度和 bbox 面积选择前 N 个目标。
- `--target-output-compact-ids`：默认开启，非扇区 `targets` JSON 的外层 key 使用本帧
  输出序号 `1..N`；内部 `id` 仍使用当前 `display_id/public_id`。原始 `display_id` 和
  `public_id` 会继续作为字段输出。
- `--no-target-output-compact-ids`：关闭本帧输出序号，恢复用 `display_id/public_id`
  作为非扇区 `targets` JSON 的外层 key。
- `--display-id-reuse`：开启输出层 display ID 复用池，默认开启。内部仍使用 public ID 做
  跟踪、遮挡继承和边界恢复；WebUI 标签显示可复用的 `display_id`，非扇区 JSON
  会同时保留原始 `display_id/public_id`。同一个 `public_id` 短暂不输出后，
  会在保留窗口内优先拿回原 `display_id`；只有全新的 `public_id` 才会尝试复用
  最近消失、空间位置接近、bbox 尺寸相近的旧 `display_id`；如果仍找不到合适候选，
  再按 fallback 分配或复用，避免刚释放的显示 ID 被远处新目标抢走。关闭时输出逻辑与原 public ID 行为一致。该模式不会
  为了强行连续而重编号仍在输出中的目标；如果需要每帧严格压缩成 `1..N`，应单独
  实现 compact display 模式。
- `--no-display-id-reuse-spatial`：关闭 display ID 的空间优先复用，退回到按 fallback
  条件复用空闲编号；仍会尊重 `public_id -> display_id` 独占保留窗口。
- `--display-id-reuse-max-seconds VALUE`：display ID 空间复用的最大回看时间，默认
  `30` 秒。`--display-id-reuse-max-frames N` 仍保留为旧帧数覆盖参数。
- `--display-id-reuse-fallback-min-seconds VALUE`：允许非空间匹配的无条件复用，但只复用
  已消失至少该时间的空闲 display ID；默认 `5` 秒。`--display-id-reuse-fallback-min-frames N`
  仍保留为旧帧数覆盖参数。
- `--display-id-public-binding-ttl-seconds VALUE`：`public_id -> display_id` 绑定的独占
  保留时间，默认 `2` 秒。在窗口内同一个 `public_id` 重新输出时直接拿回原
  `display_id`；其他 public ID 不能通过空间复用或 fallback 抢占这个 display ID。
  `--display-id-public-binding-ttl N` 仍保留为旧帧数覆盖参数。
- `--display-id-reuse-center-thresh VALUE`：display ID 空间复用的中心距离阈值，
  单位为平均框高，默认 `2.0`。
- `--display-id-reuse-size-ratio VALUE`：display ID 空间复用的尺寸比例阈值，默认
  `0.4`。
- `--display-id-reuse-ambiguity-margin VALUE`：display ID 空间复用的唯一最佳候选
  兼容参数，默认 `0.25`。当前 display ID 复用只要通过空间距离和尺寸 gate，就选择
  综合距离/时间分数最好的候选，不再因为多个近邻候选而直接分配新编号。
- `--smooth-bbox-alpha VALUE`：跟踪框 EMA 防抖系数，默认 `0.6`；检测框抖动明显
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
- 浏览器页面使用 `/ws/frame` 接收画框后的 JPEG 帧，断线后会自动重连；
  `/video/infer` 仍保留为 MJPEG 流端点，方便外部工具直接访问。
- WebUI 只在有浏览器订阅 `/ws/frame` 或 `/video/infer` 时才额外重建全景和编码
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
  --input-fps 30 \
  --track-buffer-seconds 30 \
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
--min-box-width 12
--min-box-height 12
--min-box-aspect-ratio 0.35
--tracker-new-thresh 0.7
--tracker-min-hits 3
--public-alias-confirm-frames 6
--tracker-input-small-dedup
--new-track-center-suppress 1.0
--new-track-suppress-max-side 80
--lost-track-reconnect-center-thresh 1.6
--lost-track-reconnect-multi-target-center-thresh 1.2
--lost-track-reconnect-size-ratio 0.5
--lost-track-reconnect-ambiguity-margin 0.35
--byte-residual-size-ratio 0.45
--byte-residual-max-side 40
--no-close-small-dedup
--close-small-dedup-center 2.00
--close-small-dedup-center-px 32
--close-small-dedup-x-gap-px 8
--close-small-dedup-y-center-px 12
--close-small-dedup-y-cover 0.65
--close-small-public-alias-center-px 20
--close-small-public-alias-x-gap-px 4
--close-small-public-alias-y-cover 0.85
--close-small-dedup-max-area 900
--close-small-dedup-max-side 30
--no-normal-duplicate-dedup
--normal-duplicate-center 0.25
--normal-duplicate-abs-center-px 16
--normal-duplicate-size-ratio 0.70
--normal-duplicate-iou 0.45
--normal-duplicate-cover 0.75
--raw-duplicate-retire
--raw-duplicate-retire-confirm-frames 5
--raw-duplicate-retire-center 0.45
--raw-duplicate-retire-iou 0.55
--raw-duplicate-retire-size-ratio 0.65
--raw-duplicate-retire-y-cover 0.75
--raw-duplicate-retire-local-dist 1.5
--raw-duplicate-retire-strong-public-seen 10
```

不要为了强行匹配目标总数而过度降低这些阈值。当前 3 秒对比里，很多 C++ 多出的
行是低分短 track，不一定是重叠重复框；过松的边界邻居规则可能误删真实相邻的人。
`--min-box-width`、`--min-box-height` 和 `--min-box-aspect-ratio` 用于在进入 tracker
前过滤全景边界裁剪产生的极窄残框，避免这类非真实目标占用 public ID。
`--tracker-input-small-dedup` 默认开启，会在检测框送入 native HybridSORT 前先压掉
同一小区域内的相邻小残框，避免两个相近小框同时进入 tracker 后创建多条 raw track。
`--new-track-center-suppress` 和 `--new-track-suppress-max-side` 在 native 层阻止贴着
已有小轨迹的新小框直接创建新 raw track；值为 0 可关闭该中心距离抑制。
`--lost-track-reconnect-*` 在 native 新建 raw track 前执行严格接续：当高分未匹配检测框
与已丢失 raw track 的最近可靠观测或 Kalman state 在中心距离、尺寸比例和唯一最佳候选
上都满足阈值时，直接更新旧 raw track，而不是先新建 raw track 再靠 public ID 继承补救。
raw 层接续已经按全景宽度使用环形距离；因此左右边界两侧的框会按最短环形距离比较，
而不是按普通 x 坐标差值比较。
`--byte-residual-size-ratio` 和 `--byte-residual-max-side` 用于 native 残框可靠观测
保护：高分主匹配、BYTE 低分匹配和 OCR 补匹配中，只要小框相对上一可靠观测尺寸掉得
过多，就不更新 `last_observation`、Kalman 和方向速度，只让旧轨迹继续按未匹配状态
预测。保护命中时，小残框不会替代最终输出框；正式 target 仍输出上一可靠框，避免
WebUI/JSON 的 bbox 因残框突然缩小。值为 0 可关闭该保护。
默认开启 `--coast-hold`，public coasting 会把短暂丢失目标的最后真实位置补回输出；
如需使用预测位置输出 coast 框，可加 `--no-coast-hold`。
coast 框追加到最终结果后，还会再做一层只针对左右边缘相对框的环绕边界去重：
如果 active 框和 coast-hold 框在全景左右边界按环形距离重合，优先保留 active 框；
如果两个都是 coast-hold 框，则优先保留最近出现、历史更稳定或分数更高的框。
`--close-small-dedup-*` 用于处理最大边长默认不超过 30px、绝对中心距离和横纵间隔都很近、
纵向重合较高的小残框；该逻辑在 tracker 输入前和输出层都会执行，重复小框会被压掉。
public ID 合并不再跟随普通小框去重默认发生。两个 raw track 原本已经是同一个
public ID 时可直接压重复框；否则即使满足边界/小框 alias 条件，也要连续通过
`--public-alias-confirm-frames` 帧后才真正合并，期间如果明显分离会取消 pending alias。
`--normal-duplicate-*` 用于处理大小相近、中心非常近、且 IoU
或 x/y 覆盖明显的正常框重复输出，默认 IoU 和覆盖阈值比小框去重更严格。
两类去重默认开启；如果怀疑误删真实相邻目标，可分别用
`--no-close-small-dedup` 或 `--no-normal-duplicate-dedup` 关闭。
`--raw-duplicate-retire` 复用上述输出去重结果，但要求连续确认、空间更近、尺寸更像、
纵向覆盖更高，并且附近不能有第三个目标；满足后只退役当前帧被去重删掉的 raw track。
这用于减少同一目标被长期拆成多条 native raw track 的情况，触发次数可在 debug JSON
的 `tracker.raw_duplicate_retired` 中查看。若多人靠近场景出现误合并，可先用
`--no-raw-duplicate-retire` 回退。

## Python 命令兼容性

C++ runtime 接受当前 Python 板端无头命令里常用的参数，包括：

```text
--camera-device
--input-fps
--effective-fps
--profile-interval
--profile-system-load
--system-load-interval-ms
--angle-vectorized
--force-build
--track-buffer-seconds
--track-buffer
--lost-velocity-decay
--inherit-center-dist-thresh
--inherit-multi-target-center-dist-thresh
--inherit-size-ratio-thresh
--inherit-ambiguity-margin
--public-recover-center-dist-thresh
--public-recover-max-seconds
--public-recover-max-frames
--public-alias-confirm-frames
--target-output-limit
--target-output-compact-ids
--no-target-output-compact-ids
--tracker-new-thresh
--tracker-min-hits
--display-id-reuse
--no-display-id-reuse
--display-id-reuse-spatial
--no-display-id-reuse-spatial
--display-id-reuse-max-seconds
--display-id-reuse-max-frames
--display-id-reuse-fallback-min-seconds
--display-id-reuse-fallback-min-frames
--display-id-public-binding-ttl-seconds
--display-id-public-binding-ttl
--display-id-reuse-center-thresh
--display-id-reuse-size-ratio
--display-id-reuse-ambiguity-margin
--coast-seconds
--coast-frames
--coast-hold
--no-coast-hold
--face-rec-relink
--no-face-rec-relink
--face-rec-model-preset
--face-rec-model-name
--face-rec-debug-crops
--no-face-rec-debug-crops
--face-rec-debug-crop-dir
--face-rec-align-mode
--face-rec-five-point-scale
--face-rec-relink-threshold
--face-rec-relink-margin
--face-rec-relink-confirm-frames
--face-rec-relink-max-current-samples
--face-rec-display-tentative
--no-face-rec-display-tentative
--face-rec-tentative-threshold
--face-rec-tentative-fail-frames
--face-rec-dynamic-enroll-confirm-frames
--face-rec-cooldown-seconds
--face-rec-cooldown
--face-rec-newly-bound-cooldown-seconds
--face-rec-newly-bound-cooldown
--face-rec-stable-cooldown-seconds
--face-rec-stable-cooldown
--face-rec-reappeared-cooldown-seconds
--face-rec-reappeared-cooldown
--face-rec-reappeared-probe-seconds
--face-rec-reappeared-probe-frames
--face-rec-newly-bound-checks
--face-rec-async
--no-face-rec-async
--face-rec-async-max-pending
--face-rec-async-max-age-seconds
--face-rec-async-max-age
--face-rec-binding-ttl-seconds
--face-rec-binding-ttl
--face-rec-update-similarity
--face-rec-pose-sample-similarity
--face-rec-update-max-similarity
--face-rec-min-sample-diversity
--face-rec-max-samples-per-id
--face-rec-update-min-box-size
--face-rec-update-min-keypoint-conf
--face-rec-update-min-score
--face-rec-primary-min-box-size
--face-rec-primary-min-keypoint-conf
--face-rec-primary-min-score
--face-rec-primary-max-yaw-deg
--face-rec-primary-ema-alpha
--boundary-time-window-seconds
--boundary-time-window
--min-box-width
--min-box-height
--min-box-aspect-ratio
--tracker-input-small-dedup
--no-tracker-input-small-dedup
--new-track-center-suppress
--new-track-suppress-max-side
--lost-track-reconnect-center-thresh
--lost-track-reconnect-multi-target-center-thresh
--lost-track-reconnect-size-ratio
--lost-track-reconnect-ambiguity-margin
--byte-residual-size-ratio
--byte-residual-max-side
--no-close-small-dedup
--close-small-dedup-center
--close-small-dedup-center-px
--close-small-dedup-x-gap-px
--close-small-dedup-y-center-px
--close-small-dedup-y-cover
--close-small-public-alias-center-px
--close-small-public-alias-x-gap-px
--close-small-public-alias-y-cover
--close-small-dedup-max-area
--close-small-dedup-max-side
--no-normal-duplicate-dedup
--normal-duplicate-center
--normal-duplicate-abs-center-px
--normal-duplicate-size-ratio
--normal-duplicate-iou
--normal-duplicate-cover
--raw-duplicate-retire
--no-raw-duplicate-retire
--raw-duplicate-retire-confirm-frames
--raw-duplicate-retire-center
--raw-duplicate-retire-iou
--raw-duplicate-retire-size-ratio
--raw-duplicate-retire-y-cover
--raw-duplicate-retire-local-dist
--raw-duplicate-retire-strong-public-seen
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
- `--track-buffer-seconds 30`：跟踪器内部将丢失 track 保留最多 30 秒，当前为默认值。
  在 30 FPS 下等价于约 900 帧。
- `--lost-velocity-decay 0.85`：目标丢失后逐帧削弱预测速度，降低遮挡后轨迹继续
  漂移的幅度，当前为默认值。
- `--inherit-center-dist-thresh 2.0`：短暂遮挡继承旧 ID 的中心距离门限，`2.0`
  表示新框与旧轨迹最后观测框或 Kalman 预测框的中心距离小于约 2 个平均框高，当前为默认值。
- `--inherit-multi-target-center-dist-thresh 1.2`：如果新框附近还有其他当前目标，
  短暂遮挡继承会把中心距离门限收紧到该值，避免多人靠近时把旁边的人接成旧 ID。
- `--inherit-size-ratio-thresh 0.4`：短暂遮挡继承旧 ID 的尺寸门限，新框和旧框
  的宽高比例乘积低于 0.4 时不继承，当前为默认值。
- `--inherit-ambiguity-margin 0.25`：短暂遮挡继承旧 ID 的唯一候选门限，最近旧轨迹
  需要比第二近旧轨迹至少近 0.25 个平均框高，否则不继承。
- `--public-recover-center-dist-thresh 2.0`：旧 raw track 已不在有效 lost 候选时，
  允许在局部只有一个当前目标的情况下，把新 raw track 接回最近 public ID。
- `--public-recover-max-seconds 30`：public ID 兜底恢复最多回看 30 秒，避免很久以前
  的旧 ID 被重新接上；在 30 FPS 下等价于约 900 帧。
- `--smooth-bbox-alpha 0.7`：增强 bbox 防抖；默认是 0.5，调到 0.7 会更稳但有轻微
  跟随延迟。
- `--sector-output --num-sectors 8`：输出 Python 兼容的 8 扇区 JSON。
- `--debug-jsonl board_output/board_cpp_camera_debug.jsonl`：写完整逐帧 debug
  JSON，包括耗时、检测、跟踪、扇区和 FPS。
- `--profile-system-load --system-load-interval-ms 200`：每 200 ms 后台采样
  CPU/GPU/NPU/内存/温度，并写出 `*_system_profile.jsonl` 和 summary JSON。NPU
  优先使用 `/sys/kernel/debug/rknpu/load` 中的三核负载，JSON 字段为
  `npu_core_percent`、`npu_core0_percent`、`npu_core1_percent`、`npu_core2_percent`；
  运行用户需要有 debugfs 读取权限，否则 NPU 负载字段为空。
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
