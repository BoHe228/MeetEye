# MeetEye

**面向鱼眼全景和广角摄像头的实时多人定位与人脸识别系统。**
系统可以从 360 度鱼眼画面或广角正畸画面中完成检测、跟踪、角度/距离估计、FaceID 识别、WebUI 显示和 JSON/WebSocket 输出；当前 RK3588 板端主链路已经迁移到 C++。

[English README ->](README.md)

---

## 项目概览

MeetEye 当前包含四条主要运行路径：

| 路径 | 用途 |
|------|------|
| `face_rc/board_cpp/` | 当前重点维护的 RK3588 C++ 板端链路，支持鱼眼三切片、广角整图、RKNN、WebUI、DisplayID、FaceID 和 profile |
| `mytest/` | 服务器/GPU 原型与 WebUI 运行路径，适合 RTX/Jetson/TensorRT 环境调试、广角实验和人脸识别观察面板 |
| `face_rc/` | Python 板端链路和历史实验代码，保留 RKNN/OpenCL/Python 版本调试入口 |
| `fine-tune/` | YOLO 微调与校准工作区，包含数据转换、CVAT 修正、会议视频自动标注、INT8 校准切片导出和 TensorRT INT8 对比 |

根目录 README 只保留项目总览和常用入口。板端具体命令、map 生成、模型转换、profile 报告和问题排查以 [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md) 为准。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 鱼眼全景链路 | 鱼眼画面展开为全景图后切成 3 个重叠子图，提升远距离小目标召回，并支持左右环绕边界处理 |
| 广角整图链路 | 复用 `Wide-Angle_test` 的广角相机标定，正畸后整图 letterbox 输入模型，关闭鱼眼专用环绕逻辑 |
| YOLO 检测模型 | GPU 端支持 `.pt/.engine`，板端支持 RKNN INT8；广角和鱼眼可以使用不同模型和不同 map |
| Native HybridSORT | 板端 C++ 内置 raw/public/display 多层 ID，支持小残框保护、输入前去重、lost-track reconnect 和 raw 重复轨迹退役 |
| DisplayID 固定槽 | `--display-id-max-count` 限制最终显示 ID 数量；优先保持原绑定，满槽时按 track 持续命中时间和框面积确认替换 |
| AdaFace FaceID | 支持 IR18/IR50 RKNN、五点对齐、动态特征库、主特征 + 多姿态样本、异步识别、raw track 绑定和 FaceID 后验回并 |
| WebUI | 浏览器查看实时画面、NPU/CPU 负载、JSON、显示槽、FaceID 缩略图；`--webui-frame-scale` 可降低 WebUI 传输压力 |
| 流水线执行 | 图片/摄像头解码预取、鱼眼 staging pipeline、广角双 YOLO worker、异步 AdaFace 和异步 WebUI JPEG 发送 |
| 角度/距离输出 | 输出每个目标的方位角、俯仰角、估计距离、eye distance、display/public/raw ID 和 FaceID |
| 扇区聚合输出 | `--sector-output` 按水平视场划分扇区，每个扇区输出最大目标 |
| 边缘部署 | RK3588 C++ 板端运行见 [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md) |
| 微调与校准 | YOLO 微调、校准和对比见 [`fine-tune/README.md`](fine-tune/README.md) |

---

## 快速入口

### 1 · 安装依赖

```bash
pip install -r requirements.txt
```

如果需要 OSNet ReID，需要额外安装 `torchreid`：

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

### 2 · 服务器/GPU 本地模式

```bash
cd mytest

python main.py \
    --video-path /path/to/video.mp4 \
    --model-path ../yolo26n-pose.pt \
    --map-file ../maps/3840_fisheye_maps_2026.5.18.npz \
    --save-video --video-name result.mp4
```

本地模式适合快速验证检测、跟踪和角度输出。更多参数见英文根 README 的配置说明。

---

### 3 · 服务器/GPU WebUI 模式

```bash
cd mytest

python main_GPU_webui.py \
    --model-path ../yolo26n-pose.engine \
    --map-file ../maps/3840_fisheye_maps_2026.5.18.npz
```

启动后在浏览器打开终端打印的地址即可查看 WebUI。

---

### 4 · RK3588 板端部署

RK3588 当前推荐使用 [`face_rc/board_cpp/`](face_rc/board_cpp/)：

- 鱼眼模式：OpenCL 展开 + 三切片 + RKNN 检测 + native HybridSORT；
- 广角模式：广角正畸整图输入 + 最多 2 个 YOLO NPU worker；
- 人脸识别：AdaFace IR18/IR50 RKNN，可选异步运行；
- 输出：WebUI、JSON WebSocket、JSONL、debug JSONL 和 profile summary；
- 硬件负载：读取 `/sys/kernel/debug/rknpu/load` 分别显示 NPU Core0/Core1/Core2。

鱼眼图片列表示例：

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
  --no-target-output-compact-ids \
  --webui-jpeg-quality 80 \
  --webui-frame-scale 0.5 \
  --webui-slot-fps 5 \
  --display-id-replace-area-ratio 1.50 \
  --display-id-min-box-size 20 \
  --no-output-jsonl \
  --no-stdout-json
```

鱼眼摄像头实时示例：

```bash
cd face_rc/board_cpp

sudo taskset -c 0-6 bash run_smoke.sh \
  --camera-device /dev/video0 \
  --camera-width 1920 \
  --camera-height 1080 \
  --camera-fps 30 \
  --map-dir maps/7.10_2560_yolo_slices_640_cpp \
  --projection-mode fisheye-panorama \
  --webui \
  --face-rec \
  --face-rec-dynamic-library \
  --dynamic-face-library-dir face_library_dynamic_camera_fisheye \
  --known-face-dir face_photos \
  --face-rec-model-preset ir50_webface4m \
  --face-rec-align-mode five-point \
  --face-rec-async \
  --display-id-max-count 8 \
  --no-target-output-compact-ids \
  --webui-jpeg-quality 80 \
  --webui-slot-fps 5 \
  --display-id-replace-area-ratio 1.50 \
  --display-id-min-box-size 20 \
  --no-output-jsonl \
  --no-stdout-json
```

广角摄像头实时示例：

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
  --dynamic-face-library-dir face_library_dynamic_camera_wide_angle \
  --known-face-dir face_photos \
  --face-rec-model-preset ir50_webface4m \
  --face-rec-align-mode five-point \
  --display-id-max-count 8 \
  --no-target-output-compact-ids \
  --face-rec-async \
  --webui-jpeg-quality 80 \
  --display-id-replace-area-ratio 1.50 \
  --webui-show-hidden-targets \
  --webui-slot-fps 5 \
  --display-id-min-box-size 20 \
  --no-output-jsonl \
  --no-stdout-json
```

详细板端环境、map 生成、AdaFace ONNX/RKNN 转换、已知人脸库模式、锁频检查、性能记录和常见问题见 [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md)。开/不开持久特征库的行为展示页放在 [`face_rc/board_cpp/docs/known_face_library_modes.html`](face_rc/board_cpp/docs/known_face_library_modes.html)；`board_output/` 只作为运行输出目录，默认被 Git 忽略。

---

### 5 · YOLO 微调与校准

YOLO 训练与校准工作区位于 [`fine-tune/`](fine-tune/)，包含：

- YOLO 姿态/人脸模型微调；
- 数据集转换和 CVAT 修正流程；
- 会议视频自动标注和抽帧；
- INT8 校准切片导出；
- TensorRT INT8 构建和 JSONL 召回对比。

详细流程见 [`fine-tune/README.md`](fine-tune/README.md)。

---

## 输出 JSON

常规输出按目标 ID 组织：

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

常见 ID 字段含义：

| 字段 | 含义 |
|------|------|
| `id` / `display_id` | 对外显示 ID，可通过 `--display-id-max-count` 限制最大数量 |
| `public_id` | C++ 输出层维护的 public track ID，用于短暂遮挡继承和边界恢复 |
| `raw_track_id` | native HybridSORT 底层 raw track ID；当前 FaceID 主绑定对象 |
| `face_id` | AdaFace 动态或静态人脸身份；没有识别结果时为 `null` |
| `face_recognition.known_name` | 已知照片库匹配到的姓名，来自照片文件名；与 `face_id` 共存，不替代动态 FaceID |
| `azimuth` / `elevation` | 方位角和俯仰角，单位为度 |
| `distance` | 基于人脸关键点距离估计的目标距离 |

启用 `--sector-output` 后按扇区组织：

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

## 项目结构

```text
MeetEye/
├── mytest/                 # 服务器/GPU 本地运行、WebUI、广角实验和人脸识别观察面板
│   ├── face_rec/           # AdaFace、动态 FaceID、聚类/相似度实验脚本
│   ├── local_window/       # 摄像头推流客户端和 JSON 可视化窗口
│   └── webui/              # FastAPI/WebSocket/WebUI 页面
├── face_rc/
│   ├── board_cpp/          # 当前 RK3588 C++ 主链路
│   │   ├── src/            # meeteye_cpp_smoke.cpp 拆分后的 C++ include 模块
│   │   └── tools/          # OpenCL/RKNN/native tracker/AdaFace 构建与转换工具
│   └── board/              # Python 板端链路和历史实验入口
├── fine-tune/              # YOLO 微调、数据转换和校准
├── HybridSORT/             # HybridSORT 跟踪器源码
├── maps/                   # 鱼眼展开映射矩阵
├── yolo_model/             # 模型文件目录
├── export_trt.py           # TensorRT engine 导出脚本
├── requirements.txt        # Python 依赖
├── README.md               # 英文总览
└── README_zh.md            # 中文总览
```

---

## 文档导航

| 文档 | 内容 |
|------|------|
| [`README.md`](README.md) | 英文总览、服务器/GPU 运行方式、参数和输出格式 |
| [`README_zh.md`](README_zh.md) | 中文总览和快速入口 |
| [`face_rc/board_cpp/README.md`](face_rc/board_cpp/README.md) | RK3588 C++ 板端部署、map 生成、AdaFace 转换、headless/WebUI 命令、性能优化记录 |
| [`face_rc/README.md`](face_rc/README.md) | RK3588 Python 板端历史链路和相关说明 |
| [`fine-tune/README.md`](fine-tune/README.md) | YOLO 微调、校准切片、TensorRT INT8 和结果对比 |

---

## 许可证

本项目用于研究和教育场景。  
HybridSORT 代码遵循其原始许可证，详见 `HybridSORT/`。
