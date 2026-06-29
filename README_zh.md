# MeetEye

**基于鱼眼全景摄像头的实时多人定位系统。**  
系统从 360 度鱼眼画面中完成全景展开、切片检测、多目标跟踪、角度/距离估计，并输出视频、WebUI 实时画面或 JSONL 结构化结果。

[English README ->](README.md)

---

## 项目概览

MeetEye 当前包含两条主要运行路径：

| 路径 | 用途 |
|------|------|
| `mytest/` | 服务器/GPU 原型与 WebUI 运行路径，适合 RTX/Jetson/TensorRT 环境调试和实验 |
| `face_rc/` | RK3588 板端部署路径，包含 RKNN INT8、NPU 三核切片并行、OpenCL direct-slice remap、headless JSONL、WebUI 和视频保存 |
| `fine-tune/` | YOLO 微调与校准工作区，包含数据转换、CVAT 修正、会议视频自动标注、INT8 校准切片导出和 TensorRT INT8 对比 |

服务器/GPU 主线仍以 `mytest/` 为主；板端部署和模型微调的详细流程分别维护在子目录 README 中，根目录 README 只保留入口说明。

---

## 核心能力

| 能力 | 说明 |
|------|------|
| 鱼眼全景展开 | 单鱼眼镜头展开为全景图，支持多人会议场景 |
| 全景切片检测 | 将全景图切成多个重叠子图，提升小目标人脸/人体检测召回 |
| YOLO 检测/姿态模型 | 支持 YOLOv8 / YOLO26 `.pt`、TensorRT `.engine`，板端支持 RKNN INT8 |
| 多目标跟踪 | 默认 HybridSORT，也保留 BoT-SORT 路径 |
| 角度/距离输出 | 输出每个目标的方位角、俯仰角和估计距离 |
| 扇区聚合输出 | `--sector-output` 按水平视场划分扇区，每个扇区输出最大目标 |
| WebUI | 浏览器查看实时画面、状态和 WebSocket/JSON 输出 |
| 边缘部署 | RK3588 板端运行见 [`face_rc/README.md`](face_rc/README.md) |
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

RK3588 部署路径位于 [`face_rc/`](face_rc/)，包含：

- RKNN INT8 模型加载和 NPU 三核切片并行；
- OpenCL direct-slice 鱼眼切片展开；
- headless 视频/本地摄像头 JSONL 输出；
- WebUI 预览、扇区输出和视频录制；
- C++ merge/NMS 加速库和性能统计。

详细板端环境、运行命令、锁频检查、性能记录和常见问题见 [`face_rc/README.md`](face_rc/README.md)。

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
      "azimuth": 12.5,
      "elevation": 3.1,
      "distance": 2.1
    }
  }
}
```

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
├── mytest/                 # 服务器/GPU 本地运行和 WebUI
├── face_rc/                # RK3588 板端部署运行时
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
| [`face_rc/README.md`](face_rc/README.md) | RK3588 板端部署、headless/WebUI 命令、性能优化记录 |
| [`fine-tune/README.md`](fine-tune/README.md) | YOLO 微调、校准切片、TensorRT INT8 和结果对比 |

---

## 许可证

本项目用于研究和教育场景。  
HybridSORT 代码遵循其原始许可证，详见 `HybridSORT/`。
