# MeetEye

**基于鱼眼全景摄像头的实时多人定位系统。**  
在 360° 全景画面中检测、跟踪每一位人员，并实时输出方位角、仰角和距离——单 GPU 即可达到交互帧率。

[English →](README.md)

---

## 演示

> **大会议室 · 多人开会** — 补漏检测 + 8 扇区聚合 + 轨迹续命  
> HybridSORT 跟踪 · 3 切片全景 · 1280 宽 · 前 180 秒

<!-- 内嵌播放器步骤：网页打开此 README → Edit → 把本地 docs/demo_coast.mp4 拖入编辑框
     → GitHub 自动上传生成 https://github.com/user-attachments/assets/xxxx 链接
     → 用该链接替换下面这一行占位 → 保存即显示播放器 -->
[<!-- 在此粘贴上传 docs/demo_coast.mp4 后得到的 user-attachments 链接 -->](https://github.com/user-attachments/assets/141b880a-fcd8-4ae8-ba29-a69648f7ba5c)

<br>

> **↳ 同一场景 · 扇区角度可视化窗口** — 雷达图 + 半球视图实时显示各扇区方位角/仰角（与上方结果视频配套的本地 `angle_visualizer.py` 窗口）

<!-- 内嵌播放器步骤：网页打开此 README → Edit → 把本地 docs/demo_coast_angle.mp4 拖入编辑框 → 用生成的 user-attachments 链接替换下面这一行占位 -->
<!-- 在此粘贴上传 docs/demo_coast_angle.mp4 后得到的 user-attachments 链接 -->

<br>

> **小会议室 · 4 人黑板交流** — HybridSORT 跟踪 · OSNet ReID · 3 切片全景 · 960 × 630

https://github.com/user-attachments/assets/10e2b8d3-aa76-4ed0-9236-3f568cd06181

*如需将自己的结果视频压缩后上传 GitHub，运行：*
```bash
python compress_demo.py -i your_result.mp4 -o demo.mp4 --duration 180 --scale 1280:-2 --crf 18
```

---

## 系统流程

```
鱼眼摄像头（360°）
      │
      ▼
 GPU 鱼眼展开  ────────────────────────────────────────────────┐
      │                                                        │
      ▼                                              全景图（3840 × 1080）
 全景切片  ──  3 张重叠子图                                    │
      │                                                        │
      ▼                                                        │
 YOLOv8 / YOLO26 姿态检测（GPU 批量推理）                      │
      │                                                        │
      ▼                                                        │
 跨切片去重（NMS + ReID 相似度）                               │
      │                                                        │
      ▼                                                        │
 OSNet ReID 特征提取（GPU 裁图 → 特征向量）                    │
      │                                                        │
      ▼                                                        │
 多目标跟踪                                                    │
  ├── HybridSORT（IoU + VDC + TCM，默认）                      │
  └── BoT-SORT（IoU + ReID EMA）                              │
      │                                                        │
      ▼                                                        │
 每人方位角 / 仰角 / 距离                                      │
      │                                                        │
      ▼                                                        │
 输出：标注视频  │  JSON WebSocket 流  │  WebUI 浏览器界面     │
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **360° 全覆盖** | 单鱼眼镜头 → GPU 展开全景；左右接缝处的人员通过环绕边界匹配器跨缝重识别 |
| **双跟踪器** | **HybridSORT**（默认）：IoU + 四角点速度方向一致性（VDC）+ 置信度调制（TCM），适合交叉穿越、密集人群。**BoT-SORT**：IoU + ReID EMA，在稀疏场景下更稳定 |
| **跟踪稳定性修复** | VDC 速度幅值衰减门控（往复运动时速度向量衰减而非积累错误方向）；BoT-SORT 分配前重叠检测（消除 fuse_score 置信度偏差和污染特征参与分配） |
| **GPU 全流水线** | 鱼眼展开、YOLO 批量推理、OSNet ReID 全部在 GPU 上完成；RTX 3080 端到端延迟约 30–50 ms/帧 |
| **3D 角度输出** | 通过标定多项式拟合给出每目标的方位角（°）和仰角（°）；利用双眼关键点像素间距估算距离（m） |
| **补漏检测** | 可选第二检测模型（`--recall-boost`，如 `yolo26n`）捞回 pose 漏掉的遮挡/背身目标，融合进流水线；无关键点目标用「框顶中心」合成参考点估算角度 |
| **扇区聚合输出** | `--sector-output` 将水平 360° 等分为 N 个扇区，忽略 ID，每扇区取检测框最大者输出方位/俯仰角，画面红框高亮（WebUI 模式） |
| **轨迹续命平滑** | `--coast-frames N` 目标瞬时漏检时用 Kalman 预测框续命至多 N 帧，N 帧内重现则接回、否则停止，平滑闪烁（独立开关，不改正常框） |
| **人脸识别 / 说话检测** | 可选：AdaFace 人脸识别（`--use-face-rec`）按 track_id 标注姓名；MediaPipe 嘴部开合比说话检测（`--talking-detection`） |
| **两种运行模式** | **本地模式**（`main.py`）：摄像头/视频/图片文件夹 + OpenCV 显示。**WebUI 模式**（`webui/`）：FastAPI 服务器 + 浏览器仪表盘 + JSON WebSocket |
| **TensorRT 支持** | 用 `export_trt.py` 将 YOLO `.pt` 导出为 `.engine`，推理速度约提升 3× |

---

## 快速开始

### 1 · 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：如需 OSNet ReID 特征，需额外安装 `torchreid`：
> ```bash
> pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
> ```

---

### 2 · 本地模式（`main.py`）

```bash
cd mytest

# 接鱼眼摄像头实时运行
python main.py \
    --model-path ../yolo26n-pose.pt \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz

# 视频文件输入，保存标注结果视频
python main.py \
    --video-path /path/to/video.mp4 \
    --model-path ../yolo26n-pose.pt \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz \
    --save-video --video-name result.mp4

# 图片文件夹批量处理（无显示）
python main.py \
    --folder-path /path/to/images/ \
    --model-path  ../yolo26n-pose.pt \
    --map-file    ../maps/3840_fisheye_maps_2026.5.18.npz
```

**运行时键盘快捷键**

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 保存当前帧（3 张图片） |
| `i` | 切换置信度阈值 0.3 ↔ 0.5 |
| `o` | 切换 IoU 阈值 0.3 ↔ 0.45 |
| `a` | 循环切换角度显示模式：详细 → 概览 → 关闭 |

---

### 3 · WebUI 模式

**推理服务器**（GPU 机器）：
```bash
cd mytest
python main_GPU_webui.py \
    --model-path ../yolo26n-pose.engine \
    --map-file   ../maps/3840_fisheye_maps_2026.5.18.npz
# 在局域网内任意浏览器中打开输出的 URL
```

**摄像头客户端**（摄像头机器）：
```bash
python camera_client.py ws://<SERVER_IP>:<PORT>/ws/camera
```

**角度可视化器**（可选，任意机器）：
```bash
python angle_visualizer.py ws://<SERVER_IP>:<PORT>
# 使用合成测试数据（无需摄像头）
python angle_visualizer.py --test
```

---

## 参数说明

### 跟踪器选择

```bash
--tracker hybridsort   # 默认；适合密集/交叉场景
--tracker botsort      # 备选；ReID 稳定的稀疏场景更佳
--tracker none         # 纯检测，不跟踪
```

### 跟踪器调参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-reid` / `--no-use-reid` | `True` | 在 HybridSORT 中启用 OSNet ReID（`--no-use-osnet` 时自动失效，无特征不参与关联） |
| `--reid-emb-weight-high` | `0.1` | HybridSORT 第一轮关联中 ReID 嵌入权重 |
| `--botsort-match-thresh` | `0.3` | BoT-SORT 第一阶段关联阈值 |
| `--appearance-thresh` | `0.2` | BoT-SORT ReID 门控阈值 |
| `--smooth-bbox` / `--no-smooth-bbox` | `True` | 对输出框宽高做 EMA 平滑 |
| `--smooth-bbox-alpha` | `0.5` | EMA 系数（0 = 不平滑，1 = 完全冻结） |
| `--coast-frames` | `0` | 丢失轨迹用 Kalman 预测框续命至多 N 帧（>0 启用，浅蓝细线，不改正常框） |
| `--kalman-bbox` | 关 | 输出 Kalman 状态框替代 YOLO 原始框，并持续显示丢失目标预测框 |

### 检测与全景

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-path` | `yolo26n-pose.engine` | YOLO 模型（`.pt` 或 `.engine`） |
| `--conf-threshold` | `0.1` | YOLO 置信度阈值 |
| `--num-slices` | `3` | 每帧全景切片数（2–7） |
| `--slice-overlap` | `0.1` | 相邻切片重叠比例 |
| `--crop-divisor` | `3` | 裁去全景图顶部 `1/N`（去除鱼眼畸变区域） |
| `--osnet-model` | `osnet_ain_x1_0` | ReID 骨干网络 |
| `--no-use-osnet` | — | 完全跳过 OSNet 特征提取（提速；ReID 关联自动失效） |
| `--kpt-track` | 关 | 跟踪层用关键点推导框替代 YOLO 原始框，减少大框重叠误判 |
| `--kpt-display` | 关 | 仅显示层用关键点推导框绘制，不影响跟踪逻辑 |

### 补漏检测（recall boost）

用第二个纯检测模型捞回 pose 漏掉的遮挡 / 背身目标，作为「无关键点框」融合进流水线。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--recall-boost` | 关 | 启用补漏检测 |
| `--recall-model` | `./yolo_model/yolo26n.engine` | 补漏检测模型（建议用 nano 纯检测模型，须与主模型不同） |
| `--recall-conf-threshold` | `0.4` | 补漏模型专用置信度阈值（独立于 `--conf-threshold`） |
| `--recall-match-iou` | `0.3` | 补漏框与 pose 框 IoU ≥ 此值视为已覆盖而丢弃 |
| `--recall-head-ratio` | `0.12` | 无关键点框合成鼻子点时，头部 y 取框顶往下该比例×框高 |

### 扇区聚合输出（WebUI 模式）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sector-output` | 关 | JSON 改为扇区聚合格式，每扇区取最大目标，画面红框高亮该目标 |
| `--num-sectors` | `8` | 水平 360° 等分扇区数（可改 16 等） |

### 人脸识别 / 说话检测（可选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-face-rec` / `--no-use-face-rec` | `False` | 启用 AdaFace IR-18 人脸识别，按 track_id 标注姓名 |
| `--face-library-dir` | `face_library` | 人脸特征库目录（每个 `.npy` 一人，文件名即人名） |
| `--talking-detection` / `--no-talking-detection` | `False` | 启用 MediaPipe FaceLandmarker 嘴部开合比（MAR）说话检测 |
| `--talking-mar-threshold` | `0.06` | MAR 阈值，超过判定为说话 |

### 输出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output-dir` | `yolo_pose_output` | 输出目录 |
| `--save-video` | 关 | 保存标注结果为 `.mp4`（本地与 WebUI 模式均支持；WebUI 启动即录制到 `--video-name`，退出自动 faststart） |
| `--video-name` | 自动 | 保存视频文件名 |
| `--save-frames` | 关 | 逐帧保存为 JPEG |
| `--save-crops` | 关 | 保存每人裁图 |
| `--save-json` | 关 | 将每帧推理结果追加写入 JSONL 文件（仅 WebUI 模式） |
| `--use-dual-windows` | 关 | 同时显示纯检测窗口和跟踪窗口（本地模式） |

---

## 输出 JSON 格式

每帧结果通过 `/ws/inference` 广播（WebUI 模式），也可选择写入 JSON 文件：

```json
{
  "timestamp": 1747612800.123,
  "frame_id": 42,
  "targets": {
    "1": {
      "id":             1,
      "azimuth":        12.5,
      "elevation":       3.1,
      "eye_pixel_dist": 18.4,
      "distance":        2.1,
      "features":       [0.012, -0.034, ...]
    }
  }
}
```

| 字段 | 单位 | 说明 |
|------|------|------|
| `azimuth` | ° | 水平方位角，以摄像头正前方为 0°，顺时针为正 |
| `elevation` | ° | 俯仰角，水平面为 0°，向上为正 |
| `eye_pixel_dist` | px | 全景图中双眼关键点像素距离 |
| `distance` | m | 估计距离（标定多项式，典型范围 0–5 m） |
| `features` | — | 512 维 L2 归一化 OSNet ReID 特征向量 |

### 扇区聚合格式（`--sector-output`）

开启后改为按扇区输出（忽略 track_id），每扇区一条，`has_target` 表示该扇区本帧有无目标：

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

```
MeetEye/
├── mytest/
│   ├── main.py                  # ① 本地模式入口
│   ├── config.py                # CLI 参数定义与默认值
│   ├── core/
│   │   ├── panorama.py          # GPU 鱼眼展开（grid_sample）
│   │   ├── detector.py          # YOLO 姿态检测封装
│   │   ├── slicer.py            # 全景切片、跨切片 NMS + ReID 合并、补漏融合
│   │   ├── tracker.py           # BoT-SORT 与 HybridSortTracker 封装（含续命/coasting）
│   │   ├── angle_calculator.py  # 方位角 / 仰角 / 距离估算
│   │   ├── camera.py            # 摄像头 / 视频 / 图片文件夹输入
│   │   └── boundary_matcher.py  # 环绕边界跨缝重识别
│   ├── utils/
│   │   ├── feature_extractor.py # OSNet torchreid 封装（GPU 裁图路径）
│   │   ├── visualizer.py        # 检测框 / 关键点 / 角度绘制
│   │   ├── sector.py            # 扇区聚合（共享给 webui JSON 与画面红框高亮）
│   │   ├── distance_estimator.py# 头部姿态修正的距离估算
│   │   ├── talking_detector.py  # MediaPipe 嘴部开合比说话检测
│   │   └── display.py           # OpenCV 显示 / 布局辅助
│   ├── face_rec/                # AdaFace 人脸识别（可选，按 track_id 标注姓名）
│   ├── models/                  # MediaPipe 模型权重（face_landmarker.task）
│   ├── main_GPU_webui.py        # ② WebUI 模式入口（FastAPI）
│   └── webui/                   # 推理处理器、FastAPI 路由、WebSocket、GPU 监控
├── HybridSORT/                  # Hybrid-SORT 跟踪器源码
│   └── trackers/hybrid_sort_tracker/
│       ├── hybrid_sort.py       # 核心跟踪器（含速度幅值门控补丁）
│       ├── hybrid_sort_reid.py  # ReID 变体（相同补丁）
│       └── association.py       # IoU / VDC / TCM 关联函数
├── maps/                        # 预计算的鱼眼展开映射文件（.npz）
├── compress_demo.py             # 演示视频压缩工具（ffmpeg 封装）
├── export_trt.py                # YOLO ONNX → TensorRT 引擎导出
└── requirements.txt
```

---

## 跟踪算法：设计决策与 Bug 修复

### HybridSORT — 速度幅值衰减门控

原始 HybridSORT 的 VDC（速度方向一致性）假设目标做单调运动。当近静止目标发生**小幅往复运动**（如偏头靠向邻近人员后复位）时，跟踪器会沿偏头方向积累残留速度向量。在复位阶段，VDC 会惩罚正确匹配、奖励错误匹配，导致 ID 互换。

**修复位置**：`hybrid_sort.py`、`hybrid_sort_reid.py`，在更新 `velocity_lt/rt/lb/rb` 之前，计算最老参考观测到当前检测的中心位移，以平均框高归一化：

- 位移 **≥ 体高的 5%** → 正常更新速度向量（检测到持续运动）。
- 位移 **< 5%** → 将现有速度向量 **乘以 0.5 衰减**，而非覆盖更新。经过 3–4 帧衰减后幅值趋于零，VDC 贡献接近零，分配回退为纯 IoU。由于舞蹈等密集场景中的正常运动位移远超阈值，此修复不影响密集场景跟踪性能。

### BoT-SORT — 分配前重叠检测

原始 BoT-SORT 在 `linear_assignment` **之后**才对已匹配的检测对做重叠检测，为时已晚，无法影响分配本身。还存在两个附加问题：

1. **`fuse_score` 置信度偏差** — 代价矩阵被检测置信度加权，高置信度的重叠检测对**所有**轨迹都获得不公平的低代价，直接触发 ID 互换。
2. **污染特征参与分配** — 两个框重叠时，OSNet 裁图包含邻近人体，但受污染的嵌入向量仍被用于计算分配阶段的 `emb_dists`。

**修复位置**：`tracker.py`，将重叠检测（检测对 IoU > 0.1 / > 0.3）提前到代价矩阵构建**之前**，结果用于三处：
- 重叠检测在 fuse-score 步骤中使用 `score = 1.0`（消除置信度偏差）。
- 重叠检测列的 `emb_dists` 强制置 1.0（排除污染 ReID 参与分配）。
- Kalman 更新和特征更新阶段的 `freeze_feat` / `near_other` 标志也从相同的预计算集合中读取。

---

## 硬件与性能参考

| 配置 | 典型延迟 | FPS |
|------|---------|-----|
| RTX 3080 · YOLO `.engine` · 3 切片 · HybridSORT | 30–45 ms | 22–30 |
| RTX 3080 · YOLO `.pt` · 3 切片 · HybridSORT | 55–80 ms | 12–18 |
| 纯 CPU（无 GPU） | 300–600 ms | 1–3 |

> 各步耗时示例（30 帧均值）：①CPU→GPU 2ms  ②鱼眼展开 3ms  ③GPU→CPU 1ms  ④切片 2ms  ⑤YOLO 18ms  ⑥合并+ReID 8ms  ⑦跟踪 2ms  ⑧角度计算 1ms

---

## TensorRT 导出

```bash
python export_trt.py \
    --model yolo26n-pose.pt \
    --imgsz 1280 \
    --device 0
```

导出的 `.engine` 文件绑定到创建时使用的 GPU。

---

## 依赖项

| 包 | 用途 |
|----|------|
| `torch` + `torchvision` | GPU 推理、grid_sample 鱼眼展开 |
| `ultralytics` | YOLOv8 / YOLO26 检测 |
| `torchreid` | OSNet ReID 特征提取 |
| `opencv-python` | 视频读写、标注绘制 |
| `fastapi` + `uvicorn` | WebUI 服务器 |
| `numpy` | 数组运算 |
| `lap` | BoT-SORT 匈牙利算法（可选） |

完整列表见 [`requirements.txt`](requirements.txt)

---

## 许可证

本项目供研究与学习用途。  
包含的 HybridSORT 源码遵循其原始许可证（见 `HybridSORT/`）。
