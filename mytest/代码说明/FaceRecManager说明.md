# FaceRecManager — 人脸识别模块说明

## 概述

基于 AdaFace IR-18 的轻量级人脸识别模块，集成于主跟踪流水线。  
无需额外人脸检测器（MTCNN），直接复用 YOLO 姿态关键点完成人脸对齐。

---

## 文件结构

```
mytest/
├── face_rec/
│   ├── face_rec_manager.py      ← 核心管理器（本模块）
│   └── AdaFace/
│       └── net.py               ← IR-18 backbone 网络定义（仅此文件保留）
└── face_library/                ← 人脸特征库目录（用户自建）
    ├── 张三.npy                  ← 512D float32 L2归一化特征向量
    ├── 李四.npy
    └── ...

face_rec_model/
└── adaface_ir18_webface4m.ckpt  ← 预训练权重
```

---

## 特征库格式

每个 `.npy` 文件存储一人的 512 维 float32 特征向量（需预先 L2 归一化）。  
**文件名（无扩展名）即为识别后画面上显示的人名/ID。**

### 构建特征库

**只需把照片放进文件夹，运行一条命令即可。**

```
face_photos/          ← 把照片放这里，文件名即为人名
    张三.jpg
    李四.png
    ...
```

```bash
python mytest/face_rec/build_face_library.py
```

脚本会自动：
1. 用 OpenCV Haar 级联检测人脸并裁切（无需额外依赖）
2. 检测失败时自动回退为中心裁切
3. 提取 AdaFace 512D 特征向量
4. 保存到 `face_library/张三.npy`

常用选项：
```bash
# 自定义目录和模型路径
python mytest/face_rec/build_face_library.py \
    --photo-dir my_photos \
    --library-dir face_library \
    --model face_rec_model/adaface_ir18_webface4m.ckpt

# 强制重新提取（覆盖已有 .npy）
python mytest/face_rec/build_face_library.py --overwrite
```

> 照片建议：正面、光照均匀、无遮挡效果最佳。证件照/头像照片均可。

---

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-face-rec` | False | 是否启用人脸识别 |
| `--face-library-dir` | `face_library` | 特征库目录路径 |
| `--face-rec-model` | `face_rec_model/adaface_ir18_webface4m.ckpt` | 模型权重路径 |
| `--face-rec-threshold` | 0.35 | 余弦相似度阈值，低于此值为未知人员 |
| `--face-frontal-threshold` | 0.35 | 正面判断阈值（鼻子水平偏移/眼距） |

---

## 识别策略

```
目标首次出现（新 track_id）
    → 无条件触发识别（无论朝向）

目标后续帧
    → 已确认（face_name_map 中有记录）→ 直接复用结果，不再重算
    → 未确认 + 人脸正面 → 触发识别
    → 未确认 + 人脸侧/背 → 跳过

识别成功
    → face_name_map[track_id] = "张三"
    → 画面标注改为 "张三" 代替 "ID:3"
```

### 正面判断（is_frontal）

利用 YOLO 关键点计算 yaw 代理指标：

```
yaw_proxy = |鼻子_x - 双眼中点_x| / 眼距

yaw_proxy < face_frontal_threshold  →  正面
```

值越小要求越严格：
- 0.2：只接受几乎完全正面
- 0.35（默认）：允许轻度侧转（约 ±20°）
- 0.5：允许较大侧转

---

## 人脸对齐流程

```
YOLO 关键点（左眼、右眼、鼻子）
    ↓
1. 计算双眼连线角度 → getRotationMatrix2D 旋转校正 roll
2. 以眼睛中点下方 0.4×眼距 为裁切中心
3. 裁切边长 = 3 × 眼距
4. resize → 112×112 BGR
    ↓
AdaFace 前处理：(pixel/255 - 0.5) / 0.5
    ↓
IR-18 推理 → 512D L2归一化特征
```

---

## 匹配算法

矩阵乘法批量余弦相似度（O(N)，N 为库中人数）：

```python
sims = lib_matrix @ query_feature  # [N]，两侧均已 L2 归一化，点积 = 余弦相似度
best = lib_names[argmax(sims)]
matched = sims.max() >= threshold
```

---

## 与跟踪器的集成位置

```
YOLO → slicer.merge_detections → tracker.update → 【人脸识别】→ draw_detections
                                                        ↓
                                             face_name_map[track_id] = "张三"
```

`face_name_map` 在每帧维护，已消失的 track_id 自动清理。

---

## 注意事项

1. **俯视场景**：鱼眼俯视视角下人脸往往朝上，`face_frontal_threshold` 建议从 0.35 开始调，过小会导致大部分帧跳过识别。

2. **特征库质量**：注册图像建议使用正面、光照均匀的清晰人脸，避免遮挡。每人可注册多张，取均值或保存多个 .npy 文件（需修改 `_load_library` 支持多特征聚合）。

3. **阈值调整**：`face_rec_threshold` 过低（< 0.25）容易误识别，过高（> 0.5）容易漏识别。初始建议 0.35，实测后根据混淆情况调整。

4. **性能**：IR-18 单次推理 < 5ms（GPU），正面判断纯 CPU 计算可忽略。首次出现触发一次推理，确认后不再重复计算，对整体帧率影响极小。
