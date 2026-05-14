# 项目重构说明

## 新的目录结构

```
mytest/
├── config.py                    # 配置文件（保持不变）
├── main.py                      # 主程序（整合版）
├── fisheye_calib.yaml           # 标定文件
│
├── core/                        # 核心处理模块
│   ├── __init__.py
│   ├── camera.py                # 相机/视频处理
│   ├── panorama.py              # 鱼眼全景展开（整合了旧的 fisheye_panorama.py 和 panorama_processor.py）
│   ├── detector.py              # YOLO姿态检测
│   ├── slicer.py                # 全景切片处理
│   ├── angle_calculator.py      # 角度计算
│   ├── boundary_matcher.py      # 边界ID匹配
│   └── tracker.py               # BoT-SORT跟踪器
│
├── utils/                       # 工具模块
│   ├── __init__.py
│   ├── display.py               # 显示管理
│   ├── visualizer.py            # 绘制/可视化函数（从 main.py 拆分）
│   ├── feature.py               # 相似度计算等特征工具（整合了重复代码）
│   ├── feature_extractor.py     # 特征提取器
│   └── helpers.py               # 其他工具函数
│
├── scripts/                     # 辅助脚本
│   ├── diagnose.py              # 边界匹配诊断工具
│   ├── db_tool.py               # 数据库管理工具
│   └── db.py                    # 完整的数据库模块
│
├── old/                         # 旧文件备份（可删除）
│   ├── main_with_similarity.py
│   ├── fisheye_panorama.py
│   └── panorama_processor.py
│
└── 代码说明/                    # 原有的说明文档
```

## 主要改进

1. **消除了重复代码**
   - 移除了 utils.py 和 db.py 中重复的相似度计算函数
   - 统一使用 `utils/feature.py` 中的函数

2. **按功能分层**
   - 核心处理逻辑在 `core/`
   - 工具函数在 `utils/`
   - 辅助脚本在 `scripts/`

3. **拆分了 main.py**
   - 绘制相关函数移到 `utils/visualizer.py`
   - 保持主程序简洁

4. **整合了鱼眼展开模块**
   - `fisheye_panorama.py` + `panorama_processor.py` → `core/panorama.py`

## 使用方式

和之前一样，直接运行：

```bash
python main.py
```

参数配置依然通过 `config.py` 或命令行参数。

## 注意

- 重构后的代码可能需要微调 import 路径
- 旧文件已备份在 `old/` 目录，确认无误后可以删除
