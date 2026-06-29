# Fine-Tune 数据集说明

该目录存放 MeetEye 姿态模型微调、验证和 CVAT 人工修正相关的数据集。

## 小会议室吸顶数据集制作流程

当前已经人工修正完成的小会议室吸顶测试集，制作流程如下：

1. 从小会议室吸顶视频中抽帧，并展开为全景切片。
2. 生成初始 YOLO-pose 风格数据集，放在 `small_meeting_xiding_autolabel/`。
3. 将 YOLO-pose 标签转换为 CVAT 可导入的 COCO Keypoints：

   ```bash
   python3 fine-tune/scripts/yolo_pose_to_cvat_coco.py
   ```

4. 将每个 split 打包为 CVAT 项目导入用的 ZIP：

   ```bash
   python3 fine-tune/scripts/pack_cvat_coco_keypoints.py
   ```

5. 在 CVAT 中按 `COCO Keypoints 1.0` 导入 ZIP，人工修正 test split，
   然后按 `COCO Keypoints 1.0` 导出修正后的测试集。
6. 将 CVAT 导出的 ZIP 解压到：

   ```text
   small_meeting_xiding_cvat/export_test_coco_keypoints/
   ```

7. 将修正后的 COCO Keypoints 测试集重新转换为 YOLO-pose：

   ```bash
   python3 fine-tune/scripts/cvat_coco_keypoints_to_yolo_pose.py --overwrite
   ```

最终可用于 Ultralytics YOLO pose 训练/验证的数据集是：

```text
small_meeting_xiding_cvat_test_yolo/
```

注意：该目录中的 `train`、`val`、`test` 当前都指向同一批 276 张人工修正后的测试图片。
这样做是为了方便验证当前模型、检查数据格式，或者做小规模过拟合 sanity check。
不要把它当成真正独立的训练集和验证集划分。

## 小会议室吸顶相关目录

### `small_meeting_xiding_autolabel/`

从小会议室吸顶视频生成的中间 YOLO-pose 数据集。

当前状态：

- `images/train/`：1107 张训练用全景切片。
- `labels/train/`：1107 个训练切片对应的 YOLO-pose 标签文件。
- `images/test/`：已经删除。test 部分已由人工修正版本替代。
- `labels/test/`：已经删除。test 部分已由人工修正版本替代。
- `preview/`：标签检查预览图，其中 `test_*` 预览图已经删除。
- `autolabel_preview/`：早期标注/检查阶段生成的预览图和 contact sheet。
- `small_meeting_xiding_autolabel.yaml`：原始 Ultralytics 数据集配置。
- `manifest.csv`、`autolabel_manifest.csv`、`dedup_manifest.csv`：抽帧、自动标注和去重记录。
- `summary.json`：视频来源、切片参数和数据集生成元信息。

该目录不再作为人工修正后测试集的可信来源。

### `small_meeting_xiding_cvat/`

CVAT/COCO Keypoints 工作目录，用于和 CVAT 之间导入、导出数据。

重要文件：

- `train/images/`：用于生成 train CVAT 导入包的 1107 张图片。
- `train/annotations_coco_keypoints.json`：train split 的 COCO Keypoints 标注。
- `test/images/`：CVAT 中人工修正使用的 276 张原始测试图片。
- `test/annotations_coco_keypoints.json`：人工修正前的 test COCO Keypoints 标注。
- `small_meeting_xiding_cvat_train_coco_keypoints.zip`：CVAT train 导入包，格式为 `COCO Keypoints 1.0`。
- `small_meeting_xiding_cvat_test_coco_keypoints.zip`：CVAT test 导入包，格式为 `COCO Keypoints 1.0`。
- `export_test_coco_keypoints.zip`：从 CVAT 导出的人工修正 test split。
- `export_test_coco_keypoints/annotations/person_keypoints_test.json`：解压后的人工修正 COCO Keypoints 标注。
- `summary.json`：CVAT 转换阶段的统计信息。记录 train 有 1107 张图片、1200 个标注；
  修正前 test 有 276 张图片、300 个标注。

当前 CVAT 修正后的导出 JSON 中包含：

- 276 张图片元信息
- 312 个人体姿态标注

### `small_meeting_xiding_cvat_test_yolo/`

最终人工修正后的 YOLO-pose 测试集，由 CVAT 导出的 COCO Keypoints 转换得到。

重要文件：

- `images/train/`、`images/val/`、`images/test/`：三者都包含同一批 276 张人工修正测试图片。
- `labels/train/`、`labels/val/`、`labels/test/`：三者都包含对应的 YOLO-pose 标签。
- `small_meeting_xiding_cvat_test_yolo.yaml`：Ultralytics 数据集配置文件。
- `summary.json`：转换统计：
  - 276 张图片
  - 312 个标注
  - 0 个标注被跳过
  - 3 个镜像 split

验证当前模型的示例命令：

```bash
yolo pose val \
  model=fine-tune/models/yolo26n-pose.pt \
  data=fine-tune/datasets/small_meeting_xiding_cvat_test_yolo/small_meeting_xiding_cvat_test_yolo.yaml \
  imgsz=1536 \
  device=0
```

小规模过拟合/数据格式检查训练命令：

```bash
python3 fine-tune/train_yolo26_pose.py \
  --data datasets/small_meeting_xiding_cvat_test_yolo/small_meeting_xiding_cvat_test_yolo.yaml \
  --name yolo26n-pose-small-meeting-cvat-test \
  --imgsz 1536 \
  --batch 4 \
  --epochs 20 \
  --device 0
```

## 其他数据集目录

### `omnilab_raw/`

OmniLab 原始标注和元数据目录。包含：

- `omnilab_v1.1_2023.02.05.json`
- 对应 XML 标注数据

### `omnilab_zhankai/`

OmniLab 吸顶鱼眼数据展开并切片后的 YOLO-pose 数据集。

- train：11080 张图片 / 11080 个标签文件
- test：2766 张图片 / 2766 个标签文件
- `omnilab_zhankai.yaml`：Ultralytics 数据集配置。
- `annotations_slices.json`：COCO 风格的切片标注元数据。

### `posefes_zhankai/`

PoseFES 吸顶鱼眼数据展开并切片后的 YOLO-pose 数据集。

- train：1680 张图片 / 1680 个标签文件
- test：423 张图片 / 423 个标签文件
- `posefes_zhankai.yaml`：Ultralytics 数据集配置。
- `annotations_slices.json`：COCO 风格的切片标注元数据。

### `coco-pose/`

用于混合训练的 COCO-Pose 子集和标签。

- `train2017.txt`、`val2017.txt`：图片列表。
- `annotations/`：COCO 标注文件。
- `subsets/`：构建混合数据集时使用的采样列表。

### `mixed_pose_640/`

混合数据集列表目录，用于 640 输入尺寸训练。该数据集将鱼眼数据和 COCO-Pose 采样数据混合。

- `mixed_pose_640.yaml`：Ultralytics 数据集配置。
- `train.txt`、`test.txt`：绝对路径图片列表。
- `summary.txt`：混合数据集统计信息。

### `coco2017labels-pose.zip`

COCO pose 标签压缩包。

## 相关脚本

- `fine-tune/scripts/yolo_pose_to_cvat_coco.py`：将 YOLO-pose 转为 CVAT COCO Keypoints。
- `fine-tune/scripts/pack_cvat_coco_keypoints.py`：打包 CVAT 导入用 ZIP。
- `fine-tune/scripts/cvat_coco_keypoints_to_yolo_pose.py`：将 CVAT 导出的 COCO Keypoints 转回 YOLO-pose。
- `fine-tune/scripts/prepare_meeting_video_autolabel.py`：从小会议室吸顶视频生成源数据集。
- `fine-tune/train_yolo26_pose.py`：使用 Ultralytics 数据集 YAML 训练 YOLO pose 模型。
