# PanoramaSlicer 参数说明

## 目录
- [切片相关参数](#切片相关参数)
- [去重相关参数](#去重相关参数)
- [全景环绕相关参数](#全景环绕相关参数)
- [推荐配置方案](#推荐配置方案)

---

## 切片相关参数

### `num_slices` (调用时传入，非 __init__ 参数)
- **当前值**: 3
- **意义**: 把全景图切成几个切片
- **调整方法**:
  - 3: 平衡精度和速度（推荐）
  - 2: 更快，但检测精度可能略降
  - 5: 更高精度，但速度更慢
- **建议范围**: 2 ~ 5
- **全景场景**: 建议 3

### `overlap_ratio`
- **当前值**: 0.1
- **意义**: 切片之间的重叠比例
- **调整方法**:
  - 增大: 切片重叠更多，减少漏检，但增加计算量
  - 减小: 更快，但可能漏检边界目标
- **建议范围**: 0.05 ~ 0.2
- **全景场景**: 建议 0.1 ~ 0.15

---

## 去重相关参数

### `iou_threshold`
- **当前值**: 0.3
- **意义**: IoU 阈值，用于判断两个检测是否是同一个目标
- **调整方法**:
  - 增大: 只合并高度重叠的检测（更严格，减少误合并）
  - 减小: 合并更多检测（更宽松，减少漏检）
- **建议范围**: 0.2 ~ 0.5
- **全景场景**: 建议 0.25 ~ 0.35

### `confidence_threshold`
- **当前值**: 0.5
- **意义**: 置信度阈值，但**注意**：这个参数目前没实际使用！
- **说明**: 实际的置信度过滤是在 YOLO 检测时就做了，通过 `config.py` 中的 `conf_threshold`

### `reid_similarity_threshold`
- **当前值**: 0.7
- **意义**: ReID 特征相似度阈值（余弦相似度）
  - **注意**: 这里是相似度，不是距离！值越大表示越相似！
- **调整方法**:
  - 增大: 更严格的特征匹配（例如 0.8）
  - 减小: 更宽松的特征匹配（例如 0.5）
- **建议范围**: 0.5 ~ 0.85
- **全景场景**: 建议 0.55 ~ 0.75（关键：环绕去重）

### `max_width_ratio` (方法参数，非 __init__ 参数)
- **默认值**: 0.6
- **意义**: 检测框的最大宽度占全景图宽度的比例
- **作用**: 过滤横跨左右边界的无效检测框（bug框）
- **调整方法**:
  - 增大: 允许更大的框
  - 减小: 过滤更严格
- **建议范围**: 0.4 ~ 0.7

---

## 全景环绕去重逻辑

### 环绕边界对（slice 0 & slice 2）
- **逻辑**: 只要 ReID 特征相似，就认为是同一个目标，**不限制位置**
- **目的**: 处理目标横跨全景左右边界的情况
- **关键**: `reid_similarity_threshold` 参数

### 相邻边界对（slice 0 & slice 1，slice 1 & slice 2）
- **逻辑**: 必须在边界重叠区域 + IoU > 阈值 + ReID 特征相似
- **目的**: 处理普通切片之间的重复检测

---

## 推荐配置方案

### 方案一：平衡配置（默认）
```python
num_slices=3
overlap_ratio=0.1
iou_threshold=0.3
reid_similarity_threshold=0.7
```

### 方案二：高召回（适合人员密集）
```python
num_slices=3
overlap_ratio=0.15
iou_threshold=0.25
reid_similarity_threshold=0.55  # 宽松
```

### 方案三：高精度（适合场景干净）
```python
num_slices=3
overlap_ratio=0.1
iou_threshold=0.4
reid_similarity_threshold=0.8
```

### 方案四：全景专用（推荐）
```python
num_slices=3                  # 3切片
overlap_ratio=0.12           # 12%重叠
iou_threshold=0.3            # 适中IoU
reid_similarity_threshold=0.65 # 适中特征相似度
```

---

## 调试技巧

### 查看去重效果
运行时可以看到这些打印：
- `[环绕去重]`: slice 0 和 slice 2 之间的去重
- `[相邻去重]`: slice 0&1 或 slice 1&2 之间的去重

### 如果同一目标被分成两个
- 降低 `reid_similarity_threshold`（例如 0.7 → 0.6）
- 降低 `iou_threshold`（例如 0.3 → 0.25）
- 增大 `overlap_ratio`（例如 0.1 → 0.15）

### 如果不同目标被合并
- 提高 `reid_similarity_threshold`（例如 0.7 → 0.8）
- 提高 `iou_threshold`（例如 0.3 → 0.4）
- 减小 `overlap_ratio`（例如 0.1 → 0.08）

---

## 当前配置总结

| 参数 | 配置值 | 说明 |
|------|--------|------|
| num_slices | 3 | 3切片 |
| overlap_ratio | 0.1 | 10%重叠 |
| iou_threshold | 0.3 | 适中 |
| reid_similarity_threshold | 0.7 | 适中 |

---

## 常见问题

### Q: slice 0 和 slice 2 的检测没有去重怎么办？
A: 降低 `reid_similarity_threshold`，例如 0.7 → 0.6

### Q: slice 0 和 slice 1 的检测没有去重怎么办？
A: 降低 `iou_threshold` 和 `reid_similarity_threshold`

### Q: 两个不同的人被合并成一个了怎么办？
A: 提高 `reid_similarity_threshold`，例如 0.7 → 0.8

### Q: 切片检测很慢怎么办？
A: 可以尝试 `num_slices=2`，但可能降低精度

