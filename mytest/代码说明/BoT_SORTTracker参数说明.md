# BoT_SORTTracker 参数说明

## 目录
- [检测相关参数](#检测相关参数)
- [跟踪相关参数](#跟踪相关参数)
- [特征匹配相关参数](#特征匹配相关参数)
- [边界穿越相关参数](#边界穿越相关参数)
- [推荐配置方案](#推荐配置方案)

---

## 检测相关参数

### `track_high_thresh`
- **默认值**: 0.5
- **当前值**: 0.5
- **意义**: 高置信度检测阈值，高于此值的检测参与第一级匹配
- **调整方法**:
  - 增大: 只使用更可靠的检测，减少误匹配，但可能漏检
  - 减小: 使用更多检测，增加匹配机会，但可能引入噪声
- **建议范围**: 0.3 ~ 0.7
- **全景场景**: 建议 0.4 ~ 0.6

### `track_low_thresh`
- **默认值**: 0.1
- **当前值**: 0.1
- **意义**: 低置信度检测阈值，高于此值但低于 `track_high_thresh` 的检测参与第二级匹配
- **调整方法**:
  - 增大: 减少低质量检测
  - 减小: 保留更多检测（用于遮挡/边界场景）
- **建议范围**: 0.05 ~ 0.3
- **全景场景**: 建议 0.1 ~ 0.2（关键：抗遮挡）

### `new_track_thresh`
- **默认值**: 0.3
- **当前值**: 0.3
- **意义**: 新轨迹阈值，只有高于此值的检测才能创建新轨迹
- **调整方法**:
  - 增大: 减少误检轨迹
  - 减小: 更快创建新轨迹，但可能误检
- **建议范围**: 0.2 ~ 0.5
- **全景场景**: 建议 0.2 ~ 0.3

---

## 跟踪相关参数

### `track_buffer`
- **默认值**: 30
- **当前值**: 120
- **意义**: 丢失轨迹的缓存帧数（30FPS 下，120帧 = 4秒）
- **调整方法**:
  - 增大: 轨迹缓存更久，目标重新出现时更易匹配（抗遮挡）
  - 减小: 更快清理旧轨迹，节省内存
- **建议范围**: 30 ~ 180
- **全景场景**: 建议 90 ~ 150（关键：抗遮挡）

### `frame_rate`
- **默认值**: 30
- **当前值**: 30
- **意义**: 帧率，用于时间相关计算
- **调整方法**: 根据实际视频/摄像头帧率设置
- **建议范围**: 15 ~ 60

### `use_hungarian`
- **默认值**: False
- **当前值**: False
- **意义**: 是否使用匈牙利算法进行线性分配
- **调整方法**:
  - True: 使用匈牙利算法（更准确，但需要安装 `lap` 库）
  - False: 使用贪心算法（更快，无需依赖）
- **建议**: 没有特殊需求保持 False

---

## 特征匹配相关参数

### `match_thresh`
- **默认值**: 0.7
- **当前值**: 0.3
- **意义**: 融合匹配阈值（IoU和特征的融合值），低于此值认为匹配成功
- **调整方法**:
  - 增大: 更宽松的匹配（更多匹配，但可能误匹配）
  - 减小: 更严格的匹配（更少误匹配，但可能漏匹配）
- **建议范围**: 0.2 ~ 0.8
- **全景场景**: 建议 0.2 ~ 0.4（关键：更容易匹配）

### `proximity_thresh`
- **默认值**: 0.5
- **当前值**: 0.5
- **意义**: IoU 阈值，用于第一级匹配
- **调整方法**:
  - 增大: 要求检测框更重叠（更严格）
  - 减小: 更宽松的空间匹配（适应目标移动快）
- **建议范围**: 0.3 ~ 0.7
- **全景场景**: 建议 0.4 ~ 0.6

### `appearance_thresh`
- **默认值**: 0.25
- **当前值**: 0.15
- **意义**: 外观特征（ReID）相似度阈值（余弦距离）
  - **注意**: 这里是距离，不是相似度！值越小表示越相似！
- **调整方法**:
  - 增大（例如 0.3）: 更宽松的特征匹配（0.3表示相似度 > 0.7）
  - 减小（例如 0.1）: 更严格的特征匹配（0.1表示相似度 > 0.9）
- **建议范围**: 0.1 ~ 0.5
- **全景场景**: 建议 0.15 ~ 0.3（关键：更容易匹配）

### `feat_history`
- **默认值**: 50
- **当前值**: 50
- **意义**: 特征历史长度（保存最近多少帧的特征）
- **调整方法**:
  - 增大: 使用更多历史特征，更稳定，但占用内存
  - 减小: 更快适应外观变化
- **建议范围**: 30 ~ 100
- **全景场景**: 建议 40 ~ 60

### `with_reid`
- **默认值**: True
- **当前值**: True
- **意义**: 是否使用 ReID 特征融合
- **调整方法**:
  - True: 使用 ReID 特征（推荐，边界穿越必须开启）
  - False: 只用 IoU 和运动信息
- **全景场景**: 必须为 True！

---

## 边界穿越相关参数

### `enable_boundary_matching`
- **默认值**: False
- **当前值**: True
- **意义**: 是否启用边界穿越匹配
- **调整方法**:
  - True: 启用（全景场景必须）
  - False: 禁用（普通摄像头场景）

### `frame_width` / `frame_height`
- **默认值**: 3840 / 1080
- **当前值**: 3840 / 1080
- **意义**: 画面尺寸（全景展开后的尺寸）
- **调整方法**: 根据实际分辨率设置
- **注意**: 会在 `initialize()` 中自动更新为实际值

### `boundary_margin`
- **默认值**: 0.1
- **当前值**: 0.15
- **意义**: 边界区域占比（例如 0.15 = 画面左右各15%的区域）
- **调整方法**:
  - 增大: 更大的边界区域，更容易检测到边界穿越
  - 减小: 更小的边界区域，减少干扰
- **建议范围**: 0.1 ~ 0.25
- **全景场景**: 建议 0.12 ~ 0.18

### `boundary_time_window`
- **默认值**: 30
- **当前值**: 90
- **意义**: 边界穿越的时间窗口（多少帧内的消失-出现认为是穿越）
- **调整方法**:
  - 增大: 更长的时间窗口，允许目标慢慢穿越
  - 减小: 更短的时间窗口，只快速穿越
- **建议范围**: 30 ~ 120
- **全景场景**: 建议 60 ~ 120

### `boundary_similarity_thresh`
- **默认值**: 0.6
- **当前值**: 0.2
- **意义**: 边界匹配的特征相似度阈值（余弦相似度，不是距离！）
- **调整方法**:
  - 增大: 更严格的特征匹配（例如 0.6 需要相似度 > 0.6）
  - 减小: 更宽松的特征匹配（例如 0.2 需要相似度 > 0.2）
- **建议范围**: 0.2 ~ 0.6
- **全景场景**: 建议 0.15 ~ 0.3

### `boundary_debug`
- **默认值**: False
- **当前值**: True
- **意义**: 是否打印边界匹配的调试信息
- **调整方法**:
  - True: 打印调试信息（开发/调试用）
  - False: 不打印（生产用）

### `enable_top_boundary` / `enable_bottom_boundary` / `enable_left_boundary` / `enable_right_boundary`
- **默认值**: 前两个 False，后两个 True
- **当前值**: 同上
- **意义**: 启用哪些边界的穿越匹配
- **全景场景**: 只启用左、右边界（水平穿越），不启用上、下边界

---

## 推荐配置方案

### 方案一：平衡配置（默认）
```python
track_high_thresh=0.5
track_low_thresh=0.2
new_track_thresh=0.3
track_buffer=90
match_thresh=0.4
appearance_thresh=0.2
boundary_margin=0.15
boundary_time_window=60
boundary_similarity_thresh=0.3
```

### 方案二：高召回（抗遮挡，适合人员密集）
```python
track_high_thresh=0.4
track_low_thresh=0.1
new_track_thresh=0.25
track_buffer=150
match_thresh=0.3
appearance_thresh=0.15
boundary_margin=0.15
boundary_time_window=120
boundary_similarity_thresh=0.2
```

### 方案三：高精度（误匹配少，适合场景干净）
```python
track_high_thresh=0.6
track_low_thresh=0.3
new_track_thresh=0.35
track_buffer=60
match_thresh=0.5
appearance_thresh=0.3
boundary_margin=0.12
boundary_time_window=45
boundary_similarity_thresh=0.4
```

### 方案四：全景专用（推荐）
```python
track_high_thresh=0.5          # 适中
track_low_thresh=0.1           # 低，抗遮挡
new_track_thresh=0.3           # 适中
track_buffer=120               # 长，4秒
match_thresh=0.3               # 宽松
proximity_thresh=0.5           # 适中
appearance_thresh=0.15         # 宽松
with_reid=True                 # 必须
boundary_margin=0.15           # 边界15%
boundary_time_window=90        # 3秒
boundary_similarity_thresh=0.2 # 宽松
enable_left_boundary=True      # 启用
enable_right_boundary=True     # 启用
enable_top_boundary=False      # 禁用
enable_bottom_boundary=False   # 禁用
boundary_debug=True            # 调试用
```

---

## 参数调试建议

1. **先调检测阈值**
   - 如果漏检多 → 降低 `track_high_thresh`
   - 如果误检多 → 提高 `track_high_thresh`

2. **再调匹配阈值**
   - 如果 ID 经常变 → 降低 `match_thresh` / `appearance_thresh`
   - 如果 ID 经常错配 → 提高 `match_thresh` / `appearance_thresh`

3. **最后调边界参数**
   - 如果边界穿越 ID 变 → 增大 `boundary_time_window` / 降低 `boundary_similarity_thresh`
   - 如果边界错误匹配 → 减小 `boundary_margin` / 提高 `boundary_similarity_thresh`

4. **抗遮挡优先**
   - 最有效的参数: `track_buffer` → 调大
   - 第二有效: `track_low_thresh` → 调小
   - 第三有效: `appearance_thresh` → 调小

---

## 当前配置总结

| 参数 | 配置值 | 说明 |
|------|--------|------|
| track_high_thresh | 0.5 | 适中 |
| track_low_thresh | 0.1 | 低，抗遮挡 |
| new_track_thresh | 0.3 | 适中 |
| track_buffer | 120 | 长，4秒缓存 |
| match_thresh | 0.3 | 宽松匹配 |
| appearance_thresh | 0.15 | 宽松特征匹配 |
| boundary_margin | 0.15 | 边界15% |
| boundary_time_window | 90 | 3秒窗口 |
| boundary_similarity_thresh | 0.2 | 宽松相似度 |

---

## 常见问题

### Q: 目标被遮挡后 ID 变了怎么办？
A: 调大 `track_buffer`、调小 `appearance_thresh`

### Q: 边界穿越 ID 变了怎么办？
A: 确保 `with_reid=True`，然后调大 `boundary_time_window`、调小 `boundary_similarity_thresh`

### Q: 误匹配太多怎么办？
A: 提高 `match_thresh`、提高 `appearance_thresh`、提高 `track_high_thresh`

### Q: 漏检太多怎么办？
A: 降低 `track_high_thresh`、降低 `new_track_thresh`

