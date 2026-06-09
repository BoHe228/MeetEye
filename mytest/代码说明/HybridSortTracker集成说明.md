# HybridSortTracker 集成说明

> 文件覆盖范围：`mytest/core/tracker.py`（HybridSortTracker 类）、
> `HybridSORT/trackers/hybrid_sort_tracker/hybrid_sort.py`、
> `HybridSORT/trackers/hybrid_sort_tracker/hybrid_sort_reid.py`、
> `HybridSORT/trackers/hybrid_sort_tracker/association.py`

---

## 一、原始 HybridSORT 的三阶段关联流水线

每帧 update() 按以下顺序执行，所有阶段均在纯 NumPy 上运行：

```
输入检测 [N,5] (x1,y1,x2,y2,score)
    │
    ├─ 按 det_thresh 拆成两组
    │     高分检测 dets           (score > track_high_thresh)
    │     低分检测 dets_second    (low_thresh < score ≤ track_high_thresh)
    │
    ▼
【第一阶段】高分检测 × 所有轨迹
    代价 = -(IoU + VDC) + TCM               ← Hybrid_Sort
    代价 = w0*(-(IoU + VDC) + TCM) + w1*EG  ← Hybrid_Sort_ReID（启用ReID时）
    匹配成功 → 更新 KalmanBoxTracker + smooth_feat EMA
    │
    ▼
【第二阶段 BYTE】低分检测 × 未匹配轨迹
    代价 = -(IoU - TCM_byte)                ← 只用运动，不用外观
    匹配成功 → 更新 KalmanBoxTracker（不更新 smooth_feat）
    │
    ▼
【第三阶段 OC-SORT 再关联】剩余高分检测 × 剩余未匹配轨迹
    使用 last_observation（上次实际观测框）而非 Kalman 预测框
    代价 = -IoU（纯运动）
    匹配成功 → 更新 KalmanBoxTracker（不更新 smooth_feat）
    │
    ▼
未匹配检测 → 创建新轨迹
未匹配轨迹 → update(None)（纯预测，age++）
age > max_age → 删除轨迹
```

### 关键机制说明

| 机制 | 缩写 | 位置 | 作用 |
|---|---|---|---|
| 速度方向一致性 | VDC | association.py | 用四角速度方向与运动方向夹角余弦作为额外奖励，方向一致的匹配得分更高 |
| 置信度调制 | TCM | association.py | 检测分与轨迹历史分之差会降低关联代价，惩罚"检测分突降"的匹配 |
| Embedding Gate | EG | hybrid_sort_reid.py | 将 ReID 外观余弦距离加入代价矩阵，权重为 `EG_weight_high_score` |
| BYTE 两阶段 | BYTE | hybrid_sort.py | 低分检测在第二阶段单独匹配，保住被遮挡轨迹 |
| OC-SORT 再关联 | OCR | 两个文件 | 用 last_observation 而非 Kalman 预测做第三阶段，减少预测漂移导致的漏匹配 |
| 9 维 Kalman | - | kalmanfilter_score_new.py | 状态 [x,y,s,score,r,dx,dy,ds,dscore]，含置信度预测 |

---

## 二、MeetEye 集成 vs 原始 HybridSORT：逐项对比

### 2.1 完全一致的部分

- KalmanBoxTracker 9 维状态空间与初始化参数（R、P、Q 矩阵）
- 三阶段关联顺序与代价函数（IoU+VDC+TCM、BYTE、OCR）
- `k_previous_obs` 取 delta_t 帧内历史速度的累加均值
- `update_features` 的 EMA 平滑（α=0.8）+ 自适应平滑 adapfs 分支
- `np.float` 已改为 `np.float64`（NumPy 1.24+ 兼容）
- `float(kalman_score)` 防止 shape-(1,) 数组导致数组写入错误

### 2.2 差异与扩展

| 项目 | 原始 HybridSORT | MeetEye 集成 |
|---|---|---|
| 特征提取模型 | FastReID / ResNet50，2048 维，在跟踪器内部裁图推理 | OSNet x0.25，512 维，检测阶段已提取，通过 `det['feature']` 传入 |
| 特征维度 | 硬编码 2048 | 自动从第一个非 None 特征推断，默认 512 |
| 输出格式 | numpy array [M,5] (x1,y1,x2,y2,track_id) | dict 列表，含 `track_id`、`confidence`、`keypoints`、`feature`、`_boundary_matched` 等字段 |
| 元数据恢复 | 不需要（单纯评测） | 双射 IoU 反查：按 IoU 降序贪心分配，确保每条输出轨迹唯一对应一个输入 detection |
| 边界环绕匹配 | 无 | `BoundaryCrossingTracker`：轨迹消失于左/右边界后，用 smooth_feat + 位置重新关联 |
| low_thresh | 原始代码硬编码 0.1 | 已参数化为 `track_low_thresh`，通过 `low_thresh=` 传入 `Hybrid_Sort` |
| 空帧处理 | 要求调用方传 `np.empty((0,5))` | 内部生成空矩阵；ReID 模式下始终传 `id_feature shape=(0,512)` 给 Hybrid_Sort_ReID，避免 `NoneType` 下标错误 |
| ID 编号 | 从 0 全局递增，模块级 `KalmanBoxTracker.count` | 同上；`reset()` 时重建 inner，重置计数器 |

### 2.3 `high_score_matching_thresh` 参数实际无效

原始 `association.py` 中 `thresh` 参数在 `associate_4_points_with_score_with_reid` 里**已被注释掉**：

```python
# matched_indices = linear_assignment(..., thresh=thresh)   ← 注释掉了
matched_indices = linear_assignment(...)                    ← 无 thresh
```

因此 `--reid-emb-weight-high` 控制权重有效，但 `reid_high_score_thresh` 对当前版本无实际过滤效果。

---

## 三、ID 互换（ID Swap）问题分析

### 3.1 现象

两人近距离经过彼此（交叉或并排）后，分离时 ID 发生互换：A→B、B→A。

### 3.2 根本原因

**Kalman 预测框的歧义性：**

```
遮挡前：
  Track A 预测框 ≈ det_A_new
  Track B 预测框 ≈ det_B_new

遮挡中（两人重叠）：
  两个预测框都落在重叠区域内
  IoU(pred_A, det_A) ≈ IoU(pred_A, det_B)   ← 代价矩阵模糊
  IoU(pred_B, det_A) ≈ IoU(pred_B, det_B)

分离后：
  匈牙利/线性分配选最优组合，可能分配错误
  → Track A 被挂到 det_B，Track B 被挂到 det_A
```

**VDC 无法区分同向运动：**
若两人朝同方向行走，四角速度方向几乎相同，VDC 奖励对双方相同，不能成为区分依据。

**TCM 作用有限：**
TCM 惩罚的是"检测分与轨迹历史置信度之差"。遮挡时两人检测分都降低，差值近似，TCM 同样无法区分。

### 3.3 ReID（Embedding Gate）的帮助与局限

**有帮助的场景：**
- 两人外貌差异大（衣服颜色/纹理不同）
- 遮挡时间短（smooth_feat 仍保存分离前的干净特征）

**局限：**

| 情况 | 原因 | 结果 |
|---|---|---|
| 遮挡中的裁图 | 包含两人身体混合区域，特征降质 | smooth_feat 被污染 |
| smooth_feat EMA α=0.8 | 历史特征惯性大，污染特征稀释慢 | 分离后特征仍混淆 |
| 外貌相似的人 | OSNet 余弦距离接近 0 | EG 无法区分 |
| EG_weight_high=0.1 | 外观代价占比仅 10% | IoU+VDC 仍主导决策 |

**数学表达（ReID 模式第一阶段代价）：**
```
cost = 0.9 × (-(IoU + VDC) + TCM) + 0.1 × cosine_distance(smooth_feat_track, feat_det)
```

当 IoU 差异微小时（两检测框相似），0.1 的外观权重不足以纠正匹配。

### 3.4 可以调整的参数

| 参数 | 当前默认 | 调大的效果 | 副作用 |
|---|---|---|---|
| `reid_emb_weight_high` | 0.1 | 增加外观在关联中的权重，遮挡后更依赖外观判断 | 外貌相似者误匹配概率上升 |
| `match_thresh` (iou_threshold) | 0.3 | 调大 → 要求更高 IoU 才能匹配，不确定的匹配宁可放弃 → 新轨迹创建 | 正常跟踪中断增多 |
| `inertia` (VDC 权重) | 0.2 | 增加速度方向一致性的影响 | 方向估计不稳定时误拒 |
| `delta_t` | 3 | 增大 → 速度估计使用更长历史，更稳健 | 对快速转向的响应变慢 |
| `reid_alpha` | 0.8 | 调小（如 0.5）→ 更新更快，遮挡污染消退更快 | 特征抖动加大 |
| `track_buffer` | 500 | 调小 → 轨迹更快死亡，减少长期混乱 | 真实遮挡恢复能力下降 |

**推荐调整组合（改善 ID 互换）：**
```bash
python mytest/main.py --tracker hybridsort --use-reid \
  --reid-emb-weight-high 0.3 \   # 外观权重从 0.1 提升至 0.3
  # match_thresh 在 main.py 中修改为 0.5（更严格）
```

---

## 四、已修复的 Bug 清单

| Bug | 位置 | 症状 | 修复方式 |
|---|---|---|---|
| `np.float` 弃用 | hybrid_sort_reid.py, association.py | NumPy 1.24+ 报 AttributeError | 全部替换为 `np.float64` |
| `float(kalman_score)` | hybrid_sort_reid.py | `setting an array element with a sequence` | `float()` 显式转换 |
| 空 trackers 时 `track_features.ndim==1` | hybrid_sort_reid.py | `XA and XB must have same number of columns` | `reshape(-1, feat_dim)` 代替 `reshape(1,-1)` |
| `smooth_feat is None` 时无 fallback | hybrid_sort_reid.py | 同上 | 用 `np.zeros(feat_dim)` 作为占位向量 |
| `update_features` 零向量除法 | hybrid_sort_reid.py | NaN/Inf 污染 smooth_feat | `if norm < 1e-6: return` |
| ReID 模式空帧 `id_feature=None` | tracker.py | `NoneType is not subscriptable` | 始终传 `id_feature_np shape=(0,feat_dim)` |
| `low_thresh` 硬编码 0.1 | hybrid_sort.py | `track_low_thresh` 参数无效 | 参数化为 `self.low_thresh` |
| IoU 反查重复分配 | tracker.py | 两条轨迹映射同一 detection，元数据/keypoints 混乱 | 贪心双射分配（按 IoU 降序，维护 used_trk/det 集合） |
| `temp_id=100000+` | tracker.py | 边界重映射链断裂 | 改为 `temp_id=track_id`（使用 Hybrid_Sort 内部 ID）|

---

## 五、融合时需格外关注的地方

### 5.1 `KalmanBoxTracker.count` 是模块级全局变量

`hybrid_sort.py` 和 `hybrid_sort_reid.py` 各自有一个 `KalmanBoxTracker` 类，其 `count` 是**类变量**而非实例变量。每次 `__init__` 时重置为 0。

**风险：**如果同一进程内同时存在多个 `HybridSortTracker` 实例（如多路摄像头），两者共享同一 `count`，ID 会冲突。

**缓解：** 目前 MeetEye 只有一个全局 tracker 实例，无此问题。多实例时需要手动隔离 count。

### 5.2 `_feat_cache` 键是 `trk.id + 1`，不是 track_id

`hybrid_sort_reid.py` 输出的 `row[4] = trk.id + 1`（+1 是 MOT benchmark 惯例），
`_feat_cache` 也用 `trk.id + 1` 作键（在 ③-后同步步骤中）。
`_meta_cache`、`_prev_bbox` 等同样用 `track_id = int(row[4])`。
**保持一致，勿混用 `trk.id` 和 `trk.id+1`。**

### 5.3 ReID 模式下 `_feat_cache` 由内部同步，外部 EMA 不再执行

```python
# with_reid=True：step③-后同步
for trk in self._inner.trackers:
    if trk.smooth_feat is not None:
        self._feat_cache[trk.id + 1] = trk.smooth_feat   # 直接覆盖

# with_reid=False：step⑥-c 外部 EMA
self._feat_cache[track_id] = α * old + (1-α) * new
```

两路不能同时运行，否则会互相覆盖。当前代码用 `if not self._with_reid:` 分支保证互斥。

### 5.4 边界匹配要求 `smooth_feat is not None`

新轨迹触发边界匹配的条件之一是 `smooth_feat is not None`。
对于检测结果无 `feature` 字段（`feature=None`）的帧，`smooth_feat` 不会被初始化，
该轨迹**永远不会触发边界匹配**。确保特征提取器正常运行是边界匹配生效的前提。

### 5.5 TCM 的两个独立作用机制（重要）

TCM（置信度轨迹调制）在代码中分成**两处独立逻辑**，行为完全不同：

**作用一：代价矩阵惩罚（受 `tcm_first_step_weight` 控制）**

```python
# association.py:542 / 599
score_dif = |det置信度 - 轨迹Kalman预测置信度|
angle_diff_cost -= score_dif * args.TCM_first_step_weight
```

影响线性分配的代价矩阵，使置信度差异大的匹配对竞争力下降。
调整 `tcm_first_step_weight` 可改变这个惩罚力度（0 = 完全禁用）。

**作用二：分配后阈值过滤（NOT 受 `tcm_first_step_weight` 控制，固定强度）**

```python
# association.py:632 — 仅在 with_reid 路径（当前项目走这条）
iou_matrix_thre = iou_matrix - score_dif   # ← 写死，无法通过参数调节
if (iou_matrix_thre[m[0], m[1]] < iou_threshold):
    # 匹配被拒绝
```

线性分配结束后，逐对检查。即使 `tcm_first_step_weight=0` 也无法关闭此过滤。

**两条路径的对比：**

| 函数 | 触发条件 | 分配后阈值过滤 |
|---|---|---|
| `associate_4_points_with_score` | `EG_weight=0`（无 ReID）| 用原始 IoU |
| `associate_4_points_with_score_with_reid` | `EG_weight>0`（当前路径）| 用 `IoU - score_dif`，更严苛 |

**对本项目的影响：**

当 `with_reid=True` 且 `reid_emb_weight_high>0` 时走 ReID 路径。
如果遮挡导致检测置信度下降（如从 0.85 → 0.4，score_dif=0.45），则：
```
iou_matrix_thre = IoU - 0.45
若 IoU=0.6：iou_matrix_thre = 0.15 < match_thresh(0.4) → 匹配被拒绝
```
正确的匹配被 score_dif 拖累而丢弃，轨迹进入 Lost 状态。

**修复方式：** 需直接修改 `association.py:632`：
```python
# 原来（硬编码）：
iou_matrix_thre = iou_matrix - score_dif
# 改为（可调权重）：
iou_matrix_thre = iou_matrix - score_dif * args.TCM_filter_weight  # 新增参数
```
并在 `hs_args` 中添加 `TCM_filter_weight=0.3`。

### 5.6 `dataset` 参数与 `iou_matrix_thre` 的注释代码

`association.py` 中曾有：
```python
iou_matrix_thre = iou_matrix if dataset == "dancetrack" else iou_matrix - score_dif
```
现在这行被注释，固定使用 `iou_matrix - score_dif`。
我们传入 `dataset="dancetrack"` 实际无效，保留是为了 API 兼容。
这意味着我们始终走更严苛的非 DanceTrack 路径（DanceTrack 场景原本用原始 IoU 过滤）。

---

## 六、参数速查表（HybridSortTracker）

```
track_high_thresh    = 0.5    高分检测阈值，决定进入第一阶段的检测
track_low_thresh     = 0.1    低分检测阈值，决定进入 BYTE 第二阶段的检测
match_thresh         = 0.3    IoU 最低阈值（关联通过的门槛）
inertia              = 0.2    VDC 速度方向一致性权重
delta_t              = 3      速度估计的历史帧数
use_byte             = True   启用 BYTE 第二阶段
tcm_first_step       = True   第一阶段启用 TCM 置信度调制
tcm_byte_step        = True   BYTE 第二阶段启用 TCM
track_buffer         = 500    轨迹最长存活帧数（max_age = buffer * fps/30）
min_hits             = 1      连续命中几帧后才输出轨迹

── 丢失轨迹预测框（续命/coasting） ──
kalman_bbox          = False  输出 Kalman 状态框替代 YOLO 原始框，并持续显示丢失目标预测框
coast_frames         = 0      >0：丢失轨迹用 Kalman 预测框续命至多 N 帧（独立开关，不改正常框）
                               触发输出条件改为 (kalman_bbox or coast_frames>0)，且
                               time_since_update > N 时停止输出预测框（_is_lost=True，浅蓝细线）

── TCM 置信度调制 ──
tcm_first_step       = True   是否启用第一阶段 TCM（作用一：代价矩阵）
tcm_first_step_weight= 1.0    TCM 代价矩阵惩罚的乘数（0=禁用代价惩罚）
                               ⚠ 不影响分配后阈值过滤（见 5.5）
tcm_byte_step        = True   BYTE 第二阶段是否启用 TCM
tcm_byte_step_weight = 1.0    BYTE 阶段 TCM 惩罚乘数

── ReID 专用（--use-reid 时） ──
reid_emb_weight_high = 0.1    第一阶段外观代价权重（0=纯IoU+VDC，同时切换到更宽松的IoU阈值过滤）
reid_emb_weight_low  = 0.0    BYTE 第二阶段外观代价权重（建议保持 0）
reid_alpha           = 0.8    smooth_feat EMA 动量（越小更新越快）
reid_longterm_bank   = 30     每轨迹历史特征帧数（长期 ReID 备用）
```

> **with_reid 与 OSNet 绑定**：构造时 `with_reid = args.use_reid and use_osnet`。
> `--no-use-osnet` 时无特征，若仍走 ReID 路径，`embedding_distance` 对零向量的余弦距离为
> `0/0 = NaN`，会污染关联代价矩阵导致高置信度目标也关联不上被丢。因此无特征时自动关闭
> ReID，回退纯运动（IoU+VDC）关联。
