"""
多目标跟踪器模块
包含：
  BoT_SORTTracker  —— 原有跟踪器（ByteTrack + DeepSORT 融合）
  HybridSortTracker —— 基于 Hybrid-SORT 的替换实现
                       （IoU + 四角点速度方向 VDC + 置信度 TCM）
两者均保留 BoundaryCrossingTracker 环绕边界去重逻辑；
跨切片去重由上游 PanoramaSlicer 完成，本模块不涉及。
"""
import time
import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque, OrderedDict

import sys
import os

try:
    from .boundary_matcher import BoundaryCrossingTracker
    BOUNDARY_MATCHER_AVAILABLE = True
except ImportError:
    BOUNDARY_MATCHER_AVAILABLE = False

# 尝试导入lap库用于匈牙利算法
try:
    import lap
    LAP_AVAILABLE = True
except ImportError:
    LAP_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────
# HybridSORT 导入（可选，失败时 HybridSortTracker 不可用）
# ────────────────────────────────────────────────────────────────────────────
_HYBRID_SORT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'HybridSORT')
if _HYBRID_SORT_PATH not in sys.path:
    sys.path.insert(0, _HYBRID_SORT_PATH)

try:
    from trackers.hybrid_sort_tracker.hybrid_sort import (
        Hybrid_Sort,
        KalmanBoxTracker as _HybridKalmanBoxTracker,
    )
    from trackers.hybrid_sort_tracker.hybrid_sort_reid import (
        Hybrid_Sort_ReID,
        KalmanBoxTracker as _HybridReIDKalmanBoxTracker,
    )
    HYBRID_SORT_AVAILABLE = True
except ImportError as _hybrid_err:
    HYBRID_SORT_AVAILABLE = False
    Hybrid_Sort = None
    Hybrid_Sort_ReID = None
    _HybridKalmanBoxTracker = None
    _HybridReIDKalmanBoxTracker = None
    print(f"[HybridSORT] 导入失败，HybridSortTracker 不可用: {_hybrid_err}")

# 用于跟踪算法是否已经输出过
ALGORITHM_LOGGED = False
# 性能统计
PERF_STATS = {
    'hungarian_calls': 0,
    'hungarian_total_time': 0.0,
    'greedy_calls': 0,
    'greedy_total_time': 0.0,
}
_stats_lock = threading.Lock()


class TrackState:
    """跟踪状态枚举"""
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import cosine_similarity, box_iou


class BaseTrack:
    """跟踪基类 - 支持平滑特征更新"""
    _count = 0

    def __init__(self):
        self.track_id = 0
        self.is_activated = False
        self.state = TrackState.New
        self.history = OrderedDict()

        self.score = 0
        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0
        self.location = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        BaseTrack._count = 0


class KalmanFilter:
    """
    Kalman 滤波器，用于跟踪框状态估计
    状态：[x, y, w, h, vx, vy, vw, vh]
    其中 (x,y) 是中心坐标，w 是宽度，h 是高度
    """

    def __init__(self):
        ndim, dt = 4, 1.

        # 运动矩阵 F
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        # 观测矩阵 H
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 运动和观测噪声权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def multi_predict(self, means: np.ndarray, covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = len(means)
        new_means = []
        new_covs = []
        for i in range(N):
            m, c = self.predict(means[i], covariances[i])
            new_means.append(m)
            new_covs.append(c)
        return np.array(new_means), np.array(new_covs)

    def project(self, mean: np.ndarray, covariance: np.ndarray,
                noise_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        noise_factor > 1 时放大测量噪声 R，使卡尔曼增益 K 减小，
        更新后的状态更靠近预测（而非检测），适用于遮挡/近邻期间。
        """
        std = [
            noise_factor * self._std_weight_position * mean[2],
            noise_factor * self._std_weight_position * mean[3],
            noise_factor * self._std_weight_position * mean[2],
            noise_factor * self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean: np.ndarray, covariance: np.ndarray,
               measurement: np.ndarray,
               noise_factor: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        noise_factor: 测量噪声放大倍数
          1.0  → 正常更新
          3.0  → 近邻（框有接触），适度信任预测
          10.0 → 重叠（框IoU>0.3），强信任预测，保护速度向量
        """
        projected_mean, projected_cov = self.project(mean, covariance, noise_factor)

        # 使用简单的卡尔曼增益计算（避免scipy依赖）
        K = np.dot(covariance, self._update_mat.T) @ np.linalg.inv(projected_cov)
        y = measurement - projected_mean
        new_mean = mean + np.dot(y, K.T)
        I = np.eye(8)
        new_covariance = np.dot(I - np.dot(K, self._update_mat), covariance)

        return new_mean, new_covariance




def iou_distance(tracks, detections):
    """计算IoU距离矩阵 (1 - IoU)"""
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost_matrix = np.zeros((n_tracks, n_dets))
    for i, track in enumerate(tracks):
        track_box = track.tlbr
        for j, det in enumerate(detections):
            det_box = det.tlbr
            iou = box_iou(track_box, det_box)
            cost_matrix[i, j] = 1 - iou
    return cost_matrix


def embedding_distance(tracks, detections):
    """计算外观特征距离矩阵（余弦距离）"""
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost_matrix = np.ones((n_tracks, n_dets))
    for i, track in enumerate(tracks):
        if track.smooth_feat is None:
            continue
        for j, det in enumerate(detections):
            if det.curr_feat is not None:
                sim = cosine_similarity(track.smooth_feat, det.curr_feat)
                cost_matrix[i, j] = 1 - sim
    return cost_matrix


def fuse_score(cost_matrix, detections):
    """将检测置信度融合到代价矩阵中"""
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def linear_assignment(dist_matrix: np.ndarray, thresh: float, use_hungarian: bool = False) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    线性分配 - 支持两种算法：
    - 贪心算法（默认，无依赖）
    - 匈牙利算法（lap.lapjv，更精确，参考标准BoT-SORT实现）
    """
    global ALGORITHM_LOGGED
    N, M = dist_matrix.shape

    if N == 0 or M == 0:
        return [], list(range(N)), list(range(M))

    # 尝试使用匈牙利算法（标准BoT-SORT的方式）
    if use_hungarian and LAP_AVAILABLE:
        try:
            if not ALGORITHM_LOGGED:
                print("[BoT-SORT] 使用匈牙利算法 (lap.lapjv) 进行线性分配")
                ALGORITHM_LOGGED = True

            start_time = time.time()

            # 标准BoT-SORT的方式：只使用lap.lapjv，不设置cost_limit
            _, x, _ = lap.lapjv(dist_matrix, extend_cost=True)
            matches = []
            matched_a = set()
            matched_b = set()

            for i in range(N):
                j = x[i]
                if j >= 0 and j < M and dist_matrix[i, j] <= thresh:
                    matches.append((i, j))
                    matched_a.add(i)
                    matched_b.add(j)

            unmatched_a = [i for i in range(N) if i not in matched_a]
            unmatched_b = [j for j in range(M) if j not in matched_b]

            # 统计性能
            elapsed = time.time() - start_time
            with _stats_lock:
                PERF_STATS['hungarian_calls'] += 1
                PERF_STATS['hungarian_total_time'] += elapsed

            return matches, unmatched_a, unmatched_b
        except Exception as e:
            # 如果失败回退到贪心算法
            if not ALGORITHM_LOGGED:
                print(f"[BoT-SORT] 匈牙利算法失败 ({e})，回退到贪心算法")
                ALGORITHM_LOGGED = True
            pass
    elif use_hungarian and not LAP_AVAILABLE:
        if not ALGORITHM_LOGGED:
            print("[BoT-SORT] lap库未安装，使用贪心算法 (安装: pip install lap)")
            ALGORITHM_LOGGED = True

    # 贪心算法（标准贪心：按全局最小代价依次匹配）
    if not ALGORITHM_LOGGED:
        print("[BoT-SORT] 使用贪心算法进行线性分配")
        ALGORITHM_LOGGED = True

    start_time = time.time()

    # 标准贪心算法：生成所有可能的匹配对，按代价排序，依次匹配
    pairs = []
    for i in range(N):
        for j in range(M):
            if dist_matrix[i, j] <= thresh:
                pairs.append((dist_matrix[i, j], i, j))

    # 按代价从小到大排序
    pairs.sort(key=lambda x: x[0])

    matched_a = set()
    matched_b = set()
    matches = []

    for dist, i, j in pairs:
        if i not in matched_a and j not in matched_b:
            matches.append((i, j))
            matched_a.add(i)
            matched_b.add(j)

    unmatched_a = [i for i in range(N) if i not in matched_a]
    unmatched_b = [j for j in range(M) if j not in matched_b]

    # 统计性能
    elapsed = time.time() - start_time
    with _stats_lock:
        PERF_STATS['greedy_calls'] += 1
        PERF_STATS['greedy_total_time'] += elapsed

    return matches, unmatched_a, unmatched_b


class Detection:
    """检测结果封装类，包含外观特征"""

    def __init__(self, bbox: List[float], score: float, cls: Any,
                 keypoints: Optional[List] = None, feature: Optional[np.ndarray] = None):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = score
        self.cls = cls
        self.keypoints = keypoints or []
        self.feature = feature  # ReID外观特征

    def to_xywh(self):
        """转换为 [x, y, w, h] (中心坐标)"""
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.array([cx, cy, w, h], dtype=np.float32)


class STrack(BaseTrack):
    """
    单个跟踪轨迹 - BoT-SORT风格
    支持平滑特征更新（指数移动平均）
    """
    shared_kalman = KalmanFilter()

    def __init__(self, detection: Detection, feat_history: int = 50):
        super().__init__()
        xywh = detection.to_xywh()
        self._tlwh = np.asarray(self.xywh_to_tlwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = detection.score
        self.tracklet_len = 0
        self.cls = detection.cls
        self.keypoints = detection.keypoints

        # 记录目标年龄和距离上次更新的时间
        self.age = 0  # 目标存在的总帧数
        self.time_since_update = 0  # 距离上次成功更新的帧数

        # 保存原始检测框
        self.original_xyxy = detection.bbox.copy()

        # 特征平滑相关
        self.smooth_feat = None
        self.curr_feat = None
        self.features = deque([], maxlen=feat_history)
        # alpha不再是固定值，而是动态计算
        # self.alpha = 0.9

        if detection.feature is not None:
            self.update_features(detection.feature, detection.score)

    def update_features(self, feat, confidence=1.0, near_other: bool = False):
        """
        核心：动态调整的指数移动平均更新特征
        smooth_feat = alpha * smooth_feat + (1 - alpha) * new_feat

        near_other=True 表示当前检测框与其他活跃轨迹框空间接近（潜在遮挡/混合裁图），
        此时高度保护历史特征，防止对方身体污染 smooth_feat 导致分离后 ID 互换。
        """
        feat = np.array(feat, dtype=np.float32).flatten()  # 强制 1D，避免 (1,512) 形状污染
        feat /= np.linalg.norm(feat) + 1e-6
        self.curr_feat = feat

        # 动态计算alpha
        if self.smooth_feat is None:
            alpha = 0.0  # 第一次直接使用
        else:
            if near_other:
                # 检测框与其他轨迹空间接近 → YOLO 裁图可能包含对方身体
                # 用极高 alpha 冻结特征，防止污染；curr_feat 仍更新供调试用
                alpha = 0.98
            elif self.time_since_update > 30:  # 长时间未更新（边界穿越后重新出现）
                alpha = 0.8  # 注重新特征，快速适应当前状态
            elif confidence < 0.7:  # 低置信度检测（多切片可能有低质量检测）
                alpha = 0.93  # 更信任历史特征
            elif self.age < 15:  # 新目标（前15帧快速学习）
                alpha = 0.7  # 更快学习
            else:  # 稳定跟踪
                alpha = 0.9

        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = alpha * self.smooth_feat + (1 - alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat) + 1e-6

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked     
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False,
                    freeze_feat: bool = False, near_other: bool = False):
        # 遮挡/近邻时放大测量噪声 R → 卡尔曼增益 K 减小 → 状态更靠近预测而非检测
        # 这样即使检测框因遮挡而位置偏移，速度向量也能得到保护
        noise_factor = 10.0 if freeze_feat else (3.0 if near_other else 1.0)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh),
            noise_factor=noise_factor
        )
        # freeze_feat=True  → 完全跳过（重度遮挡）
        # near_other=True   → alpha=0.98 高度保护（轻度接近）
        if not freeze_feat and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score,
                                 near_other=near_other)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0  # 重置更新计时器
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        if new_track.keypoints:
            self.keypoints = new_track.keypoints
        self.original_xyxy = new_track.original_xyxy.copy()

    def update(self, new_track: 'STrack', frame_id: int,
               freeze_feat: bool = False, near_other: bool = False):
        self.frame_id = frame_id
        self.tracklet_len += 1

        # 遮挡/近邻时放大测量噪声 R → 卡尔曼增益 K 减小 → 状态更靠近预测而非检测
        # 这样即使检测框因遮挡而位置偏移，速度向量也能得到保护
        noise_factor = 10.0 if freeze_feat else (3.0 if near_other else 1.0)
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh),
            noise_factor=noise_factor
        )

        # freeze_feat=True  → 完全跳过（重度遮挡，IoU > 0.3）
        # near_other=True   → alpha=0.98 高度保护（轻度接近，框有接触）
        if not freeze_feat and new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score,
                                 near_other=near_other)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.time_since_update = 0  # 重置更新计时器

        self.score = new_track.score
        self.cls = new_track.cls
        if new_track.keypoints:
            self.keypoints = new_track.keypoints
        self.original_xyxy = new_track.original_xyxy.copy()

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """优先使用原始检测框，避免Kalman滤波导致框变大"""
        if hasattr(self, 'original_xyxy') and self.original_xyxy is not None:
            return self.original_xyxy.copy()
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @staticmethod
    def xywh_to_tlwh(xywh):
        ret = np.asarray(xywh).copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


class BoT_SORTTracker:
    """
    BoT-SORT跟踪器 - 完整实现
    核心特性：
    1. 高/低置信度检测分离（ByteTrack风格）
    2. IoU + ReID特征融合（取最小值策略）
    3. 平滑特征更新（指数移动平均）
    4. 两级关联策略
    """

    def __init__(self,
                 track_high_thresh: float = 0.3,
                 track_low_thresh: float = 0.1,
                 new_track_thresh: float = 0.4,
                 track_buffer: int = 30,
                 match_thresh: float = 0.7,
                 proximity_thresh: float = 0.5,  # IoU阈值
                 appearance_thresh: float = 0.5,  # 第一阶段外观特征阈值（有IoU门控兜底）
                 reid_lost_thresh: float = 0.25,  # 第三阶段纯ReID阈值（无IoU门控，需更严格）
                 frame_rate: int = 30,
                 feat_history: int = 50,
                 with_reid: bool = True,
                 use_hungarian: bool = False,  # 是否使用匈牙利算法
                 # 边界穿越匹配参数
                 enable_boundary_matching: bool = False,
                 frame_width: int = 1920,
                 frame_height: int = 1080,
                 boundary_margin: float = 0.1,
                 boundary_time_window: int = 30,
                 boundary_similarity_thresh: float = 0.6,
                 boundary_debug: bool = True,  # 边界匹配调试模式
                 enable_top_boundary: bool = False,  # 是否启用顶部边界
                 enable_bottom_boundary: bool = True,  # 是否启用底部边界
                 enable_left_boundary: bool = True,  # 是否启用左侧边界
                 enable_right_boundary: bool = True,  # 是否启用右侧边界
                 kalman_bbox: bool = False):  # 是否用 Kalman 状态框替代 YOLO 原始框 + 显示 lost 预测框
        self.kalman_bbox = kalman_bbox
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        BaseTrack.reset_id()

        self.frame_id = 0
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        # 关联阈值
        self.match_thresh = match_thresh
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.reid_lost_thresh = reid_lost_thresh  # 第三阶段专用：无IoU门控，需比appearance_thresh更严
        self.with_reid = with_reid
        self.feat_history = feat_history
        self.use_hungarian = use_hungarian and LAP_AVAILABLE  # 只有lap可用时才使用

        # 边界穿越匹配
        self.enable_boundary_matching = enable_boundary_matching and BOUNDARY_MATCHER_AVAILABLE
        self.boundary_tracker = None
        self.boundary_debug = boundary_debug
        self.enable_top_boundary = enable_top_boundary
        self.enable_bottom_boundary = enable_bottom_boundary
        self.enable_left_boundary = enable_left_boundary
        self.enable_right_boundary = enable_right_boundary
        if self.enable_boundary_matching:
            self.boundary_tracker = BoundaryCrossingTracker(
                frame_width=frame_width,
                frame_height=frame_height,
                boundary_margin=boundary_margin,
                time_window=boundary_time_window,
                similarity_threshold=boundary_similarity_thresh,
                debug=boundary_debug,
                enable_top_boundary=enable_top_boundary,
                enable_bottom_boundary=enable_bottom_boundary,
                enable_left_boundary=enable_left_boundary,
                enable_right_boundary=enable_right_boundary
            )
        # 上一帧的轨迹信息（用于边界匹配）
        self.prev_track_info: Dict[int, Dict] = {}
        # 临时ID映射
        self.temp_id_map: Dict[int, int] = {}
        self._next_temp_id = 100000

    def reset_id(self):
        BaseTrack.reset_id()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilter()
        self.reset_id()
        # 重置边界匹配器
        if self.boundary_tracker is not None:
            self.boundary_tracker.reset()
        self.prev_track_info.clear()
        self.temp_id_map.clear()
        self._next_temp_id = 100000

    def set_boundary_frame_size(self, width: int, height: int):
        """设置边界匹配器的画面尺寸"""
        if self.boundary_tracker is not None:
            self.boundary_tracker.set_frame_size(width, height)

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        更新跟踪器 - BoT-SORT完整流程

        Args:
            detections: 检测结果列表，每个检测可包含 'feature' 字段(ReID特征)

        Returns:
            更新后的检测结果，包含统一、稳定的 'track_id'
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # === 更新所有轨迹的计数器 ===
        # 先把所有tracks合在一起更新
        all_tracks = self.tracked_stracks + self.lost_stracks
        for track in all_tracks:
            track.age += 1  # 所有轨迹的年龄都加1
            track.time_since_update += 1  # 距离上次更新的时间加1

        # === 边界匹配预处理 ===
        if self.enable_boundary_matching and self.boundary_tracker is not None:
            self.boundary_tracker.pre_process(self.frame_id)

        # 1. 将输入检测转换为Detection对象
        detection_list = []
        for det in detections:
            feature = det.get('feature', None)
            if feature is not None and not isinstance(feature, np.ndarray):
                feature = np.array(feature)
            detection = Detection(
                bbox=det['bbox'],
                score=det['confidence'],
                cls=det.get('class_id', 0),
                keypoints=det.get('keypoints', []),
                feature=feature
            )
            detection_list.append(detection)

        # 2. 分离高/低置信度检测
        if len(detection_list):
            scores = np.array([d.score for d in detection_list])
            bboxes = np.array([d.bbox for d in detection_list])

            # 低置信度过滤
            lowest_inds = scores > self.track_low_thresh
            low_score_detections = [detection_list[i] for i in np.where(lowest_inds)[0]]

            # 高置信度检测
            high_score_inds = scores > self.track_high_thresh
            high_score_detections = [detection_list[i] for i in np.where(high_score_inds)[0]]
        else:
            low_score_detections = []
            high_score_detections = []

        # 3. 转换为STrack
        if len(high_score_detections) > 0:
            detections_high = [STrack(d, self.feat_history) for d in high_score_detections]
        else:
            detections_high = []

        # 4. 分离已确认和未确认的轨迹
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 5. 第一阶段关联：高置信度检测 + 所有轨迹
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # 预测当前位置
        STrack.multi_predict(strack_pool)

        # ── 预计算检测框两两重叠情况（assignment 和 update 两阶段共用）────────
        # 必须在 assignment 之前完成，用于：
        #   1. fuse_score 去偏（重叠框 score→1.0）
        #   2. ReID 参与分配时屏蔽重叠列（裁图已被污染，参与分配反而帮倒忙）
        #   3. 确定 update 阶段的 freeze_feat / near_other 标志
        _FREEZE_IOU_THRESH = 0.3
        _NEAR_IOU_THRESH   = 0.1
        freeze_det_cols: set = set()
        near_det_cols:   set = set()
        _n_dh = len(detections_high)
        for _a in range(_n_dh):
            for _b in range(_a + 1, _n_dh):
                _iou_ab = box_iou(detections_high[_a].tlbr, detections_high[_b].tlbr)
                if _iou_ab > _FREEZE_IOU_THRESH:
                    freeze_det_cols.add(_a)
                    freeze_det_cols.add(_b)
                elif _iou_ab > _NEAR_IOU_THRESH:
                    near_det_cols.add(_a)
                    near_det_cols.add(_b)
        _overlap_det_cols = freeze_det_cols | near_det_cols

        # 计算IoU距离
        ious_dists = iou_distance(strack_pool, detections_high)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # 融合检测置信度
        # 重叠检测框（IoU > _NEAR_IOU_THRESH）的 score 临时设为 1.0：
        # fuse_score 会让高置信度检测对所有轨迹都降低代价，两人bbox重叠时置信度
        # 差异会把高置信度那个人错误地推给所有轨迹，直接引发 ID 互换。
        _fuse_scores = np.array([d.score for d in detections_high], dtype=np.float32)
        for _col in _overlap_det_cols:
            _fuse_scores[_col] = 1.0
        ious_dists = 1 - (1 - ious_dists) * _fuse_scores[np.newaxis, :]

        # 如果有ReID特征，融合外观距离
        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections_high)  # 余弦距离 [0,1]
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0  # IoU 不足时不信任 ReID
            # 重叠检测列：OSNet 裁图包含邻近人体，特征已被污染，不参与分配决策
            if _overlap_det_cols:
                emb_dists[:, list(_overlap_det_cols)] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # 线性分配
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh, use_hungarian=self.use_hungarian)

        for _mi, (itracked, idet) in enumerate(matches):
            track = strack_pool[itracked]
            det = detections_high[idet]
            freeze_feat = (idet in freeze_det_cols)
            near_other  = (idet in near_det_cols)
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id,
                             freeze_feat=freeze_feat, near_other=near_other)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False,
                                  freeze_feat=freeze_feat, near_other=near_other)
                refind_stracks.append(track)

        # 6. 按状态拆分第一阶段未匹配轨迹
        #    - Lost 轨迹：IoU≈0，交给第三阶段纯 ReID 尝试恢复
        #    - Tracked 轨迹：交给第二阶段与低分框做 IoU 关联
        u_lost_stracks    = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Lost]
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        # 第二阶段：低置信度检测框 + 未匹配 Tracked 轨迹（纯 IoU）
        if len(low_score_detections) > 0:
            detections_second = [STrack(d, self.feat_history) for d in low_score_detections
                                 if d.score <= self.track_high_thresh]
        else:
            detections_second = []

        if r_tracked_stracks and detections_second:
            dists = iou_distance(r_tracked_stracks, detections_second)
            stage2_matches, u_r_tracked, _ = linear_assignment(
                dists, thresh=0.5, use_hungarian=self.use_hungarian)
            for itracked, idet in stage2_matches:
                track = r_tracked_stracks[itracked]
                det = detections_second[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_stracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
            unmatched_r_tracked = [r_tracked_stracks[i] for i in u_r_tracked]
        else:
            unmatched_r_tracked = r_tracked_stracks

        # 标记未匹配的 Tracked 轨迹为丢失
        for track in unmatched_r_tracked:
            if not track.state == TrackState.Lost:
                if self.enable_boundary_matching and self.boundary_tracker is not None:
                    prev_info = self.prev_track_info.get(track.track_id)
                    prev_bbox = prev_info.get('bbox') if prev_info else None
                    feat = track.curr_feat if track.curr_feat is not None else track.smooth_feat
                    if feat is not None:
                        self.boundary_tracker.process_lost_track(
                            track_id=track.track_id,
                            bbox=track.xyxy.tolist(),
                            feature=feat,
                            frame_id=self.frame_id,
                            smooth_feat=track.smooth_feat,
                            prev_bbox=prev_bbox
                        )
                track.mark_lost()
                lost_stracks.append(track)

        # 第三阶段：纯 ReID 恢复 Lost 轨迹
        #
        # 为什么只用 ReID、不用 IoU：
        #   能到这里的 Lost 轨迹，都是 Stage 1 的 IoU+ReID 已经失败的。
        #   Stage 1 失败意味着 ious_dist ≥ match_thresh（IoU 彻底无效）。
        #   在 Stage 3 里再加 IoU，min(无效IoU, ReID) 结果和纯 ReID 完全一样，
        #   反而引入用错误 IoU 匹配的风险。
        #
        # Stage 3 的唯一价值：
        #   Stage 1 里 IoU 门控（ious_dists_mask）会在 IoU 差时把 ReID 也封掉，
        #   Stage 3 去掉这个门控，让 ReID 单独工作——专门应对：
        #     · 出画面再进来（IoU=0，但 ReID 仍能识别）
        #     · 长时间遮挡后重现（卡尔曼漂移，IoU=0）
        if self.with_reid and u_lost_stracks and u_detection:
            reid_dets = [detections_high[i] for i in u_detection]
            emb_dists_lost = embedding_distance(u_lost_stracks, reid_dets)
            emb_dists_lost[emb_dists_lost > self.reid_lost_thresh] = 1.0
            stage3_matches, _, _ = linear_assignment(
                emb_dists_lost, thresh=self.reid_lost_thresh, use_hungarian=self.use_hungarian)
            matched_det_local = set()
            for i_lost, i_det in stage3_matches:
                track = u_lost_stracks[i_lost]
                det = reid_dets[i_det]
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                matched_det_local.add(i_det)
            # 从 u_detection 中移除已被第三阶段认领的检测
            u_detection = [u_detection[i] for i in range(len(u_detection))
                           if i not in matched_det_local]

        # 7. 处理未确认的轨迹
        detections_unconfirmed = [detections_high[i] for i in u_detection]
        ious_dists = iou_distance(unconfirmed, detections_unconfirmed)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        ious_dists = fuse_score(ious_dists, detections_unconfirmed)

        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections_unconfirmed)
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7, use_hungarian=self.use_hungarian)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_unconfirmed[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        # 8. 初始化新轨迹
        for inew in u_detection:
            track = detections_unconfirmed[inew]
            if track.score < self.new_track_thresh:
                continue

            # === 边界匹配：检查新目标是否匹配到消失的目标 ===
            matched_id = None
            if self.enable_boundary_matching and self.boundary_tracker is not None and track.curr_feat is not None:
                temp_id = self._next_temp_id
                self._next_temp_id += 1
                matched_id = self.boundary_tracker.check_new_track(
                    bbox=track.original_xyxy.tolist() if hasattr(track, 'original_xyxy') else track.tlbr.tolist(),
                    feature=track.curr_feat,
                    frame_id=self.frame_id,
                    temp_id=temp_id
                )

            track.activate(self.kalman_filter, self.frame_id)

            # 如果匹配到ID，临时记录下来
            if matched_id is not None:
                self.temp_id_map[track.track_id] = matched_id
                track._boundary_matched_id = matched_id

            activated_stracks.append(track)

        # 9. 更新状态
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # 合并
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 10. 准备输出结果
        output_detections = []
        current_track_info = {}

        for track in self.tracked_stracks:
            if not track.is_activated:
                continue

            # 应用边界匹配的ID重映射
            final_track_id = track.track_id
            boundary_matched = False

            # 检查是否有匹配的ID
            if hasattr(track, '_boundary_matched_id'):
                final_track_id = track._boundary_matched_id
                boundary_matched = True
            elif track.track_id in self.temp_id_map:
                final_track_id = self.temp_id_map[track.track_id]
                boundary_matched = True

            # smooth_feat 是多帧 EMA 平均特征，比 curr_feat（单帧）更稳定
            out_feat = track.smooth_feat if track.smooth_feat is not None else track.curr_feat
            det_out = {
                'bbox': track.tlbr.tolist() if self.kalman_bbox else track.xyxy.tolist(),
                'confidence': track.score,
                'class_id': track.cls,
                'class_name': detections[0].get('class_name', 'person') if detections else 'person',
                'keypoints': track.keypoints,
                'track_id': final_track_id,
                'feature': out_feat[np.newaxis, :] if out_feat is not None else None,
                '_feature_count': len(track.features),
                '_boundary_matched': boundary_matched
            }
            output_detections.append(det_out)

            # 保存当前帧轨迹信息供下一帧使用
            current_track_info[final_track_id] = {
                'bbox': track.xyxy.tolist(),
                'track_id': final_track_id
            }

        # === 边界匹配后处理 ===
        if self.enable_boundary_matching and self.boundary_tracker is not None:
            output_detections = self.boundary_tracker.post_process(output_detections)

        # === use_track_bbox：补充 lost 轨迹的 Kalman 预测框 ===
        # lost_stracks 已经过 Kalman predict，tlbr 是本帧预测位置
        if self.kalman_bbox:
            class_name = detections[0].get('class_name', 'person') if detections else 'person'
            for track in self.lost_stracks:
                if not track.is_activated:
                    continue
                final_track_id = track.track_id
                if hasattr(track, '_boundary_matched_id'):
                    final_track_id = track._boundary_matched_id
                elif track.track_id in self.temp_id_map:
                    final_track_id = self.temp_id_map[track.track_id]
                out_feat = track.smooth_feat if track.smooth_feat is not None else track.curr_feat
                output_detections.append({
                    'bbox': track.tlbr.tolist(),
                    'confidence': track.score,
                    'class_id': track.cls,
                    'class_name': class_name,
                    'keypoints': track.keypoints,
                    'track_id': final_track_id,
                    'feature': out_feat[np.newaxis, :] if out_feat is not None else None,
                    '_feature_count': len(track.features),
                    '_boundary_matched': False,
                    '_is_lost': True,
                })

        # 更新轨迹信息缓存
        self.prev_track_info = current_track_info

        return output_detections

    def get_boundary_stats(self) -> Dict[str, Any]:
        """获取边界匹配统计信息"""
        if self.boundary_tracker is not None:
            return self.boundary_tracker.get_stats()
        return {'enabled': False}

    def draw_boundary_regions(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制边界区域（用于调试）"""
        if self.boundary_tracker is not None:
            return self.boundary_tracker.draw_boundary_regions(image)
        return image


def print_assignment_stats():
    """打印线性分配算法的性能统计"""
    print("\n" + "="*60)
    print("线性分配算法性能统计")
    print("="*60)

    if PERF_STATS['hungarian_calls'] > 0:
        avg_time = PERF_STATS['hungarian_total_time'] / PERF_STATS['hungarian_calls'] * 1000
        print(f"匈牙利算法: {PERF_STATS['hungarian_calls']} 次调用, "
              f"平均 {avg_time:.3f} ms/次, "
              f"总计 {PERF_STATS['hungarian_total_time']*1000:.3f} ms")

    if PERF_STATS['greedy_calls'] > 0:
        avg_time = PERF_STATS['greedy_total_time'] / PERF_STATS['greedy_calls'] * 1000
        print(f"贪心算法  : {PERF_STATS['greedy_calls']} 次调用, "
              f"平均 {avg_time:.3f} ms/次, "
              f"总计 {PERF_STATS['greedy_total_time']*1000:.3f} ms")

    if PERF_STATS['hungarian_calls'] > 0 and PERF_STATS['greedy_calls'] > 0:
        hungarian_avg = PERF_STATS['hungarian_total_time'] / PERF_STATS['hungarian_calls']
        greedy_avg = PERF_STATS['greedy_total_time'] / PERF_STATS['greedy_calls']
        ratio = hungarian_avg / greedy_avg
        print(f"\n匈牙利算法耗时是贪心算法的 {ratio:.2f} 倍")
        if ratio < 1:
            print("匈牙利算法更快！(因为lap是C优化实现)")
        else:
            print("贪心算法更快！")

    print("="*60 + "\n")


def reset_assignment_stats():
    """重置性能统计"""
    global ALGORITHM_LOGGED
    with _stats_lock:
        ALGORITHM_LOGGED = False
        PERF_STATS['hungarian_calls'] = 0
        PERF_STATS['hungarian_total_time'] = 0.0
        PERF_STATS['greedy_calls'] = 0
        PERF_STATS['greedy_total_time'] = 0.0


# ============================================================================
# HybridSortTracker — 基于 Hybrid-SORT 的多目标跟踪器
# ============================================================================

class HybridSortTracker:
    """
    基于 Hybrid-SORT（AAAI 2024）的多目标跟踪器，替代 BoT_SORTTracker。

    核心关联逻辑（由 Hybrid_Sort 内部完成）：
      1. 高/低置信度检测两阶段关联（BYTE 策略）
      2. IoU + 四角点速度方向一致性代价（VDC，4-point velocity direction consistency）
      3. 置信度状态调制惩罚（TCM，trajectory confidence modulation）
      4. OC-SORT 风格的第三轮基于最后真实观测框的重关联

    保留的原有逻辑：
      · 环绕边界去重：BoundaryCrossingTracker（与 BoT_SORTTracker 完全相同）
      · 跨切片去重：由上游 PanoramaSlicer.merge_detections() 完成，本类不涉及

    接口与 BoT_SORTTracker 完全兼容：
      · 构造函数参数向后兼容（BoT-SORT 专用参数静默忽略）
      · update(List[Dict]) → List[Dict]，字段含义相同
      · reset() / reset_id() / set_boundary_frame_size() / get_boundary_stats() /
        draw_boundary_regions() 均有实现
    """

    def __init__(
        self,
        # ── 置信度阈值（与 BoT_SORTTracker 同名） ──
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.5,
        new_track_overlap_thresh: float = 0.6,   # 与现有轨迹 IoU>此值的检测不新建轨迹（保持旧 ID 优先）
        # ── 轨迹生命周期 ──
        track_buffer: int = 30,
        frame_rate: int = 30,
        # ── 关联阈值 ──
        match_thresh: float = 0.3,
        # ── BoT-SORT 兼容参数（静默忽略，保留以兼容调用方） ──
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.5,
        reid_lost_thresh: float = 0.25,
        feat_history: int = 50,
        use_hungarian: bool = False,
        # ── ReID 开关与参数 ──
        with_reid: bool = False,            # True → 使用 Hybrid_Sort_ReID（外观特征参与关联）
        reid_emb_weight_high: float = 0.1, # 第一轮关联中外观代价的权重（0=纯 IoU+VDC）
        reid_emb_weight_low: float = 0.0,  # BYTE 第二轮关联中外观代价的权重
        reid_alpha: float = 0.8,           # smooth_feat EMA 动量（α·old + (1-α)·new）
        reid_longterm_bank: int = 30,      # 每条轨迹保留的历史特征帧数
        reid_adapfs: bool = False,         # 自适应特征平滑（依赖检测置信度加权）
        reid_high_score_thresh: float = 0.8,  # 外观匹配的最低相似度阈值
        # ── Hybrid-SORT 专有参数 ──
        inertia: float = 0.2,
        delta_t: int = 3,
        use_byte: bool = True,
        tcm_first_step: bool = True,
        tcm_first_step_weight: float = 1.0,
        tcm_byte_step: bool = True,
        tcm_byte_step_weight: float = 1.0,
        asso_func: str = "iou",
        min_hits: int = 1,
        # ── 框平滑 ──
        smooth_bbox: bool = False,       # 是否对输出框宽高做 EMA 平滑
        smooth_bbox_alpha: float = 0.5,  # EMA 系数，0=纯当前帧，1=纯历史
        # ── Kalman 轨迹框 ──
        kalman_bbox: bool = False,    # True → 输出 Kalman 状态框而非 YOLO 原始框
        # ── Round 3.5 中心距离兜底 ──
        cd_thresh: float = 0.5,       # 归一化中心点距离阈值（< 0 关闭）
        # ── 全景图尺寸 ──
        panorama_width: int = 3840,
        panorama_height: int = 1080,
        # ── 环绕边界匹配参数 ──
        enable_boundary_matching: bool = False,
        frame_width: int = 3840,
        frame_height: int = 1080,
        boundary_margin: float = 0.1,
        boundary_time_window: int = 30,
        boundary_similarity_thresh: float = 0.6,
        boundary_debug: bool = True,
        enable_top_boundary: bool = False,
        enable_bottom_boundary: bool = True,
        enable_left_boundary: bool = True,
        enable_right_boundary: bool = True,
    ):
        if not HYBRID_SORT_AVAILABLE:
            raise RuntimeError(
                "HybridSORT 未能导入，请确认 HybridSORT/ 目录存在且依赖已安装。\n"
                f"  期望路径：{_HYBRID_SORT_PATH}"
            )

        import argparse

        # ── 记录参数 ──────────────────────────────────────────────────────────
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.new_track_overlap_thresh = new_track_overlap_thresh
        self.panorama_width = panorama_width
        self.panorama_height = panorama_height
        self._max_age = int(frame_rate / 30.0 * track_buffer)
        self._min_hits = min_hits
        self._match_thresh = match_thresh
        self._inertia = inertia
        self._delta_t = delta_t
        self._asso_func = asso_func
        self._use_byte = use_byte
        self._with_reid = with_reid and (Hybrid_Sort_ReID is not None)

        # Hybrid_Sort / Hybrid_Sort_ReID 共用的 args namespace
        # Hybrid_Sort_ReID 需要额外的 ReID 相关字段
        self.hs_args = argparse.Namespace(
            # 基础阈值
            track_thresh=track_high_thresh,
            low_thresh=track_low_thresh,
            # TCM
            TCM_first_step=tcm_first_step,
            TCM_first_step_weight=tcm_first_step_weight,
            TCM_byte_step=tcm_byte_step,
            TCM_byte_step_weight=tcm_byte_step_weight,
            # BYTE
            use_byte=use_byte,
            # ReID（Hybrid_Sort_ReID 专用，Hybrid_Sort 忽略）
            EG_weight_high_score=reid_emb_weight_high,
            EG_weight_low_score=reid_emb_weight_low,
            high_score_matching_thresh=reid_high_score_thresh,
            alpha=reid_alpha,
            adapfs=reid_adapfs,
            longterm_bank_length=reid_longterm_bank,
            with_longterm_reid=False,
            longterm_reid_weight=0.0,
            with_longterm_reid_correction=False,
            longterm_reid_correction_thresh=1.0,
            longterm_reid_correction_thresh_low=1.0,
            dataset="dancetrack",
            ECC=False,
        )

        # ── Round 3.5 / 遮挡后 ID 继承的中心距离阈值（需在 _make_inner 之前赋值）──
        self._cd_thresh = float(cd_thresh)

        # ── 创建核心 Hybrid_Sort 实例 ────────────────────────────────────────
        self._inner: Hybrid_Sort = self._make_inner()

        self.frame_id = 0

        # ── ID 生命周期调试日志（环境变量 HS_ID_DEBUG=1 开启）──────────────
        # 将每帧"出现/消失"的最终 track_id 及其中心坐标写入 txt 文件（不打印到终端），
        # 用于定位 ID 跳变事件。文件路径由 HS_ID_DEBUG_FILE 指定，默认 ./id_debug.txt。
        self._id_debug = os.environ.get('HS_ID_DEBUG', '') not in ('', '0', 'false', 'False')
        self._prev_output_dbg: Dict[int, List] = {}
        self._id_debug_fp = None
        if self._id_debug:
            _dbg_path = os.environ.get('HS_ID_DEBUG_FILE', 'id_debug.txt')
            try:
                self._id_debug_fp = open(_dbg_path, 'w', encoding='utf-8')
                self._id_debug_fp.write("# 格式: f<帧号>\\t事件\\tid=<最终ID>\\t@(cx,cy)\\tsize=(w x h)\n")
                self._id_debug_fp.flush()
                print(f"[ID] ID 生命周期日志将写入: {os.path.abspath(_dbg_path)}")
            except Exception as _e:
                print(f"[ID] 无法打开调试日志文件，已关闭 ID 调试: {_e}")
                self._id_debug = False

        # ── 显示号连续化 ────────────────────────────────────────────────────
        # 内部计数器在"确认"时就 +1，但边界匹配可能随后把某条轨迹重映射回老 ID，
        # 导致显示号出现缺口。这里在最终输出前再套一层映射：每个（重映射后的）内部
        # final_track_id 首次出现时领取一个连续递增的公开号；被边界找回的轨迹复用其
        # 老 final_track_id 对应的公开号 → 显示号严格连续、无缺口。
        self._public_id_map: Dict[int, int] = {}
        self._public_id_counter = 0

        # ── 特征 EMA 缓存（模拟 STrack.smooth_feat，供边界匹配使用） ────────
        # alpha=0.9 固定值；如需动态调整可参考 STrack.update_features()
        self._feat_cache: Dict[int, np.ndarray] = {}
        self._feat_alpha = 0.9

        # ── 检测元数据缓存（class_id / class_name / keypoints / confidence） ─
        # 每帧通过 IoU 反查，将输出 bbox 对应到输入 detection，更新此缓存
        self._meta_cache: Dict[int, Dict] = {}

        # ── 上一帧状态（用于检测新出现/消失轨迹） ───────────────────────────
        self._prev_active_ids: set = set()
        # track_id → 上一帧该轨迹的 bbox [x1,y1,x2,y2]
        self._prev_bbox: Dict[int, List] = {}

        # ── 环绕边界匹配（与 BoT_SORTTracker 逻辑完全相同） ─────────────────
        self.enable_boundary_matching = enable_boundary_matching and BOUNDARY_MATCHER_AVAILABLE
        self.boundary_tracker = None
        if self.enable_boundary_matching:
            self.boundary_tracker = BoundaryCrossingTracker(
                frame_width=frame_width,
                frame_height=frame_height,
                boundary_margin=boundary_margin,
                time_window=boundary_time_window,
                similarity_threshold=boundary_similarity_thresh,
                debug=boundary_debug,
                enable_top_boundary=enable_top_boundary,
                enable_bottom_boundary=enable_bottom_boundary,
                enable_left_boundary=enable_left_boundary,
                enable_right_boundary=enable_right_boundary,
            )

        # 临时 ID 映射（边界匹配后将新 track_id 重映射到旧 track_id，跨帧持久）
        self.prev_track_info: Dict[int, Dict] = {}
        self.temp_id_map: Dict[int, int] = {}

        # 框宽高 EMA 平滑
        self._smooth_bbox = smooth_bbox
        self._smooth_bbox_alpha = float(np.clip(smooth_bbox_alpha, 0.0, 0.99))
        self._bbox_size_cache: Dict[int, Tuple[float, float, float, float]] = {}  # track_id → (cx, cy, w, h)
        self.kalman_bbox = kalman_bbox

    # ── 内部工具 ─────────────────────────────────────────────────────────────

    def _make_inner(self):
        """创建（或重建）内部跟踪器实例，同时重置 KalmanBoxTracker.count。"""
        common = dict(
            args=self.hs_args,
            det_thresh=self.track_high_thresh,
            max_age=self._max_age,
            min_hits=self._min_hits,
            iou_threshold=self._match_thresh,
            delta_t=self._delta_t,
            asso_func=self._asso_func,
            inertia=self._inertia,
            new_track_thresh=self.new_track_thresh,
            new_track_overlap_thresh=self.new_track_overlap_thresh,
        )
        if self._with_reid:
            inner = Hybrid_Sort_ReID(**common)
        else:
            inner = Hybrid_Sort(**common, use_byte=self._use_byte, low_thresh=self.track_low_thresh)
        inner.cd_thresh = self._cd_thresh  # Round 3.5 中心距离兜底阈值
        return inner

    # ── 公共接口（与 BoT_SORTTracker 完全相同） ──────────────────────────────

    def reset_id(self):
        """重置 track_id 计数器（兼容 BoT_SORTTracker 接口）。"""
        if self._with_reid:
            if _HybridReIDKalmanBoxTracker is not None:
                _HybridReIDKalmanBoxTracker.count = 0
        else:
            if _HybridKalmanBoxTracker is not None:
                _HybridKalmanBoxTracker.count = 0

    def reset(self):
        """重置跟踪器全部状态（等价于重新实例化）。"""
        self._inner = self._make_inner()
        self.frame_id = 0
        self._feat_cache.clear()
        self._meta_cache.clear()
        self._prev_active_ids.clear()
        self._prev_bbox.clear()
        self.temp_id_map.clear()
        self.prev_track_info.clear()
        self._bbox_size_cache.clear()
        self._public_id_map.clear()
        self._public_id_counter = 0
        self._prev_output_dbg.clear()
        if self.boundary_tracker is not None:
            self.boundary_tracker.reset()

    def set_boundary_frame_size(self, width: int, height: int):
        """设置边界匹配器的画面尺寸（兼容 BoT_SORTTracker 接口）。"""
        if self.boundary_tracker is not None:
            self.boundary_tracker.set_frame_size(width, height)

    def get_boundary_stats(self) -> Dict[str, Any]:
        """获取边界匹配统计信息（兼容 BoT_SORTTracker 接口）。"""
        if self.boundary_tracker is not None:
            return self.boundary_tracker.get_stats()
        return {'enabled': False}

    def draw_boundary_regions(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制边界区域（用于调试，兼容 BoT_SORTTracker 接口）。"""
        if self.boundary_tracker is not None:
            return self.boundary_tracker.draw_boundary_regions(image)
        return image

    # ── 核心更新逻辑 ──────────────────────────────────────────────────────────

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        每帧调用一次，更新跟踪状态并返回带稳定 track_id 的检测结果。

        Args:
            detections: 检测列表，格式与 BoT_SORTTracker 完全相同：
                [{'bbox': [x1,y1,x2,y2], 'confidence': float,
                  'class_id': int, 'class_name': str,
                  'keypoints': list, 'feature': np.ndarray | None}, ...]

        Returns:
            同格式的检测列表，每项包含稳定的 'track_id'。
        """
        self.frame_id += 1

        # ① 边界匹配帧头预处理 ────────────────────────────────────────────────
        if self.enable_boundary_matching and self.boundary_tracker is not None:
            self.boundary_tracker.pre_process(self.frame_id)

        # ② 构造输入数组 [N, 5] = [x1,y1,x2,y2,score]，以及 ReID 特征矩阵 ───
        if detections:
            dets_np = np.array(
                [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['confidence']]
                 for d in detections],
                dtype=np.float32,
            )
        else:
            dets_np = np.empty((0, 5), dtype=np.float32)

        # 预计算重叠检测框集合：IoU > 阈值则裁图包含邻近人体，特征会被污染
        # 0.05：偏头等轻微接触（IoU 0.05~0.1）在 0.1 阈值下无法被拦截，导致 ReID 特征
        # 悄悄污染；降到 0.05 可提前保护，避免高 reid_emb_weight 时污染特征主导分配。
        # with_reid 路径：置零 id_feature_np → KalmanBoxTracker.update_features() norm<1e-6 → 跳过
        # 无reid 路径：跳过外部 EMA 更新，保护 _feat_cache
        _OVERLAP_IOU_THRESH = 0.12
        _contaminated_det_indices: set = set()
        if len(dets_np) > 1:
            for _a in range(len(dets_np)):
                for _b in range(_a + 1, len(dets_np)):
                    if box_iou(dets_np[_a, :4], dets_np[_b, :4]) > _OVERLAP_IOU_THRESH:
                        _contaminated_det_indices.add(_a)
                        _contaminated_det_indices.add(_b)

        # img_info == img_size → scale = 1.0，坐标不做缩放
        img_info = [self.panorama_height, self.panorama_width]
        img_size = [self.panorama_height, self.panorama_width]

        # ③ 调用核心跟踪器 ───────────────────────────────────────────────────
        if self._with_reid:
            # Hybrid_Sort_ReID 必须始终收到 id_feature（不能为 None），
            # 空帧时传 shape=(0, feat_dim) 的空矩阵，让内部逻辑正常走完
            feat_dim = 512
            if detections:
                for d in detections:
                    f = d.get('feature')
                    if f is not None:
                        feat_dim = int(np.asarray(f).size)
                        break
            id_feature_np = np.zeros((len(detections), feat_dim), dtype=np.float32)
            for i, d in enumerate(detections):
                feat = d.get('feature')
                if feat is not None:
                    f = np.asarray(feat, dtype=np.float32).flatten()
                    if f.shape[0] == feat_dim:
                        norm = np.linalg.norm(f)
                        if norm > 1e-6:
                            id_feature_np[i] = f / norm
            # freeze_feat：重叠检测框置零 → KalmanBoxTracker.update_features() 跳过，保护 smooth_feat
            for _idx in _contaminated_det_indices:
                if _idx < len(id_feature_np):
                    id_feature_np[_idx] = 0.0
            online_targets = self._inner.update(dets_np, img_info, img_size,
                                                 id_feature=id_feature_np)
        else:
            online_targets = self._inner.update(dets_np, img_info, img_size)

        # ③-后：ReID 模式下，从内部 KalmanBoxTracker 同步 smooth_feat 到 _feat_cache
        # 内部跟踪器维护自己的 EMA（smooth_feat），比外部 EMA 更准确（含自适应平滑）
        # 仅纳入已确认轨迹（trk.id >= 0）；未确认轨迹 id=-1 尚未分配 ID，不参与映射/特征同步
        _id_to_inner_trk: Dict[int, Any] = {trk.id + 1: trk for trk in self._inner.trackers if trk.id >= 0}
        if self._with_reid:
            for tid, trk in _id_to_inner_trk.items():
                if trk.smooth_feat is not None:
                    self._feat_cache[tid] = trk.smooth_feat

        # ④ 本帧活跃 track_id 集合 ────────────────────────────────────────────
        current_active_ids: set = (
            {int(row[4]) for row in online_targets}
            if len(online_targets) > 0
            else set()
        )

        # ⑤ 处理消失轨迹 → 注册到边界匹配器，并清理框平滑缓存 ──────────────────
        #    消失轨迹 = 上帧活跃、本帧不活跃的轨迹
        lost_ids = self._prev_active_ids - current_active_ids
        for _lost in lost_ids:
            self._bbox_size_cache.pop(_lost, None)
        if self.enable_boundary_matching and self.boundary_tracker is not None:
            for lost_id in lost_ids:
                smooth_feat = self._feat_cache.get(lost_id)
                if smooth_feat is None:
                    continue
                last_bbox = self._prev_bbox.get(lost_id, [0, 0, 1, 1])
                self.boundary_tracker.process_lost_track(
                    track_id=lost_id,
                    bbox=last_bbox,
                    feature=smooth_feat,
                    frame_id=self.frame_id,
                    smooth_feat=smooth_feat,
                    prev_bbox=self._prev_bbox.get(lost_id),
                )

        # ⑥-a. 双射 IoU 反查：将每条输出轨迹唯一对应一个输入 detection ─────
        # Hybrid_Sort 输出的是 last_observation（原始检测框），IoU 通常精确为 1.0。
        # 按 IoU 降序贪心分配，避免两条轨迹争抢同一个 detection。
        track_to_det_idx: Dict[int, int] = {}
        if detections and len(online_targets) > 0:
            pairs: List[Tuple[float, int, int]] = []
            for ti, row in enumerate(online_targets):
                out_bbox = [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
                for di, det in enumerate(detections):
                    iou = box_iou(out_bbox, det['bbox'])
                    if iou > 0.3:
                        pairs.append((iou, ti, di))
            pairs.sort(key=lambda x: -x[0])
            used_trk_set: set = set()
            used_det_set: set = set()
            for _iou, ti, di in pairs:
                if ti not in used_trk_set and di not in used_det_set:
                    track_to_det_idx[ti] = di
                    used_trk_set.add(ti)
                    used_det_set.add(di)

        # ⑥ 逐轨迹后处理：更新元数据 / 特征 EMA / 注册边界新轨迹 ───────────
        new_ids = current_active_ids - self._prev_active_ids  # 本帧首次出现的轨迹
        output_detections: List[Dict] = []

        for ti, row in enumerate(online_targets):
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            track_id = int(row[4])
            out_bbox = [x1, y1, x2, y2]
            best_det: Optional[Dict] = (
                detections[track_to_det_idx[ti]] if ti in track_to_det_idx else None
            )

            # ── b. 更新元数据缓存 ─────────────────────────────────────────────
            if best_det is not None:
                self._meta_cache[track_id] = {
                    'confidence': best_det.get('confidence', 0.5),
                    'class_id':   best_det.get('class_id', 0),
                    'class_name': best_det.get('class_name', 'person'),
                    'keypoints':  best_det.get('keypoints', []),
                }
            meta = self._meta_cache.get(track_id) or {}

            # ── c. 更新特征 EMA ───────────────────────────────────────────────
            # with_reid=True：smooth_feat 由内部 KalmanBoxTracker 管理，
            #                 已在步骤③-后同步到 _feat_cache，此处直接读取
            # with_reid=False：由外部 EMA 维护（α=0.9）
            if not self._with_reid:
                curr_feat = best_det.get('feature') if best_det is not None else None
                _det_idx_for_feat = track_to_det_idx.get(ti)
                _feat_contaminated = (_det_idx_for_feat is not None
                                      and _det_idx_for_feat in _contaminated_det_indices)
                if curr_feat is not None and not _feat_contaminated:
                    feat_arr = np.asarray(curr_feat, dtype=np.float32).flatten()
                    norm = np.linalg.norm(feat_arr)
                    if norm > 1e-6:
                        feat_arr /= norm
                        if track_id in self._feat_cache:
                            self._feat_cache[track_id] = (
                                self._feat_alpha * self._feat_cache[track_id]
                                + (1 - self._feat_alpha) * feat_arr
                            )
                            n2 = np.linalg.norm(self._feat_cache[track_id])
                            if n2 > 1e-6:
                                self._feat_cache[track_id] /= n2
                        else:
                            self._feat_cache[track_id] = feat_arr
            smooth_feat = self._feat_cache.get(track_id)

            # ── d. 边界匹配：仅对本帧首次出现的轨迹检查是否与消失轨迹吻合 ───
            # 用 track_id（Hybrid_Sort 内部 ID）作为 temp_id，使得：
            #   · pending_remaps[track_id] = matched_id → post_process() 可直接重映射
            #   · id_remap[track_id] = matched_id → 该轨迹消失时 process_lost_track 能
            #     追溯到正确的 matched_id，保持边界匹配链的一致性
            # temp_id_map 跨帧持久保存映射，确保连续帧都输出正确的 final_track_id。
            is_new_track = track_id in new_ids
            if (is_new_track
                    and self.enable_boundary_matching
                    and self.boundary_tracker is not None
                    and smooth_feat is not None):
                matched_id = self.boundary_tracker.check_new_track(
                    bbox=out_bbox,
                    feature=smooth_feat,
                    frame_id=self.frame_id,
                    temp_id=track_id,
                )
                if matched_id is not None:
                    self.temp_id_map[track_id] = matched_id

            # temp_id_map 持久映射（post_process 仅负责首帧的 pending_remap 重映射）
            if track_id in self.temp_id_map:
                final_track_id = self.temp_id_map[track_id]
                boundary_matched = True
            else:
                final_track_id = track_id
                boundary_matched = False

            # ── e-0. 可选：用 Kalman 状态框替代 YOLO 原始框 ─────────────────
            if self.kalman_bbox and track_id in _id_to_inner_trk:
                try:
                    ks = _id_to_inner_trk[track_id].get_state()[0]
                    out_bbox = [float(ks[0]), float(ks[1]), float(ks[2]), float(ks[3])]
                except Exception:
                    pass  # get_state 失败时保持原始框

            # ── e. 全框 EMA 平滑（--smooth-bbox）─────────────────────────────
            # kalman_bbox 时 Kalman 本身已平滑，跳过 EMA 避免过度平滑
            if self._smooth_bbox and not self.kalman_bbox:
                cx = (out_bbox[0] + out_bbox[2]) / 2
                cy = (out_bbox[1] + out_bbox[3]) / 2
                w  = out_bbox[2] - out_bbox[0]
                h  = out_bbox[3] - out_bbox[1]
                _cached = self._bbox_size_cache.get(track_id)
                if _cached is not None and len(_cached) == 4:  # 防御旧格式 2-tuple
                    pcx, pcy, pw, ph = _cached
                    a = self._smooth_bbox_alpha
                    cx = a * pcx + (1 - a) * cx
                    cy = a * pcy + (1 - a) * cy
                    w  = a * pw  + (1 - a) * w
                    h  = a * ph  + (1 - a) * h
                self._bbox_size_cache[track_id] = (cx, cy, w, h)
                out_bbox = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

            # ── f. 构建输出 dict（字段与 BoT_SORTTracker 完全相同） ──────────
            out_feat = smooth_feat[np.newaxis, :] if smooth_feat is not None else None
            det_out = {
                'bbox':              out_bbox,
                'confidence':        meta.get('confidence', 0.5),
                'class_id':          meta.get('class_id', 0),
                'class_name':        meta.get('class_name', 'person'),
                'keypoints':         meta.get('keypoints', []),
                'track_id':          final_track_id,
                'feature':           out_feat,
                '_feature_count':    1 if smooth_feat is not None else 0,
                '_boundary_matched': boundary_matched,
            }
            output_detections.append(det_out)

        # ⑦ 边界匹配帧尾后处理：对本帧首次出现的轨迹应用 pending_remaps ─────
        # post_process() 通过 pending_remaps[track_id] 完成首帧映射（与 temp_id_map 一致），
        # 并更新 boundary_tracker 的 prev_tracks（供下一帧的消失检测使用）。
        if self.enable_boundary_matching and self.boundary_tracker is not None:
            output_detections = self.boundary_tracker.post_process(output_detections)

        # === use_track_bbox：补充 lost 轨迹的 Kalman 预测框 ===
        if self.kalman_bbox:
            output_internal_ids = {int(row[4]) for row in online_targets}
            for trk in self._inner.trackers:
                if trk.id < 0:
                    continue  # 未确认轨迹（尚未分配 ID），不输出预测框
                tid = trk.id + 1
                if tid in output_internal_ids:
                    continue  # 本帧已匹配，已在输出里
                try:
                    ks = trk.get_state()[0]
                    out_bbox = [float(ks[0]), float(ks[1]), float(ks[2]), float(ks[3])]
                except Exception:
                    continue
                final_tid = self.temp_id_map.get(tid, tid)
                meta = self._meta_cache.get(tid) or {}
                smooth_feat = self._feat_cache.get(tid)
                output_detections.append({
                    'bbox': out_bbox,
                    'confidence': meta.get('confidence', 0.5),
                    'class_id': meta.get('class_id', 0),
                    'class_name': meta.get('class_name', 'person'),
                    'keypoints': meta.get('keypoints', []),
                    'track_id': final_tid,
                    'feature': smooth_feat[np.newaxis, :] if smooth_feat is not None else None,
                    '_feature_count': 0,
                    '_boundary_matched': False,
                    '_is_lost': True,
                })

        # ⑦-末-pre 短暂遮挡后 ID 恢复
        #
        # 触发条件：新确认轨迹（raw_id 首次出现，不在 _public_id_map）+ 附近有最近丢失的旧轨迹
        #
        # 保护措施（仅保留时间 + 位置两个约束，不排除"附近有活跃目标"的情况）：
        #   ① 时间约束：旧轨迹 time_since_update ≤ _max_frames_lost（丢失时间短）
        #   ② 位置约束：新旧轨迹中心距离 < _INHERIT_CD_THRESH 倍平均框高（位置足够近）
        #
        # 注意：被其他目标遮挡时，遮挡者本身是活跃目标——若加"周边活跃排除"会把被遮挡后
        # 重现的情况也拦住，因此不做活跃邻居排除。两个约束合力保证：只有"短时间内在同一
        # 位置消失又出现"才触发继承；人真的走远了（位置远）或消失太久（时间长）则不继承。
        _MAX_FRAMES_LOST = self._max_age   # 与 max_age 对齐：旧轨迹还活着就参与比较
        _INHERIT_CD_THRESH = 1.0           # 中心距离 / 平均框高，新旧轨迹位置必须足够近
        _INHERIT_DEBUG = os.environ.get('HS_INHERIT_DEBUG', '') not in ('', '0', 'false')

        # 找出"已确认但本帧未匹配"的旧轨迹
        _lost_confirmed = {
            tid: trk for tid, trk in _id_to_inner_trk.items()
            if tid not in current_active_ids
        }

        if _lost_confirmed:
            for det in output_detections:
                if det.get('_is_lost'):
                    continue
                raw_id = det['track_id']
                if raw_id in self._public_id_map:
                    continue  # 已知 ID，无需继承

                det_bbox = det['bbox']
                cx_n = (det_bbox[0] + det_bbox[2]) / 2.0
                cy_n = (det_bbox[1] + det_bbox[3]) / 2.0
                h_n = max(det_bbox[3] - det_bbox[1], 1.0)

                if _INHERIT_DEBUG:
                    print(f"[INHERIT] f{self.frame_id} new_raw={raw_id} "
                          f"pos=({cx_n:.0f},{cy_n:.0f}) h={h_n:.0f} "
                          f"lost_ids={list(_lost_confirmed.keys())}")

                # 寻找满足条件的最近旧轨迹
                # 同时比较 last_observation（最后检测位置）和 Kalman 预测位置，取最小距离：
                # · last_observation：人没动时最准
                # · Kalman 预测：人在遮挡期间匀速运动时，预测位置更接近重现位置
                best_old_raw, best_dist = None, _INHERIT_CD_THRESH
                for old_tid, old_trk in _lost_confirmed.items():
                    # ① 时间约束：丢失太久的旧轨迹不参与（可能已是不同目标）
                    tsu = old_trk.time_since_update
                    if tsu > _MAX_FRAMES_LOST:
                        if _INHERIT_DEBUG:
                            print(f"[INHERIT]   skip old_raw={old_tid} tsu={tsu} > {_MAX_FRAMES_LOST}")
                        continue
                    if old_trk.last_observation.sum() < 0:
                        if _INHERIT_DEBUG:
                            print(f"[INHERIT]   skip old_raw={old_tid} no obs")
                        continue
                    obs = old_trk.last_observation
                    cx_o = (obs[0] + obs[2]) / 2.0
                    cy_o = (obs[1] + obs[3]) / 2.0
                    h_o = max(obs[3] - obs[1], 1.0)
                    avg_h = (h_n + h_o) / 2.0
                    d_obs = (np.sqrt((cx_n - cx_o) ** 2 + (cy_n - cy_o) ** 2) / avg_h)
                    # Kalman 预测位置（已连续 predict tsu 帧，跟踪器内部状态）
                    d_kalman = d_obs
                    try:
                        ks = old_trk.get_state()[0]
                        cx_k = (ks[0] + ks[2]) / 2.0
                        cy_k = (ks[1] + ks[3]) / 2.0
                        d_kalman = np.sqrt((cx_n - cx_k) ** 2 + (cy_n - cy_k) ** 2) / avg_h
                    except Exception:
                        pass
                    d = min(d_obs, d_kalman)
                    if _INHERIT_DEBUG:
                        print(f"[INHERIT]   old_raw={old_tid} pub={self._public_id_map.get(old_tid)} "
                              f"tsu={tsu} obs=({cx_o:.0f},{cy_o:.0f}) "
                              f"d_obs={d_obs:.3f} d_kalman={d_kalman:.3f} d={d:.3f}")
                    if d < best_dist:
                        best_dist, best_old_raw = d, old_tid

                if best_old_raw is not None and best_old_raw in self._public_id_map:
                    old_pub = self._public_id_map.pop(best_old_raw)
                    self._public_id_map[raw_id] = old_pub  # 继承旧公开号
                    if _INHERIT_DEBUG:
                        print(f"[INHERIT] ✓ f{self.frame_id} new_raw={raw_id} → pub={old_pub} "
                              f"(from old_raw={best_old_raw} dist={best_dist:.3f})")
                elif _INHERIT_DEBUG:
                    reason = "no_match" if best_old_raw is None else f"old_raw={best_old_raw}_not_in_pubmap"
                    print(f"[INHERIT] ✗ f{self.frame_id} new_raw={raw_id} ({reason})")

        # ⑦-末 显示号连续化：把（边界重映射后的）final_track_id 映射到连续递增的公开号。
        # 仅改写对外输出的 track_id；内部状态（_prev_active_ids/temp_id_map/_meta_cache/
        # boundary_tracker 等）仍用 Hybrid_Sort 内部 ID，不受影响。被边界找回的轨迹其
        # final_track_id 等于老内部 ID，命中已有映射 → 复用老公开号，因此不再产生缺号。
        for det in output_detections:
            _raw = det['track_id']
            _pub = self._public_id_map.get(_raw)
            if _pub is None:
                self._public_id_counter += 1
                _pub = self._public_id_counter
                self._public_id_map[_raw] = _pub
            det['track_id'] = _pub

        # ⑧ 更新帧间状态（_prev_active_ids / _prev_bbox 使用 Hybrid_Sort 内部 ID） ──
        self._prev_active_ids = current_active_ids
        self._prev_bbox = {
            int(row[4]): [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
            for row in online_targets
        }
        self.prev_track_info = {
            det['track_id']: {'bbox': det['bbox'], 'track_id': det['track_id']}
            for det in output_detections
        }

        # ── ID 生命周期日志：将本帧最终 ID 的"出现/消失"事件写入 txt 文件 ────
        if self._id_debug and self._id_debug_fp is not None:
            cur_boxes = {d['track_id']: d['bbox'] for d in output_detections}
            lines = []
            for tid in sorted(set(cur_boxes) - set(self._prev_output_dbg)):
                x1, y1, x2, y2 = cur_boxes[tid]
                lines.append(f"f{self.frame_id}\t+出现\tid={tid}\t@({(x1+x2)/2:.0f},{(y1+y2)/2:.0f})\t"
                             f"size=({x2-x1:.0f}x{y2-y1:.0f})")
            for tid in sorted(set(self._prev_output_dbg) - set(cur_boxes)):
                x1, y1, x2, y2 = self._prev_output_dbg[tid]
                lines.append(f"f{self.frame_id}\t-消失\tid={tid}\t@({(x1+x2)/2:.0f},{(y1+y2)/2:.0f})")
            if lines:
                self._id_debug_fp.write("\n".join(lines) + "\n")
                self._id_debug_fp.flush()
            self._prev_output_dbg = cur_boxes

        return output_detections
