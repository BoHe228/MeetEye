"""
BoT-SORT多目标跟踪器
整合ByteTrack和DeepSORT的优势：
1. 高/低置信度检测分离（ByteTrack策略）
2. IoU + ReID特征融合（DeepSORT优势）
3. 平滑特征更新（指数移动平均）
4. 支持相机运动补偿（GMC）- 可选
5. 边界穿越ID连续性匹配（新增）
"""
import time
import threading
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque, OrderedDict

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


import sys
import os
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
                 enable_right_boundary: bool = True):  # 是否启用右侧边界
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

        # 计算IoU距离
        ious_dists = iou_distance(strack_pool, detections_high)
        ious_dists_mask = (ious_dists > self.proximity_thresh)

        # 融合检测置信度
        ious_dists = fuse_score(ious_dists, detections_high)

        # 如果有ReID特征，融合外观距离
        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections_high)  # 余弦距离 [0,1]
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0  # IoU 不足时不信任 ReID
            dists = np.minimum(ious_dists, emb_dists)
            # ⚠️ 注意：不做 reid_veto（用 ReID 强制封锁 IoU 支持的匹配）
            # 原因：特征在近邻/遮挡期间会被污染，污染的 smooth_feat 会导致
            # emb_dist(Track_A, Det_A) > appearance_thresh，触发错误否决，
            # 把正确匹配封掉，反而造成 ID 互换。
            # 应对近邻污染的手段是特征冻结（near_other / freeze_feat），而不是事后否决。
        else:
            dists = ious_dists

        # 线性分配
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh, use_hungarian=self.use_hungarian)

        # ── 近邻检测：找出任意两个匹配框之间空间接近的对 ──────────────────
        # 判断标准：IoU > 0 (框有重叠) 或 中心距离 < 均值框尺寸的 1.5 倍
        # 两个层级：
        #   near_other   (轻度接近, IoU > 0 或 距离较近) → alpha=0.98 保护特征
        #   freeze_feat  (重度重叠, IoU > 0.3)          → 完全跳过特征更新
        _FREEZE_IOU_THRESH = 0.3   # 重度：完全跳过特征更新
        _NEAR_IOU_THRESH   = 0.1   # 轻度：IoU > 0.1 即视为框有接触
        matched_dets = [detections_high[idet] for _, idet in matches]
        freeze_det_indices: set = set()
        near_det_indices:   set = set()
        n_md = len(matched_dets)
        for _a in range(n_md):
            for _b in range(_a + 1, n_md):
                _iou = box_iou(matched_dets[_a].tlbr, matched_dets[_b].tlbr)
                if _iou > _FREEZE_IOU_THRESH:
                    freeze_det_indices.add(_a)
                    freeze_det_indices.add(_b)
                elif _iou > _NEAR_IOU_THRESH:
                    near_det_indices.add(_a)
                    near_det_indices.add(_b)

        for _mi, (itracked, idet) in enumerate(matches):
            track = strack_pool[itracked]
            det = detections_high[idet]
            freeze_feat = (_mi in freeze_det_indices)
            near_other  = (_mi in near_det_indices)
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
                'bbox': track.xyxy.tolist(),
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
