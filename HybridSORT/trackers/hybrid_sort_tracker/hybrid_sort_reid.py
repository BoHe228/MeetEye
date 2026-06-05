"""
    This script is adopted from the SORT script by Alex Bewley alex@bewley.ai
"""
from __future__ import print_function

import numpy as np
import copy
from .association import *
from collections import deque       # [hgx0418] deque for reid feature
np.random.seed(0)

def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age-dt]
    max_age = max(observations.keys())
    return observations[max_age]


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h  # scale is just area
    r = w / float(h+1e-6)
    score = bbox[4]
    if score:
        return np.array([x, y, s, score, r]).reshape((5, 1))
    else:
        return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[4])
    h = x[2] / w
    score = x[3]
    if(score == None):
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1, 4))
    else:
      return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1, 5))


def speed_direction(bbox1, bbox2):
    cx1, cy1 = (bbox1[0]+bbox1[2]) / 2.0, (bbox1[1]+bbox1[3])/2.0
    cx2, cy2 = (bbox2[0]+bbox2[2]) / 2.0, (bbox2[1]+bbox2[3])/2.0
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[1]
    cx2, cy2 = bbox2[0], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rt(bbox1, bbox2):
    cx1, cy1 = bbox1[0], bbox1[3]
    cx2, cy2 = bbox2[0], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_lb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[1]
    cx2, cy2 = bbox2[2], bbox2[1]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def speed_direction_rb(bbox1, bbox2):
    cx1, cy1 = bbox1[2], bbox1[3]
    cx2, cy2 = bbox2[2], bbox2[3]
    speed = np.array([cy2-cy1, cx2-cx1])
    norm = np.sqrt((cy2-cy1)**2 + (cx2-cx1)**2) + 1e-6
    return speed / norm

def _normalize_vel(vel):
    """将速度方向向量单位化（模长归一）。

    上游实现把最近 delta_t 帧的方向向量直接累加（未除以帧数），模长可达 ~delta_t，
    使 association.cost_vel 中 inertia·dir 的点积超出 [-1,1] 被 np.clip 截断饱和，
    VDC 方向一致性代价退化为近二值。单位化后点积恢复为真实余弦，方向代价随夹角平滑变化。
    """
    if vel is None:
        return vel
    n = np.sqrt((vel ** 2).sum()) + 1e-6
    return vel / n


def _center_dist_normalized(dets, trks):
    """归一化中心点距离矩阵（以框高度归一化），对框大小变化鲁棒。"""
    n_d, n_t = len(dets), len(trks)
    mat = np.full((n_d, n_t), np.inf, dtype=np.float32)
    for i in range(n_d):
        cx_d = (dets[i, 0] + dets[i, 2]) / 2.0
        cy_d = (dets[i, 1] + dets[i, 3]) / 2.0
        h_d = max(dets[i, 3] - dets[i, 1], 1.0)
        for j in range(n_t):
            if trks[j, 0] < 0:
                continue
            cx_t = (trks[j, 0] + trks[j, 2]) / 2.0
            cy_t = (trks[j, 1] + trks[j, 3]) / 2.0
            h_t = max(trks[j, 3] - trks[j, 1], 1.0)
            dist = np.sqrt((cx_d - cx_t) ** 2 + (cy_d - cy_t) ** 2)
            mat[i, j] = dist / ((h_d + h_t) / 2.0)
    return mat

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox, temp_feat, delta_t=3, orig=False, buffer_size=30, args=None):     # 'temp_feat' and 'buffer_size' for reid feature
        """
        Initialises a tracker using initial bounding box.

        """
        # define constant velocity model
        # if not orig and not args.kalman_GPR:
        if not orig:
          from .kalmanfilter_score_new import KalmanFilterNew_score_new as KalmanFilter_score_new
          self.kf = KalmanFilter_score_new(dim_x=9, dim_z=5)
        else:
          from filterpy.kalman import KalmanFilter
          self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # u, v, s, c, r, ~u, ~v, ~s, ~c
        self.kf.F = np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.
        self.kf.P[5:, 5:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[5:, 5:] *= 0.01

        self.kf.x[:5] = convert_bbox_to_z(bbox)

        self.time_since_update = 0
        # 延迟 ID 分配：创建时不占用全局计数器，待轨迹确认（连续命中达到 min_hits）后再分配。
        # -1 表示尚未确认；避免瞬时误检消耗 ID 号导致编号跳变/膨胀。
        self.id = -1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        """
        NOTE: [-1,-1,-1,-1,-1] is a compromising placeholder for non-observation status, the same for the return of 
        function k_previous_obs. It is ugly and I do not like it. But to support generate observation array in a 
        fast and unified way, which you would see below k_observations = np.array([k_previous_obs(...]]), let's bear it for now.
        """
        self.last_observation = np.array([-1, -1, -1, -1, -1])  # placeholder
        self.last_observation_save = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []
        self.velocity_lt = None
        self.velocity_rt = None
        self.velocity_lb = None
        self.velocity_rb = None
        self.delta_t = delta_t
        self.confidence_pre = None
        self.confidence = bbox[-1]
        self.args = args
        self.kf.args = args

        # add the following values and functions
        self.smooth_feat = None
        buffer_size = args.longterm_bank_length
        self.features = deque([], maxlen=buffer_size)
        self.update_features(temp_feat)

        # momentum of embedding update
        self.alpha = self.args.alpha

    # ReID. for update embeddings during tracking
    def update_features(self, feat, score=-1):
        norm = np.linalg.norm(feat)
        if norm < 1e-6:
            return  # skip zero/near-zero feature vectors
        feat = feat / norm
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            if self.args.adapfs:
                assert score > 0
                pre_w = self.alpha * (self.confidence / (self.confidence + score))
                cur_w = (1 - self.alpha) * (score / (self.confidence + score))
                sum_w = pre_w + cur_w
                pre_w = pre_w / sum_w
                cur_w = cur_w / sum_w
                self.smooth_feat = pre_w * self.smooth_feat + cur_w * feat
            else:
                self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def camera_update(self, warp_matrix):
        """
        update 'self.mean' of current tracklet with ecc results.
        Parameters
        ----------
        warp_matrix: warp matrix computed by ECC.
        """
        x1, y1, x2, y2, s = convert_x_to_bbox(self.kf.x)[0]
        x1_, y1_, _ = warp_matrix @ np.array([x1, y1, 1]).T
        x2_, y2_, _ = warp_matrix @ np.array([x2, y2, 1]).T
        # w, h = x2_ - x1_, y2_ - y1_
        # cx, cy = x1_ + w / 2, y1_ + h / 2
        self.kf.x[:5] = convert_bbox_to_z([x1_, y1_, x2_, y2_, s])

    def update(self, bbox, id_feature, update_feature=True):
        """
        Updates the state vector with observed bbox.
        """
        velocity_lt = None
        velocity_rt = None
        velocity_lb = None
        velocity_rb = None
        if bbox is not None:
            _kf_damp = False
            if self.last_observation.sum() >= 0:  # no previous observation
                previous_box = None
                for i in range(self.delta_t):
                    # dt = self.delta_t - i
                    if self.age - i - 1 in self.observations:
                        previous_box = self.observations[self.age - i - 1]
                        if velocity_lt is not None:
                            velocity_lt += speed_direction_lt(previous_box, bbox)
                            velocity_rt += speed_direction_rt(previous_box, bbox)
                            velocity_lb += speed_direction_lb(previous_box, bbox)
                            velocity_rb += speed_direction_rb(previous_box, bbox)
                        else:
                            velocity_lt = speed_direction_lt(previous_box, bbox)
                            velocity_rt = speed_direction_rt(previous_box, bbox)
                            velocity_lb = speed_direction_lb(previous_box, bbox)
                            velocity_rb = speed_direction_rb(previous_box, bbox)
                        # break
                if previous_box is None:
                    previous_box = self.last_observation
                    new_vel_lt = speed_direction_lt(previous_box, bbox)
                    new_vel_rt = speed_direction_rt(previous_box, bbox)
                    new_vel_lb = speed_direction_lb(previous_box, bbox)
                    new_vel_rb = speed_direction_rb(previous_box, bbox)
                else:
                    new_vel_lt = velocity_lt
                    new_vel_rt = velocity_rt
                    new_vel_lb = velocity_lb
                    new_vel_rb = velocity_rb

                # 速度向量单位化（修正上游多帧累加导致的模长>1、VDC 余弦饱和问题）
                new_vel_lt = _normalize_vel(new_vel_lt)
                new_vel_rt = _normalize_vel(new_vel_rt)
                new_vel_lb = _normalize_vel(new_vel_lb)
                new_vel_rb = _normalize_vel(new_vel_rb)

                _cx_ref = (previous_box[0] + previous_box[2]) * 0.5
                _cy_ref = (previous_box[1] + previous_box[3]) * 0.5
                _cx_cur = (bbox[0] + bbox[2]) * 0.5
                _cy_cur = (bbox[1] + bbox[3]) * 0.5
                _disp = np.sqrt((_cx_cur - _cx_ref) ** 2 + (_cy_cur - _cy_ref) ** 2)
                _avg_h = max((previous_box[3] - previous_box[1] + bbox[3] - bbox[1]) * 0.5, 1.0)
                _kf_damp = False
                if _disp >= 0.05 * _avg_h:
                    self.velocity_lt = new_vel_lt
                    self.velocity_rt = new_vel_rt
                    self.velocity_lb = new_vel_lb
                    self.velocity_rb = new_vel_rb
                elif self.velocity_lt is not None:
                    self.velocity_lt = self.velocity_lt * 0.3
                    self.velocity_rt = self.velocity_rt * 0.3
                    self.velocity_lb = self.velocity_lb * 0.3
                    self.velocity_rb = self.velocity_rb * 0.3
                    _kf_damp = True
            """
              Insert new observations. This is a ugly way to maintain both self.observations
              and self.history_observations. Bear it for the moment.
            """
            self.last_observation = bbox
            self.last_observation_save = bbox
            self.observations[self.age] = bbox
            self.history_observations.append(bbox)

            self.time_since_update = 0
            self.history = []
            self.hits += 1
            self.hit_streak += 1
            self.kf.update(convert_bbox_to_z(bbox))
            if _kf_damp:
                self.kf.x[5] *= 0.3  # vx
                self.kf.x[6] *= 0.3  # vy
            # add interface for update feature or not
            if update_feature:
                if self.args.adapfs:
                    self.update_features(id_feature, score=bbox[-1])
                else:
                    self.update_features(id_feature)
            self.confidence_pre = self.confidence
            self.confidence = bbox[-1]
        else:
            self.kf.update(bbox)
            self.confidence_pre = None

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[7]+self.kf.x[2]) <= 0):
            self.kf.x[7] *= 0.0

        self.kf.predict()
        # 卡尔曼预测后 x[2](面积) 或 x[4](宽高比) 可能被速度项拉成负数
        # 钳位到最小正值，避免 convert_x_to_bbox 中 sqrt(负数) 产生 nan
        if self.kf.x[2] <= 0:
            self.kf.x[2] = 1e-6
        if self.kf.x[4] <= 0:
            self.kf.x[4] = 1e-6
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        if not self.confidence_pre:
            return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence, 0.1, self.args.track_thresh)
        else:
            return self.history[-1], np.clip(self.kf.x[3], self.args.track_thresh, 1.0), np.clip(self.confidence - (self.confidence_pre - self.confidence), 0.1, self.args.track_thresh)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


"""
    We support multiple ways for association cost calculation, by default
    we use IoU. GIoU may have better performance in some situations. We note 
    that we hardly normalize the cost by all methods to (0,1) which may not be 
    the best practice.
"""
ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist,
                "Height_Modulated_IoU": hmiou
                }


class Hybrid_Sort_ReID(object):
    def __init__(self, args, det_thresh, max_age=30, min_hits=3,
        iou_threshold=0.3, delta_t=3, asso_func="iou", inertia=0.2, new_track_thresh=None,
        new_track_overlap_thresh=0.6):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        # 新轨迹生成阈值：仅置信度 >= 此值的未匹配高分检测才会创建新轨迹。
        # 默认回退到 det_thresh（与原行为一致）；设为更高值可抑制低质量检测起新 ID。
        self.new_track_thresh = new_track_thresh if new_track_thresh is not None else det_thresh
        # 保持旧 ID 优先：未匹配检测若与任一现有轨迹 IoU > 此值，则不新建轨迹（视为重复/抖动）。
        # 1.0 表示关闭该抑制。越小越倾向"宁可不出新 ID 也不抢号"。
        self.new_track_overlap_thresh = new_track_overlap_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = args.use_byte
        self.args = args
        KalmanBoxTracker.count = 0

    # ECC for CMC
    def camera_update(self, trackers, warp_matrix):
        for tracker in trackers:
            tracker.camera_update(warp_matrix)

    def update(self, output_results, img_info, img_size, id_feature=None, warp_matrix=None):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.
        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        if output_results is None:
            return np.empty((0, 5))

        if self.args.ECC:
            # camera update for all stracks
            if warp_matrix is not None:
                self.camera_update(self.trackers, warp_matrix)

        self.frame_count += 1
        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]  # x1y1x2y2
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)
        inds_low = scores > self.args.low_thresh
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)  # self.det_thresh > score > 0.1, for second matching
        dets_second = dets[inds_second]  # detections for second matching
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]
        id_feature_keep = id_feature[remain_inds]  # ID feature of 1st stage matching
        id_feature_second = id_feature[inds_second]  # ID feature of 2nd stage matching

        trks = np.zeros((len(self.trackers), 6))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos, kalman_score, simple_score = self.trackers[t].predict()
            trk[:] = [pos[0][0], pos[0][1], pos[0][2], pos[0][3],
                      float(kalman_score), float(simple_score)]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities_lt = np.array(
            [trk.velocity_lt if trk.velocity_lt is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_rt = np.array(
            [trk.velocity_rt if trk.velocity_rt is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_lb = np.array(
            [trk.velocity_lb if trk.velocity_lb is not None else np.array((0, 0)) for trk in self.trackers])
        velocities_rb = np.array(
            [trk.velocity_rb if trk.velocity_rb is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
            First round of association
        """
        if self.args.EG_weight_high_score > 0 and self.args.TCM_first_step:
            feat_dim = id_feature_keep.shape[1] if id_feature_keep.ndim == 2 else 512
            track_features = np.asarray([
                t.smooth_feat if t.smooth_feat is not None else np.zeros(feat_dim)
                for t in self.trackers
            ], dtype=np.float64)
            # np.asarray([]) gives shape (0,) when trackers is empty; reshape to (0, feat_dim)
            if track_features.ndim != 2:
                track_features = track_features.reshape(-1, feat_dim)
            emb_dists = embedding_distance(track_features, id_feature_keep).T
            if self.args.with_longterm_reid or self.args.with_longterm_reid_correction:
                long_track_features = np.asarray([
                    np.vstack(list(t.features)).mean(0) if len(t.features) > 0
                    else np.zeros(feat_dim)
                    for t in self.trackers
                ], dtype=np.float64)
                if long_track_features.ndim != 2:
                    long_track_features = long_track_features.reshape(-1, feat_dim)
                assert track_features.shape == long_track_features.shape
                long_emb_dists = embedding_distance(long_track_features, id_feature_keep).T
                assert emb_dists.shape == long_emb_dists.shape
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.asso_func, self.args,emb_cost=emb_dists,
                    weights=(1.0, self.args.EG_weight_high_score), thresh=self.args.high_score_matching_thresh,
                    long_emb_dists=long_emb_dists, with_longterm_reid=self.args.with_longterm_reid,
                    longterm_reid_weight=self.args.longterm_reid_weight,
                    with_longterm_reid_correction=self.args.with_longterm_reid_correction,
                    longterm_reid_correction_thresh=self.args.longterm_reid_correction_thresh,
                    dataset=self.args.dataset)
            else:
                matched, unmatched_dets, unmatched_trks = associate_4_points_with_score_with_reid(
                    dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                    k_observations, self.inertia, self.asso_func, self.args,emb_cost=emb_dists,
                    weights=(1.0, self.args.EG_weight_high_score), thresh=self.args.high_score_matching_thresh)
        elif self.args.TCM_first_step:
            matched, unmatched_dets, unmatched_trks = associate_4_points_with_score(
                dets, trks, self.iou_threshold, velocities_lt, velocities_rt, velocities_lb, velocities_rb,
                k_observations, self.inertia, self.asso_func, self.args)

        # update with id feature
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :], id_feature_keep[m[0], :])

        """
            Second round of associaton by OCR
        """
        # BYTE association
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            u_tracklets = [self.trackers[index] for index in unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                if self.args.TCM_byte_step:
                    iou_left_ori = copy.deepcopy(iou_left)
                    iou_left -= np.array(cal_score_dif_batch_two_score(dets_second, u_trks) * self.args.TCM_byte_step_weight)
                    iou_left_thre = iou_left
                if self.args.EG_weight_low_score > 0:
                    low_feat_dim = id_feature_second.shape[1] if id_feature_second.ndim == 2 else 512
                    u_track_features = np.asarray([
                        t.smooth_feat if t.smooth_feat is not None else np.zeros(low_feat_dim)
                        for t in u_tracklets
                    ], dtype=np.float64)
                    if u_track_features.ndim == 1:
                        u_track_features = u_track_features.reshape(1, -1)
                    emb_dists_low_score = embedding_distance(u_track_features, id_feature_second).T
                    matched_indices = linear_assignment(-iou_left + self.args.EG_weight_low_score * emb_dists_low_score,
                                                        )
                else:
                    matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if self.args.with_longterm_reid_correction and self.args.EG_weight_low_score > 0:
                        if iou_left_thre[m[0], m[1]] < self.iou_threshold or emb_dists_low_score[m[0], m[1]] > self.args.longterm_reid_correction_thresh_low:
                            print("correction 2nd:", emb_dists_low_score[m[0], m[1]])
                            continue
                    else:
                        if iou_left_thre[m[0], m[1]] < self.iou_threshold:
                            continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :], id_feature_second[det_ind, :], update_feature=False)     # [hgx0523] do not update with id feature
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            # left_id_feature = id_feature_keep[unmatched_dets]       # update id feature, if needed
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)

            if iou_left.max() > self.iou_threshold:
                """
                    NOTE: by using a lower threshold, e.g., self.iou_threshold - 0.1, you may
                    get a higher performance especially on MOT17/MOT20 datasets. But we keep it
                    uniform here for simplicity
                """
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :], id_feature_keep[det_ind, :], update_feature=False)
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None, None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            if dets[i, 4] < self.new_track_thresh:
                continue  # 置信度低于新轨迹阈值，不生成新 ID
            # 保持旧 ID 优先：与任一现有轨迹（含本帧刚新建的）显著重叠的检测不另起新轨迹，
            # 避免边界重复检测/抖动生成抢号的重复轨迹。
            if self.new_track_overlap_thresh < 1.0 and len(self.trackers) > 0:
                exist_boxes = np.array([t.get_state()[0][:4] for t in self.trackers], dtype=np.float32)
                if iou_batch(dets[i, :4][None, :], exist_boxes).max() > self.new_track_overlap_thresh:
                    continue
            trk = KalmanBoxTracker(dets[i, :], id_feature_keep[i, :], delta_t=self.delta_t, args=self.args)
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]
            else:
                """
                    this is optional to use the recent observation or the kalman filter prediction,
                    we didn't notice significant difference here
                """
                d = trk.last_observation[:4]
            # 确认 vs 显示解耦：
            #   · 未确认轨迹(id<0)需连续命中达到 min_hits 才"确认"并首次分配 ID（严苛新增）；
            #   · 已确认轨迹(id>=0)只要本帧有匹配就显示，不因遮挡后 hit_streak 被 predict 重置
            #     而被再次抑制——否则每次遮挡后已确认目标会凭空消失 min_hits 帧（仍保留 ID）。
            is_confirmed = (trk.id >= 0) or (trk.hit_streak >= self.min_hits)
            if (trk.time_since_update < 1) and is_confirmed:
                if trk.id < 0:
                    trk.id = KalmanBoxTracker.count
                    KalmanBoxTracker.count += 1
                # +1 as MOT benchmark requires positive
                ret.append(np.concatenate((d, [trk.id+1])).reshape(1, -1))
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
            # 未确认轨迹（尚未分配 ID）一旦发生漏检立即删除：强制连续 min_hits 帧命中才确认，
            # 杜绝闪烁式误检（检到-丢-检到）在多帧后侥幸累积确认并占用新 ID。
            elif trk.id < 0 and trk.time_since_update >= 1:
                self.trackers.pop(i)
        if(len(ret) > 0):
            return np.concatenate(ret)
        return np.empty((0, 5))

    def update_public(self, dets, cates, scores):
        self.frame_count += 1

        det_scores = np.ones((dets.shape[0], 1))
        dets = np.concatenate((dets, det_scores), axis=1)

        remain_inds = scores > self.det_thresh
        
        cates = cates[remain_inds]
        dets = dets[remain_inds]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            cat = self.trackers[t].cate
            trk[:] = [pos[0], pos[1], pos[2], pos[3], cat]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array([trk.velocity if trk.velocity is not None else np.array((0,0)) for trk in self.trackers])
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        matched, unmatched_dets, unmatched_trks = associate_kitti\
              (dets, trks, cates, self.iou_threshold, velocities, k_observations, self.inertia)
          
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])
          
        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            """
                The re-association stage by OCR.
                NOTE: at this stage, adding other strategy might be able to continue improve
                the performance, such as BYTE association by ByteTrack. 
            """
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            left_dets_c = left_dets.copy()
            left_trks_c = left_trks.copy()

            iou_left = self.asso_func(left_dets_c, left_trks_c)
            iou_left = np.array(iou_left)
            det_cates_left = cates[unmatched_dets]
            trk_cates_left = trks[unmatched_trks][:,4]
            num_dets = unmatched_dets.shape[0]
            num_trks = unmatched_trks.shape[0]
            cate_matrix = np.zeros((num_dets, num_trks))
            for i in range(num_dets):
                for j in range(num_trks):
                    if det_cates_left[i] != trk_cates_left[j]:
                            """
                                For some datasets, such as KITTI, there are different categories,
                                we have to avoid associate them together.
                            """
                            cate_matrix[i][j] = -1e6
            iou_left = iou_left + cate_matrix
            if iou_left.max() > self.iou_threshold - 0.1:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold - 0.1:
                          continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind) 
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            trk.cate = cates[i]
            self.trackers.append(trk)
        i = len(self.trackers)

        for trk in reversed(self.trackers):
            if trk.last_observation.sum() > 0:
                d = trk.last_observation[:4]
            else:
                d = trk.get_state()[0]
            if (trk.time_since_update < 1):
                if (self.frame_count <= self.min_hits) or (trk.hit_streak >= self.min_hits):
                    # id+1 as MOT benchmark requires positive
                    ret.append(np.concatenate((d, [trk.id+1], [trk.cate], [0])).reshape(1,-1)) 
                if trk.hit_streak == self.min_hits:
                    # Head Padding (HP): recover the lost steps during initializing the track
                    for prev_i in range(self.min_hits - 1):
                        prev_observation = trk.history_observations[-(prev_i+2)]
                        ret.append((np.concatenate((prev_observation[:4], [trk.id+1], [trk.cate], 
                            [-(prev_i+1)]))).reshape(1,-1))
            i -= 1 
            if (trk.time_since_update > self.max_age):
                  self.trackers.pop(i)
        
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0, 7))


