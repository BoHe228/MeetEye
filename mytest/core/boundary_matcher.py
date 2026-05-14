"""
边界穿越ID匹配器

解决目标在监控画面边界穿越时身份ID不连续的问题。
基于外观特征匹配的轻量级判定策略：
1. 事件检测：检测目标在边界区域消失和出现
2. 特征比对：计算外观特征相似度
3. 身份判定：根据阈值判断是否为同一目标
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from dataclasses import dataclass
from enum import Enum

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class BoundarySide(Enum):
    """边界位置枚举"""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass
class DisappearedTarget:
    """消失的目标记录"""
    track_id: int
    feature: np.ndarray
    bbox: List[float]  # [x1, y1, x2, y2]
    disappear_frame: int
    boundary_side: BoundarySide
    smooth_feat: Optional[np.ndarray] = None
    last_known_velocity: Optional[Tuple[float, float]] = None  # (vx, vy)


@dataclass
class AppearedTarget:
    """新出现的目标记录"""
    feature: np.ndarray
    bbox: List[float]  # [x1, y1, x2, y2]
    appear_frame: int
    boundary_side: BoundarySide


class BoundaryIDMatcher:
    """
    边界穿越ID匹配器

    核心功能：
    1. 检测目标是否在边界区域消失或出现
    2. 维护消失目标的特征缓存
    3. 当新目标在边界出现时，与消失目标进行特征匹配
    4. 若匹配成功，复用原ID保持连续性
    """

    def __init__(self,
                 frame_width: int = 1920,
                 frame_height: int = 1080,
                 boundary_margin: float = 0.1,  # 边界区域占画面宽度/高度的比例
                 time_window: int = 30,  # 时间窗口（帧数）
                 similarity_threshold: float = 0.6,  # 特征相似度阈值
                 min_bbox_overlap: float = 0.3,  # 边界检测框最小重叠比例
                 enable_velocity_check: bool = True,  # 是否启用速度一致性检查
                 max_velocity_deviation: float = 2.0,  # 速度最大允许偏差倍数
                 debug: bool = False,  # 调试模式
                 enable_top_boundary: bool = False,  # 是否启用顶部边界
                 enable_bottom_boundary: bool = True,  # 是否启用底部边界
                 enable_left_boundary: bool = True,  # 是否启用左侧边界
                 enable_right_boundary: bool = True):  # 是否启用右侧边界
        """
        初始化边界ID匹配器

        Args:
            frame_width: 画面宽度
            frame_height: 画面高度
            boundary_margin: 边界区域占比（0-0.5）
            time_window: 消失目标缓存时间窗口（帧数）
            similarity_threshold: 余弦相似度阈值，超过此值认为是同一目标
            min_bbox_overlap: 检测框与边界的最小重叠比例
            enable_velocity_check: 是否启用运动一致性检查
            max_velocity_deviation: 速度最大允许偏差倍数
            debug: 是否启用调试输出
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.boundary_margin = max(0.01, min(0.4, boundary_margin))
        self.time_window = time_window
        self.similarity_threshold = similarity_threshold
        self.min_bbox_overlap = min_bbox_overlap
        self.enable_velocity_check = enable_velocity_check
        self.max_velocity_deviation = max_velocity_deviation
        self.debug = debug

        # 边界启用配置
        self.enable_top_boundary = enable_top_boundary
        self.enable_bottom_boundary = enable_bottom_boundary
        self.enable_left_boundary = enable_left_boundary
        self.enable_right_boundary = enable_right_boundary

        # 计算边界区域坐标
        self._update_boundaries()

        # 消失目标缓存（按边界侧分类存储，只包含启用的边界）
        self.disappeared_by_side: Dict[BoundarySide, deque] = {}
        if self.enable_left_boundary:
            self.disappeared_by_side[BoundarySide.LEFT] = deque(maxlen=50)
        if self.enable_right_boundary:
            self.disappeared_by_side[BoundarySide.RIGHT] = deque(maxlen=50)
        if self.enable_top_boundary:
            self.disappeared_by_side[BoundarySide.TOP] = deque(maxlen=50)
        if self.enable_bottom_boundary:
            self.disappeared_by_side[BoundarySide.BOTTOM] = deque(maxlen=50)

        # 统计信息
        self.stats = {
            'disappeared_count': 0,
            'appeared_count': 0,
            'matched_count': 0,
            'failed_matches': 0
        }

    def _update_boundaries(self):
        """更新边界区域坐标"""
        margin_w = int(self.frame_width * self.boundary_margin)
        margin_h = int(self.frame_height * self.boundary_margin)

        self.boundaries = {}
        if self.enable_left_boundary:
            self.boundaries[BoundarySide.LEFT] = (0, 0, margin_w, self.frame_height)
        if self.enable_right_boundary:
            self.boundaries[BoundarySide.RIGHT] = (self.frame_width - margin_w, 0, self.frame_width, self.frame_height)
        if self.enable_top_boundary:
            self.boundaries[BoundarySide.TOP] = (0, 0, self.frame_width, margin_h)
        if self.enable_bottom_boundary:
            self.boundaries[BoundarySide.BOTTOM] = (0, self.frame_height - margin_h, self.frame_width, self.frame_height)

    def set_frame_size(self, width: int, height: int):
        """设置画面尺寸"""
        self.frame_width = width
        self.frame_height = height
        self._update_boundaries()

    def _check_boundary_overlap(self, bbox: List[float], boundary: Tuple[int, int, int, int]) -> float:
        """
        计算检测框与边界区域的重叠比例

        Args:
            bbox: 检测框 [x1, y1, x2, y2]
            boundary: 边界区域 [x1, y1, x2, y2]

        Returns:
            重叠比例（0-1）
        """
        x1, y1, x2, y2 = bbox
        bx1, by1, bx2, by2 = boundary

        # 计算交集
        ix1 = max(x1, bx1)
        iy1 = max(y1, by1)
        ix2 = min(x2, bx2)
        iy2 = min(y2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        bbox_area = (x2 - x1) * (y2 - y1)

        return intersection / (bbox_area + 1e-6)

    def get_boundary_side(self, bbox: List[float]) -> Optional[BoundarySide]:
        """
        判断检测框位于哪个边界区域

        Args:
            bbox: 检测框 [x1, y1, x2, y2]

        Returns:
            边界侧枚举，如果不在任何边界区域返回None
        """
        max_overlap = 0.0
        max_side = None

        for side, boundary in self.boundaries.items():
            overlap = self._check_boundary_overlap(bbox, boundary)
            if overlap > max(self.min_bbox_overlap, max_overlap):
                max_overlap = overlap
                max_side = side

        if self.debug and max_side is not None:
            print(f"  [边界检测] 框={bbox} 在 {max_side.value} 边界, 重叠={max_overlap:.2f}")

        return max_side

    def _get_opposite_side(self, side: BoundarySide) -> BoundarySide:
        """获取对侧边界"""
        opposite_map = {
            BoundarySide.LEFT: BoundarySide.RIGHT,
            BoundarySide.RIGHT: BoundarySide.LEFT,
            BoundarySide.TOP: BoundarySide.BOTTOM,
            BoundarySide.BOTTOM: BoundarySide.TOP
        }
        return opposite_map[side]

    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """计算余弦相似度"""
        if feat1 is None or feat2 is None:
            return 0.0
        feat1 = feat1.flatten().astype(np.float32)
        feat2 = feat2.flatten().astype(np.float32)
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def add_disappeared_target(self,
                                track_id: int,
                                bbox: List[float],
                                feature: np.ndarray,
                                frame_id: int,
                                smooth_feat: Optional[np.ndarray] = None,
                                prev_bbox: Optional[List[float]] = None):
        """
        记录消失在边界的目标

        Args:
            track_id: 轨迹ID
            bbox: 最后出现的检测框
            feature: 外观特征
            frame_id: 消失的帧号
            smooth_feat: 平滑后的特征（优先使用）
            prev_bbox: 前一帧的检测框（用于计算速度）
        """
        # 检查是否在边界区域
        side = self.get_boundary_side(bbox)
        if side is None:
            return

        # 计算速度（如果有前一帧位置）
        velocity = None
        if prev_bbox is not None:
            x1_prev, y1_prev, x2_prev, y2_prev = prev_bbox
            x1_curr, y1_curr, x2_curr, y2_curr = bbox
            cx_prev = (x1_prev + x2_prev) / 2
            cy_prev = (y1_prev + y2_prev) / 2
            cx_curr = (x1_curr + x2_curr) / 2
            cy_curr = (y1_curr + y2_curr) / 2
            velocity = (cx_curr - cx_prev, cy_curr - cy_prev)

        target = DisappearedTarget(
            track_id=track_id,
            feature=feature.copy() if feature is not None else None,
            bbox=list(bbox),
            disappear_frame=frame_id,
            boundary_side=side,
            smooth_feat=smooth_feat.copy() if smooth_feat is not None else None,
            last_known_velocity=velocity
        )

        # 添加到对应边界的缓存
        self.disappeared_by_side[side].append(target)
        self.stats['disappeared_count'] += 1

        if self.debug:
            print(f"  [消失目标] ID={track_id}, 边界={side.value}, 帧={frame_id}")

    def find_matching_id(self,
                         bbox: List[float],
                         feature: np.ndarray,
                         frame_id: int) -> Optional[int]:
        """
        为新出现的目标寻找匹配的消失目标ID

        Args:
            bbox: 新目标的检测框
            feature: 新目标的外观特征
            frame_id: 当前帧号

        Returns:
            匹配的track_id，如果没有匹配返回None
        """
        # 检查新目标是否在边界区域
        appear_side = self.get_boundary_side(bbox)
        if appear_side is None:
            return None

        self.stats['appeared_count'] += 1

        if self.debug:
            print(f"  [新目标] 边界={appear_side.value}, 框={bbox}, 帧={frame_id}")

        # 计算新目标的位置（用于速度检查）
        x1, y1, x2, y2 = bbox
        cx_new = (x1 + x2) / 2
        cy_new = (y1 + y2) / 2

        # 搜索策略：
        # 1. 优先搜索同侧边界 - 处理检测抖动导致的"消失"后立即重现（用速度检查）
        # 2. 然后搜索对侧边界 - 处理真正的边界穿越（360度全景）（不用速度检查）
        # 注意：只搜索启用的边界
        candidate_sides = [appear_side, self._get_opposite_side(appear_side)]
        search_sides = [side for side in candidate_sides if side in self.disappeared_by_side]

        best_match_id = None
        best_similarity = 0.0

        if self.debug:
            cache_counts = {side.value: len(q) for side, q in self.disappeared_by_side.items()}
            print(f"  [缓存状态] {cache_counts}")
            print(f"  [搜索顺序] {[s.value for s in search_sides]}")

        for search_side in search_sides:
            candidates = self.disappeared_by_side.get(search_side, [])

            # 判断是否是对侧边界匹配（环绕穿越）
            is_wrap_around = (search_side != appear_side)

            if self.debug and len(candidates) > 0:
                match_type = "环绕穿越" if is_wrap_around else "同侧抖动"
                print(f"  [搜索] 在{search_side.value}边界找到{len(candidates)}个候选目标 ({match_type})")

            # 倒序遍历（优先检查最近消失的）
            for target in reversed(candidates):
                # 检查时间窗口
                if frame_id - target.disappear_frame > self.time_window:
                    if self.debug:
                        print(f"    [跳过] ID={target.track_id}, 超出时间窗口")
                    continue

                # 使用平滑特征（如果有）
                target_feat = target.smooth_feat if target.smooth_feat is not None else target.feature
                if target_feat is None or feature is None:
                    if self.debug:
                        print(f"    [跳过] ID={target.track_id}, 特征为空")
                    continue

                # 计算特征相似度
                similarity = self._cosine_similarity(target_feat, feature)

                # === 智能速度检查：只在同侧匹配时使用
                velocity_ok = True
                if not is_wrap_around and self.enable_velocity_check and target.last_known_velocity is not None:
                    # 同侧匹配：使用速度一致性检查
                    # 计算消失位置
                    tx1, ty1, tx2, ty2 = target.bbox
                    cx_last = (tx1 + tx2) / 2
                    cy_last = (ty1 + ty2) / 2

                    # 预测的当前位置（基于最后已知速度）
                    vx, vy = target.last_known_velocity
                    dt = frame_id - target.disappear_frame
                    cx_pred = cx_last + vx * dt
                    cy_pred = cy_last + vy * dt

                    # 实际位置和预测位置的距离
                    dist = np.sqrt((cx_new - cx_pred)**2 + (cy_new - cy_pred)**2)

                    max_dist = max(tx2 - tx1, ty2 - ty1) * self.max_velocity_deviation

                    if self.debug:
                        print(f"    [速度检查] ID={target.track_id}, 预测距离={dist:.1f}, 允许距离={max_dist:.1f}")

                    if dist > max_dist:
                        velocity_ok = False
                        if self.debug:
                            print(f"    [跳过] ID={target.track_id}, 速度不一致")

                if self.debug:
                    print(f"    [比较] ID={target.track_id}, 相似度={similarity:.3f}, 速度检查={velocity_ok}")

                # 记录最佳匹配
                if velocity_ok and similarity > max(self.similarity_threshold, best_similarity):
                    best_similarity = similarity
                    best_match_id = target.track_id
                    if self.debug:
                        match_type = "环绕穿越" if is_wrap_around else "同侧匹配"
                        print(f"    [候选] ID={target.track_id} 成为最佳匹配 ({match_type})")

        if best_match_id is not None:
            self.stats['matched_count'] += 1
            # === 注意：不立即从缓存中移除已匹配的目标 ===
            # 原因：目标可能在边界附近反复出现（检测抖动）
            # 让时间窗口自动清理过期目标
            # self._remove_matched_target(best_match_id)
            if self.debug:
                print(f"  [匹配成功] 新目标匹配到ID={best_match_id}, 相似度={best_similarity:.2f}")
        else:
            self.stats['failed_matches'] += 1
            if self.debug:
                print(f"  [匹配失败] 未找到匹配, 最佳相似度={best_similarity:.2f}")

        return best_match_id

    def _remove_matched_target(self, track_id: int):
        """从缓存中移除已匹配的目标"""
        for side in self.disappeared_by_side:
            self.disappeared_by_side[side] = deque(
                [t for t in self.disappeared_by_side[side] if t.track_id != track_id],
                maxlen=50
            )

    def cleanup_old_targets(self, current_frame: int):
        """
        清理过期的消失目标

        Args:
            current_frame: 当前帧号
        """
        for side in self.disappeared_by_side:
            self.disappeared_by_side[side] = deque(
                [t for t in self.disappeared_by_side[side]
                 if current_frame - t.disappear_frame <= self.time_window],
                maxlen=50
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats['cache_size'] = sum(len(q) for q in self.disappeared_by_side.values())
        stats['cache_by_side'] = {
            side.value: len(q) for side, q in self.disappeared_by_side.items()
        }
        return stats

    def reset(self):
        """重置匹配器"""
        for side in self.disappeared_by_side:
            self.disappeared_by_side[side].clear()
        self.stats = {
            'disappeared_count': 0,
            'appeared_count': 0,
            'matched_count': 0,
            'failed_matches': 0
        }

    def draw_boundary_regions(self, image: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255),
                              thickness: int = 2, alpha: float = 0.3) -> np.ndarray:
        """
        在图像上绘制边界区域（可视化调试用）

        Args:
            image: 输入图像
            color: 边界颜色 (BGR格式)
            thickness: 线条粗细
            alpha: 填充透明度 (0-1)

        Returns:
            绘制了边界区域的图像
        """
        if not CV2_AVAILABLE:
            return image

        result = image.copy()

        # 绘制半透明填充
        overlay = result.copy()
        for side, boundary in self.boundaries.items():
            x1, y1, x2, y2 = boundary
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, -1)

        # 混合图像
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)

        # 绘制边界线条和标签
        for side, boundary in self.boundaries.items():
            x1, y1, x2, y2 = boundary
            cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

            # 添加标签
            label = side.value.upper()
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.putText(result, label, (int(cx - 20), int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    def draw_matched_targets(self, image: np.ndarray,
                             disappeared_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        绘制匹配的目标（可视化调试用）

        Args:
            image: 输入图像
            disappeared_color: 消失目标颜色

        Returns:
            绘制了目标的图像
        """
        if not CV2_AVAILABLE:
            return image

        result = image.copy()

        # 绘制消失目标
        for side in self.disappeared_by_side:
            for target in self.disappeared_by_side[side]:
                x1, y1, x2, y2 = target.bbox
                cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), disappeared_color, 2)
                cv2.putText(result, f"ID:{target.track_id}", (int(x1), int(y1 - 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, disappeared_color, 2)

        return result


class BoundaryCrossingTracker:
    """
    边界穿越跟踪增强器

    与BoT-SORTTracker集成的高层接口，提供：
    1. 自动检测消失和出现的目标
    2. 自动ID匹配和复用
    3. 透明集成到现有跟踪流程
    """

    def __init__(self,
                 frame_width: int = 1920,
                 frame_height: int = 1080,
                 boundary_margin: float = 0.1,
                 time_window: int = 30,
                 similarity_threshold: float = 0.6,
                 debug: bool = True,
                 enable_top_boundary: bool = False,
                 enable_bottom_boundary: bool = True,
                 enable_left_boundary: bool = True,
                 enable_right_boundary: bool = True):
        """
        初始化边界穿越跟踪增强器

        Args:
            frame_width: 画面宽度
            frame_height: 画面高度
            boundary_margin: 边界区域占比
            time_window: 时间窗口（帧数）
            similarity_threshold: 特征相似度阈值
            debug: 是否启用调试输出
            enable_top_boundary: 是否启用顶部边界
            enable_bottom_boundary: 是否启用底部边界
            enable_left_boundary: 是否启用左侧边界
            enable_right_boundary: 是否启用右侧边界
        """
        self.matcher = BoundaryIDMatcher(
            frame_width=frame_width,
            frame_height=frame_height,
            boundary_margin=boundary_margin,
            time_window=time_window,
            similarity_threshold=similarity_threshold,
            debug=debug,
            enable_top_boundary=enable_top_boundary,
            enable_bottom_boundary=enable_bottom_boundary,
            enable_left_boundary=enable_left_boundary,
            enable_right_boundary=enable_right_boundary
        )

        # 上一帧的轨迹
        self.prev_tracks: Dict[int, Dict] = {}

        # ID映射表：新ID -> 原始ID
        self.id_remap: Dict[int, int] = {}

        # 下一帧要强制复用的ID
        self.pending_remaps: Dict[int, int] = {}  # temp_id -> original_id

    def set_frame_size(self, width: int, height: int):
        """设置画面尺寸"""
        self.matcher.set_frame_size(width, height)

    def pre_process(self, frame_id: int):
        """
        跟踪前预处理

        Args:
            frame_id: 当前帧号
        """
        self.matcher.cleanup_old_targets(frame_id)

    def process_lost_track(self,
                           track_id: int,
                           bbox: List[float],
                           feature: np.ndarray,
                           frame_id: int,
                           smooth_feat: Optional[np.ndarray] = None,
                           prev_bbox: Optional[List[float]] = None):
        """
        处理丢失的轨迹

        当轨迹丢失时调用此方法，如果在边界区域则记录下来用于后续匹配

        Args:
            track_id: 轨迹ID
            bbox: 最后检测框
            feature: 外观特征
            frame_id: 当前帧号
            smooth_feat: 平滑特征
            prev_bbox: 前一帧检测框
        """
        # 检查是否已在ID映射表中（如果是，使用原始ID）
        original_id = self.id_remap.get(track_id, track_id)

        self.matcher.add_disappeared_target(
            track_id=original_id,
            bbox=bbox,
            feature=feature,
            frame_id=frame_id,
            smooth_feat=smooth_feat,
            prev_bbox=prev_bbox
        )

    def check_new_track(self,
                        bbox: List[float],
                        feature: np.ndarray,
                        frame_id: int,
                        temp_id: int) -> Optional[int]:
        """
        检查新轨迹是否可以匹配到消失的目标

        当初始化新轨迹前调用此方法

        Args:
            bbox: 新目标检测框
            feature: 新目标特征
            frame_id: 当前帧号
            temp_id: 临时分配的新ID（用于后续重映射）

        Returns:
            匹配到的原始ID，如果没有匹配返回None
        """
        matched_id = self.matcher.find_matching_id(bbox, feature, frame_id)

        if matched_id is not None:
            # 记录ID映射
            self.pending_remaps[temp_id] = matched_id
            self.id_remap[temp_id] = matched_id

        return matched_id

    def remap_track_id(self, new_id: int) -> int:
        """
        获取重映射后的ID

        Args:
            new_id: 跟踪器分配的新ID

        Returns:
            重映射后的原始ID
        """
        return self.id_remap.get(new_id, new_id)

    def post_process(self, tracks: List[Dict]) -> List[Dict]:
        """
        跟踪后处理，应用ID重映射

        Args:
            tracks: 跟踪器输出的轨迹列表

        Returns:
            ID重映射后的轨迹列表
        """
        # 处理待定的重映射
        for track in tracks:
            track_id = track.get('track_id')
            if track_id in self.pending_remaps:
                track['track_id'] = self.pending_remaps[track_id]
                track['_boundary_matched'] = True

        # 清理已处理的待定映射
        self.pending_remaps.clear()

        # 更新当前轨迹记录
        self.prev_tracks = {
            t['track_id']: {
                'bbox': t['bbox'],
                'feature': t.get('feature')
            }
            for t in tracks
        }

        return tracks

    def get_remap_count(self) -> int:
        """获取已重映射的ID数量"""
        return len(self.id_remap)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.matcher.get_stats()
        stats['remap_count'] = len(self.id_remap)
        return stats

    def reset(self):
        """重置状态"""
        self.matcher.reset()
        self.prev_tracks.clear()
        self.id_remap.clear()
        self.pending_remaps.clear()

    def draw_boundary_regions(self, image: np.ndarray) -> np.ndarray:
        """在图像上绘制边界区域（用于调试）"""
        return self.matcher.draw_boundary_regions(image)
