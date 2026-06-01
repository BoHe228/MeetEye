import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
from ultralytics.engine.results import Results

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import cosine_similarity, box_iou, box_iou_batch


class PanoramaSlicer:
    """
    全景图切片处理器 - 支持基于ReID特征的智能去重
    核心改进：
    1. 在切片检测阶段就提取ReID特征
    2. 合并去重时：基于ReID特征相似度判断是否为同一目标
    3. 避免将邻近的不同目标误判为重复检测
    """

    def __init__(self, overlap_ratio: float = 0.2, iou_threshold: float = 0.3,
                 nms_iou_thresh: float = 0.6,
                 confidence_threshold: float = 0.5,
                 reid_similarity_threshold: float = 0.7,
                 wrap_reid_threshold: float = 0.5,
                 max_width_ratio: float = 0.6,
                 verbose: bool = False):
        """
        初始化切片器

        Args:
            overlap_ratio: 切片之间的重叠比例 (0-1)
            iou_threshold: 跨切片去重（Step 3）的 IoU 阈值，需较低以捕获重叠区域重复检测
            nms_iou_thresh: 最终 NMS（Step 4）的 IoU 阈值，须高于 iou_threshold，
                            避免将物理上接近或被包裹的不同目标误判为重复检测。
            confidence_threshold: 置信度阈值
            reid_similarity_threshold: 相邻切片对（slice0/1、slice1/2）去重的 ReID 阈值，
                                       这两对有 IoU 前置门控，可以使用较严格的值
            wrap_reid_threshold: 环绕切片对（slice0 & slice2）专用阈值，须低于
                                 reid_similarity_threshold。环绕对没有位置门控，
                                 且两侧裁图上下文不同会导致特征略有偏差，需要宽松判断。
            max_width_ratio: 检测框最大允许宽度占全景图宽度的比例（过滤横跨边界的无效框）
            verbose: 是否打印去重/过滤调试日志（默认关闭）
        """
        self.overlap_ratio = overlap_ratio
        self.iou_threshold = iou_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.confidence_threshold = confidence_threshold
        self.reid_similarity_threshold = reid_similarity_threshold
        self.wrap_reid_threshold = wrap_reid_threshold
        self.max_width_ratio = max_width_ratio
        self.verbose = verbose

    def slice_panorama(self, panorama: np.ndarray, num_slices: int = 3) -> Tuple[List[np.ndarray], List[dict]]:
        """
        将全景图切分为多个有重叠的切片
        """
        slices = []
        slice_infos = []

        height, width = panorama.shape[:2]
        slice_width = width // num_slices
        overlap_width = int(slice_width * self.overlap_ratio)

        for i in range(num_slices):
            start_x = i * slice_width - overlap_width
            end_x = (i + 1) * slice_width + overlap_width

            if i == 0:
                if start_x < 0:
                    left_part = panorama[:, start_x:]
                    right_part = panorama[:, :end_x]
                    slice_img = np.concatenate([left_part, right_part], axis=1)
                    actual_start_x = width + start_x
                    wrap_around = True
                else:
                    slice_img = panorama[:, start_x:end_x]
                    actual_start_x = start_x
                    wrap_around = False
            elif i == num_slices - 1:
                if end_x > width:
                    right_part = panorama[:, start_x:width]
                    left_part = panorama[:, :end_x - width]
                    slice_img = np.concatenate([right_part, left_part], axis=1)
                    actual_start_x = start_x
                    wrap_around = True
                else:
                    slice_img = panorama[:, start_x:end_x]
                    actual_start_x = start_x
                    wrap_around = False
            else:
                slice_img = panorama[:, start_x:end_x]
                actual_start_x = start_x
                wrap_around = False

            slices.append(slice_img)
            slice_infos.append({
                'slice_idx': i,
                'start_x': start_x,
                'actual_start_x': actual_start_x,
                'end_x': end_x,
                'slice_width': slice_img.shape[1],
                'original_width': width,
                'wrap_around': wrap_around
            })

        return slices, slice_infos

    def extract_detections_from_yolo_results(self, yolo_results, slice_img=None,
                                              feature_extractor=None, slice_info=None):
        """
        从YOLO结果中提取检测信息 - 支持ReID特征提取

        Args:
            yolo_results: YOLO检测结果
            slice_img: 切片图像（用于特征提取）
            feature_extractor: 特征提取器（如果提供，则提取ReID特征）
            slice_info: 切片信息（用于坐标转换，当前未使用但保留接口）
        """
        """
        从YOLO结果中提取检测信息 - 支持ReID特征提取

        Args:
            yolo_results: YOLO检测结果
            slice_img: 切片图像（用于特征提取）
            feature_extractor: 特征提取器（如果提供，则提取ReID特征）
            slice_info: 切片信息（用于坐标转换）
        """
        detections = []
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                boxes_data = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                keypoints_data = []
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()

                for i, (box, conf, cls_id) in enumerate(zip(boxes_data, confidences, class_ids)):
                    det = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': result.names[int(cls_id)],
                    }

                    if i < len(keypoints_data):
                        det['keypoints'] = keypoints_data[i].tolist()

                    # === 关键：如果提供了特征提取器，在这里就提取ReID特征 ===
                    if feature_extractor is not None and slice_img is not None:
                        x1, y1, x2, y2 = box
                        x1 = max(0, int(x1))
                        y1 = max(0, int(y1))
                        x2 = min(slice_img.shape[1], int(x2))
                        y2 = min(slice_img.shape[0], int(y2))

                        if x2 > x1 and y2 > y1:
                            crop_img = slice_img[y1:y2, x1:x2]
                            try:
                                features = feature_extractor.extract_features_from_image_array(crop_img)
                                if features is not None and len(features) > 0:
                                    det['feature'] = features[0].numpy()
                            except Exception as e:
                                print(f"切片内特征提取失败: {e}")
                                det['feature'] = None
                        else:
                            det['feature'] = None

                    detections.append(det)

        return detections

    def merge_detections(
        self,
        all_yolo_results: List[List[Results]],
        slice_infos: List[dict],
        slice_images: List[np.ndarray] = None,
        slice_tensors=None,          # GPU [3,H,W] float 0-1 RGB 张量列表（优先级高于 slice_images）
        feature_extractor=None
    ) -> List[Dict[str, Any]]:
        """
        合并所有切片的检测结果 - 基于ReID特征的智能去重

        核心改进：
        1. 先在每个切片上提取ReID特征
        2. 对重叠区域的检测，通过ReID特征相似度判断是否为同一目标
        3. 只有特征相似度高且空间重叠高时，才认为是重复检测
        """
        if not all_yolo_results:
            return []

        # === 第一步：从每个切片提取检测框/关键点（不逐一提取特征）===
        all_detections = []
        for slice_idx, (yolo_results, info) in enumerate(zip(all_yolo_results, slice_infos)):
            detections = self.extract_detections_from_yolo_results(
                yolo_results,
                slice_img=None,
                feature_extractor=None,  # 不在此处单次提取，见下方批量提取
                slice_info=info
            )
            all_detections.append(detections)

        # === 第一步(b)：批量 OSNet 特征提取（所有切片 crop 一次 forward pass）===
        # 优先使用 GPU 张量路径（省去 numpy→PIL→transform CPU 开销，约 5ms）
        if feature_extractor is not None:
            if slice_tensors is not None and hasattr(feature_extractor, 'extract_batch_gpu_crops'):
                # GPU 路径：直接从 GPU slice tensor 裁切，全程不落 CPU
                gpu_crops, crop_refs = [], []
                for slice_idx, detections in enumerate(all_detections):
                    st = slice_tensors[slice_idx] if slice_idx < len(slice_tensors) else None
                    if st is None:
                        continue
                    sh, sw = st.shape[1], st.shape[2]
                    for det_idx, det in enumerate(detections):
                        x1, y1, x2, y2 = det['bbox']
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(sw, int(x2)), min(sh, int(y2))
                        if x2 > x1 and y2 > y1:
                            gpu_crops.append(st[:, y1:y2, x1:x2])
                            crop_refs.append((slice_idx, det_idx))
                if gpu_crops:
                    feats = feature_extractor.extract_batch_gpu_crops(gpu_crops)
                    for (si, di), feat in zip(crop_refs, feats):
                        all_detections[si][di]['feature'] = feat.numpy()
            elif slice_images is not None:
                # CPU 回退路径：numpy→PIL→transform→GPU
                crops, crop_refs = [], []
                for slice_idx, detections in enumerate(all_detections):
                    slice_img = slice_images[slice_idx] if slice_idx < len(slice_images) else None
                    if slice_img is None:
                        continue
                    for det_idx, det in enumerate(detections):
                        x1, y1, x2, y2 = det['bbox']
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(slice_img.shape[1], int(x2)), min(slice_img.shape[0], int(y2))
                        if x2 > x1 and y2 > y1:
                            crops.append(slice_img[y1:y2, x1:x2])
                            crop_refs.append((slice_idx, det_idx))
                if crops:
                    feats = feature_extractor.extract_batch_arrays(crops)
                    for (si, di), feat in zip(crop_refs, feats):
                        all_detections[si][di]['feature'] = feat.numpy()

        # === 第二步：转换所有检测框到原始全景图坐标 ===
        all_boxes = []
        all_scores = []
        all_labels = []
        all_areas = []
        all_keypoints = []
        all_features = []  # 新增：ReID特征列表
        all_detection_infos = []

        for slice_idx, (detections, info) in enumerate(zip(all_detections, slice_infos)):
            for det in detections:
                original_bbox = self._convert_to_original_coords(det['bbox'], info)

                x1, y1, x2, y2 = original_bbox
                area = (x2 - x1) * (y2 - y1)

                original_keypoints = []
                if 'keypoints' in det:
                    original_keypoints = self._convert_keypoints_to_original_coords(det['keypoints'], info)

                all_boxes.append(original_bbox)
                all_scores.append(det['confidence'])
                all_labels.append(det['class_id'])
                all_areas.append(area)
                all_keypoints.append(original_keypoints)
                all_features.append(det.get('feature', None))  # 保存ReID特征
                all_detection_infos.append({
                    'detection': det,
                    'slice_idx': slice_idx
                })

        if not all_boxes:
            return []

        # === 第三步：基于ReID特征的智能去重 ===
        suppressed = np.zeros(len(all_boxes), dtype=bool)

        # 获取全景图宽度（用于处理环绕边界）
        panorama_width = slice_infos[0]['original_width']

        # 首先计算每个检测框是否位于切片边界重叠区域（用于相邻切片对）
        in_boundary_overlap = self._check_in_boundary_overlap(
            all_boxes,
            all_detection_infos,
            slice_infos,
            num_slices=3
        )

        for i in range(len(all_boxes)):
            if suppressed[i]:
                continue

            for j in range(i + 1, len(all_boxes)):
                if suppressed[j]:
                    continue

                # 来自不同切片才可能是重复检测
                if all_detection_infos[i]['slice_idx'] == all_detection_infos[j]['slice_idx']:
                    continue

                # 检查是否是环绕边界对（slice0 & slice2）
                slice_i = all_detection_infos[i]['slice_idx']
                slice_j = all_detection_infos[j]['slice_idx']
                is_wrap_around_pair = (slice_i == 0 and slice_j == 2) or (slice_i == 2 and slice_j == 0)

                # 检查是否是相邻边界对（slice0&slice1 或 slice1&slice2）
                is_adjacent_pair = (abs(slice_i - slice_j) == 1)

                # 环绕边界对：只要特征相似就去重，不限制位置
                # 相邻边界对：需要在边界重叠区域才去重
                if is_wrap_around_pair:
                    # 环绕边界对（slice0 & slice2）：无位置门控，用专用宽松阈值
                    feat1 = all_features[i]
                    feat2 = all_features[j]

                    is_same_target = False

                    if feat1 is not None and feat2 is not None:
                        similarity = cosine_similarity(feat1, feat2)
                        if similarity >= self.wrap_reid_threshold:
                            is_same_target = True
                            if self.verbose:
                                print(f"[环绕去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：特征相似度={similarity:.3f} → 同一目标，去重")
                    else:
                        # 无特征时：用全景坐标下的环绕近邻距离代替 IoU
                        # 环绕对的框经过坐标转换后可能一个在 x≈0、一个在 x≈W，
                        # 正常 IoU 永远为 0，改用全景最短距离判断是否同一区域
                        b1 = all_boxes[i]
                        b2 = all_boxes[j]
                        cx1 = (b1[0] + b1[2]) / 2
                        cx2 = (b2[0] + b2[2]) / 2
                        cy1 = (b1[1] + b1[3]) / 2
                        cy2 = (b2[1] + b2[3]) / 2
                        wrap_dx = min(abs(cx1 - cx2), panorama_width - abs(cx1 - cx2))
                        box_w = max((b1[2] - b1[0] + b2[2] - b2[0]) / 2, 1)
                        if wrap_dx < box_w * 1.5 and abs(cy1 - cy2) < box_w:
                            is_same_target = True
                            if self.verbose:
                                print(f"[环绕去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：无特征，环绕中心距={wrap_dx:.1f} → 同一目标，去重")

                    if is_same_target:
                        if all_scores[i] >= all_scores[j]:
                            suppressed[j] = True
                        else:
                            suppressed[i] = True
                            break
                elif is_adjacent_pair:
                    # 相邻边界对：不限制必须在边界区域，只要IoU高就去重
                    # 计算正常IoU
                    iou = box_iou(all_boxes[i], all_boxes[j])

                    if iou > self.iou_threshold:
                        feat1 = all_features[i]
                        feat2 = all_features[j]

                        is_same_target = False

                        if feat1 is not None and feat2 is not None:
                            similarity = cosine_similarity(feat1, feat2)
                            if similarity >= self.reid_similarity_threshold:
                                is_same_target = True
                                if self.verbose:
                                    print(f"[相邻去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：IoU={iou:.3f}, 特征相似度={similarity:.3f} → 同一目标，去重")
                        else:
                            if self._are_keypoints_similar(all_keypoints[i], all_keypoints[j]):
                                is_same_target = True
                                if self.verbose:
                                    print(f"[相邻去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：IoU={iou:.3f}, 无特征但关键点相似 → 同一目标，去重")

                        if is_same_target:
                            if all_scores[i] >= all_scores[j]:
                                suppressed[j] = True
                            else:
                                suppressed[i] = True
                                break
                    # else:
                        # 不是同一目标（可能是邻近的不同人）：都保留
                        # print(f"[边界去重] 检测{i}和{j}（都在边界）：IoU={iou:.3f}，但不是同一目标 → 都保留")

        # === 第四步：应用面积加权NMS处理剩余检测 ===
        keep_indices = []
        boxes_array = np.array(all_boxes, dtype=np.float32)
        scores_array = np.array(all_scores, dtype=np.float32)
        areas_array = np.array(all_areas, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int32)

        unique_labels = np.unique(labels_array)

        for label in unique_labels:
            mask = (labels_array == label) & (~suppressed)
            if np.any(mask):
                label_boxes = boxes_array[mask]
                label_scores = scores_array[mask]
                label_areas = areas_array[mask]
                original_indices = np.where(mask)[0]

                if len(label_boxes) == 1:
                    keep_indices.extend(original_indices)
                else:
                    keep_indices.extend(
                        self._area_weighted_nms(label_boxes, label_scores, label_areas, original_indices)
                    )

        # === 第五步：构建合并后的检测结果 ===
        merged_detections = []
        for idx in keep_indices:
            det_info = all_detection_infos[idx]
            det = det_info['detection']

            merged_det = {
                'bbox': all_boxes[idx],
                'confidence': det['confidence'],
                'class_id': det['class_id'],
                'class_name': det['class_name'],
                'feature': all_features[idx]  # 传递ReID特征给后续跟踪
            }

            if all_keypoints[idx]:
                merged_det['keypoints'] = all_keypoints[idx]

            merged_detections.append(merged_det)

        return merged_detections

    def _are_keypoints_similar(self, kp1, kp2, threshold=0.8):
        """检查两个关键点集是否相似（备用方案，当无ReID特征时使用）"""
        if not kp1 or not kp2:
            return False

        distances = []
        for (x1, y1, conf1), (x2, y2, conf2) in zip(kp1, kp2):
            if conf1 > 0.3 and conf2 > 0.3:
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                distances.append(distance)

        if not distances:
            return False

        avg_distance = np.mean(distances)
        return avg_distance < 30

    def _convert_to_original_coords(
        self,
        bbox: List[float],
        slice_info: dict
    ) -> List[float]:
        x1, y1, x2, y2 = bbox
        original_width = slice_info['original_width']

        if not slice_info['wrap_around']:
            original_x1 = x1 + slice_info['start_x']
            original_x2 = x2 + slice_info['start_x']
        else:
            if slice_info['start_x'] < 0:
                right_width = -slice_info['start_x']
                if x2 <= right_width:
                    original_x1 = x1 + (original_width + slice_info['start_x'])
                    original_x2 = x2 + (original_width + slice_info['start_x'])
                elif x1 >= right_width:
                    original_x1 = x1 - right_width
                    original_x2 = x2 - right_width
                else:
                    original_x1 = x1 - right_width
                    original_x2 = x2 - right_width
                    if original_x2 > original_width:
                        original_x2 = original_width
            else:
                left_width = original_width - slice_info['start_x']
                if x2 <= left_width:
                    original_x1 = x1 + slice_info['start_x']
                    original_x2 = x2 + slice_info['start_x']
                elif x1 >= left_width:
                    original_x1 = x1 - left_width
                    original_x2 = x2 - left_width
                else:
                    original_x1 = x1 + slice_info['start_x']
                    original_x2 = x2 + slice_info['start_x']
                    if original_x2 > original_width:
                        original_x2 = original_width

        original_x1 = max(0, min(original_x1, original_width - 1))
        original_x2 = max(0, min(original_x2, original_width - 1))

        if original_x1 > original_x2:
            original_x1, original_x2 = original_x2, original_x1

        return [original_x1, y1, original_x2, y2]

    def _convert_keypoints_to_original_coords(
        self,
        keypoints: List[List[float]],
        slice_info: dict
    ) -> List[List[float]]:
        if not keypoints:
            return []

        original_keypoints = []

        for kp in keypoints:
            if len(kp) < 3:
                original_keypoints.append(kp)
                continue

            x, y, conf = kp
            original_width = slice_info['original_width']

            if not slice_info['wrap_around']:
                original_x = x + slice_info['start_x']
            else:
                if slice_info['start_x'] < 0:
                    right_width = -slice_info['start_x']
                    if x <= right_width:
                        original_x = x + (original_width + slice_info['start_x'])
                    else:
                        original_x = x - right_width
                else:
                    left_width = original_width - slice_info['start_x']
                    if x <= left_width:
                        original_x = x + slice_info['start_x']
                    else:
                        original_x = x - left_width

            original_x = max(0, min(original_x, original_width - 1))
            original_keypoints.append([original_x, y, conf])

        return original_keypoints

    def _area_weighted_nms(self, boxes, scores, areas, mask_indices):
        normalized_areas = areas / (areas.max() + 1e-6)
        weighted_scores = scores * (0.6 + 0.4 * normalized_areas)
        sorted_indices = np.argsort(-weighted_scores)

        keep = []
        while len(sorted_indices) > 0:
            current_idx = sorted_indices[0]
            keep.append(mask_indices[current_idx])

            if len(sorted_indices) == 1:
                break

            current_box = boxes[current_idx]
            other_boxes = boxes[sorted_indices[1:]]

            ious = box_iou_batch(current_box, other_boxes)
            # 使用 nms_iou_thresh 而非 iou_threshold：
            # Step 3 已处理跨切片重复，到达这里的检测若仍重叠极大概率才是同一目标。
            # 较高阈值防止将"大框包裹小框"的不同人物误压制（IoU = 小框面积/大框面积）。
            keep_indices = np.where(ious < self.nms_iou_thresh)[0]
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep


    def _calculate_wrap_around_iou(self, box1, box2, panorama_width):
        """
        计算360°环绕边界的IoU

        当两个检测框分别位于全景图左右两侧的环绕边界时，
        正常计算IoU会得到很低的值，需要特殊处理。

        策略：
        1. 先尝试正常IoU
        2. 如果很低，则尝试平移一个框后计算IoU（两种方向）
        3. 取最大值
        """
        # 先计算正常IoU
        normal_iou = box_iou(box1, box2)

        # 计算平移后的IoU
        # 方案1：将box1向右平移一个全景宽度
        box1_shifted_right = [box1[0] + panorama_width, box1[1], box1[2] + panorama_width, box1[3]]
        iou_right = box_iou(box1_shifted_right, box2)

        # 方案2：将box1向左平移一个全景宽度
        box1_shifted_left = [box1[0] - panorama_width, box1[1], box1[2] - panorama_width, box1[3]]
        iou_left = box_iou(box1_shifted_left, box2)

        # 取最大值
        max_iou = max(normal_iou, iou_right, iou_left)

        if self.verbose and max_iou > normal_iou + 0.1:  # 如果环绕方式明显更好
            print(f"[环绕IoU] 正常IoU={normal_iou:.3f}, 右移IoU={iou_right:.3f}, 左移IoU={iou_left:.3f} → 使用最大值={max_iou:.3f}")

        return max_iou

    def _check_in_boundary_overlap(self, all_boxes, all_detection_infos, slice_infos, num_slices=3):
        """
        === 关键方法：检查检测框是否位于切片边界重叠区域 ===

        核心思想：
        - 只有位于切片边界重叠区域的检测框才可能是重复检测
        - 切片内部的检测框不需要去重，可以避免误删
        - 优化：每个切片只检查一侧边界，避免重复处理

        优化后的边界检查策略：
        - slice0: 只检查右侧边界（与slice1衔接）
        - slice1: 只检查右侧边界（与slice2衔接）
        - slice2: 只检查右侧边界（与slice0的环绕边界衔接）

        然后在merge阶段，slice0和slice2都在边界的检测对会被特殊处理

        参数:
            all_boxes: 所有检测框 [x1, y1, x2, y2]
            all_detection_infos: 检测信息，包含 slice_idx
            slice_infos: 切片信息
            num_slices: 切片数量

        返回:
            in_boundary_overlap: 布尔数组，表示每个检测框是否在边界重叠区域
        """
        in_boundary_overlap = np.zeros(len(all_boxes), dtype=bool)

        if not slice_infos:
            return in_boundary_overlap

        # 获取全景图宽度
        panorama_width = slice_infos[0]['original_width']
        slice_width = panorama_width // num_slices
        overlap_width = int(slice_width * self.overlap_ratio)

        # 计算每个切片的边界重叠区域
        boundary_regions = []
        for i in range(num_slices):
            # 切片的理论起始和结束位置
            slice_start = i * slice_width - overlap_width
            slice_end = (i + 1) * slice_width + overlap_width

            # 这个切片的边界重叠区域（与相邻切片的重叠部分）
            # 左边界重叠区域
            left_boundary_start = max(0, i * slice_width - overlap_width)
            left_boundary_end = i * slice_width + overlap_width

            # 右边界重叠区域
            right_boundary_start = (i + 1) * slice_width - overlap_width
            right_boundary_end = min(panorama_width, (i + 1) * slice_width + overlap_width)

            boundary_regions.append({
                'slice_idx': i,
                'left_boundary': (left_boundary_start, left_boundary_end),
                'right_boundary': (right_boundary_start, right_boundary_end)
            })

        # 检查每个检测框是否位于边界重叠区域
        for idx, (box, det_info) in enumerate(zip(all_boxes, all_detection_infos)):
            x1, y1, x2, y2 = box
            slice_idx = det_info['slice_idx']

            # 获取该切片的边界区域
            if slice_idx < len(boundary_regions):
                region = boundary_regions[slice_idx]

                # 计算检测框中心点
                center_x = (x1 + x2) / 2

                if slice_idx == 0:
                    # slice0：检查右侧边界（与slice1衔接）
                    right_start, right_end = region['right_boundary']
                    in_right_overlap = (center_x >= right_start) and (center_x <= right_end)
                    in_boundary_overlap[idx] = in_right_overlap
                elif slice_idx == 1:
                    # slice1：检查右侧边界（与slice2衔接）
                    right_start, right_end = region['right_boundary']
                    in_right_overlap = (center_x >= right_start) and (center_x <= right_end)
                    in_boundary_overlap[idx] = in_right_overlap
                elif slice_idx == 2:
                    # slice2：检查右侧边界（与slice0的环绕边界衔接）
                    # slice2的右侧边界包括环绕到全景图左侧的部分
                    right_start, right_end = region['right_boundary']
                    # 正常右侧部分
                    in_right_overlap = (center_x >= right_start) and (center_x <= right_end)
                    # 环绕到左侧的部分（slice2右侧超出panorama_width后会环绕到0开始）
                    in_wrap_right = (center_x >= 0) and (center_x <= overlap_width)
                    in_boundary_overlap[idx] = in_right_overlap or in_wrap_right

        return in_boundary_overlap

    def filter_wide_detections(self, detections: List[Dict], panorama_width: int) -> List[Dict]:
        """
        过滤横跨全景图左右边界的超宽无效检测框

        问题描述：
        在360°全景图中，左右边界是连续的。当一个目标横跨这个边界时，
        可能被检测为一个从最左侧横跨到最右侧的超宽无效框。

        过滤策略：
        1. 检测框宽度超过 panorama_width * max_width_ratio 的认为是无效框
        2. 或者检测框同时接触左右边界（x1 < threshold 且 x2 > panorama_width - threshold）

        Args:
            detections: 检测结果列表
            panorama_width: 全景图宽度

        Returns:
            过滤后的检测结果列表
        """
        if not detections:
            return []

        filtered = []
        threshold = panorama_width * 0.05  # 边界5%范围
        max_allowed_width = panorama_width * self.max_width_ratio

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            box_width = x2 - x1

            # 检查1：宽度是否超过阈值
            is_too_wide = box_width > max_allowed_width

            # 检查2：是否同时接触左右边界（横跨边界的特征）
            is_across_boundary = (x1 < threshold) and (x2 > panorama_width - threshold)

            if is_too_wide or is_across_boundary:
                # 这是一个无效检测，跳过
                if self.verbose:
                    print(f"[边界过滤] 跳过超宽检测框：宽度={box_width:.0f}px，位置=[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                continue

            # 通过检查，保留
            filtered.append(det)

        return filtered
