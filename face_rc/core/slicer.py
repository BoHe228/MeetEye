import numpy as np
from typing import List, Tuple, Dict, Any
import cv2
import time
from math import sqrt
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
                 dedup_use_reid: bool = True,
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
        self.dedup_use_reid = dedup_use_reid
        self.max_width_ratio = max_width_ratio
        self.verbose = verbose
        self.last_merge_profile = {}
        self._merge_fast = None
        self._merge_fast_failed = False

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
        if isinstance(yolo_results, list) and (
            not yolo_results or isinstance(yolo_results[0], dict)
        ):
            return yolo_results
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                boxes_data = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                keypoints_data = []
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()

                _multiclass = len(result.names) > 1
                for i, (box, conf, cls_id) in enumerate(zip(boxes_data, confidences, class_ids)):
                    class_name = result.names[int(cls_id)]
                    # 多类模型（如 yolo26n.pt）只保留 person 类
                    if _multiclass and class_name != 'person':
                        continue
                    det = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': class_name,
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
        feature_extractor=None,
        recall_yolo_results: List[List[Results]] = None,  # 补漏检测模型的逐切片结果（--recall-boost）
        recall_match_iou: float = 0.4,                    # 补漏框与 pose 框的 IoU 关联阈值
    ) -> List[Dict[str, Any]]:
        """
        合并所有切片的检测结果 - 基于ReID特征的智能去重

        核心改进：
        1. 先在每个切片上提取ReID特征
        2. 对重叠区域的检测，通过ReID特征相似度判断是否为同一目标
        3. 只有特征相似度高且空间重叠高时，才认为是重复检测
        """
        if not all_yolo_results:
            self.last_merge_profile = {}
            return []

        t0 = time.perf_counter()
        if (
            not self.dedup_use_reid
            and feature_extractor is None
            and recall_yolo_results is None
            and self._can_use_merge_fast(all_yolo_results)
        ):
            try:
                return self._merge_detections_fast(all_yolo_results, slice_infos, t0)
            except Exception as exc:
                if not self._merge_fast_failed:
                    print(f"[merge-fast] 不可用，回退 Python merge: {type(exc).__name__}: {exc}")
                    self._merge_fast_failed = True

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
        t_extract = time.perf_counter()

        # === 第一步(a)：补漏融合（--recall-boost）===
        # 对每个切片，提取补漏检测模型的 person 框；凡是与本切片任何 pose 框
        # IoU >= recall_match_iou 的，视为同一人已被 pose 覆盖 → 丢弃（保留带关键点的 pose 版本）；
        # 与所有 pose 框都不重叠的，作为「无关键点检测」补入（遮挡/背身等 pose 漏检目标）。
        # 在切片局部坐标下做关联，补入的框随后与 pose 框一起做特征提取、坐标转换、去重与 NMS。
        if recall_yolo_results is not None:
            _added = 0
            for slice_idx, recall_results in enumerate(recall_yolo_results):
                if slice_idx >= len(all_detections):
                    break
                recall_dets = self.extract_detections_from_yolo_results(
                    recall_results,
                    slice_img=None,
                    feature_extractor=None,
                    slice_info=slice_infos[slice_idx]
                )
                pose_boxes = [d['bbox'] for d in all_detections[slice_idx]]
                for rdet in recall_dets:
                    rb = rdet['bbox']
                    matched = any(box_iou(rb, pb) >= recall_match_iou for pb in pose_boxes)
                    if not matched:
                        rdet['from_recall'] = True   # 标记来源，便于调试/下游区分
                        all_detections[slice_idx].append(rdet)
                        _added += 1
            if self.verbose and _added:
                print(f"[补漏融合] 本帧补入 {_added} 个 pose 漏检目标（无关键点）")

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
        t_features = time.perf_counter()

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
            self.last_merge_profile = {
                "extract": (t_extract - t0) * 1000,
                "features": (t_features - t_extract) * 1000,
                "coords": (time.perf_counter() - t_features) * 1000,
                "dedup": 0.0,
                "nms": 0.0,
                "build": 0.0,
                "raw": 0,
                "kept": 0,
            }
            return []
        t_coords = time.perf_counter()

        # === 第三步：基于ReID特征的智能去重 ===
        suppressed = np.zeros(len(all_boxes), dtype=bool)

        # 获取全景图宽度与切片数量（切片数随 --num-slices 变化，不再写死 3）
        panorama_width = slice_infos[0]['original_width']
        num_slices = len(slice_infos)
        last_slice_idx = num_slices - 1

        # 首先计算每个检测框是否位于切片边界重叠区域（用于相邻切片对）
        in_boundary_overlap = self._check_in_boundary_overlap(
            all_boxes,
            all_detection_infos,
            slice_infos,
            num_slices=num_slices
        )

        for i in range(len(all_boxes)):
            if suppressed[i]:
                continue

            for j in range(i + 1, len(all_boxes)):
                if suppressed[j]:
                    continue

                # 同切片内：YOLO 偶尔对同一人输出两个框（如全身+头部），
                # 标准 IoU 可能只有 0.09，但 min-IoU（小框面积为分母）趋近 1.0。
                # 用 min-IoU > 0.7 压制"小框大部分落在大框内"的同人重复检测。
                if all_detection_infos[i]['slice_idx'] == all_detection_infos[j]['slice_idx']:
                    _b1 = all_boxes[i]; _b2 = all_boxes[j]
                    _ix1 = max(_b1[0], _b2[0]); _iy1 = max(_b1[1], _b2[1])
                    _ix2 = min(_b1[2], _b2[2]); _iy2 = min(_b1[3], _b2[3])
                    _inter = max(0.0, _ix2 - _ix1) * max(0.0, _iy2 - _iy1)
                    _min_a = min((_b1[2]-_b1[0])*(_b1[3]-_b1[1]),
                                 (_b2[2]-_b2[0])*(_b2[3]-_b2[1]))
                    _intra_min_iou = _inter / (_min_a + 1e-6)
                    if _intra_min_iou > 0.7:
                        if self.verbose:
                            _cx1 = (_b1[0]+_b1[2])/2; _cx2 = (_b2[0]+_b2[2])/2
                            print(f"[同切片去重] 检测{i}和{j}"
                                  f"(slice{all_detection_infos[i]['slice_idx']},"
                                  f"cx={_cx1:.0f}/{_cx2:.0f})："
                                  f"min_iou={_intra_min_iou:.3f} → 同人重复框，去重")
                        if all_scores[i] >= all_scores[j]:
                            suppressed[j] = True
                        else:
                            suppressed[i] = True
                            break
                    continue

                # 检查是否是环绕边界对（首切片 & 末切片，即 slice0 与 slice(N-1)）
                # 仅 num_slices>=3 时成立；num_slices==2 时首末切片本身相邻，归为相邻对处理
                slice_i = all_detection_infos[i]['slice_idx']
                slice_j = all_detection_infos[j]['slice_idx']
                is_wrap_around_pair = (
                    num_slices >= 3
                    and {slice_i, slice_j} == {0, last_slice_idx}
                )

                # 检查是否是相邻边界对（相邻切片，slice_k & slice_(k+1)）
                is_adjacent_pair = (abs(slice_i - slice_j) == 1)

                # 环绕边界对：只要特征相似就去重，不限制位置
                # 相邻边界对：需要在边界重叠区域才去重
                if is_wrap_around_pair:
                    # 环绕边界对（slice0 & sliceN-1）：
                    # 两框在全景坐标下分别位于 x≈0 和 x≈W 两端，标准 IoU=0。
                    # 将 x 较大的框向左平移 panorama_width，使两框在接缝处对齐，
                    # 再计算 min_iou，与相邻切片逻辑统一，精度优于中心距离启发式。
                    b1 = all_boxes[i]
                    b2 = all_boxes[j]
                    # 判断哪个框在右侧（x 中心更大），平移至左侧接缝
                    cx1 = (b1[0] + b1[2]) / 2
                    cx2 = (b2[0] + b2[2]) / 2
                    if cx2 > cx1:
                        b2_shifted = [b2[0] - panorama_width, b2[1],
                                      b2[2] - panorama_width, b2[3]]
                        ba, bb = b1, b2_shifted
                    else:
                        b1_shifted = [b1[0] - panorama_width, b1[1],
                                      b1[2] - panorama_width, b1[3]]
                        ba, bb = b1_shifted, b2

                    inter_x1 = max(ba[0], bb[0]); inter_y1 = max(ba[1], bb[1])
                    inter_x2 = min(ba[2], bb[2]); inter_y2 = min(ba[3], bb[3])
                    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    min_area = min((b1[2]-b1[0])*(b1[3]-b1[1]),
                                   (b2[2]-b2[0])*(b2[3]-b2[1]))
                    wrap_min_iou = inter / (min_area + 1e-6)

                    is_same_target = False

                    if not self.dedup_use_reid:
                        # 纯空间：平移后 min_iou 达标即去重
                        if wrap_min_iou > self.iou_threshold:
                            is_same_target = True
                            if self.verbose:
                                print(f"[环绕去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})："
                                      f"wrap_min_iou={wrap_min_iou:.3f}（纯空间）→ 同一目标，去重")
                    else:
                        feat1 = all_features[i]
                        feat2 = all_features[j]
                        if feat1 is not None and feat2 is not None:
                            similarity = cosine_similarity(feat1, feat2)
                            if similarity >= self.wrap_reid_threshold:
                                is_same_target = True
                                if self.verbose:
                                    print(f"[环绕去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})："
                                          f"wrap_min_iou={wrap_min_iou:.3f}, 特征相似度={similarity:.3f} → 同一目标，去重")
                        else:
                            if wrap_min_iou > self.iou_threshold:
                                is_same_target = True
                                if self.verbose:
                                    print(f"[环绕去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})："
                                          f"wrap_min_iou={wrap_min_iou:.3f}，无特征 → 同一目标，去重")

                    if is_same_target:
                        if all_scores[i] >= all_scores[j]:
                            suppressed[j] = True
                        else:
                            suppressed[i] = True
                            break
                elif is_adjacent_pair:
                    # min-IoU = 交集 / min(面积i, 面积j)：小框被大框覆盖时趋近1.0，
                    # 解决切片边缘"一高一矮"导致标准IoU偏低、去重失败的问题
                    b1 = all_boxes[i]
                    b2 = all_boxes[j]
                    cx1 = (b1[0] + b1[2]) / 2
                    cx2 = (b2[0] + b2[2]) / 2

                    # 边界近邻门控：只对靠近共享切片边界的框做去重
                    # 距边界超过 3×overlap_width 的框不可能是跨切片重复检测
                    _sw = panorama_width // num_slices
                    _boundary_x = max(slice_i, slice_j) * _sw
                    _max_dist = 3.0 * _sw * self.overlap_ratio
                    if abs(cx1 - _boundary_x) > _max_dist or abs(cx2 - _boundary_x) > _max_dist:
                        continue

                    inter_x1 = max(b1[0], b2[0]); inter_y1 = max(b1[1], b2[1])
                    inter_x2 = min(b1[2], b2[2]); inter_y2 = min(b1[3], b2[3])
                    inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                    min_area = min((b1[2]-b1[0])*(b1[3]-b1[1]), (b2[2]-b2[0])*(b2[3]-b2[1]))
                    min_iou = inter / (min_area + 1e-6)

                    if min_iou > self.iou_threshold:
                        is_same_target = False

                        if not self.dedup_use_reid:
                            # 纯空间判据：IoU 满足即视为同一目标，不引入特征
                            is_same_target = True
                            if self.verbose:
                                print(f"[相邻去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：min_iou={min_iou:.3f}（纯空间）→ 同一目标，去重")
                        else:
                            feat1 = all_features[i]
                            feat2 = all_features[j]
                            if feat1 is not None and feat2 is not None:
                                similarity = cosine_similarity(feat1, feat2)
                                if similarity >= self.reid_similarity_threshold:
                                    is_same_target = True
                                    if self.verbose:
                                        print(f"[相邻去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：min_iou={min_iou:.3f}, 特征相似度={similarity:.3f} → 同一目标，去重")
                            else:
                                if self._are_keypoints_similar(all_keypoints[i], all_keypoints[j]):
                                    is_same_target = True
                                    if self.verbose:
                                        print(f"[相邻去重] 检测{i}(slice{slice_i})和{j}(slice{slice_j})：min_iou={min_iou:.3f}, 无特征但关键点相似 → 同一目标，去重")

                        if is_same_target:
                            if all_scores[i] >= all_scores[j]:
                                suppressed[j] = True
                            else:
                                suppressed[i] = True
                                break
                    else:
                        # min_iou 未达阈值时的中心距离兜底：
                        # 适用于同一人在切片边界被检测为两个不同高度的框（如一个只检测到头部、
                        # 一个检测到全身），导致 YOLO 原始框重叠少，但同一人归一化中心距很小。
                        cy1 = (b1[1] + b1[3]) / 2
                        cy2 = (b2[1] + b2[3]) / 2
                        avg_h = ((b1[3] - b1[1]) + (b2[3] - b2[1])) / 2.0
                        cd_norm = sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / max(avg_h, 1.0)
                        is_same_target = False
                        if cd_norm < 0.8:
                            if not self.dedup_use_reid:
                                is_same_target = True
                                if self.verbose:
                                    print(f"[相邻去重(CD)] 检测{i}(slice{slice_i},cx={cx1:.0f})和{j}(slice{slice_j},cx={cx2:.0f})："
                                          f"min_iou={min_iou:.3f}, cd={cd_norm:.2f} → 中心距近，去重")
                            else:
                                feat1 = all_features[i]
                                feat2 = all_features[j]
                                if feat1 is not None and feat2 is not None:
                                    similarity = cosine_similarity(feat1, feat2)
                                    if similarity >= self.reid_similarity_threshold:
                                        is_same_target = True
                                        if self.verbose:
                                            print(f"[相邻去重(CD)] 检测{i}(slice{slice_i})和{j}(slice{slice_j})："
                                                  f"cd={cd_norm:.2f}, 特征={similarity:.3f} → 去重")
                                else:
                                    if self._are_keypoints_similar(all_keypoints[i], all_keypoints[j]):
                                        is_same_target = True
                        elif self.verbose:
                            print(f"[相邻未去重] 检测{i}(slice{slice_i},cx={cx1:.0f})和{j}(slice{slice_j},cx={cx2:.0f})："
                                  f"min_iou={min_iou:.3f}, cd={cd_norm:.2f} [边界x={_boundary_x}]")
                        if is_same_target:
                            if all_scores[i] >= all_scores[j]:
                                suppressed[j] = True
                            else:
                                suppressed[i] = True
                                break

        t_dedup = time.perf_counter()
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
        t_nms = time.perf_counter()

        nms_keep_count = len(keep_indices)
        keep_indices = self._old_spatial_final_dedup(
            keep_indices,
            boxes_array,
            scores_array,
            areas_array,
            labels_array,
        )
        t_final_dedup = time.perf_counter()

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

        t_build = time.perf_counter()
        self.last_merge_profile = {
            "extract": (t_extract - t0) * 1000,
            "features": (t_features - t_extract) * 1000,
            "coords": (t_coords - t_features) * 1000,
            "dedup": (t_dedup - t_coords) * 1000,
            "nms": (t_nms - t_dedup) * 1000,
            "final_dedup": (t_final_dedup - t_nms) * 1000,
            "build": (t_build - t_final_dedup) * 1000,
            "raw": len(all_boxes),
            "nms_kept": nms_keep_count,
            "kept": len(merged_detections),
        }
        return merged_detections

    @staticmethod
    def _can_use_merge_fast(all_yolo_results: List[List[Results]]) -> bool:
        if not isinstance(all_yolo_results, list):
            return False
        for item in all_yolo_results:
            if not isinstance(item, list):
                return False
            for det in item:
                if not isinstance(det, dict):
                    return False
                if "bbox" not in det or "keypoints" not in det:
                    return False
        return True

    def _get_merge_fast(self):
        if self._merge_fast is None:
            try:
                from core.merge_fast import MergeFast
            except Exception:
                from merge_fast import MergeFast
            self._merge_fast = MergeFast()
        return self._merge_fast

    def _merge_detections_fast(
        self,
        all_detections: List[List[Dict[str, Any]]],
        slice_infos: List[dict],
        t0: float,
    ) -> List[Dict[str, Any]]:
        merge_fast = self._get_merge_fast()
        merged, profile = merge_fast.merge(
            all_detections,
            slice_infos,
            overlap_ratio=self.overlap_ratio,
            iou_threshold=self.iou_threshold,
            nms_iou_thresh=self.nms_iou_thresh,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        profile.setdefault("extract", 0.0)
        profile.setdefault("features", 0.0)
        profile["total"] = total_ms
        self.last_merge_profile = profile
        return merged

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

    @staticmethod
    def _intersection_stats(box1, box2):
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        w1 = max(0.0, float(box1[2]) - float(box1[0]))
        h1 = max(0.0, float(box1[3]) - float(box1[1]))
        w2 = max(0.0, float(box2[2]) - float(box2[0]))
        h2 = max(0.0, float(box2[3]) - float(box2[1]))
        area1 = w1 * h1
        area2 = w2 * h2
        min_area = min(area1, area2)
        min_w = min(w1, w2)
        min_h = min(h1, h2)
        return inter, min_area, iw, ih, min_w, min_h, area1, area2

    @staticmethod
    def _center_distance_norm(box1, box2):
        cx1 = (float(box1[0]) + float(box1[2])) / 2.0
        cy1 = (float(box1[1]) + float(box1[3])) / 2.0
        cx2 = (float(box2[0]) + float(box2[2])) / 2.0
        cy2 = (float(box2[1]) + float(box2[3])) / 2.0
        h1 = max(float(box1[3]) - float(box1[1]), 1.0)
        h2 = max(float(box2[3]) - float(box2[1]), 1.0)
        return sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) / ((h1 + h2) / 2.0)

    def _old_spatial_duplicate(self, box1, box2) -> bool:
        """旧版 no-OSNet 去重判据的最终兜底版，只使用空间关系。"""
        inter, min_area, iw, ih, min_w, min_h, area1, area2 = self._intersection_stats(box1, box2)
        if min_area <= 0:
            return False

        min_iou = inter / (min_area + 1e-6)
        if min_iou > 0.7:
            return True

        # 等价于旧版相邻切片中心距离兜底，但加轴向重叠门控，避免把相邻不同人误压掉。
        cd_norm = self._center_distance_norm(box1, box2)
        x_cover = iw / (min_w + 1e-6) if min_w > 0 else 0.0
        y_cover = ih / (min_h + 1e-6) if min_h > 0 else 0.0
        area_ratio = min(area1, area2) / (max(area1, area2) + 1e-6)
        if cd_norm < 0.8 and x_cover > 0.35 and y_cover > 0.35 and area_ratio > 0.25:
            return True

        return False

    def _old_spatial_final_dedup(self, keep_indices, boxes, scores, areas, labels):
        """
        NMS 后的旧逻辑兜底：RKNN/INT8 输出可能让同一目标以多个框保留下来。
        这里仍按旧版 no-OSNet 的空间关系压重，不使用 ReID/OSNet。
        """
        if len(keep_indices) <= 1:
            return list(keep_indices)

        selected = []
        for label in np.unique(labels[keep_indices]):
            label_indices = [int(idx) for idx in keep_indices if labels[int(idx)] == label]
            if len(label_indices) <= 1:
                selected.extend(label_indices)
                continue

            label_areas = areas[label_indices]
            normalized_areas = label_areas / (label_areas.max() + 1e-6)
            weighted_scores = scores[label_indices] * (0.6 + 0.4 * normalized_areas)
            ordered = [label_indices[i] for i in np.argsort(-weighted_scores)]

            kept_for_label = []
            for idx in ordered:
                duplicate = False
                for kept_idx in kept_for_label:
                    if self._old_spatial_duplicate(boxes[idx], boxes[kept_idx]):
                        duplicate = True
                        break
                if not duplicate:
                    kept_for_label.append(idx)
            selected.extend(kept_for_label)

        selected_set = set(selected)
        return [int(idx) for idx in keep_indices if int(idx) in selected_set]


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

        # 检查每个检测框是否位于边界重叠区域（对任意切片数通用）
        # 策略：每个切片只检查其右侧边界（与下一切片衔接）；
        #       末切片的右侧会环绕到全景图最左侧，额外检查 [0, overlap_width]。
        for idx, (box, det_info) in enumerate(zip(all_boxes, all_detection_infos)):
            x1, y1, x2, y2 = box
            slice_idx = det_info['slice_idx']

            # 获取该切片的边界区域
            if slice_idx < len(boundary_regions):
                region = boundary_regions[slice_idx]

                # 计算检测框中心点
                center_x = (x1 + x2) / 2

                # 右侧边界（与下一切片衔接），所有切片通用
                right_start, right_end = region['right_boundary']
                in_right_overlap = (center_x >= right_start) and (center_x <= right_end)

                # 末切片：右侧超出 panorama_width 后环绕到全景图最左侧 [0, overlap_width]
                if slice_idx == num_slices - 1:
                    in_wrap_right = (center_x >= 0) and (center_x <= overlap_width)
                    in_right_overlap = in_right_overlap or in_wrap_right

                in_boundary_overlap[idx] = in_right_overlap

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
