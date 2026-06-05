"""
可视化工具模块 - 包含绘制相关函数
从 main.py 拆分出来
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple
import sys
import os
# 添加父目录到 path 以便导入 config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from config import KEYPOINT_COLORS, SKELETON_CONNECTIONS
except ImportError:
    # 如果导入失败，使用默认值
    print("警告: 无法从 config 导入关键点颜色和骨架连接，使用默认值")
    KEYPOINT_COLORS = [
        (0, 255, 0), (0, 255, 255), (0, 255, 255),
        (0, 0, 255), (0, 0, 255), (255, 0, 0),
        (255, 0, 0), (255, 0, 255), (255, 0, 255),
        (128, 0, 128), (128, 0, 128), (0, 165, 255),
        (0, 165, 255), (0, 128, 128), (0, 128, 128),
        (255, 255, 0), (255, 255, 0)
    ]
    SKELETON_CONNECTIONS = [
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
        (5, 7), (6, 8), (7, 9), (8, 10), (5, 11),
        (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)
    ]


def compute_stable_bbox_from_keypoints(
    keypoints: List,
    conf_thresh: float = 0.3,
    padding: float = 0.15,
    fallback_bbox: List = None,
    min_visible: int = 3,
    upper_body_only: bool = True,
    padding_v: float = None,
) -> List:
    """
    从高置信度关键点推导稳定的边界框。

    关键点坐标由多帧骨架结构约束，比 bbox 回归头更稳定，
    不会因手臂伸展、局部遮挡等原因引起框大小频繁抖动。

    COCO 头肩点索引（upper_body_only=True 时仅用这 7 个）：
      0=鼻, 1=左眼, 2=右眼, 3=左耳, 4=右耳, 5=左肩, 6=右肩

    参数:
        keypoints:       COCO 17 关键点列表，每项 [x, y, conf]
        conf_thresh:     关键点可见性阈值（默认 0.3）
        padding:         左右(水平)扩展比例（相对于关键点跨度）
        fallback_bbox:   可见点不足时的兜底框（通常传 YOLO 原始框）
        min_visible:     最少需要几个可见关键点，不足则使用 fallback
        upper_body_only: 仅使用头肩关键点（索引 0-6），排除手臂/腿部（默认 True）
        padding_v:       上下(垂直)扩展比例；None 时等同 padding

    返回:
        [x1, y1, x2, y2] 或 fallback_bbox
    """
    if padding_v is None:
        padding_v = padding

    if not keypoints:
        return fallback_bbox

    indices = range(7) if upper_body_only else range(len(keypoints))
    visible = [
        (keypoints[i][0], keypoints[i][1])
        for i in indices
        if i < len(keypoints) and len(keypoints[i]) >= 3 and keypoints[i][2] > conf_thresh
    ]

    if len(visible) < min_visible:
        return fallback_bbox

    xs = [x for x, y in visible]
    ys = [y for x, y in visible]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    w, h = x2 - x1, y2 - y1
    if w < 5 or h < 5:
        return fallback_bbox

    # 防呆：关键点跨 360° 接缝分裂时（部分点在 x≈0、部分在 x≈W），min/max 会得到
    # 横跨整幅全景的巨框（宽度远超人体、中心落在画面正中），产生数千像素宽的畸形框
    # 污染跟踪。此时回退到原始框（YOLO 框在拼缝处是正常的一侧框）。
    if fallback_bbox is not None:
        fb_w = fallback_bbox[2] - fallback_bbox[0]
        if fb_w > 0 and w > 2.0 * fb_w:
            return fallback_bbox

    x1 -= w * padding
    x2 += w * padding
    y1 -= h * padding_v
    y2 += h * padding_v

    return [x1, y1, x2, y2]


def draw_yolo_only(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    在图像上绘制纯YOLO检测结果（不含跟踪ID、ReID等额外信息）

    参数:
        image: 输入图像
        detections: 检测结果列表

    返回:
        绘制后的图像
    """
    annotated = image.copy()

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = map(int, bbox)
        confidence = det['confidence']
        class_name = det['class_name']

        if confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 200, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {confidence:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

    return annotated


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict],
                        show_id: bool = True, show_conf: bool = True,
                        face_name_map: dict = None,
                        use_kpt_bbox: bool = False,
                        kpt_bbox_conf: float = 0.3,
                        kpt_bbox_padding: float = 0.15,
                        kpt_bbox_upper_only: bool = True,
                        kpt_bbox_padding_v: float = None) -> np.ndarray:
    """
    在图像上绘制边界框（包含ReID ID）

    参数:
        image: 输入图像
        detections: 检测结果列表
        show_id: 是否显示 Track ID
        show_conf: 是否显示置信度

    返回:
        绘制后的图像
    """
    annotated = image.copy()

    for det in detections:
        bbox = det['bbox']
        if use_kpt_bbox:
            bbox = compute_stable_bbox_from_keypoints(
                det.get('keypoints', []),
                conf_thresh=kpt_bbox_conf,
                padding=kpt_bbox_padding,
                fallback_bbox=bbox,
                upper_body_only=kpt_bbox_upper_only,
                padding_v=kpt_bbox_padding_v,
            )
        x1, y1, x2, y2 = map(int, bbox)
        confidence = det['confidence']
        class_name = det['class_name']
        track_id = det.get('track_id', -1)
        is_lost = det.get('_is_lost', False)

        if is_lost:
            color = (255, 191, 0)  # 灰色表示 Kalman 预测（非当前帧检测）
        elif confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 200, 255)
        else:
            color = (0, 165, 255)

        thickness = 1 if is_lost else 2
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label_parts = []
        face_name = (face_name_map or {}).get(track_id)
        if face_name:
            label_parts.append(face_name)
        elif show_id and track_id != -1:
            label_parts.append(f"ID:{track_id}")
        if show_conf:
            label_parts.append(f"{confidence:.2f}")
        label = " ".join(label_parts)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )

        cv2.rectangle(
            annotated,
            (x1, y1 - text_height - 5),
            (x1 + text_width, y1),
            color,
            -1
        )

        cv2.putText(
            annotated,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        # 说话状态标签（框右下角）
        talking = det.get('talking')
        if talking is not None:
            t_label = "Speaking" if talking else "Silent"
            t_color = (0, 200, 80) if talking else (180, 180, 180)
            (tw, th), _ = cv2.getTextSize(t_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx1 = x2 - tw - 4
            ty1 = y2 - th - 4
            cv2.rectangle(annotated, (tx1 - 2, ty1 - 2), (x2, y2), t_color, -1)
            cv2.putText(annotated, t_label, (tx1, y2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return annotated


def draw_keypoints(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    在图像上绘制关键点和骨架

    参数:
        image: 输入图像
        detections: 检测结果列表

    返回:
        绘制后的图像
    """
    annotated = image.copy()
    seam_threshold = image.shape[1] // 2  # 跨接缝的骨架线水平跨度必然超过半幅宽

    for det in detections:
        if 'keypoints' not in det:
            continue

        keypoints = det['keypoints']

        for i, kp in enumerate(keypoints):
            if len(kp) >= 3:
                x, y, conf = kp
                if conf > 0.3:
                    cv2.circle(annotated, (int(x), int(y)), 4, KEYPOINT_COLORS[i], -1)

        for connection in SKELETON_CONNECTIONS:
            start_idx, end_idx = connection
            if (len(keypoints) > max(start_idx, end_idx) and
                len(keypoints[start_idx]) >= 3 and
                len(keypoints[end_idx]) >= 3):

                x1, y1, conf1 = keypoints[start_idx]
                x2, y2, conf2 = keypoints[end_idx]

                if conf1 > 0.3 and conf2 > 0.3:
                    if abs(x1 - x2) > seam_threshold:
                        continue  # 跨越全景接缝的伪连线，跳过
                    cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 255, 0), 2)

    return annotated


def draw_detections(image: np.ndarray, detections: List[Dict],
                   tracker=None,
                   show_id: bool = True, show_conf: bool = True,
                   face_name_map: dict = None,
                   use_kpt_bbox: bool = False,
                   kpt_bbox_conf: float = 0.3,
                   kpt_bbox_padding: float = 0.15,
                   kpt_bbox_upper_only: bool = True,
                   kpt_bbox_padding_v: float = None) -> np.ndarray:
    """
    在图像上绘制检测结果（包括关键点）

    参数:
        image: 输入图像
        detections: 检测结果列表
        tracker: 跟踪器（可选，用于绘制边界区域）
        show_id: 是否显示 Track ID
        show_conf: 是否显示置信度

    返回:
        绘制后的图像
    """
    # if tracker is not None and hasattr(tracker, 'enable_boundary_matching') and tracker.enable_boundary_matching:
    #     annotated = tracker.draw_boundary_regions(image)
    # else:
    annotated = image.copy()

    annotated = draw_bounding_boxes(annotated, detections, show_id=show_id, show_conf=show_conf,
                                    face_name_map=face_name_map, use_kpt_bbox=use_kpt_bbox,
                                    kpt_bbox_conf=kpt_bbox_conf, kpt_bbox_padding=kpt_bbox_padding,
                                    kpt_bbox_upper_only=kpt_bbox_upper_only,
                                    kpt_bbox_padding_v=kpt_bbox_padding_v)
    annotated = draw_keypoints(annotated, detections)

    return annotated


def filter_cross_boundary_detections(detections: List[Dict],
                                    image_shape: Tuple[int, int]) -> List[Dict]:
    """
    过滤跨边界的检测框，处理坐标转换异常

    参数:
        detections: 检测结果列表
        image_shape: 图像形状

    返回:
        过滤后的检测结果
    """
    if not detections:
        return []

    filtered = []
    height, width = image_shape[:2]

    for det in detections:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox

        if x1 >= width or x2 <= 0 or y1 >= height or y2 <= 0:
            continue

        if x2 < x1:
            continue

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        det['bbox'] = [x1, y1, x2, y2]
        filtered.append(det)

    return filtered
