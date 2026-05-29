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


def draw_bounding_boxes(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    在图像上绘制边界框（包含ReID ID）

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
        track_id = det.get('track_id', -1)

        if confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 200, 255)
        else:
            color = (0, 165, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label_parts = []
        if track_id != -1:
            label_parts.append(f"ID:{track_id}")
        label_parts.append(f"{class_name}: {confidence:.2f}")
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
                    cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                            (0, 255, 0), 2)

    return annotated


def draw_detections(image: np.ndarray, detections: List[Dict],
                   tracker=None) -> np.ndarray:
    """
    在图像上绘制检测结果（包括关键点）

    参数:
        image: 输入图像
        detections: 检测结果列表
        tracker: 跟踪器（可选，用于绘制边界区域）

    返回:
        绘制后的图像
    """
    # if tracker is not None and hasattr(tracker, 'enable_boundary_matching') and tracker.enable_boundary_matching:
    #     annotated = tracker.draw_boundary_regions(image)
    # else:
    annotated = image.copy()

    annotated = draw_bounding_boxes(annotated, detections)
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
