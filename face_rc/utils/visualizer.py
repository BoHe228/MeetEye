from typing import Dict, List, Tuple

import cv2
import numpy as np

from config import KEYPOINT_COLORS, SKELETON_CONNECTIONS


def compute_stable_bbox_from_keypoints(
    keypoints: List,
    conf_thresh: float = 0.3,
    padding: float = 0.15,
    fallback_bbox: List = None,
    min_visible: int = 3,
    upper_body_only: bool = True,
    padding_v: float = None,
) -> List:
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

    xs = [x for x, _y in visible]
    ys = [y for _x, y in visible]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    if w < 5 or h < 5:
        return fallback_bbox

    if fallback_bbox is not None:
        fb_w = fallback_bbox[2] - fallback_bbox[0]
        if fb_w > 0 and w > 2.0 * fb_w:
            return fallback_bbox

    x1 -= w * padding
    x2 += w * padding
    y1 -= h * padding_v
    y2 += h * padding_v
    return [x1, y1, x2, y2]


def draw_keypoints(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated = image.copy()
    seam_threshold = image.shape[1] // 2
    for det in detections:
        keypoints = det.get("keypoints")
        if not keypoints:
            continue
        for i, kp in enumerate(keypoints):
            if len(kp) >= 3 and kp[2] > 0.3 and i < len(KEYPOINT_COLORS):
                cv2.circle(annotated, (int(kp[0]), int(kp[1])), 4, KEYPOINT_COLORS[i], -1)

        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if len(keypoints) <= max(start_idx, end_idx):
                continue
            a, b = keypoints[start_idx], keypoints[end_idx]
            if len(a) < 3 or len(b) < 3 or a[2] <= 0.3 or b[2] <= 0.3:
                continue
            if abs(a[0] - b[0]) > seam_threshold:
                continue
            cv2.line(
                annotated,
                (int(a[0]), int(a[1])),
                (int(b[0]), int(b[1])),
                (0, 255, 0),
                2,
            )
    return annotated


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    tracker=None,
    show_id: bool = True,
    show_conf: bool = True,
    use_kpt_bbox: bool = False,
    kpt_bbox_conf: float = 0.3,
    kpt_bbox_padding: float = 0.15,
    kpt_bbox_upper_only: bool = True,
    kpt_bbox_padding_v: float = None,
    draw_kpt: bool = False,
) -> np.ndarray:
    annotated = image.copy()

    for det in detections:
        bbox = det["bbox"]
        if use_kpt_bbox:
            bbox = compute_stable_bbox_from_keypoints(
                det.get("keypoints", []),
                conf_thresh=kpt_bbox_conf,
                padding=kpt_bbox_padding,
                fallback_bbox=bbox,
                upper_body_only=kpt_bbox_upper_only,
                padding_v=kpt_bbox_padding_v,
            )
        x1, y1, x2, y2 = map(int, bbox)
        confidence = float(det.get("confidence", 0.0))
        track_id = int(det.get("track_id", -1))
        is_lost = bool(det.get("_is_lost", False))

        if is_lost:
            color = (255, 191, 0)
        elif confidence > 0.8:
            color = (0, 255, 0)
        elif confidence > 0.6:
            color = (0, 200, 255)
        else:
            color = (0, 165, 255)
        thickness = 1 if is_lost else 2
        if det.get("_sector_rep"):
            color = (0, 0, 255)
            thickness = 3

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        lines = []
        first_line = []
        if show_id and track_id != -1:
            first_line.append(f"ID:{track_id}")
        if show_conf:
            first_line.append(f"{confidence:.2f}")
        if first_line:
            lines.append(" ".join(first_line))
        if not lines:
            continue

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        font_thickness = 2
        sizes = [cv2.getTextSize(line, font, scale, font_thickness)[0] for line in lines]
        text_w = max(w for w, _h in sizes)
        line_h = max(h for _w, h in sizes) + 6
        box_h = line_h * len(lines) + 4
        label_y2 = max(box_h, y1)
        label_y1 = label_y2 - box_h
        label_x2 = min(annotated.shape[1] - 1, x1 + text_w + 8)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), color, -1)
        for idx, line in enumerate(lines):
            ty = label_y1 + 4 + (idx + 1) * line_h - 4
            cv2.putText(
                annotated,
                line,
                (x1 + 4, ty),
                font,
                scale,
                (0, 0, 0),
                font_thickness,
                cv2.LINE_AA,
            )

    if draw_kpt:
        annotated = draw_keypoints(annotated, detections)
    return annotated


def filter_cross_boundary_detections(
    detections: List[Dict],
    image_shape: Tuple[int, int],
) -> List[Dict]:
    if not detections:
        return []
    filtered = []
    height, width = image_shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
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
        det["bbox"] = [x1, y1, x2, y2]
        filtered.append(det)
    return filtered
