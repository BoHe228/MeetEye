#!/usr/bin/env python3
"""Convert OmniLab fisheye pose data to unwrapped panorama slices.

The output is an Ultralytics YOLO pose dataset plus transformed COCO-style
metadata for inspection:

  omnilab_zhankai/
    images/train|val/*.png
    labels/train|test/*.txt
    panorama/*.png
    debug/*.jpg
    annotations_slices.json
    omnilab_zhankai.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO_SKELETON = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
]

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


@dataclass(frozen=True)
class UnwrapConfig:
    output_width: int
    output_height: int
    cut_position: str
    view_type: str
    crop_divisor: int
    num_slices: int
    slice_overlap: float


@dataclass(frozen=True)
class FisheyeCircle:
    center_x: float
    center_y: float
    radius: float


@dataclass(frozen=True)
class SliceInfo:
    index: int
    start_x: int
    end_x: int
    actual_start_x: int
    width: int
    wrap_around: bool


def detect_fisheye_region(img: np.ndarray) -> FisheyeCircle:
    """Detect the fisheye circle, using the xiding_zhankai.py fallback behavior."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    center_x = w // 2
    center_y = h // 2
    radius = min(w, h) // 2

    try:
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min(w, h) // 4,
            param1=50,
            param2=30,
            minRadius=min(w, h) // 4,
            maxRadius=min(w, h) // 2,
        )
        if circles is not None:
            detected = np.round(circles[0, 0]).astype(int)
            candidate_center = (int(detected[0]), int(detected[1]))
            candidate_radius = int(detected[2])
            if validate_circle(candidate_center, candidate_radius, w, h):
                center_x, center_y = candidate_center
                radius = candidate_radius
    except Exception:
        pass

    return FisheyeCircle(float(center_x), float(center_y), float(radius))


def validate_circle(center: tuple[int, int], radius: int, img_w: int, img_h: int) -> bool:
    max_offset = min(img_w, img_h) // 3
    if abs(center[0] - img_w // 2) > max_offset or abs(center[1] - img_h // 2) > max_offset:
        return False

    min_radius = min(img_w, img_h) // 4
    max_radius = min(img_w, img_h) // 2 + 50
    return min_radius <= radius <= max_radius


def create_unwrap_maps(
    img_width: int,
    img_height: int,
    circle: FisheyeCircle,
    cfg: UnwrapConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the ceiling-fisheye unwrap map used by maps/xiding_zhankai.py."""
    x_out = np.arange(cfg.output_width, dtype=np.float32)[None, :]
    y_out = np.arange(cfg.output_height, dtype=np.float32)[:, None]
    y_norm = y_out / float(max(1, cfg.output_height - 1))
    if cfg.view_type == "bottom":
        # Output bottom = fisheye center, output top = fisheye edge.
        r_ratio = 1.0 - y_norm
    else:
        # Output top = fisheye center, output bottom = fisheye edge.
        r_ratio = y_norm

    # xiding_zhankai.py uses 0 -> 2pi from the right side counter-clockwise.
    source_x = x_out
    if cfg.cut_position == "left":
        # xiding applies np.roll(panorama, output_width // 2, axis=1)
        # after remap. Integrate that roll into the map for one-pass remap.
        source_x = (x_out - cfg.output_width // 2) % cfg.output_width
    angle = 2.0 * math.pi * source_x / float(cfg.output_width)

    map_x = circle.center_x + r_ratio * circle.radius * np.cos(angle)
    map_y = circle.center_y + r_ratio * circle.radius * np.sin(angle)
    map_x = np.clip(map_x, 0, img_width - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, img_height - 1).astype(np.float32)
    return map_x, map_y


def unwrap_image(img: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def fish_point_to_panorama(
    x: float,
    y: float,
    circle: FisheyeCircle,
    cfg: UnwrapConfig,
) -> tuple[float, float] | None:
    dx = float(x) - circle.center_x
    dy = float(y) - circle.center_y
    radius = math.hypot(dx, dy)
    if radius > circle.radius * 1.02:
        return None

    if cfg.view_type == "bottom":
        y_out = (1.0 - radius / circle.radius) * float(cfg.output_height - 1)
    else:
        y_out = radius / circle.radius * float(cfg.output_height - 1)
    angle = math.atan2(dy, dx)
    x_out = (angle / (2.0 * math.pi) * float(cfg.output_width)) % float(cfg.output_width)
    if cfg.cut_position == "left":
        x_out = (x_out + cfg.output_width // 2) % float(cfg.output_width)
    return x_out, y_out


def bbox_sample_points(bbox: list[float], samples_per_edge: int = 12) -> list[tuple[float, float]]:
    x, y, w, h = bbox
    x2 = x + w
    y2 = y + h
    points: list[tuple[float, float]] = []
    for t in np.linspace(0.0, 1.0, samples_per_edge):
        points.append((x + t * w, y))
        points.append((x + t * w, y2))
        points.append((x, y + t * h))
        points.append((x2, y + t * h))
    return points


def crop_top(panorama: np.ndarray, cfg: UnwrapConfig) -> tuple[np.ndarray, int]:
    if cfg.crop_divisor <= 0:
        return panorama, 0
    crop_y = cfg.output_height // cfg.crop_divisor
    return panorama[crop_y:, :], crop_y


def make_slice_infos(width: int, cfg: UnwrapConfig) -> list[SliceInfo]:
    slice_width = width // cfg.num_slices
    overlap_width = int(slice_width * cfg.slice_overlap)
    infos: list[SliceInfo] = []

    for i in range(cfg.num_slices):
        start_x = i * slice_width - overlap_width
        end_x = (i + 1) * slice_width + overlap_width
        wrap_around = start_x < 0 or end_x > width
        actual_start_x = width + start_x if start_x < 0 else start_x
        infos.append(
            SliceInfo(
                index=i,
                start_x=start_x,
                end_x=end_x,
                actual_start_x=actual_start_x,
                width=end_x - start_x,
                wrap_around=wrap_around,
            )
        )
    return infos


def slice_panorama(panorama: np.ndarray, infos: list[SliceInfo]) -> list[np.ndarray]:
    width = panorama.shape[1]
    slices: list[np.ndarray] = []
    for info in infos:
        if info.start_x < 0:
            left_part = panorama[:, info.start_x :]
            right_part = panorama[:, : info.end_x]
            slices.append(np.concatenate([left_part, right_part], axis=1))
        elif info.end_x > width:
            right_part = panorama[:, info.start_x : width]
            left_part = panorama[:, : info.end_x - width]
            slices.append(np.concatenate([right_part, left_part], axis=1))
        else:
            slices.append(panorama[:, info.start_x : info.end_x])
    return slices


def panorama_x_to_slice_x(x: float, panorama_width: int, info: SliceInfo) -> float | None:
    local_x = (float(x) - float(info.actual_start_x)) % float(panorama_width)
    if 0.0 <= local_x < float(info.width):
        return local_x
    return None


def transform_keypoints(
    keypoints: list[float],
    circle: FisheyeCircle,
    cfg: UnwrapConfig,
    crop_y: int,
) -> list[tuple[float, float, int]]:
    transformed: list[tuple[float, float, int]] = []
    for i in range(17):
        x, y, v = keypoints[i * 3 : i * 3 + 3]
        if int(v) <= 0:
            transformed.append((0.0, 0.0, 0))
            continue
        point = fish_point_to_panorama(x, y, circle, cfg)
        if point is None:
            transformed.append((0.0, 0.0, 0))
            continue
        px, py = point
        transformed.append((px, py - crop_y, int(v)))
    return transformed


def transform_bbox_samples(
    bbox: list[float],
    circle: FisheyeCircle,
    cfg: UnwrapConfig,
    crop_y: int,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for x, y in bbox_sample_points(bbox):
        point = fish_point_to_panorama(x, y, circle, cfg)
        if point is not None:
            px, py = point
            points.append((px, py - crop_y))
    return points


def build_slice_label(
    bbox_points: list[tuple[float, float]],
    keypoints: list[tuple[float, float, int]],
    panorama_width: int,
    slice_height: int,
    info: SliceInfo,
    min_box_size: float,
    min_keypoints: int,
    min_keypoint_box_ratio: float,
    bbox_source: str,
    keypoint_bbox_padding: float,
    keypoint_bbox_padding_y: float,
) -> tuple[list[float], list[tuple[float, float, int]]] | None:
    local_bbox_points: list[tuple[float, float]] = []
    for px, py in bbox_points:
        local_x = panorama_x_to_slice_x(px, panorama_width, info)
        if local_x is None:
            continue
        if -2.0 <= py <= slice_height + 2.0:
            local_bbox_points.append((local_x, min(max(py, 0.0), float(slice_height - 1))))

    local_keypoints: list[tuple[float, float, int]] = []
    visible_count = 0
    for px, py, v in keypoints:
        if v <= 0:
            local_keypoints.append((0.0, 0.0, 0))
            continue
        local_x = panorama_x_to_slice_x(px, panorama_width, info)
        if local_x is None or not (0.0 <= py < float(slice_height)):
            local_keypoints.append((0.0, 0.0, 0))
            continue
        local_keypoints.append((local_x, py, v))
        visible_count += 1
        local_bbox_points.append((local_x, py))

    if len(local_bbox_points) < 2 or visible_count < min_keypoints:
        return None

    visible_points = [(x, y) for x, y, v in local_keypoints if v > 0]
    if bbox_source == "keypoints":
        xs = [p[0] for p in visible_points]
        ys = [p[1] for p in visible_points]
        kx1 = min(xs)
        ky1 = min(ys)
        kx2 = max(xs)
        ky2 = max(ys)
        kw = max(kx2 - kx1, min_box_size)
        kh = max(ky2 - ky1, min_box_size)
        pad_x = max(kw * keypoint_bbox_padding, min_box_size)
        pad_y = max(kh * keypoint_bbox_padding_y, min_box_size)
        x1 = min(max(kx1 - pad_x, 0.0), float(info.width - 1))
        y1 = min(max(ky1 - pad_y, 0.0), float(slice_height - 1))
        x2 = min(max(kx2 + pad_x, 0.0), float(info.width - 1))
        y2 = min(max(ky2 + pad_y, 0.0), float(slice_height - 1))
    else:
        xs = [p[0] for p in local_bbox_points]
        ys = [p[1] for p in local_bbox_points]
        x1 = min(max(min(xs), 0.0), float(info.width - 1))
        y1 = min(max(min(ys), 0.0), float(slice_height - 1))
        x2 = min(max(max(xs), 0.0), float(info.width - 1))
        y2 = min(max(max(ys), 0.0), float(slice_height - 1))

    if x2 - x1 < min_box_size or y2 - y1 < min_box_size:
        return None

    if len(visible_points) >= 2 and min_keypoint_box_ratio > 0.0:
        kx = [p[0] for p in visible_points]
        ky = [p[1] for p in visible_points]
        keypoint_area = max(max(kx) - min(kx), 1.0) * max(max(ky) - min(ky), 1.0)
        bbox_area = max(x2 - x1, 1.0) * max(y2 - y1, 1.0)
        if keypoint_area / bbox_area < min_keypoint_box_ratio:
            return None

    return [x1, y1, x2, y2], local_keypoints


def yolo_pose_line(
    bbox_xyxy: list[float],
    keypoints: list[tuple[float, float, int]],
    image_width: int,
    image_height: int,
) -> str:
    x1, y1, x2, y2 = bbox_xyxy
    cx = ((x1 + x2) / 2.0) / image_width
    cy = ((y1 + y2) / 2.0) / image_height
    bw = (x2 - x1) / image_width
    bh = (y2 - y1) / image_height
    values: list[float | int] = [0, cx, cy, bw, bh]
    for x, y, v in keypoints:
        if v <= 0:
            values.extend([0.0, 0.0, 0])
        else:
            values.extend([x / image_width, y / image_height, int(v)])
    return " ".join(format_label_value(v) for v in values)


def format_label_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def draw_pose_debug(
    image: np.ndarray,
    labels: list[tuple[list[float], list[tuple[float, float, int]]]],
) -> np.ndarray:
    out = image.copy()
    for bbox, keypoints in labels:
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        for start, end in COCO_SKELETON:
            s = start - 1
            e = end - 1
            if keypoints[s][2] > 0 and keypoints[e][2] > 0:
                p1 = (int(round(keypoints[s][0])), int(round(keypoints[s][1])))
                p2 = (int(round(keypoints[e][0])), int(round(keypoints[e][1])))
                cv2.line(out, p1, p2, (0, 200, 0), 2)
        for x, y, v in keypoints:
            if v > 0:
                color = (0, 0, 255) if v == 2 else (255, 0, 255)
                cv2.circle(out, (int(round(x)), int(round(y))), 4, color, -1)
    return out


def split_name(index: int, eval_ratio: float, eval_split: str) -> str:
    if eval_ratio <= 0.0:
        return "train"
    period = max(1, round(1.0 / eval_ratio))
    return eval_split if index % period == 0 else "train"


def write_dataset_yaml(path: Path, cfg: UnwrapConfig, eval_split: str) -> None:
    yaml_text = f"""path: {path.resolve()}
train: images/train
val: images/{eval_split}
test: images/{eval_split}

kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}

names:
  0: person

kpt_names:
  0:
"""
    yaml_text += "".join(f"    - {name}\n" for name in COCO_KEYPOINT_NAMES)
    yaml_text += f"""
# Generated from OmniLab ceiling fisheye images.
# unwrap: {cfg.output_width}x{cfg.output_height}, view_type={cfg.view_type}, cut_position={cfg.cut_position}
# crop_divisor: {cfg.crop_divisor}
# slices: {cfg.num_slices}, overlap={cfg.slice_overlap}
"""
    (path / "omnilab_zhankai.yaml").write_text(yaml_text, encoding="utf-8")


def ensure_output_dirs(out_dir: Path) -> None:
    (out_dir / "panorama").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)


def load_omnilab(json_path: Path) -> tuple[dict, dict[int, list[dict]]]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    anns_by_image: dict[int, list[dict]] = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return data, anns_by_image


def iter_images(data: dict, limit: int | None) -> Iterable[tuple[int, dict]]:
    images = sorted(data["images"], key=lambda item: int(item["id"]))
    if limit is not None:
        images = images[:limit]
    for index, image_info in enumerate(images):
        yield index, image_info


def convert_dataset(args: argparse.Namespace) -> None:
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    image_dir = src_dir / "concatenated"
    json_path = src_dir / args.annotation
    data, anns_by_image = load_omnilab(json_path)
    ensure_output_dirs(out_dir)

    first_image = cv2.imread(str(image_dir / data["images"][0]["file_name"]))
    if first_image is None:
        raise FileNotFoundError(image_dir / data["images"][0]["file_name"])

    if args.center_x is not None and args.center_y is not None and args.radius is not None:
        circle = FisheyeCircle(float(args.center_x), float(args.center_y), float(args.radius))
    elif args.circle_mode == "center":
        h, w = first_image.shape[:2]
        circle = FisheyeCircle(float(w // 2), float(h // 2), float(min(w, h) // 2))
    else:
        circle = detect_fisheye_region(first_image)

    cfg = UnwrapConfig(
        output_width=args.output_width,
        output_height=args.output_height,
        cut_position=args.cut_position,
        view_type=args.view_type,
        crop_divisor=args.crop_divisor,
        num_slices=args.num_slices,
        slice_overlap=args.slice_overlap,
    )
    map_x, map_y = create_unwrap_maps(first_image.shape[1], first_image.shape[0], circle, cfg)
    slice_infos = make_slice_infos(cfg.output_width, cfg)
    crop_y = cfg.output_height // cfg.crop_divisor if cfg.crop_divisor > 0 else 0
    slice_height = cfg.output_height - crop_y

    print(
        "Using fisheye circle: "
        f"center=({circle.center_x:.1f}, {circle.center_y:.1f}), radius={circle.radius:.1f}"
    )
    print(
        "Output: "
        f"panorama={cfg.output_width}x{cfg.output_height}, "
        f"view_type={cfg.view_type}, "
        f"cropped={cfg.output_width}x{slice_height}, "
        f"slices={cfg.num_slices}x{slice_infos[0].width}x{slice_height}"
    )

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 0
    counts = {"images": 0, "slices": 0, "labels": 0}

    for index, image_info in iter_images(data, args.limit):
        file_name = image_info["file_name"]
        stem = Path(file_name).stem
        if args.labels_only:
            slices = [None] * len(slice_infos)
        else:
            img = cv2.imread(str(image_dir / file_name))
            if img is None:
                print(f"Skip unreadable image: {file_name}")
                continue

            panorama = unwrap_image(img, map_x, map_y)
            if not args.no_panorama:
                cv2.imwrite(str(out_dir / "panorama" / f"{stem}.png"), panorama)
            panorama_cropped, crop_y = crop_top(panorama, cfg)
            slices = slice_panorama(panorama_cropped, slice_infos)
        split = split_name(index, args.eval_ratio, args.eval_split)
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        transformed_anns = []
        for ann in anns_by_image.get(int(image_info["id"]), []):
            kpts = transform_keypoints(ann["keypoints"], circle, cfg, crop_y)
            bbox_points = transform_bbox_samples(ann["bbox"], circle, cfg, crop_y)
            transformed_anns.append((ann, bbox_points, kpts))

        for slice_img, info in zip(slices, slice_infos):
            slice_name = f"{stem}_s{info.index}.png"
            label_name = f"{stem}_s{info.index}.txt"
            image_out = out_dir / "images" / split / slice_name
            label_out = out_dir / "labels" / split / label_name
            slice_width = info.width if slice_img is None else slice_img.shape[1]
            if not args.labels_only:
                cv2.imwrite(str(image_out), slice_img)

            labels_for_debug: list[tuple[list[float], list[tuple[float, float, int]]]] = []
            yolo_lines: list[str] = []
            for ann, bbox_points, kpts in transformed_anns:
                label = build_slice_label(
                    bbox_points,
                    kpts,
                    cfg.output_width,
                    slice_height,
                    info,
                    args.min_box_size,
                    args.min_keypoints,
                    args.min_keypoint_box_ratio,
                    args.bbox_source,
                    args.keypoint_bbox_padding,
                    args.keypoint_bbox_padding_y,
                )
                if label is None:
                    continue
                bbox_xyxy, local_kpts = label
                yolo_lines.append(yolo_pose_line(bbox_xyxy, local_kpts, slice_width, slice_height))
                labels_for_debug.append((bbox_xyxy, local_kpts))

                x1, y1, x2, y2 = bbox_xyxy
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": len(coco_images),
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                        "num_keypoints": sum(1 for _, _, v in local_kpts if v > 0),
                        "keypoints": [
                            value
                            for x, y, v in local_kpts
                            for value in (round(x, 3), round(y, 3), int(v))
                        ],
                    }
                )
                ann_id += 1

            label_out.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            counts["labels"] += len(yolo_lines)
            counts["slices"] += 1
            coco_images.append(
                {
                    "id": len(coco_images),
                    "file_name": str(Path("images") / split / slice_name),
                    "width": slice_width,
                    "height": slice_height,
                    "source_file_name": file_name,
                    "source_image_id": image_info["id"],
                    "slice_index": info.index,
                    "split": split,
                    "action": image_info.get("action"),
                    "performer": image_info.get("performer"),
                    "room": image_info.get("room"),
                }
            )

            if args.debug and slice_img is not None:
                debug = draw_pose_debug(slice_img, labels_for_debug)
                cv2.imwrite(str(out_dir / "debug" / f"{stem}_s{info.index}.jpg"), debug)

        counts["images"] += 1
        if counts["images"] % args.log_every == 0:
            print(f"Converted {counts['images']} source images, {counts['slices']} slices")

    annotations = {
        "info": {
            "description": "OmniLab unwrapped panorama slices for YOLO pose",
            "source": str(src_dir),
            "unwrap": cfg.__dict__,
            "circle": circle.__dict__,
        },
        "licenses": data.get("licenses", []),
        "categories": data.get("categories", []),
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (out_dir / "annotations_slices.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    write_dataset_yaml(out_dir, cfg, args.eval_split)

    print(
        "Done: "
        f"{counts['images']} source images, {counts['slices']} slices, {counts['labels']} pose labels -> {out_dir}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default="data/fine-tune_dataset/omnilab")
    parser.add_argument("--out-dir", default="data/fine-tune_dataset/omnilab_zhankai")
    parser.add_argument("--annotation", default="omnilab_v1.1_2023.02.05.json")
    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--cut-position", choices=["right", "left"], default="right")
    parser.add_argument(
        "--view-type",
        choices=["top", "bottom"],
        default="top",
        help="Ceiling fisheye unwrap view from maps/xiding_zhankai.py.",
    )
    parser.add_argument(
        "--crop-divisor",
        type=int,
        default=0,
        help="Top crop divisor after unwrap. 0 disables cropping; recommended for OmniLab.",
    )
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--slice-overlap", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument(
        "--circle-mode",
        choices=["center", "detect"],
        default="center",
        help="How to choose the fisheye circle when explicit center/radius are not provided.",
    )
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument("--min-keypoints", type=int, default=5)
    parser.add_argument(
        "--min-keypoint-box-ratio",
        type=float,
        default=0.03,
        help="Drop slice labels where visible keypoints occupy too little of the bbox.",
    )
    parser.add_argument(
        "--bbox-source",
        choices=["keypoints", "projected-bbox"],
        default="keypoints",
        help="Generate bbox from visible keypoints by default; projected-bbox keeps the old projected COCO bbox behavior.",
    )
    parser.add_argument("--keypoint-bbox-padding", type=float, default=0.25)
    parser.add_argument("--keypoint-bbox-padding-y", type=float, default=0.30)
    parser.add_argument("--labels-only", action="store_true", help="Rewrite labels/metadata without rewriting images.")
    parser.add_argument("--no-panorama", action="store_true", help="Do not save unwrapped panorama images.")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    convert_dataset(parse_args())
