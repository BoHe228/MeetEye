#!/usr/bin/env python3
"""Convert a CVAT-exported COCO Keypoints split to a YOLO-pose dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


KEYPOINT_NAMES = [
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

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coco",
        default=(
            "fine-tune/datasets/small_meeting_xiding_cvat/export_test_coco_keypoints/"
            "annotations/person_keypoints_test.json"
        ),
        help="CVAT-exported COCO Keypoints JSON.",
    )
    parser.add_argument(
        "--images",
        default="fine-tune/datasets/small_meeting_xiding_cvat/test/images",
        help="Directory containing the image files referenced by the COCO JSON.",
    )
    parser.add_argument(
        "--out",
        default="fine-tune/datasets/small_meeting_xiding_cvat_test_yolo",
        help="Output YOLO-pose dataset directory.",
    )
    parser.add_argument("--split", default="test", help="Source split name to write first.")
    parser.add_argument(
        "--mirror-to-train-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also make train and val point at the same corrected test split.",
    )
    parser.add_argument("--copy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bbox-padding", type=float, default=0.10)
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def valid_bbox(bbox: list[float], width: int, height: int) -> list[float] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox]
    if w <= 0 or h <= 0:
        return None
    x1 = clamp(x, 0.0, float(width - 1))
    y1 = clamp(y, 0.0, float(height - 1))
    x2 = clamp(x + w, 0.0, float(width))
    y2 = clamp(y + h, 0.0, float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def keypoint_bbox(
    keypoints: list[tuple[float, float, int]], width: int, height: int, padding: float
) -> list[float] | None:
    visible = [(x, y) for x, y, v in keypoints if v > 0]
    if not visible:
        return None
    xs = [p[0] for p in visible]
    ys = [p[1] for p in visible]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw = max(x2 - x1, 2.0)
    bh = max(y2 - y1, 2.0)
    pad_x = max(bw * padding, 2.0)
    pad_y = max(bh * padding, 2.0)
    return [
        clamp(x1 - pad_x, 0.0, float(width - 1)),
        clamp(y1 - pad_y, 0.0, float(height - 1)),
        clamp(x2 + pad_x, 0.0, float(width)),
        clamp(y2 + pad_y, 0.0, float(height)),
    ]


def format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return f"{value:.6f}"


def yolo_pose_line(annotation: dict, image: dict, padding: float) -> str | None:
    width = int(image["width"])
    height = int(image["height"])
    raw_keypoints = annotation.get("keypoints", [])
    if len(raw_keypoints) != 51:
        return None

    keypoints: list[tuple[float, float, int]] = []
    for index in range(17):
        x, y, visibility = raw_keypoints[index * 3 : index * 3 + 3]
        visibility = int(visibility)
        if visibility <= 0:
            keypoints.append((0.0, 0.0, 0))
        else:
            keypoints.append(
                (
                    clamp(float(x), 0.0, float(width - 1)),
                    clamp(float(y), 0.0, float(height - 1)),
                    min(visibility, 2),
                )
            )

    bbox = valid_bbox(annotation.get("bbox", []), width, height)
    if bbox is None:
        bbox = keypoint_bbox(keypoints, width, height, padding)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    values: list[float | int] = [
        0,
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    ]
    for x, y, visibility in keypoints:
        if visibility <= 0:
            values.extend([0.0, 0.0, 0])
        else:
            values.extend([x / width, y / height, visibility])
    return " ".join(format_value(value) for value in values)


def write_yaml(out_dir: Path) -> None:
    yaml_text = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
test: images/test

kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}

names:
  0: person

kpt_names:
  0:
"""
    yaml_text += "".join(f"    - {name}\n" for name in KEYPOINT_NAMES)
    yaml_text += """
# Corrected CVAT test split converted from COCO Keypoints to YOLO-pose.
# train/val/test point to the same corrected image set by default, so this
# dataset is suitable for quick verification or overfit checks.
"""
    (out_dir / "small_meeting_xiding_cvat_test_yolo.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    coco_path = Path(args.coco)
    images_dir = Path(args.images)
    out_dir = Path(args.out)

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(coco_path.read_text(encoding="utf-8"))
    images_by_id = {image["id"]: image for image in data.get("images", [])}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        annotations_by_image[annotation["image_id"]].append(annotation)

    splits = [args.split]
    if args.mirror_to_train_val:
        splits = sorted(set([args.split, "train", "val", "test"]))

    counts = {"images": 0, "annotations": 0, "skipped_annotations": 0}
    label_lines_by_image: dict[int, list[str]] = {}
    for image_id, image in images_by_id.items():
        lines = []
        for annotation in annotations_by_image.get(image_id, []):
            line = yolo_pose_line(annotation, image, args.bbox_padding)
            if line is None:
                counts["skipped_annotations"] += 1
                continue
            lines.append(line)
        label_lines_by_image[image_id] = lines
        counts["annotations"] += len(lines)

    for split in splits:
        image_out_dir = out_dir / "images" / split
        label_out_dir = out_dir / "labels" / split
        image_out_dir.mkdir(parents=True, exist_ok=True)
        label_out_dir.mkdir(parents=True, exist_ok=True)

        for image_id, image in sorted(images_by_id.items()):
            name = Path(image["file_name"]).name
            source_image = images_dir / name
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            link_or_copy(source_image, image_out_dir / name, args.copy)
            (label_out_dir / f"{Path(name).stem}.txt").write_text(
                "\n".join(label_lines_by_image[image_id])
                + ("\n" if label_lines_by_image[image_id] else ""),
                encoding="utf-8",
            )

    counts["images"] = len(images_by_id)
    counts["mirrored_splits"] = len(splits)
    write_yaml(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"out: {out_dir}")
    print(f"yaml: {out_dir / 'small_meeting_xiding_cvat_test_yolo.yaml'}")


if __name__ == "__main__":
    main()
