#!/usr/bin/env python3
"""Convert a YOLO-pose dataset to CVAT-friendly COCO Keypoints tasks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from PIL import Image


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

SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 3],
    [1, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
    [5, 7],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="fine-tune/datasets/small_meeting_xiding_autolabel")
    parser.add_argument("--out", default="fine-tune/datasets/small_meeting_xiding_cvat")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--copy", action=argparse.BooleanOptionalAction, default=False, help="Copy images instead of hard-linking.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def yolo_to_coco_annotation(line: str, ann_id: int, image_id: int, width: int, height: int) -> dict | None:
    vals = [float(v) for v in line.split()]
    if len(vals) != 56:
        return None

    cx, cy, bw, bh = vals[1:5]
    x = (cx - bw / 2) * width
    y = (cy - bh / 2) * height
    w = bw * width
    h = bh * height
    if w <= 0 or h <= 0:
        return None

    keypoints = []
    visible = 0
    for i in range(17):
        kx = vals[5 + i * 3] * width
        ky = vals[5 + i * 3 + 1] * height
        kv = int(vals[5 + i * 3 + 2])
        if kv > 0:
            visible += 1
        keypoints.extend([round(kx, 3), round(ky, 3), kv])

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
        "area": round(w * h, 3),
        "iscrowd": 0,
        "num_keypoints": visible,
        "keypoints": keypoints,
    }


def convert_split(src_root: Path, out_root: Path, split: str, copy: bool) -> dict:
    image_src_dir = src_root / "images" / split
    label_src_dir = src_root / "labels" / split
    image_out_dir = out_root / split / "images"
    annotation_out = out_root / split / "annotations_coco_keypoints.json"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    ann_id = 1
    for image_id, src_image in enumerate(sorted(image_src_dir.glob("*.png")), 1):
        dst_image = image_out_dir / src_image.name
        link_or_copy(src_image, dst_image, copy)
        with Image.open(src_image) as im:
            width, height = im.size
        images.append(
            {
                "id": image_id,
                "file_name": f"images/{src_image.name}",
                "width": width,
                "height": height,
            }
        )

        label_path = label_src_dir / f"{src_image.stem}.txt"
        if not label_path.exists():
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for line in text.splitlines():
            ann = yolo_to_coco_annotation(line, ann_id, image_id, width, height)
            if ann is None:
                continue
            annotations.append(ann)
            ann_id += 1

    coco = {
        "info": {"description": f"{src_root.name} {split} converted from YOLO-pose for CVAT"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": KEYPOINT_NAMES,
                "skeleton": SKELETON,
            }
        ],
    }
    annotation_out.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"split": split, "images": len(images), "annotations": len(annotations), "annotation_file": str(annotation_out)}


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    out_root = Path(args.out)
    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = [convert_split(src_root, out_root, split, args.copy) for split in args.splits]
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in summary:
        print(f"{item['split']}: images={item['images']} annotations={item['annotations']} file={item['annotation_file']}")
    print(f"out: {out_root}")


if __name__ == "__main__":
    main()
