#!/usr/bin/env python3
"""Fill YOLO-pose labels for an existing image dataset using a pose model."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FINE_TUNE_ROOT = ROOT / "fine-tune"
sys.path.insert(0, str(FINE_TUNE_ROOT))


SKELETON = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fine-tune/datasets/small_meeting_xiding_autolabel")
    parser.add_argument("--model", default="fine-tune/models/yolo26n-pose.pt")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--kpt-conf", type=float, default=0.25)
    parser.add_argument("--min-visible-kpts", type=int, default=5)
    parser.add_argument("--device", default=0)
    parser.add_argument("--preview-dir", default="autolabel_preview")
    parser.add_argument("--preview-limit", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def yolo_pose_line(box_xyxy: np.ndarray, keypoints: np.ndarray, width: int, height: int, kpt_conf_thr: float) -> tuple[str | None, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 <= x1 or y2 <= y1:
        return None, 0

    vals = [0, ((x1 + x2) / 2.0) / width, ((y1 + y2) / 2.0) / height, (x2 - x1) / width, (y2 - y1) / height]
    visible = 0
    for x, y, conf in keypoints:
        if conf >= kpt_conf_thr and 0 <= x < width and 0 <= y < height:
            vals.extend([x / width, y / height, 2])
            visible += 1
        else:
            vals.extend([0.0, 0.0, 0])
    if visible == 0:
        return None, 0
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals), visible


def draw_preview(image_path: Path, label_path: Path, out_path: Path) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        return
    h, w = image.shape[:2]
    text = label_path.read_text(encoding="utf-8").strip()
    rows = [[float(v) for v in line.split()] for line in text.splitlines()] if text else []
    for obj_idx, row in enumerate(rows):
        cx, cy, bw, bh = row[1:5]
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(image, f"person {obj_idx}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        kpts = []
        for i in range(17):
            x = int(round(row[5 + i * 3] * w))
            y = int(round(row[5 + i * 3 + 1] * h))
            v = int(row[5 + i * 3 + 2])
            kpts.append((x, y, v))
        for a, b in SKELETON:
            if kpts[a][2] > 0 and kpts[b][2] > 0:
                cv2.line(image, kpts[a][:2], kpts[b][:2], (0, 210, 0), 2, cv2.LINE_AA)
        for i, (x, y, v) in enumerate(kpts):
            if v > 0:
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
                if i in (0, 5, 6, 11, 12):
                    cv2.putText(image, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    if not rows:
        cv2.putText(image, "AUTO-LABEL EMPTY", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main() -> None:
    args = parse_args()
    dataset = (ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    model_path = (ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    rng = random.Random(args.seed)

    image_paths = []
    for split in args.splits:
        image_paths.extend(sorted((dataset / "images" / split).glob("*.png")))
    if not args.overwrite:
        image_paths = [p for p in image_paths if not label_for_image(p).exists()]

    preview_candidates = set(rng.sample(image_paths, min(args.preview_limit, len(image_paths))))
    preview_dir = dataset / args.preview_dir
    preview_dir.mkdir(parents=True, exist_ok=True)

    counts = {"images": 0, "positive_images": 0, "empty_images": 0, "objects": 0, "filtered_low_kpts": 0}
    manifest_rows = []
    for start in range(0, len(image_paths), args.batch):
        batch_paths = image_paths[start : start + args.batch]
        batch_images = [cv2.imread(str(p)) for p in batch_paths]
        if any(img is None for img in batch_images):
            bad = [str(p) for p, img in zip(batch_paths, batch_images) if img is None]
            raise FileNotFoundError(f"Could not read images: {bad[:3]}")
        results = model.predict(
            batch_images,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            device=args.device,
            half=True,
        )
        for image_path, image, result in zip(batch_paths, batch_images, results):
            h, w = image.shape[:2]
            lines = []
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                keypoints = result.keypoints.data.cpu().numpy()
                for box, cls_id, kpts in zip(boxes, cls_ids, keypoints):
                    if result.names.get(int(cls_id), str(cls_id)) != "person":
                        continue
                    line, visible = yolo_pose_line(box, kpts, w, h, args.kpt_conf)
                    if line is None or visible < args.min_visible_kpts:
                        counts["filtered_low_kpts"] += 1
                        continue
                    lines.append(line)

            label_path = label_for_image(image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            counts["images"] += 1
            counts["objects"] += len(lines)
            if lines:
                counts["positive_images"] += 1
            else:
                counts["empty_images"] += 1

            preview_path = ""
            if image_path in preview_candidates or lines:
                if len(list(preview_dir.glob("*.jpg"))) < args.preview_limit:
                    preview_path = preview_dir / f"{image_path.parent.name}_{image_path.stem}.jpg"
                    draw_preview(image_path, label_path, preview_path)
            manifest_rows.append(
                {
                    "image": str(image_path),
                    "label": str(label_path),
                    "split": image_path.parent.name,
                    "objects": len(lines),
                    "preview": str(preview_path),
                }
            )
        print(f"{min(start + args.batch, len(image_paths))}/{len(image_paths)} images done", flush=True)

    with (dataset / "autolabel_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "label", "split", "objects", "preview"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(counts, flush=True)
    print(f"preview: {preview_dir}", flush=True)


if __name__ == "__main__":
    main()
