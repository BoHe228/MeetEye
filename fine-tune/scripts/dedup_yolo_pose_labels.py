#!/usr/bin/env python3
"""Remove duplicate YOLO-pose labels in each image label file."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Obj:
    index: int
    line: str
    vals: list[float]
    box: tuple[float, float, float, float]
    area: float
    visible: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="fine-tune/datasets/small_meeting_xiding_autolabel")
    parser.add_argument("--iou-thresh", type=float, default=0.55)
    parser.add_argument("--mincov-thresh", type=float, default=0.85)
    parser.add_argument("--backup-suffix", default=".before_dedup")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_obj(index: int, line: str) -> Obj:
    vals = [float(v) for v in line.split()]
    cx, cy, w, h = vals[1:5]
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    visible = sum(1 for i in range(5, 56, 3) if vals[i + 2] > 0)
    return Obj(index=index, line=line, vals=vals, box=box, area=w * h, visible=visible)


def overlap(a: Obj, b: Obj) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = a.box
    bx1, by1, bx2, by2 = b.box
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    iou = inter / union if union > 0 else 0.0
    mincov = inter / min(a.area, b.area) if min(a.area, b.area) > 0 else 0.0
    return iou, mincov


def score(obj: Obj) -> tuple[int, float, float]:
    # Prefer more keypoints; for tied duplicates, prefer the tighter box.
    return (obj.visible, -obj.area, obj.vals[3] * obj.vals[4])


def dedup_objects(objects: list[Obj], iou_thresh: float, mincov_thresh: float) -> tuple[list[Obj], list[dict]]:
    suppressed: set[int] = set()
    removals = []
    for i in range(len(objects)):
        if i in suppressed:
            continue
        for j in range(i + 1, len(objects)):
            if j in suppressed:
                continue
            iou, mincov = overlap(objects[i], objects[j])
            if iou < iou_thresh and mincov < mincov_thresh:
                continue
            keep_i = score(objects[i]) >= score(objects[j])
            keep_idx, drop_idx = (i, j) if keep_i else (j, i)
            suppressed.add(drop_idx)
            removals.append(
                {
                    "kept_line": objects[keep_idx].index + 1,
                    "dropped_line": objects[drop_idx].index + 1,
                    "iou": f"{iou:.6f}",
                    "mincov": f"{mincov:.6f}",
                    "kept_visible": objects[keep_idx].visible,
                    "dropped_visible": objects[drop_idx].visible,
                    "kept_area": f"{objects[keep_idx].area:.6f}",
                    "dropped_area": f"{objects[drop_idx].area:.6f}",
                }
            )
            if i in suppressed:
                break
    kept = [obj for idx, obj in enumerate(objects) if idx not in suppressed]
    return kept, removals


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    label_files = sorted((dataset / "labels").glob("*/*.txt"))
    manifest_rows = []
    changed = 0
    removed = 0
    for label_path in label_files:
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) < 2:
            continue
        objects = [parse_obj(i, line) for i, line in enumerate(lines)]
        kept, removals = dedup_objects(objects, args.iou_thresh, args.mincov_thresh)
        if not removals:
            continue
        changed += 1
        removed += len(removals)
        for row in removals:
            row = {"label": str(label_path), "before": len(lines), "after": len(kept), **row}
            manifest_rows.append(row)
        if not args.dry_run:
            backup = label_path.with_name(label_path.name + args.backup_suffix)
            if not backup.exists():
                shutil.copy2(label_path, backup)
            label_path.write_text("\n".join(obj.line for obj in kept) + ("\n" if kept else ""), encoding="utf-8")

    manifest_path = dataset / ("dedup_manifest_dry_run.csv" if args.dry_run else "dedup_manifest.csv")
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "label",
            "before",
            "after",
            "kept_line",
            "dropped_line",
            "iou",
            "mincov",
            "kept_visible",
            "dropped_visible",
            "kept_area",
            "dropped_area",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"changed_files: {changed}")
    print(f"removed_objects: {removed}")
    print(f"manifest: {manifest_path}")
    print(f"dry_run: {args.dry_run}")


if __name__ == "__main__":
    main()
