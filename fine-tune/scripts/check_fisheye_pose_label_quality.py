#!/usr/bin/env python3
"""Rank and visualize suspicious fisheye YOLO-pose labels before training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2


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


@dataclass(frozen=True)
class LabelMetric:
    dataset: str
    split: str
    image_path: Path
    label_path: Path
    line_index: int
    visible: int
    box_area: float
    box_w: float
    box_h: float
    kpt_w: float
    kpt_h: float
    kpt_box_ratio: float
    expand_w: float
    expand_h: float
    center_offset: float
    row: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "fine-tune/datasets/omnilab_zhankai",
            "fine-tune/datasets/posefes_zhankai",
        ],
    )
    parser.add_argument("--out-dir", default="fine-tune/label_check_samples/pretrain_fisheye_quality_check")
    parser.add_argument("--per-group", type=int, default=12)
    return parser.parse_args()


def label_to_image(label_path: Path) -> Path:
    parts = list(label_path.parts)
    parts[parts.index("labels")] = "images"
    return Path(*parts).with_suffix(".png")


def read_metrics(dataset: Path) -> list[LabelMetric]:
    metrics = []
    for label_path in sorted((dataset / "labels").glob("*/*.txt")):
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        image_path = label_to_image(label_path)
        split = label_path.parent.name
        for line_index, line in enumerate(text.splitlines(), 1):
            row = [float(v) for v in line.split()]
            if len(row) != 56:
                continue
            cx, cy, bw, bh = row[1:5]
            pts = []
            for i in range(17):
                x, y, v = row[5 + i * 3 : 8 + i * 3]
                if v > 0:
                    pts.append((x, y))
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            kpt_w = max(xs) - min(xs)
            kpt_h = max(ys) - min(ys)
            box_area = bw * bh
            kpt_box_ratio = (kpt_w * kpt_h / box_area) if box_area > 0 else 0.0
            expand_w = bw / max(kpt_w, 1e-6)
            expand_h = bh / max(kpt_h, 1e-6)
            kcx = (min(xs) + max(xs)) / 2
            kcy = (min(ys) + max(ys)) / 2
            center_offset = (((cx - kcx) / max(bw, 1e-6)) ** 2 + ((cy - kcy) / max(bh, 1e-6)) ** 2) ** 0.5
            metrics.append(
                LabelMetric(
                    dataset=dataset.name,
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    line_index=line_index,
                    visible=len(pts),
                    box_area=box_area,
                    box_w=bw,
                    box_h=bh,
                    kpt_w=kpt_w,
                    kpt_h=kpt_h,
                    kpt_box_ratio=kpt_box_ratio,
                    expand_w=expand_w,
                    expand_h=expand_h,
                    center_offset=center_offset,
                    row=row,
                )
            )
    return metrics


def yolo_to_pixels(row: list[float], width: int, height: int):
    cx, cy, bw, bh = row[1:5]
    bbox = (
        int(round((cx - bw / 2) * width)),
        int(round((cy - bh / 2) * height)),
        int(round((cx + bw / 2) * width)),
        int(round((cy + bh / 2) * height)),
    )
    keypoints = []
    for i in range(17):
        x, y, v = row[5 + i * 3 : 8 + i * 3]
        keypoints.append((int(round(x * width)), int(round(y * height)), int(v)))
    return bbox, keypoints


def draw_metric(metric: LabelMetric, out_path: Path) -> None:
    image = cv2.imread(str(metric.image_path))
    if image is None:
        raise FileNotFoundError(metric.image_path)
    height, width = image.shape[:2]
    bbox, keypoints = yolo_to_pixels(metric.row, width, height)
    x1, y1, x2, y2 = bbox

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
    for start, end in SKELETON:
        if keypoints[start][2] > 0 and keypoints[end][2] > 0:
            cv2.line(image, keypoints[start][:2], keypoints[end][:2], (0, 210, 0), 2, cv2.LINE_AA)
    for idx, (x, y, v) in enumerate(keypoints):
        if v <= 0:
            continue
        color = (0, 0, 255) if v == 2 else (255, 0, 255)
        cv2.circle(image, (x, y), 4, color, -1, cv2.LINE_AA)
        cv2.putText(image, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    lines = [
        f"{metric.dataset}/{metric.split}/{metric.label_path.stem} line {metric.line_index}",
        f"vis={metric.visible} area={metric.box_area:.4f} ratio={metric.kpt_box_ratio:.3f}",
        f"expand_w/h={metric.expand_w:.2f}/{metric.expand_h:.2f} center={metric.center_offset:.3f}",
    ]
    y = 26
    for text in lines:
        cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def select_groups(metrics: list[LabelMetric], per_group: int) -> dict[str, list[LabelMetric]]:
    groups = {
        "low_visible_large_box": sorted(
            [m for m in metrics if m.visible <= 7],
            key=lambda m: (-m.box_area, m.kpt_box_ratio),
        )[:per_group],
        "lowest_kpt_box_ratio": sorted(metrics, key=lambda m: (m.kpt_box_ratio, -m.box_area))[:per_group],
        "widest_box": sorted(metrics, key=lambda m: (-m.box_w, m.visible))[:per_group],
        "tallest_box": sorted(metrics, key=lambda m: (-m.box_h, m.visible))[:per_group],
        "largest_center_offset": sorted(metrics, key=lambda m: (-m.center_offset, -m.box_area))[:per_group],
    }
    return groups


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for dataset in args.datasets:
        all_metrics.extend(read_metrics(Path(dataset)))

    groups = select_groups(all_metrics, args.per_group)
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "rank",
                "dataset",
                "split",
                "label",
                "line",
                "image",
                "output",
                "visible",
                "box_area",
                "box_w",
                "box_h",
                "kpt_box_ratio",
                "expand_w",
                "expand_h",
                "center_offset",
            ]
        )
        for group_name, selected in groups.items():
            for rank, metric in enumerate(selected, 1):
                out_name = (
                    f"{group_name}_{rank:02d}_{metric.dataset}_{metric.split}_"
                    f"{metric.label_path.stem}_l{metric.line_index}.jpg"
                )
                out_path = out_dir / out_name
                draw_metric(metric, out_path)
                writer.writerow(
                    [
                        group_name,
                        rank,
                        metric.dataset,
                        metric.split,
                        metric.label_path,
                        metric.line_index,
                        metric.image_path,
                        out_path,
                        metric.visible,
                        f"{metric.box_area:.8f}",
                        f"{metric.box_w:.8f}",
                        f"{metric.box_h:.8f}",
                        f"{metric.kpt_box_ratio:.8f}",
                        f"{metric.expand_w:.8f}",
                        f"{metric.expand_h:.8f}",
                        f"{metric.center_offset:.8f}",
                    ]
                )

    print(f"metrics: {len(all_metrics)}")
    for group_name, selected in groups.items():
        print(f"{group_name}: {len(selected)}")
    print(f"out_dir: {out_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
