#!/usr/bin/env python3
"""Convert PoseFES ceiling-fisheye pose data to unwrapped YOLO pose slices."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2

from prepare_omnilab_zhankai import (
    COCO_KEYPOINT_NAMES,
    FLIP_IDX,
    FisheyeCircle,
    UnwrapConfig,
    build_slice_label,
    create_unwrap_maps,
    crop_top,
    detect_fisheye_region,
    draw_pose_debug,
    make_slice_infos,
    slice_panorama,
    split_name,
    transform_bbox_samples,
    transform_keypoints,
    unwrap_image,
    yolo_pose_line,
)


@dataclass(frozen=True)
class PoseFESItem:
    scenario: str
    index: int
    image_info: dict
    annotations: list[dict]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", default="data/fine-tune_dataset/PoseFES")
    parser.add_argument("--out-dir", default="fine-tune/datasets/posefes_zhankai")
    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--cut-position", choices=["right", "left"], default="right")
    parser.add_argument("--view-type", choices=["top", "bottom"], default="bottom")
    parser.add_argument("--crop-divisor", type=int, default=0)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--slice-overlap", type=float, default=0.1)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--eval-split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--center-x", type=float, default=None)
    parser.add_argument("--center-y", type=float, default=None)
    parser.add_argument("--radius", type=float, default=None)
    parser.add_argument("--circle-mode", choices=["center", "detect"], default="center")
    parser.add_argument("--min-box-size", type=float, default=4.0)
    parser.add_argument("--min-keypoints", type=int, default=7)
    parser.add_argument("--min-keypoint-box-ratio", type=float, default=0.03)
    parser.add_argument("--bbox-source", choices=["keypoints", "projected-bbox"], default="keypoints")
    parser.add_argument("--keypoint-bbox-padding", type=float, default=0.25)
    parser.add_argument("--keypoint-bbox-padding-y", type=float, default=0.30)
    parser.add_argument("--no-panorama", action="store_true")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_posefes(src_dir: Path, limit: int | None) -> list[PoseFESItem]:
    ann_dir = src_dir / "coco_annotations_final_corrected_2022"
    items: list[PoseFESItem] = []

    for scenario in ("scenario1", "scenario2"):
        json_path = ann_dir / f"person_keypoints_{scenario}.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        anns_by_image: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

        images = sorted(data["images"], key=lambda image: image["file_name"])
        for image_info in images:
            items.append(
                PoseFESItem(
                    scenario=scenario,
                    index=len(items),
                    image_info=image_info,
                    annotations=anns_by_image.get(int(image_info["id"]), []),
                )
            )

    if limit is not None:
        return items[:limit]
    return items


def iter_items(items: list[PoseFESItem]) -> Iterable[tuple[int, PoseFESItem]]:
    for index, item in enumerate(items):
        yield index, item


def source_image_path(src_dir: Path, item: PoseFESItem) -> Path:
    return src_dir / item.scenario / "JPEGImages" / item.image_info["file_name"]


def output_stem(item: PoseFESItem) -> str:
    return f"{item.scenario}_{Path(item.image_info['file_name']).stem}"


def prepare_output(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)

    for split in ("train", "test", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "panorama").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(out_dir: Path, cfg: UnwrapConfig, eval_split: str) -> None:
    yaml_text = f"""path: {out_dir.resolve()}
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
# Generated from PoseFES ceiling fisheye images.
# unwrap: {cfg.output_width}x{cfg.output_height}, view_type={cfg.view_type}, cut_position={cfg.cut_position}
# crop_divisor: {cfg.crop_divisor}
# slices: {cfg.num_slices}, overlap={cfg.slice_overlap}
"""
    (out_dir / "posefes_zhankai.yaml").write_text(yaml_text, encoding="utf-8")


def choose_circle(first_image, args: argparse.Namespace) -> FisheyeCircle:
    if args.center_x is not None and args.center_y is not None and args.radius is not None:
        return FisheyeCircle(float(args.center_x), float(args.center_y), float(args.radius))
    if args.circle_mode == "center":
        h, w = first_image.shape[:2]
        return FisheyeCircle(float(w // 2), float(h // 2), float(min(w, h) // 2))
    return detect_fisheye_region(first_image)


def convert_dataset(args: argparse.Namespace) -> None:
    src_dir = Path(args.src_dir)
    out_dir = Path(args.out_dir)
    items = load_posefes(src_dir, args.limit)
    if not items:
        raise RuntimeError(f"No PoseFES images found under {src_dir}")

    prepare_output(out_dir, args.overwrite)

    first_image = cv2.imread(str(source_image_path(src_dir, items[0])))
    if first_image is None:
        raise FileNotFoundError(source_image_path(src_dir, items[0]))

    circle = choose_circle(first_image, args)
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
        f"slices={cfg.num_slices}x{slice_infos[0].width}x{slice_height}, "
        f"min_keypoints={args.min_keypoints}"
    )

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    ann_id = 0
    counts = {
        "source_images": 0,
        "slices": 0,
        "labels": 0,
        "source_annotations": 0,
        "filtered_annotations": 0,
    }

    for index, item in iter_items(items):
        img_path = source_image_path(src_dir, item)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip unreadable image: {img_path}")
            continue

        panorama = unwrap_image(img, map_x, map_y)
        stem = output_stem(item)
        if not args.no_panorama:
            cv2.imwrite(str(out_dir / "panorama" / f"{stem}.png"), panorama)
        panorama_cropped, crop_y = crop_top(panorama, cfg)
        slices = slice_panorama(panorama_cropped, slice_infos)
        split = split_name(index, args.eval_ratio, args.eval_split)

        transformed_anns = []
        for ann in item.annotations:
            kpts = transform_keypoints(ann["keypoints"], circle, cfg, crop_y)
            bbox_points = transform_bbox_samples(ann["bbox"], circle, cfg, crop_y)
            transformed_anns.append((ann, bbox_points, kpts))

        counts["source_annotations"] += len(item.annotations)

        for slice_img, info in zip(slices, slice_infos):
            slice_name = f"{stem}_s{info.index}.png"
            label_name = f"{stem}_s{info.index}.txt"
            image_out = out_dir / "images" / split / slice_name
            label_out = out_dir / "labels" / split / label_name
            cv2.imwrite(str(image_out), slice_img)

            labels_for_debug: list[tuple[list[float], list[tuple[float, float, int]]]] = []
            yolo_lines: list[str] = []
            filtered_on_slice = 0
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
                    filtered_on_slice += 1
                    continue

                bbox_xyxy, local_kpts = label
                yolo_lines.append(yolo_pose_line(bbox_xyxy, local_kpts, slice_img.shape[1], slice_height))
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
                        "source_annotation_id": ann.get("id"),
                    }
                )
                ann_id += 1

            label_out.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            counts["labels"] += len(yolo_lines)
            counts["filtered_annotations"] += filtered_on_slice
            counts["slices"] += 1
            coco_images.append(
                {
                    "id": len(coco_images),
                    "file_name": str(Path("images") / split / slice_name),
                    "width": slice_img.shape[1],
                    "height": slice_height,
                    "source_file_name": item.image_info["file_name"],
                    "source_image_id": item.image_info["id"],
                    "scenario": item.scenario,
                    "slice_index": info.index,
                    "split": split,
                }
            )

            if args.debug:
                debug = draw_pose_debug(slice_img, labels_for_debug)
                cv2.imwrite(str(out_dir / "debug" / f"{stem}_s{info.index}.jpg"), debug)

        counts["source_images"] += 1
        if counts["source_images"] % args.log_every == 0:
            print(f"Converted {counts['source_images']} source images, {counts['slices']} slices")

    annotations = {
        "info": {
            "description": "PoseFES unwrapped panorama slices for YOLO pose",
            "source": str(src_dir),
            "unwrap": cfg.__dict__,
            "circle": circle.__dict__,
            "min_keypoints": args.min_keypoints,
            "bbox_source": args.bbox_source,
        },
        "licenses": [],
        "categories": [
            {
                "supercategory": "person",
                "id": 1,
                "name": "person",
                "keypoints": COCO_KEYPOINT_NAMES,
            }
        ],
        "images": coco_images,
        "annotations": coco_annotations,
    }
    (out_dir / "annotations_slices.json").write_text(json.dumps(annotations, indent=2), encoding="utf-8")
    write_dataset_yaml(out_dir, cfg, args.eval_split)

    print(
        "Done: "
        f"{counts['source_images']} source images, {counts['slices']} slices, "
        f"{counts['labels']} pose labels -> {out_dir}"
    )


if __name__ == "__main__":
    convert_dataset(parse_args())
