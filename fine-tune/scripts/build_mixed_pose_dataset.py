#!/usr/bin/env python3
"""Build a sampled mixed YOLO-pose dataset list from fisheye slices and COCO-Pose."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


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

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="fine-tune/datasets/mixed_pose_640")
    parser.add_argument("--omnilab", default="fine-tune/datasets/omnilab_zhankai")
    parser.add_argument("--posefes", default="fine-tune/datasets/posefes_zhankai")
    parser.add_argument("--coco", default="fine-tune/datasets/coco-pose")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--include-backgrounds", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-coco-label", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def image_paths(root: Path, split: str) -> list[Path]:
    return sorted((root / "images" / split).glob("*.png"))


def label_for_image(path: Path) -> Path:
    parts = list(path.parts)
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def filter_backgrounds(paths: list[Path], include_backgrounds: bool) -> list[Path]:
    if include_backgrounds:
        return paths
    return [p for p in paths if label_for_image(p).read_text(encoding="utf-8").strip()]


def load_coco_list(coco_root: Path, list_name: str, require_label: bool) -> list[Path]:
    list_path = coco_root / list_name
    if not list_path.exists():
        raise FileNotFoundError(list_path)

    paths = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if not p.is_absolute():
            p = coco_root / p
        if not p.exists():
            continue
        if require_label and not label_for_image(p).exists():
            continue
        paths.append(p)
    return sorted(paths)


def load_optional_subset(path: Path, require_label: bool) -> list[Path] | None:
    if not path.exists():
        return None

    paths = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        if not p.exists():
            continue
        if require_label and not label_for_image(p).exists():
            continue
        paths.append(p)
    return sorted(paths)


def sample_exact(paths: list[Path], count: int, rng: random.Random, name: str) -> list[Path]:
    if len(paths) < count:
        raise RuntimeError(f"{name} has {len(paths)} available images, need {count}")
    return sorted(rng.sample(paths, count))


def write_list(path: Path, paths: list[Path]) -> None:
    text = "\n".join(str(p.resolve()) for p in paths)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def write_yaml(out_dir: Path) -> None:
    yaml_text = f"""path: {out_dir.resolve()}
train: train.txt
val: test.txt
test: test.txt

kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}

names:
  0: person

kpt_names:
  0:
"""
    yaml_text += "".join(f"    - {name}\n" for name in COCO_KEYPOINT_NAMES)
    yaml_text += """
# Mixed dataset for 640-imgsz fine-tuning.
# Fisheye samples are already unwrapped and sliced.
# COCO samples are original images and are resized by the dataloader at train time.
"""
    (out_dir / "mixed_pose_640.yaml").write_text(yaml_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    omnilab = Path(args.omnilab)
    posefes = Path(args.posefes)
    coco = Path(args.coco)

    fish_train = filter_backgrounds(image_paths(omnilab, "train"), args.include_backgrounds)
    fish_train += filter_backgrounds(image_paths(posefes, "train"), args.include_backgrounds)
    fish_test = filter_backgrounds(image_paths(omnilab, "test"), args.include_backgrounds)
    fish_test += filter_backgrounds(image_paths(posefes, "test"), args.include_backgrounds)

    subset_dir = coco / "subsets"
    coco_train_all = load_optional_subset(
        subset_dir / "mixed_pose_640_train2017_download.txt",
        args.require_coco_label,
    )
    if coco_train_all is None:
        coco_train_all = load_coco_list(coco, "train2017.txt", args.require_coco_label)

    coco_val_all = load_optional_subset(
        subset_dir / "mixed_pose_640_val2017_used.txt",
        args.require_coco_label,
    )
    if coco_val_all is None:
        coco_val_all = load_coco_list(coco, "val2017.txt", args.require_coco_label)

    coco_test_from_val_count = min(len(coco_val_all), len(fish_test))
    coco_test = sample_exact(coco_val_all, coco_test_from_val_count, rng, "COCO val")

    coco_train_pool = list(coco_train_all)
    coco_test_holdout = []
    test_shortage = len(fish_test) - len(coco_test)
    if test_shortage > 0:
        # COCO-Pose val is smaller than the fisheye validation set. Hold out the
        # extra COCO test samples from train2017, then remove them from train.
        coco_test_holdout = sample_exact(
            coco_train_pool,
            test_shortage,
            rng,
            "COCO train holdout for test",
        )
        holdout_set = set(coco_test_holdout)
        coco_train_pool = [p for p in coco_train_pool if p not in holdout_set]
        coco_test += coco_test_holdout

    coco_train = sample_exact(coco_train_pool, len(fish_train), rng, "COCO train")

    train = sorted(fish_train) + coco_train
    test = sorted(fish_test) + coco_test
    rng.shuffle(train)
    rng.shuffle(test)

    write_list(out_dir / "train.txt", train)
    write_list(out_dir / "test.txt", test)
    write_yaml(out_dir)

    summary = {
        "fish_train": len(fish_train),
        "fish_test": len(fish_test),
        "coco_train": len(coco_train),
        "coco_test": len(coco_test),
        "coco_test_from_val": coco_test_from_val_count,
        "coco_test_from_train_holdout": len(coco_test_holdout),
        "coco_train_available_before_holdout": len(coco_train_all),
        "coco_train_available_after_holdout": len(coco_train_pool),
        "coco_val_available": len(coco_val_all),
        "total_train": len(train),
        "total_test": len(test),
        "include_backgrounds": args.include_backgrounds,
        "seed": args.seed,
    }
    lines = [f"{k}: {v}" for k, v in summary.items()]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for line in lines:
        print(line)
    print(f"yaml: {out_dir / 'mixed_pose_640.yaml'}")


if __name__ == "__main__":
    main()
