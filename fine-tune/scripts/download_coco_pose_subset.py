#!/usr/bin/env python3
"""Download only the COCO-Pose images needed by the mixed fine-tune dataset."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


COCO_URL_ROOT = "http://images.cocodataset.org"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omnilab", default="fine-tune/datasets/omnilab_zhankai")
    parser.add_argument("--posefes", default="fine-tune/datasets/posefes_zhankai")
    parser.add_argument("--coco", default="fine-tune/datasets/coco-pose")
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--include-backgrounds", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def label_for_image(path: Path) -> Path:
    parts = list(path.parts)
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def fisheye_images(root: Path, split: str, include_backgrounds: bool) -> list[Path]:
    paths = sorted((root / "images" / split).glob("*.png"))
    if include_backgrounds:
        return paths
    return [p for p in paths if label_for_image(p).read_text(encoding="utf-8").strip()]


def coco_labeled_paths(coco_root: Path, list_name: str) -> list[Path]:
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
        if label_for_image(p).exists():
            paths.append(p)
    return sorted(paths)


def download_one(path: Path, coco_root: Path, retries: int, timeout: float) -> tuple[Path, bool, str]:
    path = path.resolve()
    coco_root = coco_root.resolve()
    if path.exists() and path.stat().st_size > 0:
        return path, True, "exists"

    rel = path.relative_to(coco_root).as_posix()
    if rel.startswith("images/"):
        rel = rel[len("images/") :]
    url = f"{COCO_URL_ROOT}/{rel}"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MeetEye fine-tune downloader"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response")
            tmp.write_bytes(data)
            tmp.replace(path)
            return path, True, "downloaded"
        except (OSError, urllib.error.URLError, RuntimeError) as exc:
            if attempt == retries:
                return path, False, str(exc)
            time.sleep(min(2.0 * attempt, 8.0))
    return path, False, "unreachable"


def write_subset(path: Path, paths: list[Path]) -> None:
    text = "\n".join(str(p.resolve()) for p in paths)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    omnilab = Path(args.omnilab)
    posefes = Path(args.posefes)
    coco = Path(args.coco)

    fish_train = fisheye_images(omnilab, "train", args.include_backgrounds)
    fish_train += fisheye_images(posefes, "train", args.include_backgrounds)
    fish_test = fisheye_images(omnilab, "test", args.include_backgrounds)
    fish_test += fisheye_images(posefes, "test", args.include_backgrounds)

    coco_train_all = coco_labeled_paths(coco, "train2017.txt")
    coco_val_all = coco_labeled_paths(coco, "val2017.txt")

    coco_val_used = min(len(coco_val_all), len(fish_test))
    train_holdout_needed = len(fish_test) - coco_val_used
    coco_train_needed = len(fish_train) + train_holdout_needed
    if len(coco_train_all) < coco_train_needed:
        raise RuntimeError(f"COCO train has {len(coco_train_all)} labeled images, need {coco_train_needed}")

    selected_train = sorted(rng.sample(coco_train_all, coco_train_needed))
    selected_val = sorted(rng.sample(coco_val_all, coco_val_used))

    subset_dir = coco / "subsets"
    write_subset(subset_dir / "mixed_pose_640_train2017_download.txt", selected_train)
    write_subset(subset_dir / "mixed_pose_640_val2017_used.txt", selected_val)

    existing_val = sum(1 for p in selected_val if p.exists() and p.stat().st_size > 0)
    targets = [p for p in selected_train + selected_val if not (p.exists() and p.stat().st_size > 0)]

    print(f"fish_train: {len(fish_train)}")
    print(f"fish_test: {len(fish_test)}")
    print(f"coco_train_needed: {coco_train_needed}")
    print(f"coco_val_used: {coco_val_used}")
    print(f"coco_train_holdout_for_test: {train_holdout_needed}")
    print(f"selected_val_existing: {existing_val}/{len(selected_val)}")
    print(f"images_to_download: {len(targets)}")

    ok = 0
    failed: list[tuple[Path, str]] = []
    failed_path = subset_dir / "mixed_pose_640_download_failed.txt"
    if failed_path.exists():
        failed_path.unlink()

    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(download_one, p, coco, args.retries, args.timeout): p for p in targets
        }
        for i, future in enumerate(futures.as_completed(future_map), start=1):
            path, success, message = future.result()
            if success:
                ok += 1
            else:
                failed.append((path, message))
            if i == 1 or i % 200 == 0 or i == len(targets):
                print(f"progress: {i}/{len(targets)} done, ok={ok}, failed={len(failed)}")

    if failed:
        failed_path.write_text(
            "".join(f"{p.resolve()}\t{msg}\n" for p, msg in failed),
            encoding="utf-8",
        )
        raise RuntimeError(f"{len(failed)} downloads failed; see {failed_path}")

    print("Done")


if __name__ == "__main__":
    main()
