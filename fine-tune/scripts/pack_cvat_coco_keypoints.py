#!/usr/bin/env python3
"""Pack split COCO Keypoints annotations into CVAT import ZIP files."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        default="fine-tune/datasets/small_meeting_xiding_cvat",
        help="Dataset root with <split>/images and <split>/annotations_coco_keypoints.json.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--prefix", default=None, help="Output ZIP name prefix. Defaults to source directory name.")
    parser.add_argument("--compress-level", type=int, default=6)
    return parser.parse_args()


def pack_split(src_root: Path, split: str, prefix: str, compress_level: int) -> Path:
    split_dir = src_root / split
    annotation_path = split_dir / "annotations_coco_keypoints.json"
    image_dir = split_dir / "images"
    if not annotation_path.exists():
        raise FileNotFoundError(annotation_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    for image in data.get("images", []):
        name = Path(image["file_name"]).name
        image["file_name"] = name

    out_zip = src_root / f"{prefix}_{split}_coco_keypoints.zip"
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compress_level) as zf:
        zf.writestr(
            f"annotations/person_keypoints_{split}.json",
            json.dumps(data, ensure_ascii=False, indent=2),
        )
        for image in data.get("images", []):
            name = Path(image["file_name"]).name
            rel_path = f"images/{split}/{name}"
            source_image = image_dir / name
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            zf.write(source_image, rel_path)

    return out_zip


def main() -> None:
    args = parse_args()
    src_root = Path(args.src)
    prefix = args.prefix or src_root.name
    for split in args.splits:
        out_zip = pack_split(src_root, split, prefix, args.compress_level)
        print(f"{split}: {out_zip} ({out_zip.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
