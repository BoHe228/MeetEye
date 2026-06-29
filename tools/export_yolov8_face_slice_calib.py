#!/usr/bin/env python3
"""Export runtime panorama slices for YOLOv8-face INT8 calibration.

The server inference path feeds YOLO with panorama slices, not raw fisheye
frames. This script mirrors that pre-YOLO path and writes slice images plus a
dataset.txt that TensorRT calibration can consume.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_SOURCE_MAPS = [
    (
        "data/大会议室_6.4_多人开会_40秒.mp4",
        "maps/3840_fisheye_maps_6.4.npz",
    ),
    (
        "data/小会议室_3人_3min.mp4",
        "maps/3840_fisheye_maps_小会议室+办公室.npz",
    ),
    (
        "Wide-Angle_test/data/广角_小会议室_6.15.mp4",
        "maps/3840_fisheye_maps_6.10.npz",
    ),
]


def existing_source_maps(sources: Sequence[str], maps: Sequence[str]) -> List[tuple[Path, Path]]:
    if sources:
        if maps and len(maps) not in {1, len(sources)}:
            raise ValueError("--source-map must be passed once or the same number of times as --source")
        if maps:
            mapped = list(maps) if len(maps) == len(sources) else [maps[0]] * len(sources)
        else:
            mapped = []
            for source in sources:
                match = next((m for s, m in DEFAULT_SOURCE_MAPS if Path(s) == Path(source)), None)
                if match is None:
                    raise ValueError(f"no map known for source; pass --source-map: {source}")
                mapped.append(match)
        pairs = [(Path(source), Path(map_file)) for source, map_file in zip(sources, mapped)]
    else:
        pairs = [(Path(source), Path(map_file)) for source, map_file in DEFAULT_SOURCE_MAPS]

    existing = [(source, map_file) for source, map_file in pairs if source.exists() and map_file.exists()]
    if not existing:
        raise FileNotFoundError("no calibration video source exists")
    return existing


def allocate_counts(n_sources: int, total: int) -> List[int]:
    base = total // n_sources
    extra = total % n_sources
    return [base + (1 if i < extra else 0) for i in range(n_sources)]


def sample_indices(frame_count: int, count: int, skip_start: int, skip_end: int) -> List[int]:
    start = min(max(0, skip_start), max(0, frame_count - 1))
    end = max(start, frame_count - 1 - max(0, skip_end))
    if count <= 1:
        return [(start + end) // 2]
    return np.linspace(start, end, count, dtype=np.int64).astype(int).tolist()


def safe_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def write_image(path: Path, image: np.ndarray, ext: str, jpg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ext.lower() in {"jpg", "jpeg"}:
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
    elif ext.lower() == "png":
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        raise ValueError(f"unsupported image extension: {ext}")
    if not ok:
        raise RuntimeError(f"failed to write image: {path}")


def load_panorama_maps(map_file: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(map_file)
    if not path.exists():
        raise FileNotFoundError(f"map file not found: {path}")
    data = np.load(str(path), allow_pickle=True)
    return data["map_x"].astype(np.float32), data["map_y"].astype(np.float32)


def slice_panorama(
    panorama: np.ndarray,
    num_slices: int,
    overlap_ratio: float,
) -> List[np.ndarray]:
    slices: List[np.ndarray] = []
    height, width = panorama.shape[:2]
    del height
    slice_width = width // num_slices
    overlap_width = int(slice_width * overlap_ratio)

    for i in range(num_slices):
        start_x = i * slice_width - overlap_width
        end_x = (i + 1) * slice_width + overlap_width

        if i == 0 and start_x < 0:
            slices.append(np.concatenate([panorama[:, start_x:], panorama[:, :end_x]], axis=1))
        elif i == num_slices - 1 and end_x > width:
            slices.append(np.concatenate([panorama[:, start_x:width], panorama[:, :end_x - width]], axis=1))
        else:
            slices.append(panorama[:, start_x:end_x])
    return slices


def iter_video_frames(source: Path, indices: Iterable[int]):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {source}")
    try:
        for frame_idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"[warn] skip unreadable frame {frame_idx}: {source}")
                continue
            yield int(frame_idx), frame
    finally:
        cap.release()


def export_dataset(args: argparse.Namespace) -> Path:
    source_maps = existing_source_maps(args.source, args.source_map)
    frame_counts = allocate_counts(len(source_maps), args.total_frames)
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "images"
    dataset_path = out_dir / args.dataset_name
    ext = args.image_ext.lower().lstrip(".")
    dataset_entries: List[str] = []
    saved = 0

    print("[export] runtime slice calibration dataset")
    print(f"[export] output: {out_dir}")
    print(
        f"[export] panorama={args.output_width}x{args.output_height} "
        f"crop_divisor={args.crop_divisor} slices={args.num_slices} "
        f"overlap={args.slice_overlap}"
    )

    for source_idx, ((source, map_file), count) in enumerate(zip(source_maps, frame_counts)):
        map_x, map_y = load_panorama_maps(map_file)
        print(f"[export] source{source_idx} map: {map_file}", flush=True)
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"[warn] skip unreadable video: {source}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if frame_count <= 0 or count <= 0:
            print(f"[warn] skip empty video: {source}")
            continue

        indices = sample_indices(frame_count, count, args.skip_start, args.skip_end)
        print(f"[export] source{source_idx}: {source} frames={frame_count} samples={len(indices)}", flush=True)

        for frame_idx, frame in iter_video_frames(source, indices):
            panorama = cv2.remap(
                frame,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
            if args.crop_divisor > 0:
                crop_height = panorama.shape[0] // args.crop_divisor
                panorama = panorama[crop_height:, :]

            slices = slice_panorama(panorama, args.num_slices, args.slice_overlap)
            for slice_idx, slice_img in enumerate(slices):
                name = f"src{source_idx:02d}_f{frame_idx:06d}_s{slice_idx}.{ext}"
                path = image_dir / name
                write_image(path, slice_img, ext, args.jpg_quality)
                dataset_entries.append(safe_rel(path))
                saved += 1

            if saved and saved % max(1, args.log_interval) == 0:
                print(f"[export] saved slices={saved}", flush=True)

    if not dataset_entries:
        raise RuntimeError("no calibration slices were exported")

    usable = (len(dataset_entries) // args.batch) * args.batch
    if usable != len(dataset_entries):
        print(
            f"[export] trim dataset entries {len(dataset_entries)} -> {usable} "
            f"to align batch={args.batch}"
        )
        dataset_entries = dataset_entries[:usable]

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text("\n".join(dataset_entries) + "\n", encoding="utf-8")
    print(f"[export] wrote dataset: {dataset_path} images={len(dataset_entries)}", flush=True)
    return dataset_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export server runtime panorama slices for TensorRT INT8 calibration."
    )
    parser.add_argument("--source", action="append", default=[], help="Calibration video source.")
    parser.add_argument("--source-map", action="append", default=[],
                        help="Panorama map for --source. Pass once for all sources or once per source.")
    parser.add_argument("--output-dir", default="yolo_model/int8_calib_slices_864")
    parser.add_argument("--dataset-name", default="dataset.txt")
    parser.add_argument("--total-frames", type=int, default=300,
                        help="Total source frames to sample before slicing.")
    parser.add_argument("--batch", type=int, default=3,
                        help="Trim dataset size to a multiple of this batch.")
    parser.add_argument("--output-width", type=int, default=3840)
    parser.add_argument("--output-height", type=int, default=1080)
    parser.add_argument("--vertical-fov", type=float, default=100.0)
    parser.add_argument("--map-file", default=None,
                        help="Deprecated alias for --source-map when only one map is needed.")
    parser.add_argument("--cam-index", type=int, default=1)
    parser.add_argument("--crop-divisor", type=int, default=3)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--slice-overlap", type=float, default=0.1)
    parser.add_argument("--skip-start", type=int, default=0)
    parser.add_argument("--skip-end", type=int, default=0)
    parser.add_argument("--image-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--log-interval", type=int, default=300)
    args = parser.parse_args()
    if args.map_file:
        args.source_map.append(args.map_file)
    if args.total_frames <= 0:
        raise ValueError("--total-frames must be positive")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if args.num_slices <= 0:
        raise ValueError("--num-slices must be positive")
    return args


if __name__ == "__main__":
    export_dataset(parse_args())
