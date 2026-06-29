#!/usr/bin/env python3
"""Create an auto-labeled YOLO-pose dataset from ceiling-fisheye meeting videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FINE_TUNE_ROOT = ROOT / "fine-tune"
sys.path.insert(0, str(FINE_TUNE_ROOT))


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

FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


@dataclass(frozen=True)
class VideoJob:
    path: Path
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--videos",
        nargs="+",
        default=[
            "data/小会议室_吸顶_非真实会议场景.avi",
            "data/小会议室_吸顶_真实会议_退出会议.mp4",
            "data/小会议室_吸顶_真实会议_3min短视频.mp4",
            "data/小会议室_吸顶_摄像头摇晃_1min.mp4",
        ],
    )
    parser.add_argument("--map-file", default="maps/xiding_maps_bottom.npz")
    parser.add_argument("--model", default="fine-tune/models/yolo26n-pose.pt")
    parser.add_argument("--out-dir", default="fine-tune/datasets/small_meeting_xiding_autolabel")
    parser.add_argument("--fps", type=float, default=1.0, help="Frame sampling rate per second.")
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260624)
    parser.add_argument("--num-slices", type=int, default=3)
    parser.add_argument("--slice-overlap", type=float, default=0.1)
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO auto-label confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-visible-kpts", type=int, default=5)
    parser.add_argument("--preview-limit", type=int, default=160)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-autolabel", action="store_true", help="Only unwrap/slice frames and create empty labels.")
    parser.add_argument("--device", default=0)
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    keep = []
    for ch in path.stem:
        if ch.isalnum() or ch in ("_", "-"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") + "_" + digest


def load_maps(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    meta = {}
    for key in data.files:
        if key in {"map_x", "map_y"}:
            continue
        value = data[key]
        if getattr(value, "shape", ()) == ():
            meta[key] = value.item()
        elif key in {"center"}:
            meta[key] = value.tolist()
        else:
            meta[key] = str(value)
    return data["map_x"], data["map_y"], meta


def sample_indices(frame_count: int, source_fps: float, target_fps: float) -> list[int]:
    if frame_count <= 0:
        return []
    if source_fps <= 0 or target_fps <= 0:
        step = max(1, int(round(source_fps or 30)))
        return list(range(0, frame_count, step))
    duration = frame_count / source_fps
    count = max(1, int(duration * target_fps))
    indices = [min(frame_count - 1, int(round(i * source_fps / target_fps))) for i in range(count)]
    return sorted(set(indices))


def split_source_frames(keys: list[str], eval_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    keys = sorted(keys)
    rng.shuffle(keys)
    eval_count = int(round(len(keys) * eval_ratio))
    eval_set = set(keys[:eval_count])
    return {key: ("test" if key in eval_set else "train") for key in keys}


def slice_panorama(panorama: np.ndarray, num_slices: int, overlap_ratio: float) -> list[np.ndarray]:
    height, width = panorama.shape[:2]
    slice_width = width // num_slices
    overlap_width = int(slice_width * overlap_ratio)
    slices = []
    for i in range(num_slices):
        start_x = i * slice_width - overlap_width
        end_x = (i + 1) * slice_width + overlap_width
        if i == 0 and start_x < 0:
            slices.append(np.concatenate([panorama[:, start_x:], panorama[:, :end_x]], axis=1))
        elif i == num_slices - 1 and end_x > width:
            slices.append(np.concatenate([panorama[:, start_x:width], panorama[:, : end_x - width]], axis=1))
        else:
            slices.append(panorama[:, start_x:end_x])
    return slices


def yolo_pose_line(box_xyxy: np.ndarray, keypoints: np.ndarray, width: int, height: int, kpt_conf_thr: float) -> str | None:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy[:4]]
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 <= x1 or y2 <= y1:
        return None

    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    vals = [0, cx, cy, bw, bh]

    visible = 0
    for x, y, conf in keypoints:
        if conf >= kpt_conf_thr and 0 <= x < width and 0 <= y < height:
            vals.extend([x / width, y / height, 2])
            visible += 1
        else:
            vals.extend([0.0, 0.0, 0])
    if visible == 0:
        return None
    return " ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in vals)


def draw_preview(image: np.ndarray, rows: list[list[float]], out_path: Path) -> None:
    preview = image.copy()
    h, w = preview.shape[:2]
    for obj_idx, row in enumerate(rows):
        cx, cy, bw, bh = row[1:5]
        x1 = int(round((cx - bw / 2) * w))
        y1 = int(round((cy - bh / 2) * h))
        x2 = int(round((cx + bw / 2) * w))
        y2 = int(round((cy + bh / 2) * h))
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(preview, f"person {obj_idx}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        kpts = []
        for i in range(17):
            x = int(round(row[5 + i * 3] * w))
            y = int(round(row[5 + i * 3 + 1] * h))
            v = int(row[5 + i * 3 + 2])
            kpts.append((x, y, v))
        for a, b in SKELETON:
            if kpts[a][2] > 0 and kpts[b][2] > 0:
                cv2.line(preview, kpts[a][:2], kpts[b][:2], (0, 210, 0), 2, cv2.LINE_AA)
        for i, (x, y, v) in enumerate(kpts):
            if v > 0:
                cv2.circle(preview, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
                if i in (0, 5, 6, 11, 12):
                    cv2.putText(preview, str(i), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    if not rows:
        cv2.putText(preview, "AUTO-LABEL EMPTY", (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), preview)


def write_yaml(out_dir: Path) -> None:
    text = f"""path: {out_dir.resolve()}
train: images/train
val: images/test
test: images/test

kpt_shape: [17, 3]
flip_idx: {FLIP_IDX}

names:
  0: person

kpt_names:
  0:
"""
    text += "".join(f"    - {name}\n" for name in COCO_KEYPOINT_NAMES)
    text += """
# Auto-labeled small meeting ceiling-fisheye dataset.
# Images are xiding unwrapped panorama slices. Labels are YOLO-pose prelabels
# and should be manually checked before high-trust fine-tuning.
"""
    (out_dir / "small_meeting_xiding_autolabel.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "preview").mkdir(parents=True, exist_ok=True)

    map_x, map_y, map_meta = load_maps(ROOT / args.map_file)
    videos = [VideoJob((ROOT / v).resolve() if not Path(v).is_absolute() else Path(v), safe_stem(Path(v))) for v in args.videos]
    source_keys = []
    video_infos = []
    for job in videos:
        cap = cv2.VideoCapture(str(job.path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {job.path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        indices = sample_indices(frame_count, fps, args.fps)
        cap.release()
        video_infos.append({"job": job, "fps": fps, "frame_count": frame_count, "width": width, "height": height, "indices": indices})
        source_keys.extend(f"{job.stem}_f{idx:06d}" for idx in indices)

    split_map = split_source_frames(source_keys, args.eval_ratio, args.seed)

    model_path = (ROOT / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    model = None
    if not args.no_autolabel:
        from ultralytics import YOLO

        print(f"Loading auto-label model: {model_path}", flush=True)
        model = YOLO(str(model_path))

    manifest_rows = []
    counts = {
        "source_frames": 0,
        "slices": 0,
        "positive_slices": 0,
        "empty_slices": 0,
        "labels": 0,
        "filtered_low_kpts": 0,
    }

    for info in video_infos:
        job = info["job"]
        cap = cv2.VideoCapture(str(job.path))
        print(f"Processing {job.path.name}: {len(info['indices'])} sampled frames", flush=True)
        for sample_idx, frame_idx in enumerate(info["indices"], 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"  warning: could not read frame {frame_idx}", flush=True)
                continue

            key = f"{job.stem}_f{frame_idx:06d}"
            split = split_map[key]
            panorama = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            slices = slice_panorama(panorama, num_slices=args.num_slices, overlap_ratio=args.slice_overlap)
            if model is not None:
                results = model.predict(
                    slices,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    iou=args.iou,
                    verbose=False,
                    device=args.device,
                    half=True,
                )
            else:
                results = [None] * len(slices)

            counts["source_frames"] += 1
            for slice_idx, (slice_img, result) in enumerate(zip(slices, results)):
                image_name = f"{key}_s{slice_idx}.png"
                label_name = f"{key}_s{slice_idx}.txt"
                image_out = out_dir / "images" / split / image_name
                label_out = out_dir / "labels" / split / label_name
                cv2.imwrite(str(image_out), slice_img)

                h, w = slice_img.shape[:2]
                lines = []
                rows_for_preview = []
                if result is not None and result.boxes is not None and result.keypoints is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
                    keypoints = result.keypoints.data.cpu().numpy()
                    for box, conf, cls_id, kpts in zip(boxes, confs, cls_ids, keypoints):
                        class_name = result.names.get(int(cls_id), str(cls_id))
                        if class_name != "person":
                            continue
                        visible = int((kpts[:, 2] >= args.conf).sum())
                        if visible < args.min_visible_kpts:
                            counts["filtered_low_kpts"] += 1
                            continue
                        line = yolo_pose_line(box, kpts, w, h, args.conf)
                        if line is None:
                            continue
                        lines.append(line)
                        rows_for_preview.append([float(v) for v in line.split()])

                label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                if lines:
                    counts["positive_slices"] += 1
                    counts["labels"] += len(lines)
                else:
                    counts["empty_slices"] += 1
                counts["slices"] += 1

                if len(manifest_rows) < args.preview_limit or lines:
                    if len(manifest_rows) < args.preview_limit:
                        preview_out = out_dir / "preview" / f"{split}_{image_name.replace('.png', '.jpg')}"
                        draw_preview(slice_img, rows_for_preview, preview_out)
                    else:
                        preview_out = ""
                else:
                    preview_out = ""

                manifest_rows.append(
                    {
                        "video": str(job.path),
                        "source_stem": job.stem,
                        "frame_index": frame_idx,
                        "time_sec": frame_idx / info["fps"] if info["fps"] else "",
                        "split": split,
                        "slice_index": slice_idx,
                        "image": str(image_out),
                        "label": str(label_out),
                        "objects": len(lines),
                        "preview": str(preview_out),
                    }
                )

            if sample_idx % 25 == 0 or sample_idx == len(info["indices"]):
                print(f"  {sample_idx}/{len(info['indices'])} frames done", flush=True)
        cap.release()

    write_yaml(out_dir)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "source_stem", "frame_index", "time_sec", "split", "slice_index", "image", "label", "objects", "preview"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "videos": [
            {
                "path": str(info["job"].path),
                "fps": info["fps"],
                "frame_count": info["frame_count"],
                "width": info["width"],
                "height": info["height"],
                "sampled_frames": len(info["indices"]),
            }
            for info in video_infos
        ],
        "map_file": str((ROOT / args.map_file).resolve()),
        "map_meta": map_meta,
        "model": str(model_path),
        "fps": args.fps,
        "num_slices": args.num_slices,
        "slice_overlap": args.slice_overlap,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "min_visible_kpts": args.min_visible_kpts,
        "autolabel": not args.no_autolabel,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "counts": counts,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2), flush=True)
    print(f"dataset: {out_dir}", flush=True)
    print(f"yaml: {out_dir / 'small_meeting_xiding_autolabel.yaml'}", flush=True)


if __name__ == "__main__":
    main()
