#!/usr/bin/env python3
"""Benchmark one RKNN input on one RKNNLite instance.

This measures only the RKNNLite inference wall time for a single YOLO input
bound to a selected NPU core mask. It intentionally skips panorama remap,
slicing, YOLO decode/postprocess, tracking, and JSON/WebUI work.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def find_rknn_file(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_file() and path.suffix.lower() == ".rknn":
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    files = sorted(path.glob("*.rknn"))
    if not files:
        raise FileNotFoundError(f"no .rknn file found in: {model_path}")
    return files[0]


def read_metadata_int(model_path: str, key: str, default: int) -> int:
    meta = Path(model_path) / "metadata.yaml"
    try:
        lines = meta.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return default
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}:"):
            value = stripped.split(":", 1)[1].strip()
            if value.isdigit():
                return int(value)
            for next_line in lines[idx + 1:idx + 4]:
                next_value = next_line.strip()
                if next_value.startswith("-"):
                    item = next_value[1:].strip()
                    if item.isdigit():
                        return int(item)
                elif next_value and not next_value.startswith("#"):
                    break
    return default


def resolve_core_mask(RKNNLite, name: str) -> Tuple[Optional[int], str]:
    mapping = {
        "auto": "NPU_CORE_AUTO",
        "core0": "NPU_CORE_0",
        "core1": "NPU_CORE_1",
        "core2": "NPU_CORE_2",
        "core01": "NPU_CORE_0_1",
        "core012": "NPU_CORE_0_1_2",
        "all": "NPU_CORE_ALL",
    }
    attr = mapping.get(str(name).lower())
    if not attr:
        return None, "default"
    return getattr(RKNNLite, attr, None), attr


def letterbox_bgr(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    gain = min(size / float(h), size / float(w))
    new_w, new_h = int(round(w * gain)), int(round(h * gain))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = size - new_w
    pad_h = size - new_h
    left = int(round(pad_w / 2.0 - 0.1))
    right = int(round(pad_w / 2.0 + 0.1))
    top = int(round(pad_h / 2.0 - 0.1))
    bottom = int(round(pad_h / 2.0 + 0.1))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded


def load_input(args: argparse.Namespace, size: int) -> np.ndarray:
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {args.image}")
        image = letterbox_bgr(image, size)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {args.video}")
        if args.frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, image = cap.read()
        cap.release()
        if not ok or image is None:
            raise RuntimeError(f"cannot read frame {args.frame} from: {args.video}")
        image = letterbox_bgr(image, size)
    elif args.random:
        rng = np.random.default_rng(args.seed)
        image = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    else:
        image = np.full((size, size, 3), 114, dtype=np.uint8)

    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if args.input_format == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(image[None, ...].astype(np.uint8))


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark single-input RKNNLite inference on RK3588 NPU."
    )
    parser.add_argument(
        "--model-path",
        default="face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model",
        help="RKNN file or *_rknn_model directory",
    )
    parser.add_argument("--imgsz", type=int, default=0, help="input size; 0 reads metadata or uses 608")
    parser.add_argument(
        "--core-mask",
        default="all",
        choices=["default", "auto", "core0", "core1", "core2", "core01", "core012", "all"],
        help="RKNNLite runtime core mask",
    )
    parser.add_argument("--loops", type=int, default=300, help="timed inference loops")
    parser.add_argument("--warmup", type=int, default=30, help="warmup inference loops")
    parser.add_argument("--image", default=None, help="optional image path")
    parser.add_argument("--video", default=None, help="optional video path")
    parser.add_argument("--frame", type=int, default=0, help="video frame index")
    parser.add_argument("--random", action="store_true", help="use random input instead of gray image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-format",
        default="rgb",
        choices=["rgb", "bgr"],
        help="format sent to RKNN; current runtime normally uses RGB",
    )
    args = parser.parse_args()

    if args.loops <= 0:
        raise ValueError("--loops must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    from rknnlite.api import RKNNLite

    rknn_file = find_rknn_file(args.model_path)
    imgsz = int(args.imgsz) if args.imgsz > 0 else read_metadata_int(args.model_path, "imgsz", 608)
    batch = read_metadata_int(args.model_path, "batch", 1)
    if batch != 1:
        print(f"[warn] metadata batch={batch}; this benchmark sends one input")

    rknn = RKNNLite()
    ret = rknn.load_rknn(str(rknn_file))
    print(f"load_rknn ret={ret}: {rknn_file}")
    if ret != 0:
        raise RuntimeError(f"load_rknn failed: {ret}")

    core_mask, core_mask_name = resolve_core_mask(RKNNLite, args.core_mask)
    if core_mask is None:
        ret = rknn.init_runtime()
        core_mask_name = "default"
    else:
        ret = rknn.init_runtime(core_mask=core_mask)
    print(f"init_runtime ret={ret}, core_mask={core_mask_name}")
    if ret != 0:
        raise RuntimeError(f"init_runtime failed: {ret}")

    input_data = load_input(args, imgsz)
    print(
        f"input shape={input_data.shape} dtype={input_data.dtype} "
        f"min={int(input_data.min())} max={int(input_data.max())}"
    )
    print(f"warmup={args.warmup}, loops={args.loops}")

    try:
        for _ in range(args.warmup):
            outputs = rknn.inference(
                inputs=[input_data],
                data_type=["uint8"],
                data_format=["nhwc"],
            )
            if outputs is None:
                raise RuntimeError("RKNN inference returned None during warmup")

        times_ms = np.empty(args.loops, dtype=np.float64)
        t_total0 = time.perf_counter()
        for i in range(args.loops):
            t0 = time.perf_counter()
            outputs = rknn.inference(
                inputs=[input_data],
                data_type=["uint8"],
                data_format=["nhwc"],
            )
            t1 = time.perf_counter()
            if outputs is None:
                raise RuntimeError(f"RKNN inference returned None at loop {i}")
            times_ms[i] = (t1 - t0) * 1000.0
        total_s = time.perf_counter() - t_total0
    finally:
        rknn.release()

    mean_ms = float(times_ms.mean())
    fps_mean_latency = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    fps_wall = args.loops / total_s if total_s > 0 else 0.0
    print("")
    print("===== Single RKNN Input Benchmark =====")
    print(f"model:       {rknn_file}")
    print(f"core_mask:   {core_mask_name}")
    print(f"imgsz:       {imgsz}")
    print(f"loops:       {args.loops}")
    print(f"mean:        {mean_ms:.3f} ms  ({fps_mean_latency:.2f} FPS by mean latency)")
    print(f"median:      {percentile(times_ms, 50):.3f} ms")
    print(f"p90 / p95:   {percentile(times_ms, 90):.3f} / {percentile(times_ms, 95):.3f} ms")
    print(f"min / max:   {float(times_ms.min()):.3f} / {float(times_ms.max()):.3f} ms")
    print(f"wall FPS:    {fps_wall:.2f}")


if __name__ == "__main__":
    main()
