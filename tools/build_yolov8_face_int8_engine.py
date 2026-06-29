#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import tensorrt as trt
import torch


DEFAULT_CALIB_SOURCES = [
    "data/大会议室_6.4_多人开会_40秒.mp4",
    "data/小会议室_3人_3min.mp4",
    "Wide-Angle_test/data/广角_小会议室_6.15.mp4",
]


def letterbox_bgr(image: np.ndarray, size: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = size - new_w
    pad_h = size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )


def preprocess_bgr(image: np.ndarray, size: int) -> np.ndarray:
    image = letterbox_bgr(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(image)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def existing_sources(paths: Sequence[str]) -> List[Path]:
    found = [Path(p) for p in paths if Path(p).exists()]
    if not found:
        raise FileNotFoundError("no calibration source exists")
    return found


def read_dataset(dataset_path: Path) -> List[Path]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"calibration dataset not found: {dataset_path}")
    paths: List[Path] = []
    for raw in dataset_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path = Path(line)
        if not path.exists() and not path.is_absolute():
            candidate = dataset_path.parent / path
            if candidate.exists():
                path = candidate
        if not path.exists():
            raise FileNotFoundError(f"dataset image not found: {line}")
        if not is_image(path):
            raise ValueError(f"dataset entry must be an image: {line}")
        paths.append(path)
    if not paths:
        raise RuntimeError(f"empty calibration dataset: {dataset_path}")
    return paths


def allocate_counts(n_sources: int, total: int) -> List[int]:
    base = total // n_sources
    extra = total % n_sources
    return [base + (1 if i < extra else 0) for i in range(n_sources)]


def build_samples(sources: Sequence[Path], total_images: int) -> List[Tuple[Path, int]]:
    counts = allocate_counts(len(sources), total_images)
    samples: List[Tuple[Path, int]] = []
    for source, count in zip(sources, counts):
        if count <= 0:
            continue
        if is_image(source):
            samples.extend((source, -1) for _ in range(count))
            continue
        if not is_video(source):
            continue

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"[calib] skip unreadable video: {source}")
            continue
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if frame_count <= 0:
            print(f"[calib] skip empty video: {source}")
            continue

        if count == 1:
            indices = [frame_count // 2]
        else:
            indices = np.linspace(0, frame_count - 1, count, dtype=np.int64).tolist()
        samples.extend((source, int(idx)) for idx in indices)

    if not samples:
        raise RuntimeError("no usable calibration samples")
    return samples


class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        batch_size: int,
        image_size: int,
        cache_file: Path,
    ) -> None:
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.samples = list(samples)
        self.batch_size = int(batch_size)
        self.image_size = int(image_size)
        self.cache_file = Path(cache_file)
        self.index = 0
        self.device_batch = None
        self._cap_path: Path | None = None
        self._cap = None
        print(
            f"[calib] samples={len(self.samples)} "
            f"batch={self.batch_size} image_size={self.image_size}"
        )

    def get_batch_size(self) -> int:
        return self.batch_size

    def _read_image(self, source: Path, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image: {source}")
            return image

        if self._cap is None or self._cap_path != source:
            if self._cap is not None:
                self._cap.release()
            self._cap = cv2.VideoCapture(str(source))
            self._cap_path = source
            if not self._cap.isOpened():
                raise RuntimeError(f"failed to open video: {source}")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"failed to read frame {frame_index}: {source}")
        return frame

    def get_batch(self, names: Iterable[str]):
        del names
        if self.index >= len(self.samples):
            return None

        end = min(self.index + self.batch_size, len(self.samples))
        if end - self.index < self.batch_size:
            return None

        batch = np.empty(
            (self.batch_size, 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        for row, (source, frame_index) in enumerate(self.samples[self.index:end]):
            batch[row] = preprocess_bgr(self._read_image(source, frame_index), self.image_size)

        self.index = end
        self.device_batch = torch.from_numpy(batch).cuda(non_blocking=False)
        return [int(self.device_batch.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            data = self.cache_file.read_bytes()
            print(f"[calib] use cache: {self.cache_file}")
            return data
        return None

    def write_calibration_cache(self, cache) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(bytes(cache))
        print(f"[calib] wrote cache: {self.cache_file}")

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def parse_onnx_shape(onnx_path: Path) -> Tuple[int, int, int, int]:
    import onnx

    model = onnx.load(str(onnx_path))
    if len(model.graph.input) != 1:
        raise RuntimeError("expected a single ONNX input")
    dims = model.graph.input[0].type.tensor_type.shape.dim
    shape = [d.dim_value for d in dims]
    if len(shape) != 4 or any(v <= 0 for v in shape):
        raise RuntimeError(f"expected static NCHW input shape, got {shape}")
    return tuple(int(v) for v in shape)  # type: ignore[return-value]


def read_engine_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        meta_len = int.from_bytes(f.read(4), byteorder="little")
        if meta_len <= 0 or meta_len > 1_000_000:
            return {}
        try:
            return json.loads(f.read(meta_len).decode("utf-8"))
        except Exception:
            return {}


def write_ultralytics_engine(path: Path, engine_bytes: bytes, metadata: dict) -> None:
    meta_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.write_bytes(len(meta_bytes).to_bytes(4, byteorder="little") + meta_bytes + engine_bytes)


def build_engine(args: argparse.Namespace) -> None:
    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    cache_path = Path(args.cache)
    batch, channels, height, width = parse_onnx_shape(onnx_path)
    if channels != 3 or height != width:
        raise RuntimeError(f"expected NCHW with square image, got {(batch, channels, height, width)}")

    dataset_path = Path(args.calib_dataset) if args.calib_dataset else None
    if dataset_path is not None:
        images = read_dataset(dataset_path)
        requested = len(images) if args.calib_images is None or args.calib_images <= 0 else int(args.calib_images)
        total_images = min(len(images), requested)
        total_images = (total_images // batch) * batch
        if total_images <= 0:
            raise RuntimeError(f"dataset has fewer than one full batch: {dataset_path}")
        samples = [(p, -1) for p in images[:total_images]]
        print(f"[calib] dataset={dataset_path} images={len(images)} used={len(samples)}")
    else:
        sources = existing_sources(args.calib_source or DEFAULT_CALIB_SOURCES)
        requested = 300 if args.calib_images is None else int(args.calib_images)
        total_images = max(batch, requested)
        total_images = int(math.ceil(total_images / batch) * batch)
        samples = build_samples(sources, total_images)
        samples = samples[: total_images]

    if not torch.cuda.is_available():
        raise RuntimeError("torch CUDA is not available; TensorRT INT8 calibration needs GPU access")

    if args.force_recalibrate and cache_path.exists():
        cache_path.unlink()
        print(f"[calib] removed old cache: {cache_path}")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"failed to parse ONNX:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.INT8)
    if args.fp16_fallback:
        config.set_flag(trt.BuilderFlag.FP16)

    calibrator = TorchEntropyCalibrator(samples, batch, height, cache_path)
    config.int8_calibrator = calibrator

    print(f"[build] onnx:   {onnx_path}")
    print(f"[build] engine: {engine_path}")
    print(f"[build] input:  batch={batch} channels={channels} size={height}")
    print(f"[build] flags:  INT8{' + FP16 fallback' if args.fp16_fallback else ''}")
    print("[build] building TensorRT engine...")
    serialized = builder.build_serialized_network(network, config)
    calibrator.close()
    if serialized is None:
        raise RuntimeError("TensorRT build_serialized_network returned None")

    metadata = read_engine_metadata(Path(args.metadata_source))
    if not metadata:
        metadata = {
            "description": "Ultralytics YOLOv8n-face INT8 TensorRT engine",
            "author": "Ultralytics",
            "version": "8.4.37",
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "stride": 32,
            "task": "pose",
            "batch": batch,
            "imgsz": [height, width],
            "names": {"0": "face"},
            "channels": channels,
            "end2end": False,
            "kpt_shape": [5, 3],
        }
    metadata["batch"] = batch
    metadata["imgsz"] = [height, width]
    metadata["task"] = "pose"
    metadata["names"] = {"0": "face"}
    metadata["channels"] = channels
    metadata["end2end"] = False
    metadata["kpt_shape"] = [5, 3]
    metadata["args"] = dict(metadata.get("args") or {})
    metadata["args"].update({
        "batch": batch,
        "half": bool(args.fp16_fallback),
        "int8": True,
        "dynamic": False,
        "nms": False,
    })
    if dataset_path is not None:
        metadata["calibration_dataset"] = str(dataset_path)
        metadata["calibration_images"] = len(samples)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    write_ultralytics_engine(engine_path, bytes(serialized), metadata)
    print(f"[build] wrote: {engine_path} ({engine_path.stat().st_size / 1024 / 1024:.2f} MiB)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build INT8 TensorRT engine for yolov8n-face ONNX.")
    parser.add_argument("--onnx", default="yolo_model/yolov8n-face.onnx")
    parser.add_argument("--engine", default="yolo_model/yolov8n-face_int8.engine")
    parser.add_argument("--cache", default="yolo_model/yolov8n-face_int8.calib")
    parser.add_argument("--metadata-source", default="yolo_model/yolov8n-face.engine")
    parser.add_argument("--calib-dataset", default=None,
                        help="dataset.txt containing runtime slice images for calibration.")
    parser.add_argument("--calib-source", action="append", default=[],
                        help="Image/video calibration source. Can be passed multiple times.")
    parser.add_argument("--calib-images", type=int, default=None,
                        help="Number of calibration images. Default: 300 for video sources, "
                             "all images for --calib-dataset. Use 0 with --calib-dataset for all.")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--fp16-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-recalibrate", action="store_true",
                        help="Delete the calibration cache before building.")
    return parser.parse_args()


if __name__ == "__main__":
    build_engine(parse_args())
