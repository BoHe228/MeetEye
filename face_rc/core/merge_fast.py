from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class MergeFast:
    """ctypes wrapper for the native spatial merge fast path."""

    def __init__(self):
        self._root = Path(__file__).resolve().parents[2]
        self._lib_path = self._root / "face_rc" / "tools" / "bin" / "libmerge_fast.so"
        self._lib = None
        self._ensure_library()
        self._load_library()

    def _ensure_library(self) -> None:
        script = self._root / "face_rc" / "tools" / "build_merge_fast.sh"
        source = self._root / "face_rc" / "tools" / "merge_fast.cpp"
        if self._lib_path.exists() and (
            not source.exists() or self._lib_path.stat().st_mtime >= source.stat().st_mtime
        ):
            return
        if not script.exists():
            raise RuntimeError(f"merge fast build script not found: {script}")
        result = subprocess.run(
            ["bash", str(script)],
            cwd=str(self._root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("build libmerge_fast.so failed:\n" + result.stdout[-4000:])

    def _load_library(self) -> None:
        self._lib = ctypes.CDLL(str(self._lib_path))
        self._lib.face_merge_fast.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.face_merge_fast.restype = ctypes.c_int

    @staticmethod
    def _errbuf() -> ctypes.Array:
        return ctypes.create_string_buffer(1024)

    @staticmethod
    def _errmsg(buf: ctypes.Array) -> str:
        return buf.value.decode("utf-8", errors="replace")

    def merge(
        self,
        all_detections: List[List[Dict[str, Any]]],
        slice_infos: List[dict],
        overlap_ratio: float,
        iou_threshold: float,
        nms_iou_thresh: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        rows = []
        for slice_idx, detections in enumerate(all_detections):
            for det in detections:
                bbox = det.get("bbox")
                keypoints = det.get("keypoints", [])
                if bbox is None or len(bbox) < 4 or len(keypoints) < 5:
                    continue
                rows.append((slice_idx, det))

        n = len(rows)
        if n == 0:
            return [], {"raw": 0, "nms_kept": 0, "kept": 0}

        boxes = np.empty((n, 4), dtype=np.float32)
        scores = np.empty((n,), dtype=np.float32)
        labels = np.empty((n,), dtype=np.int32)
        keypoints = np.empty((n, 15), dtype=np.float32)
        slice_indices = np.empty((n,), dtype=np.int32)
        for idx, (slice_idx, det) in enumerate(rows):
            boxes[idx] = np.asarray(det["bbox"], dtype=np.float32)
            scores[idx] = float(det.get("confidence", 0.0))
            labels[idx] = int(det.get("class_id", 0))
            keypoints[idx] = np.asarray(det["keypoints"], dtype=np.float32).reshape(5, 3).reshape(15)
            slice_indices[idx] = int(slice_idx)

        start_x = np.asarray([float(info["start_x"]) for info in slice_infos], dtype=np.float32)
        wrap = np.asarray([1 if info.get("wrap_around") else 0 for info in slice_infos], dtype=np.int32)
        original_width = float(slice_infos[0]["original_width"])

        out_boxes = np.empty((n, 4), dtype=np.float32)
        out_scores = np.empty((n,), dtype=np.float32)
        out_labels = np.empty((n,), dtype=np.int32)
        out_keypoints = np.empty((n, 15), dtype=np.float32)
        out_count = np.zeros((1,), dtype=np.int32)
        stats = np.zeros((3,), dtype=np.int32)
        timings = np.zeros((5,), dtype=np.float64)
        err = self._errbuf()

        ret = self._lib.face_merge_fast(
            boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            keypoints.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            slice_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            n,
            start_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            wrap.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(slice_infos),
            ctypes.c_float(original_width),
            ctypes.c_float(float(overlap_ratio)),
            ctypes.c_float(float(iou_threshold)),
            ctypes.c_float(float(nms_iou_thresh)),
            out_boxes.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_labels.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            out_keypoints.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            timings.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            err,
            len(err),
        )
        if ret != 0:
            raise RuntimeError(self._errmsg(err) or f"face_merge_fast failed: {ret}")

        count = int(out_count[0])
        merged = []
        for idx in range(count):
            label = int(out_labels[idx])
            merged.append({
                "bbox": out_boxes[idx].tolist(),
                "confidence": float(out_scores[idx]),
                "class_id": label,
                "class_name": "face" if label == 0 else str(label),
                "feature": None,
                "keypoints": out_keypoints[idx].reshape(5, 3).tolist(),
            })

        profile = {
            "extract": 0.0,
            "features": 0.0,
            "coords": float(timings[0]),
            "dedup": float(timings[1]),
            "nms": float(timings[2]),
            "final_dedup": float(timings[3]),
            "build": float(timings[4]),
            "raw": int(stats[0]),
            "nms_kept": int(stats[1]),
            "kept": int(stats[2]),
        }
        return merged, profile
