from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np


class ParallelRknnCAPI:
    """ctypes wrapper for the native three-core RKNN C API backend."""

    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        self._root = Path(__file__).resolve().parents[2]
        self._lib_path = self._root / "face_rc" / "tools" / "bin" / "librknn_capi_parallel.so"
        self._handle = ctypes.c_void_p()
        self._lib = None
        self.input_shape: Tuple[int, int, int] = (0, 0, 0)
        self.output_shape: Tuple[int, int] = (0, 0)
        self.has_decoded_interface = False
        self.has_merged_interface = False

        self._ensure_library()
        self._load_library()
        self._create()

    def _ensure_library(self) -> None:
        script = self._root / "face_rc" / "tools" / "build_rknn_capi_parallel.sh"
        source = self._root / "face_rc" / "tools" / "rknn_capi_parallel.cpp"
        if self._lib_path.exists() and (
            not source.exists() or self._lib_path.stat().st_mtime >= source.stat().st_mtime
        ):
            return
        if not script.exists():
            raise RuntimeError(f"C API build script not found: {script}")
        result = subprocess.run(
            ["bash", str(script)],
            cwd=str(self._root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "build librknn_capi_parallel.so failed:\n"
                + result.stdout[-4000:]
            )

    def _load_library(self) -> None:
        self._lib = ctypes.CDLL(str(self._lib_path))
        self._lib.face_rknn_parallel_create.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.face_rknn_parallel_create.restype = ctypes.c_int
        self._lib.face_rknn_parallel_destroy.argtypes = [ctypes.c_void_p]
        self._lib.face_rknn_parallel_destroy.restype = None
        self._lib.face_rknn_parallel_get_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self._lib.face_rknn_parallel_get_shape.restype = ctypes.c_int
        self._lib.face_rknn_parallel_infer.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_char_p,
            ctypes.c_int,
        ]
        self._lib.face_rknn_parallel_infer.restype = ctypes.c_int
        self.has_decoded_interface = hasattr(self._lib, "face_rknn_parallel_infer_decoded")
        if self.has_decoded_interface:
            self._lib.face_rknn_parallel_infer_decoded.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            self._lib.face_rknn_parallel_infer_decoded.restype = ctypes.c_int
        self.has_merged_interface = hasattr(self._lib, "face_rknn_parallel_infer_merged")
        if self.has_merged_interface:
            self._lib.face_rknn_parallel_infer_merged.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_int),
                ctypes.c_int,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_float,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_int),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_char_p,
                ctypes.c_int,
            ]
            self._lib.face_rknn_parallel_infer_merged.restype = ctypes.c_int

    @staticmethod
    def _errbuf() -> ctypes.Array:
        return ctypes.create_string_buffer(1024)

    @staticmethod
    def _errmsg(buf: ctypes.Array) -> str:
        return buf.value.decode("utf-8", errors="replace")

    def _create(self) -> None:
        err = self._errbuf()
        ret = self._lib.face_rknn_parallel_create(
            self.model_path.encode("utf-8"),
            ctypes.byref(self._handle),
            err,
            len(err),
        )
        if ret != 0:
            raise RuntimeError(self._errmsg(err) or f"face_rknn_parallel_create failed: {ret}")

        h = ctypes.c_int()
        w = ctypes.c_int()
        c = ctypes.c_int()
        channels = ctypes.c_int()
        anchors = ctypes.c_int()
        ret = self._lib.face_rknn_parallel_get_shape(
            self._handle,
            ctypes.byref(h),
            ctypes.byref(w),
            ctypes.byref(c),
            ctypes.byref(channels),
            ctypes.byref(anchors),
        )
        if ret != 0:
            raise RuntimeError(f"face_rknn_parallel_get_shape failed: {ret}")
        self.input_shape = (int(h.value), int(w.value), int(c.value))
        self.output_shape = (int(channels.value), int(anchors.value))

    def infer(self, inputs: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        h, w, c = self.input_shape
        if isinstance(inputs, np.ndarray):
            packed = np.ascontiguousarray(inputs, dtype=np.uint8)
            if packed.shape != (3, h, w, c):
                raise ValueError(f"C API packed input shape {packed.shape} != {(3, h, w, c)}")
        else:
            if len(inputs) != 3:
                raise ValueError(f"C API parallel backend requires 3 inputs, got {len(inputs)}")
            packed = np.empty((3, h, w, c), dtype=np.uint8)
            for idx, item in enumerate(inputs):
                arr = np.asarray(item)
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.shape != (h, w, c):
                    raise ValueError(f"C API input[{idx}] shape {arr.shape} != {(h, w, c)}")
                packed[idx] = np.ascontiguousarray(arr, dtype=np.uint8)

        channels, anchors = self.output_shape
        outputs = np.empty((3, channels, anchors), dtype=np.float32)
        timings = np.zeros(9, dtype=np.float64)
        err = self._errbuf()
        ret = self._lib.face_rknn_parallel_infer(
            self._handle,
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            3,
            h,
            w,
            c,
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            timings.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            err,
            len(err),
        )
        if ret != 0:
            raise RuntimeError(self._errmsg(err) or f"face_rknn_parallel_infer failed: {ret}")
        return [outputs[i] for i in range(3)], timings

    def infer_decoded(
        self,
        inputs: List[np.ndarray],
        metas: List[Tuple[Tuple[int, int], float, Tuple[int, int]]],
        conf_threshold: float,
        iou_threshold: float,
        max_det: int = 100,
        max_nms: int = 300,
    ) -> Tuple[List[List[dict]], np.ndarray]:
        if not self.has_decoded_interface:
            raise RuntimeError("native decoded RKNN interface is not available; rebuild librknn_capi_parallel.so")
        if len(metas) != 3:
            raise ValueError(f"C API decoded backend requires 3 metas, got {len(metas)}")

        h, w, c = self.input_shape
        if isinstance(inputs, np.ndarray):
            packed = np.ascontiguousarray(inputs, dtype=np.uint8)
            if packed.shape != (3, h, w, c):
                raise ValueError(f"C API packed input shape {packed.shape} != {(3, h, w, c)}")
        else:
            if len(inputs) != 3:
                raise ValueError(f"C API parallel backend requires 3 inputs, got {len(inputs)}")
            packed = np.empty((3, h, w, c), dtype=np.uint8)
            for idx, item in enumerate(inputs):
                arr = np.asarray(item)
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.shape != (h, w, c):
                    raise ValueError(f"C API input[{idx}] shape {arr.shape} != {(h, w, c)}")
                packed[idx] = np.ascontiguousarray(arr, dtype=np.uint8)

        slice_shapes = np.empty((3, 2), dtype=np.int32)
        gains = np.empty((3,), dtype=np.float32)
        pads = np.empty((3, 2), dtype=np.int32)
        for idx, (slice_shape, gain, pad) in enumerate(metas):
            slice_shapes[idx, 0] = int(slice_shape[0])
            slice_shapes[idx, 1] = int(slice_shape[1])
            gains[idx] = float(gain)
            pads[idx, 0] = int(pad[0])
            pads[idx, 1] = int(pad[1])

        max_det = int(max_det)
        max_nms = int(max_nms)
        outputs = np.empty((3, max_det, 20), dtype=np.float32)
        counts = np.zeros((3,), dtype=np.int32)
        timings = np.zeros(13, dtype=np.float64)
        err = self._errbuf()
        ret = self._lib.face_rknn_parallel_infer_decoded(
            self._handle,
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            3,
            h,
            w,
            c,
            slice_shapes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            gains.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pads.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(float(conf_threshold)),
            ctypes.c_float(float(iou_threshold)),
            max_det,
            max_nms,
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            timings.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            err,
            len(err),
        )
        if ret != 0:
            raise RuntimeError(self._errmsg(err) or f"face_rknn_parallel_infer_decoded failed: {ret}")

        decoded: List[List[dict]] = []
        for slice_idx in range(3):
            slice_dets = []
            count = int(counts[slice_idx])
            for row in outputs[slice_idx, :count]:
                slice_dets.append({
                    "bbox": row[:4].tolist(),
                    "confidence": float(row[4]),
                    "class_id": 0,
                    "class_name": "face",
                    "keypoints": row[5:20].reshape(5, 3).tolist(),
                })
            decoded.append(slice_dets)
        return decoded, timings

    def infer_merged(
        self,
        inputs: List[np.ndarray],
        metas: List[Tuple[Tuple[int, int], float, Tuple[int, int]]],
        slice_infos: List[dict],
        conf_threshold: float,
        decode_iou_threshold: float,
        overlap_ratio: float,
        merge_iou_threshold: float,
        nms_iou_thresh: float,
        max_det: int = 100,
        max_nms: int = 300,
    ) -> Tuple[List[dict], np.ndarray, dict]:
        if not self.has_merged_interface:
            raise RuntimeError("native merged RKNN interface is not available; rebuild librknn_capi_parallel.so")
        if len(metas) != 3 or len(slice_infos) != 3:
            raise ValueError(f"C API merged backend requires 3 metas/slice_infos, got {len(metas)}/{len(slice_infos)}")

        h, w, c = self.input_shape
        if isinstance(inputs, np.ndarray):
            packed = np.ascontiguousarray(inputs, dtype=np.uint8)
            if packed.shape != (3, h, w, c):
                raise ValueError(f"C API packed input shape {packed.shape} != {(3, h, w, c)}")
        else:
            if len(inputs) != 3:
                raise ValueError(f"C API parallel backend requires 3 inputs, got {len(inputs)}")
            packed = np.empty((3, h, w, c), dtype=np.uint8)
            for idx, item in enumerate(inputs):
                arr = np.asarray(item)
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
                if arr.shape != (h, w, c):
                    raise ValueError(f"C API input[{idx}] shape {arr.shape} != {(h, w, c)}")
                packed[idx] = np.ascontiguousarray(arr, dtype=np.uint8)

        slice_shapes = np.empty((3, 2), dtype=np.int32)
        gains = np.empty((3,), dtype=np.float32)
        pads = np.empty((3, 2), dtype=np.int32)
        for idx, (slice_shape, gain, pad) in enumerate(metas):
            slice_shapes[idx, 0] = int(slice_shape[0])
            slice_shapes[idx, 1] = int(slice_shape[1])
            gains[idx] = float(gain)
            pads[idx, 0] = int(pad[0])
            pads[idx, 1] = int(pad[1])

        start_x = np.asarray([float(info["start_x"]) for info in slice_infos], dtype=np.float32)
        wrap = np.asarray([1 if info.get("wrap_around") else 0 for info in slice_infos], dtype=np.int32)
        original_width = float(slice_infos[0]["original_width"])

        max_det = int(max_det)
        max_nms = int(max_nms)
        max_output_dets = max(1, max_det * 3)
        outputs = np.empty((max_output_dets, 20), dtype=np.float32)
        out_count = np.zeros((1,), dtype=np.int32)
        stats = np.zeros((3,), dtype=np.int32)
        timings = np.zeros(19, dtype=np.float64)
        err = self._errbuf()
        ret = self._lib.face_rknn_parallel_infer_merged(
            self._handle,
            packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            3,
            h,
            w,
            c,
            slice_shapes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            gains.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pads.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            start_x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            wrap.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            3,
            ctypes.c_float(original_width),
            ctypes.c_float(float(overlap_ratio)),
            ctypes.c_float(float(merge_iou_threshold)),
            ctypes.c_float(float(nms_iou_thresh)),
            ctypes.c_float(float(conf_threshold)),
            ctypes.c_float(float(decode_iou_threshold)),
            max_det,
            max_nms,
            outputs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            max_output_dets,
            out_count.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            timings.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            err,
            len(err),
        )
        if ret != 0:
            raise RuntimeError(self._errmsg(err) or f"face_rknn_parallel_infer_merged failed: {ret}")

        count = int(out_count[0])
        merged: List[dict] = []
        for row in outputs[:count]:
            merged.append({
                "bbox": row[:4].tolist(),
                "confidence": float(row[4]),
                "class_id": 0,
                "class_name": "face",
                "feature": None,
                "keypoints": row[5:20].reshape(5, 3).tolist(),
            })

        profile = {
            "extract": 0.0,
            "features": 0.0,
            "coords": float(timings[5]),
            "dedup": float(timings[6]),
            "nms": float(timings[7]),
            "final_dedup": float(timings[8]),
            "build": float(timings[9]),
            "raw": int(stats[0]),
            "nms_kept": int(stats[1]),
            "kept": int(stats[2]),
            "total_native": float(timings[4]),
        }
        return merged, timings, profile

    def release(self) -> None:
        if self._handle and self._handle.value:
            self._lib.face_rknn_parallel_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass
