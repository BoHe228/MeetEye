"""
YOLO姿态检测器
"""
import cv2
import numpy as np
import os
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple
import config

_USE_HALF = torch.cuda.is_available()  # FP16 only on GPU; set once at import time


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


def _as_probability(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    if float(np.nanmin(values)) < 0.0 or float(np.nanmax(values)) > 1.0:
        return _sigmoid(values)
    return values


def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    out = np.empty_like(xywh, dtype=np.float32)
    out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return out


def _box_iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_thres: float, max_det: int) -> List[int]:
    if boxes.size == 0:
        return []
    order = np.argsort(-scores)
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_det:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break
        rest = order[1:]
        ious = _box_iou_one(boxes[idx], boxes[rest])
        order = rest[ious <= iou_thres]
    return keep


def _letterbox_bgr(image: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
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
    return padded, gain, (left, top)


def _normalize_rknn_output(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported RKNN output shape: {arr.shape}")
    return arr


def _normalize_rknn_outputs(outputs: List[np.ndarray]) -> np.ndarray:
    if len(outputs) == 1:
        return _normalize_rknn_output(outputs[0])
    if len(outputs) == 3:
        parts = [_normalize_rknn_output(output) for output in outputs]
        channels = [part.shape[1] for part in parts]
        order = []
        for expected in (4, 1, 15):
            try:
                idx = channels.index(expected)
            except ValueError as exc:
                raise ValueError(f"Cannot combine RKNN split outputs, channels={channels}") from exc
            order.append(idx)
        return np.concatenate([parts[idx] for idx in order], axis=1)
    raise ValueError(f"Unsupported RKNN output count: {len(outputs)}")


def _find_rknn_file(model_path: str) -> Optional[Path]:
    path = Path(model_path)
    if path.is_file() and path.suffix.lower() == ".rknn":
        return path
    if not path.is_dir():
        return None
    files = sorted(path.glob("*.rknn"))
    return files[0] if files else None


def _read_metadata_int(model_path: str, key: str, default: int) -> int:
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


def _resolve_rknn_core_mask(RKNNLite, name: str):
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


class YOLOPoseDetector:
    """YOLO姿态检测器"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45, imgsz: int = 864,
                 rknn_core_mask: str = "default",
                 rknn_parallel_slices: bool = False):
        """
        初始化YOLO姿态检测器
        """
        print(f"加载YOLO姿态估计模型: {model_path}")
        self.model = None
        self.rknn = None
        self.rknn_parallel = []
        self.rknn_parallel_executor = None
        self.rknn_capi_parallel = None
        self.rknn_capi_parallel_enabled = False
        self.rknn_capi_decode_warned = False
        self.rknn_parallel_slices = bool(rknn_parallel_slices)
        self.direct_rknn = False
        self.rknn_batch = 0
        self.rknn_imgsz = 0
        _p = str(model_path).lower().rstrip("/\\")
        rknn_file = _find_rknn_file(model_path) if _p.endswith("_rknn_model") or _p.endswith(".rknn") else None
        if self.rknn_parallel_slices and rknn_file is None:
            raise RuntimeError("--rknn-parallel-slices 只能用于 .rknn 文件或 *_rknn_model 目录")
        if rknn_file is not None:
            try:
                self.rknn_batch = _read_metadata_int(model_path, "batch", 0)
                self.rknn_imgsz = _read_metadata_int(model_path, "imgsz", 0)
                if self.rknn_parallel_slices:
                    if self.rknn_batch != 1:
                        raise RuntimeError(
                            "--rknn-parallel-slices 需要 batch=1 RKNN 模型；"
                            f"当前 metadata batch={self.rknn_batch}"
                        )
                    try:
                        from core.rknn_capi import ParallelRknnCAPI
                    except Exception:
                        from rknn_capi import ParallelRknnCAPI

                    try:
                        self.rknn_capi_parallel = ParallelRknnCAPI(str(rknn_file))
                        h, w, c = self.rknn_capi_parallel.input_shape
                        channels, anchors = self.rknn_capi_parallel.output_shape
                        if self.rknn_imgsz <= 0:
                            self.rknn_imgsz = int(h)
                        if h != self.rknn_imgsz or w != self.rknn_imgsz or c != 3:
                            raise RuntimeError(
                                f"C API RKNN 输入尺寸不匹配: {(h, w, c)} vs imgsz={self.rknn_imgsz}"
                            )
                        if channels != 20:
                            raise RuntimeError(f"C API RKNN 输出通道不匹配: channels={channels}")
                        self.rknn_capi_parallel_enabled = True
                        print(
                            "  RKNN C API 并行切片模式已启用: "
                            f"batch=1 x core0/core1/core2, output=({channels},{anchors})"
                        )
                    except Exception as capi_exc:
                        print(
                            "  RKNN C API 并行切片不可用，回退 RKNNLite: "
                            f"{type(capi_exc).__name__}: {capi_exc}"
                        )
                        self.rknn_capi_parallel = None
                        self.rknn_capi_parallel_enabled = False

                    if not self.rknn_capi_parallel_enabled:
                        from rknnlite.api import RKNNLite
                        core_names = ("core0", "core1", "core2")
                        for core_name in core_names:
                            inst = RKNNLite()
                            ret = inst.load_rknn(str(rknn_file))
                            print(f"  RKNNLite[{core_name}] load_rknn ret={ret}: {rknn_file}")
                            if ret != 0:
                                raise RuntimeError(f"load_rknn failed on {core_name}: {ret}")
                            core_mask, core_mask_attr = _resolve_rknn_core_mask(RKNNLite, core_name)
                            if core_mask is None:
                                raise RuntimeError(f"RKNNLite 不支持 {core_name}")
                            ret = inst.init_runtime(core_mask=core_mask)
                            print(f"  RKNNLite[{core_name}] init_runtime ret={ret}, core_mask={core_mask_attr}")
                            if ret != 0:
                                raise RuntimeError(f"init_runtime failed on {core_name}: {ret}")
                            self.rknn_parallel.append(inst)
                        self.rknn_parallel_executor = ThreadPoolExecutor(
                            max_workers=3,
                            thread_name_prefix="RKNNCore",
                        )
                    self.rknn = None
                    print("  RKNN 并行切片模式已启用: batch=1 x core0/core1/core2")
                else:
                    from rknnlite.api import RKNNLite
                    self.rknn = RKNNLite()
                    ret = self.rknn.load_rknn(str(rknn_file))
                    print(f"  RKNNLite load_rknn ret={ret}: {rknn_file}")
                    if ret != 0:
                        raise RuntimeError(f"load_rknn failed: {ret}")
                    core_mask, core_mask_name = _resolve_rknn_core_mask(RKNNLite, rknn_core_mask)
                    if core_mask is not None:
                        ret = self.rknn.init_runtime(core_mask=core_mask)
                        print(f"  RKNNLite init_runtime ret={ret}, core_mask={core_mask_name}")
                    else:
                        ret = self.rknn.init_runtime()
                        print(f"  RKNNLite init_runtime ret={ret}, core_mask=default")
                    if ret != 0:
                        raise RuntimeError(f"init_runtime failed: {ret}")
                self.direct_rknn = True
                print("  RKNN 模型使用直接 RKNNLite 后端，绕过 Ultralytics wrapper")
            except Exception as exc:
                print(f"  直接 RKNNLite 后端不可用，回退 Ultralytics: {type(exc).__name__}: {exc}")
                if self.rknn is not None:
                    try:
                        self.rknn.release()
                    except Exception:
                        pass
                    self.rknn = None
                self.release()
                if self.rknn_parallel_slices:
                    raise

        # .engine 不带 task 元数据时 ultralytics 默认按 detect 解析：pose/face 模型的
        # 关键点通道会被误当作类别，argmax 落到无效类别 → KeyError。按文件名显式指定 task：
        #   名含 pose/face → 'pose'（17 点 / 人脸 5 点）；其余（如 yolo26n.engine）→ 'detect'
        # RKNN 目录同样显式指定 task，避免只靠目录名/metadata 推断导致任务类型漂移。
        if self.direct_rknn:
            self.model = None
        else:
            from ultralytics import YOLO
            if _p.endswith('.engine') or _p.endswith('_rknn_model'):
                _task = 'pose' if ('pose' in _p or 'face' in _p) else 'detect'
                self.model = YOLO(model_path, task=_task)
                print(f"  模型任务类型推断为: {_task}")
            else:
                self.model = YOLO(model_path)  # .pt/.onnx 自带 task 元数据
        # self.model = self.model.half()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = int(imgsz)
        if self.direct_rknn and self.rknn_imgsz > 0 and self.imgsz != self.rknn_imgsz:
            print(f"  RKNN 静态输入尺寸为 {self.rknn_imgsz}，覆盖命令行 imgsz={self.imgsz}")
            self.imgsz = self.rknn_imgsz

        # 性能统计
        self.inference_times = []
        self.frame_count = 0
        self.total_inference_time = 0
        self.last_batch_ms = 0.0
        self.last_batch_shapes = []
        self.last_rknn_infer_ms = 0.0
        self.last_rknn_run_ms = 0.0
        self.last_rknn_output_ms = 0.0
        self.last_rknn_decode_ms = 0.0
        self.last_rknn_pre_ms = 0.0
        self.last_rknn_post_ms = 0.0
        self.last_native_merge_profile = None

    def _decode_rknn_slice(
        self,
        output_for_slice: np.ndarray,
        slice_shape: Tuple[int, int],
        gain: float,
        pad: Tuple[int, int],
        max_det: int = 100,
        max_nms: int = 300,
    ) -> List[Dict[str, Any]]:
        pred = output_for_slice
        if pred.shape[0] <= 64 and pred.shape[0] < pred.shape[1]:
            pred = pred.T
        pred = pred.astype(np.float32, copy=False)
        if pred.shape[1] < 20:
            raise ValueError(f"Expected at least 20 output channels, got {pred.shape}")

        scores = _as_probability(pred[:, 4])
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return []
        pred = pred[keep]
        scores = scores[keep]
        if scores.size > max_nms:
            top = np.argsort(-scores)[:max_nms]
            pred = pred[top]
            scores = scores[top]

        boxes = _xywh_to_xyxy(pred[:, :4])
        left, top = pad
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / gain
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / gain

        slice_h, slice_w = slice_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, slice_w - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, slice_h - 1)
        valid = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
        if not np.any(valid):
            return []

        boxes = boxes[valid]
        pred = pred[valid]
        scores = scores[valid]

        keep_indices = _nms_indices(boxes, scores, self.iou_threshold, max_det)
        if not keep_indices:
            return []

        keep_indices_arr = np.asarray(keep_indices, dtype=np.int64)
        boxes = boxes[keep_indices_arr]
        pred = pred[keep_indices_arr]
        scores = scores[keep_indices_arr]

        # Decode keypoints only after NMS; most candidates are discarded by this point.
        keypoints = pred[:, 5:20].reshape(-1, 5, 3).astype(np.float32, copy=False)
        keypoints[:, :, 0] = (keypoints[:, :, 0] - left) / gain
        keypoints[:, :, 1] = (keypoints[:, :, 1] - top) / gain
        keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0, slice_w - 1)
        keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0, slice_h - 1)
        keypoints[:, :, 2] = _as_probability(keypoints[:, :, 2])

        detections = []
        for idx in range(len(keep_indices)):
            detections.append({
                "bbox": boxes[idx].tolist(),
                "confidence": float(scores[idx]),
                "class_id": 0,
                "class_name": "face",
                "keypoints": keypoints[idx].tolist(),
            })
        return detections

    def _detect_batch_rknn(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        if self.rknn is None:
            raise RuntimeError("RKNNLite backend is not initialized")
        if self.rknn_batch and len(images) != self.rknn_batch:
            raise ValueError(f"RKNN static batch={self.rknn_batch}, got {len(images)} images")

        t0 = time.perf_counter()
        batch_images = []
        metas = []
        for image in images:
            padded, gain, pad = _letterbox_bgr(image, self.imgsz)
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            batch_images.append(rgb)
            metas.append((image.shape[:2], gain, pad))
        input_data = np.ascontiguousarray(np.stack(batch_images, axis=0).astype(np.uint8))
        t_pre = time.perf_counter()

        outputs = self.rknn.inference(inputs=[input_data], data_type=["uint8"], data_format=["nhwc"])
        t_infer = time.perf_counter()
        if outputs is None:
            raise RuntimeError("RKNN inference returned None")
        raw = _normalize_rknn_outputs(outputs)
        if raw.shape[0] != len(images):
            raise ValueError(f"RKNN output batch mismatch: output={raw.shape}, input={len(images)}")

        decoded = []
        for idx, meta in enumerate(metas):
            slice_shape, gain, pad = meta
            decoded.append(self._decode_rknn_slice(raw[idx], slice_shape, gain, pad))
        t_post = time.perf_counter()

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = (t_infer - t_pre) * 1000
        self.last_rknn_run_ms = 0.0
        self.last_rknn_output_ms = 0.0
        self.last_rknn_decode_ms = 0.0
        self.last_rknn_post_ms = (t_post - t_infer) * 1000
        return decoded

    @staticmethod
    def _decode_meta_from_direct_slice(meta: Dict[str, Any]) -> Tuple[Tuple[int, int], float, Tuple[int, int]]:
        slice_shape = meta.get("slice_shape")
        if slice_shape is None:
            slice_shape = (meta["slice_height"], meta["slice_width"])
        slice_h, slice_w = slice_shape
        pad = meta.get("pad")
        if pad is None:
            pad = (meta.get("left", 0), meta.get("top", 0))
        return (int(slice_h), int(slice_w)), float(meta["gain"]), (int(pad[0]), int(pad[1]))

    def _detect_batch_rknn_preletterboxed(
        self,
        images: List[np.ndarray],
        metas: List[Dict[str, Any]],
        input_format: str = "bgr",
    ) -> List[List[Dict[str, Any]]]:
        if self.rknn is None:
            raise RuntimeError("RKNNLite backend is not initialized")
        if self.rknn_batch and len(images) != self.rknn_batch:
            raise ValueError(f"RKNN static batch={self.rknn_batch}, got {len(images)} images")
        if len(images) != len(metas):
            raise ValueError(f"direct slice meta mismatch: images={len(images)}, metas={len(metas)}")

        t0 = time.perf_counter()
        if input_format == "rgb":
            input_data = np.asarray(images)
            if input_data.ndim != 4 or input_data.shape[1:3] != (self.imgsz, self.imgsz):
                raise ValueError(f"direct RGB batch must be Nx{self.imgsz}x{self.imgsz}x3, got {input_data.shape}")
            input_data = np.ascontiguousarray(input_data, dtype=np.uint8)
        else:
            batch_images = []
            for image in images:
                if image.shape[0] != self.imgsz or image.shape[1] != self.imgsz:
                    raise ValueError(f"direct slice image must be {self.imgsz}x{self.imgsz}, got {image.shape[:2]}")
                batch_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            input_data = np.ascontiguousarray(np.stack(batch_images, axis=0).astype(np.uint8))
        decode_metas = [self._decode_meta_from_direct_slice(meta) for meta in metas]
        t_pre = time.perf_counter()

        outputs = self.rknn.inference(inputs=[input_data], data_type=["uint8"], data_format=["nhwc"])
        t_infer = time.perf_counter()
        if outputs is None:
            raise RuntimeError("RKNN inference returned None")
        raw = _normalize_rknn_outputs(outputs)
        if raw.shape[0] != len(images):
            raise ValueError(f"RKNN output batch mismatch: output={raw.shape}, input={len(images)}")

        decoded = []
        for idx, meta in enumerate(decode_metas):
            slice_shape, gain, pad = meta
            decoded.append(self._decode_rknn_slice(raw[idx], slice_shape, gain, pad))
        t_post = time.perf_counter()

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = (t_infer - t_pre) * 1000
        self.last_rknn_run_ms = 0.0
        self.last_rknn_output_ms = 0.0
        self.last_rknn_decode_ms = 0.0
        self.last_rknn_post_ms = (t_post - t_infer) * 1000
        return decoded

    def _prepare_rknn_input(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[Tuple[int, int], float, Tuple[int, int]]]:
        padded, gain, pad = _letterbox_bgr(image, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = np.ascontiguousarray(rgb[None, ...].astype(np.uint8))
        return input_data, (image.shape[:2], gain, pad)

    def _prepare_rknn_preletterboxed_input(self, image: np.ndarray, input_format: str = "bgr") -> np.ndarray:
        if image.shape[0] != self.imgsz or image.shape[1] != self.imgsz:
            raise ValueError(f"direct slice image must be {self.imgsz}x{self.imgsz}, got {image.shape[:2]}")
        if input_format == "rgb":
            return np.ascontiguousarray(image[None, ...], dtype=np.uint8)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb[None, ...].astype(np.uint8))

    @staticmethod
    def _infer_rknn_instance(rknn, input_data: np.ndarray) -> np.ndarray:
        outputs = rknn.inference(inputs=[input_data], data_type=["uint8"], data_format=["nhwc"])
        if outputs is None:
            raise RuntimeError("RKNN inference returned None")
        raw = _normalize_rknn_outputs(outputs)
        if raw.shape[0] != 1:
            raise ValueError(f"RKNN batch=1 output mismatch: {raw.shape}")
        return raw[0]

    def _detect_parallel_rknn(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        if len(images) != 3:
            raise ValueError(f"--rknn-parallel-slices requires exactly 3 images, got {len(images)}")
        if self.rknn_capi_parallel_enabled:
            return self._detect_parallel_rknn_capi(images)
        if len(self.rknn_parallel) != 3 or self.rknn_parallel_executor is None:
            raise RuntimeError("RKNN parallel backend is not initialized")

        t0 = time.perf_counter()
        prepared = [self._prepare_rknn_input(image) for image in images]
        input_data = [item[0] for item in prepared]
        metas = [item[1] for item in prepared]
        t_pre = time.perf_counter()

        futures = [
            self.rknn_parallel_executor.submit(self._infer_rknn_instance, rknn, inp)
            for rknn, inp in zip(self.rknn_parallel, input_data)
        ]
        raw_slices = [future.result() for future in futures]
        t_infer = time.perf_counter()

        decoded = []
        for raw, meta in zip(raw_slices, metas):
            slice_shape, gain, pad = meta
            decoded.append(self._decode_rknn_slice(raw, slice_shape, gain, pad))
        t_post = time.perf_counter()

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = (t_infer - t_pre) * 1000
        self.last_rknn_run_ms = 0.0
        self.last_rknn_output_ms = 0.0
        self.last_rknn_decode_ms = 0.0
        self.last_rknn_post_ms = (t_post - t_infer) * 1000
        return decoded

    def _detect_parallel_rknn_capi(self, images: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        if self.rknn_capi_parallel is None:
            raise RuntimeError("RKNN C API parallel backend is not initialized")

        t0 = time.perf_counter()
        prepared = [self._prepare_rknn_input(image) for image in images]
        input_data = [item[0] for item in prepared]
        metas = [item[1] for item in prepared]
        t_pre = time.perf_counter()

        try:
            decoded, timings = self.rknn_capi_parallel.infer_decoded(
                input_data,
                metas,
                self.conf_threshold,
                self.iou_threshold,
            )
            t_done = time.perf_counter()
            native_decoded = True
        except RuntimeError as exc:
            if "native decoded RKNN interface is not available" not in str(exc):
                raise
            if not self.rknn_capi_decode_warned:
                print("  RKNN C API 原生 decode 不可用，使用 Python decode 回退；请重编译 librknn_capi_parallel.so")
                self.rknn_capi_decode_warned = True
            raw_slices, timings = self.rknn_capi_parallel.infer(input_data)
            t_infer_done = time.perf_counter()
            decoded = []
            for raw, meta in zip(raw_slices, metas):
                slice_shape, gain, pad = meta
                decoded.append(self._decode_rknn_slice(raw, slice_shape, gain, pad))
            t_done = time.perf_counter()
            native_decoded = False

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = float(timings[0]) if len(timings) else (t_done - t_pre) * 1000
        self.last_rknn_run_ms = float(timings[1]) if len(timings) > 1 else 0.0
        self.last_rknn_output_ms = float(timings[2]) if len(timings) > 2 else 0.0
        self.last_rknn_decode_ms = float(timings[3]) if native_decoded and len(timings) > 3 else 0.0
        if native_decoded:
            self.last_rknn_post_ms = max(0.0, (t_done - t_pre) * 1000 - self.last_rknn_infer_ms)
        else:
            self.last_rknn_post_ms = (t_done - t_infer_done) * 1000
        return decoded

    def _detect_parallel_rknn_preletterboxed(
        self,
        images: List[np.ndarray],
        metas: List[Dict[str, Any]],
        input_format: str = "bgr",
    ) -> List[List[Dict[str, Any]]]:
        if len(images) != 3:
            raise ValueError(f"--rknn-parallel-slices requires exactly 3 images, got {len(images)}")
        if len(images) != len(metas):
            raise ValueError(f"direct slice meta mismatch: images={len(images)}, metas={len(metas)}")
        if self.rknn_capi_parallel_enabled:
            return self._detect_parallel_rknn_capi_preletterboxed(images, metas, input_format=input_format)
        if len(self.rknn_parallel) != 3 or self.rknn_parallel_executor is None:
            raise RuntimeError("RKNN parallel backend is not initialized")

        t0 = time.perf_counter()
        input_data = [self._prepare_rknn_preletterboxed_input(image, input_format=input_format) for image in images]
        decode_metas = [self._decode_meta_from_direct_slice(meta) for meta in metas]
        t_pre = time.perf_counter()

        futures = [
            self.rknn_parallel_executor.submit(self._infer_rknn_instance, rknn, inp)
            for rknn, inp in zip(self.rknn_parallel, input_data)
        ]
        raw_slices = [future.result() for future in futures]
        t_infer = time.perf_counter()

        decoded = []
        for raw, meta in zip(raw_slices, decode_metas):
            slice_shape, gain, pad = meta
            decoded.append(self._decode_rknn_slice(raw, slice_shape, gain, pad))
        t_post = time.perf_counter()

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = (t_infer - t_pre) * 1000
        self.last_rknn_run_ms = 0.0
        self.last_rknn_output_ms = 0.0
        self.last_rknn_decode_ms = 0.0
        self.last_rknn_post_ms = (t_post - t_infer) * 1000
        return decoded

    def _detect_parallel_rknn_capi_preletterboxed(
        self,
        images: List[np.ndarray],
        metas: List[Dict[str, Any]],
        input_format: str = "bgr",
    ) -> List[List[Dict[str, Any]]]:
        if self.rknn_capi_parallel is None:
            raise RuntimeError("RKNN C API parallel backend is not initialized")

        t0 = time.perf_counter()
        if input_format == "rgb":
            input_data = np.asarray(images)
            if input_data.ndim != 4 or input_data.shape[0] != 3 or input_data.shape[1:3] != (self.imgsz, self.imgsz):
                raise ValueError(f"C API RGB batch must be 3x{self.imgsz}x{self.imgsz}x3, got {input_data.shape}")
            input_data = np.ascontiguousarray(input_data, dtype=np.uint8)
        else:
            input_data = [self._prepare_rknn_preletterboxed_input(image) for image in images]
        decode_metas = [self._decode_meta_from_direct_slice(meta) for meta in metas]
        t_pre = time.perf_counter()

        try:
            decoded, timings = self.rknn_capi_parallel.infer_decoded(
                input_data,
                decode_metas,
                self.conf_threshold,
                self.iou_threshold,
            )
            t_done = time.perf_counter()
            native_decoded = True
        except RuntimeError as exc:
            if "native decoded RKNN interface is not available" not in str(exc):
                raise
            if not self.rknn_capi_decode_warned:
                print("  RKNN C API 原生 decode 不可用，使用 Python decode 回退；请重编译 librknn_capi_parallel.so")
                self.rknn_capi_decode_warned = True
            raw_slices, timings = self.rknn_capi_parallel.infer(input_data)
            t_infer_done = time.perf_counter()
            decoded = []
            for raw, meta in zip(raw_slices, decode_metas):
                slice_shape, gain, pad = meta
                decoded.append(self._decode_rknn_slice(raw, slice_shape, gain, pad))
            t_done = time.perf_counter()
            native_decoded = False

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = float(timings[0]) if len(timings) else (t_done - t_pre) * 1000
        self.last_rknn_run_ms = float(timings[1]) if len(timings) > 1 else 0.0
        self.last_rknn_output_ms = float(timings[2]) if len(timings) > 2 else 0.0
        self.last_rknn_decode_ms = float(timings[3]) if native_decoded and len(timings) > 3 else 0.0
        if native_decoded:
            self.last_rknn_post_ms = max(0.0, (t_done - t_pre) * 1000 - self.last_rknn_infer_ms)
        else:
            self.last_rknn_post_ms = (t_done - t_infer_done) * 1000
        return decoded

    def detect_preletterboxed_merged(
        self,
        images: List[np.ndarray],
        metas: List[Dict[str, Any]],
        slice_infos: List[dict],
        input_format: str = "bgr",
        overlap_ratio: float = 0.1,
        merge_iou_threshold: Optional[float] = None,
        nms_iou_thresh: float = 0.5,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run direct-slice RKNN and native merge in one C API call when available."""
        self.last_native_merge_profile = None
        if (
            not self.direct_rknn
            or not self.rknn_parallel_slices
            or not self.rknn_capi_parallel_enabled
            or self.rknn_capi_parallel is None
            or not getattr(self.rknn_capi_parallel, "has_merged_interface", False)
        ):
            return None
        if len(images) != 3 or len(metas) != 3 or len(slice_infos) != 3:
            return None

        start_time = time.perf_counter()
        t0 = time.perf_counter()
        if input_format == "rgb":
            input_data = np.asarray(images)
            if input_data.ndim != 4 or input_data.shape[0] != 3 or input_data.shape[1:3] != (self.imgsz, self.imgsz):
                raise ValueError(f"C API RGB batch must be 3x{self.imgsz}x{self.imgsz}x3, got {input_data.shape}")
            input_data = np.ascontiguousarray(input_data, dtype=np.uint8)
        else:
            input_data = [self._prepare_rknn_preletterboxed_input(image) for image in images]
        decode_metas = [self._decode_meta_from_direct_slice(meta) for meta in metas]
        t_pre = time.perf_counter()

        merge_iou = self.iou_threshold if merge_iou_threshold is None else float(merge_iou_threshold)
        merged, timings, merge_profile = self.rknn_capi_parallel.infer_merged(
            input_data,
            decode_metas,
            slice_infos,
            self.conf_threshold,
            self.iou_threshold,
            float(overlap_ratio),
            merge_iou,
            float(nms_iou_thresh),
        )
        t_done = time.perf_counter()

        self.last_rknn_pre_ms = (t_pre - t0) * 1000
        self.last_rknn_infer_ms = float(timings[0]) if len(timings) else (t_done - t_pre) * 1000
        self.last_rknn_run_ms = float(timings[1]) if len(timings) > 1 else 0.0
        self.last_rknn_output_ms = float(timings[2]) if len(timings) > 2 else 0.0
        self.last_rknn_decode_ms = float(timings[3]) if len(timings) > 3 else 0.0
        self.last_rknn_post_ms = max(0.0, (t_done - t_pre) * 1000 - self.last_rknn_infer_ms)
        merge_profile["total"] = float(merge_profile.get("total_native", 0.0))
        merge_profile["fused"] = 1
        self.last_native_merge_profile = merge_profile

        self.last_batch_shapes = [tuple(img.shape[:2]) for img in images]
        self.last_batch_ms = (time.perf_counter() - start_time) * 1000
        self.inference_times.append(self.last_batch_ms / 1000.0)
        self.frame_count += 1
        self.total_inference_time += self.last_batch_ms / 1000.0
        return merged

    def detect_batch(self, images: List[np.ndarray]) -> List[List]:
        """
        批量推理多张图像（一次 GPU forward pass 处理所有切片）。
        返回格式与多次调用 detect() 一致：List[List[Result]]
        """
        if isinstance(images, np.ndarray) and images.ndim == 4:
            self.last_batch_shapes = [tuple(images.shape[1:3]) for _ in range(images.shape[0])]
        else:
            self.last_batch_shapes = [tuple(img.shape[:2]) for img in images]
        start_time = time.perf_counter()
        if self.direct_rknn:
            if self.rknn_parallel_slices:
                results = self._detect_parallel_rknn(images)
            else:
                results = self._detect_batch_rknn(images)
        else:
            results = self.model.predict(
                images,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                half=_USE_HALF,
                device=0 if _USE_HALF else 'cpu',
            )
        inference_time = time.perf_counter() - start_time
        self.last_batch_ms = inference_time * 1000
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_inference_time += inference_time
        if self.direct_rknn:
            return results
        # 每个元素包成单元素列表，与 merge_detections 期望的接口一致
        return [[r] for r in results]

    def detect_preletterboxed_batch(
        self,
        images: List[np.ndarray],
        metas: List[Dict[str, Any]],
        input_format: str = "bgr",
    ) -> List[List]:
        """
        Run RKNN on images that are already YOLO-sized letterboxed inputs.

        The metadata describes the original slice shape and letterbox transform,
        so RKNN outputs are decoded back to slice-local coordinates.
        """
        if not self.direct_rknn:
            raise RuntimeError("--direct-slice-remap currently requires direct RKNNLite backend")

        self.last_batch_shapes = [tuple(img.shape[:2]) for img in images]
        start_time = time.perf_counter()
        if self.rknn_parallel_slices:
            results = self._detect_parallel_rknn_preletterboxed(images, metas, input_format=input_format)
        else:
            results = self._detect_batch_rknn_preletterboxed(images, metas, input_format=input_format)
        inference_time = time.perf_counter() - start_time
        self.last_batch_ms = inference_time * 1000
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_inference_time += inference_time
        return results

    def release(self) -> None:
        if self.rknn_parallel_executor is not None:
            self.rknn_parallel_executor.shutdown(wait=True)
            self.rknn_parallel_executor = None
        if self.rknn_capi_parallel is not None:
            try:
                self.rknn_capi_parallel.release()
            except Exception:
                pass
            self.rknn_capi_parallel = None
            self.rknn_capi_parallel_enabled = False
        for inst in self.rknn_parallel:
            try:
                inst.release()
            except Exception:
                pass
        self.rknn_parallel = []
        if self.rknn is not None:
            try:
                self.rknn.release()
            except Exception:
                pass
            self.rknn = None

    def detect(self, image: np.ndarray, use_tracking: bool = False) -> List:
        """
        检测图像中的姿态
        参数:
            image: 输入图像
            use_tracking: 是否使用跟踪（对于切片检测建议设为False）
        返回: 检测结果列表
        """
        start_time = time.time()

        if use_tracking:
            # 使用跟踪（仅用于单图/完整图检测）
            results = self.model.track(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                tracker="bytetrack.yaml",
                half=_USE_HALF,
            )
        else:
            # 纯检测（用于切片检测，避免多切片ID冲突）
            results = self.model.predict(
                image,
                imgsz=self.imgsz,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False,
                half=_USE_HALF,
            )

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_inference_time += inference_time

        return results

    def detect_with_global_tracking(self, image: np.ndarray) -> List:
        """
        在全景图上直接进行检测 + bytetrack跟踪，获得全局一致的ID

        Args:
            image: 全景图像

        Returns:
            带有全局track_id的检测结果
        """
        start_time = time.time()

        # 直接在全景图上运行track，获得全局一致的ID
        results = self.model.track(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            tracker="bytetrack.yaml",
            half=_USE_HALF,
        )

        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_inference_time += inference_time

        return results

    def extract_detections_from_results(self, results) -> List[Dict]:
        """
        从YOLO结果中提取检测信息（包含track_id）
        """
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                boxes_data = boxes.xyxy.cpu().numpy()
                confidences = boxes.conf.cpu().numpy()
                class_ids = boxes.cls.cpu().numpy().astype(int)

                # 获取跟踪ID
                track_ids = None
                if hasattr(boxes, 'id') and boxes.id is not None:
                    track_ids = boxes.id.cpu().numpy().astype(int)

                # 获取关键点
                keypoints_data = []
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints_data = result.keypoints.data.cpu().numpy()

                for i, (box, conf, cls_id) in enumerate(zip(boxes_data, confidences, class_ids)):
                    det = {
                        'bbox': box.tolist(),
                        'confidence': float(conf),
                        'class_id': int(cls_id),
                        'class_name': result.names[int(cls_id)],
                    }

                    # 添加跟踪ID
                    if track_ids is not None and i < len(track_ids):
                        det['track_id'] = int(track_ids[i])

                    # 添加关键点
                    if i < len(keypoints_data):
                        det['keypoints'] = keypoints_data[i].tolist()

                    detections.append(det)

        return detections

    def draw_detections(self, image: np.ndarray, results: List) -> np.ndarray:
        """
        在图像上绘制检测结果
        参数:
            image: 原始图像
            results: 检测结果
        返回: 绘制后的图像
        """
        annotated_image = image.copy()

        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                # 获取关键点数据
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else []

                # 绘制每个检测到的姿态
                for i, kpts in enumerate(keypoints):
                    # 绘制关键点
                    for j, kp in enumerate(kpts):
                        if kp[2] > 0.1:  # 可见性阈值
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(annotated_image, (x, y), 4,
                                      config.KEYPOINT_COLORS[j], -1)

                    # 绘制骨架连线
                    for (start_idx, end_idx) in config.SKELETON_CONNECTIONS:
                        if (start_idx < len(kpts) and end_idx < len(kpts) and
                            kpts[start_idx][2] > 0.1 and kpts[end_idx][2] > 0.1):
                            start_pt = (int(kpts[start_idx][0]), int(kpts[start_idx][1]))
                            end_pt = (int(kpts[end_idx][0]), int(kpts[end_idx][1]))
                            cv2.line(annotated_image, start_pt, end_pt, (0, 255, 0), 2)

                # 绘制边界框
                for box in boxes:
                    if len(box) >= 4:  # 确保有足够的元素
                        x1, y1, x2, y2, conf, cls = box[:6]
                        cv2.rectangle(annotated_image, (int(x1), int(y1)),
                                     (int(x2), int(y2)), (255, 0, 0), 2)
                        label = f"Person: {conf:.2f}"
                        cv2.putText(annotated_image, label, (int(x1), int(y1)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return annotated_image

    def get_detection_info(self, results: List) -> Dict:
        """
        获取检测信息
        参数:
            results: 检测结果
        返回: 检测信息字典
        """
        info = {
            'num_people': 0,
            'keypoints': [],
            'boxes': []
        }

        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy() if result.boxes is not None else []

                info['num_people'] = len(keypoints)
                info['keypoints'] = keypoints
                info['boxes'] = boxes

        return info

    def get_performance_stats(self) -> Dict:
        """
        获取性能统计
        返回: 性能统计字典
        """
        if self.frame_count == 0:
            return {'avg_inference_time': 0, 'fps': 0}

        avg_inference = self.total_inference_time / self.frame_count * 1000

        if len(self.inference_times) > 0:
            recent_fps = 1.0 / self.inference_times[-1] if self.inference_times[-1] > 0 else 0
        else:
            recent_fps = 0

        return {
            'avg_inference_time_ms': avg_inference,
            'recent_fps': recent_fps,
            'total_frames': self.frame_count
        }

    def update_thresholds(self, conf_threshold: float = None, iou_threshold: float = None):
        """更新阈值"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
