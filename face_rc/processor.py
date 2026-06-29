"""
Fish-eye WebUI core inference pipeline.

This module is the deployable version of the original WebUI processor. It keeps
only the path required by the fish-eye meeting camera:

  raw fisheye frame
    -> GPU panorama unwrap
    -> panorama crop + overlapping slices
    -> batched YOLO face/keypoint detection
    -> slice merge and duplicate removal
    -> HybridSort TrackID
    -> angle/sector drawing

This deployment target matches mytest/main_GPU_webui.py. It does not include
the face recognition module; TrackID is the only identity label shown and
serialized.
"""
import os
import json
import threading
import time
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from core import (
    AngleCalculator,
    FisheyePanorama,
    FisheyePanoramaGPU,
    HybridSortTracker,
    PanoramaSlicer,
    YOLOPoseDetector,
    print_assignment_stats,
)
from utils import (
    compute_stable_bbox_from_keypoints,
    draw_detections,
    filter_cross_boundary_detections,
)
from utils.sector import aggregate_sectors, draw_sector_grid

_CUDA = torch.cuda.is_available()

_SLICE_INFO_FIELDS = (
    "slice_idx",
    "start_x",
    "actual_start_x",
    "end_x",
    "slice_width",
    "slice_height",
    "original_width",
    "original_height",
    "wrap_around",
)

_LETTERBOX_INFO_FIELDS = (
    "gain",
    "left",
    "top",
    "new_width",
    "new_height",
)


class FaceRCPipeline:
    """Minimal fisheye WebUI runtime used by face_rc/main.py."""

    def __init__(self, args):
        self.args = args
        self.panorama_processor: Optional[object] = None
        self.yolo_detector: Optional[YOLOPoseDetector] = None
        self.angle_calculator: Optional[AngleCalculator] = None

        self.num_slices = int(getattr(args, "num_slices", 3))
        self.slice_overlap = float(getattr(args, "slice_overlap", 0.1))
        self.use_tracker = getattr(args, "tracker", "hybridsort") != "none"

        self.show_id = bool(getattr(args, "show_id", True))
        self.show_conf = bool(getattr(args, "show_conf", True))
        self.show_kpt = bool(getattr(args, "show_kpt", False))
        self.show_angle = bool(getattr(args, "show_angle", False))
        self.show_arrow = bool(getattr(args, "show_arrow", False))
        self.show_sectors = bool(getattr(args, "show_sectors", False))
        self.kpt_display = bool(getattr(args, "kpt_display", False))
        self.kpt_track = bool(getattr(args, "kpt_track", False))
        self.headless = not bool(getattr(args, "webui", False))
        self.kpt_bbox_conf = float(getattr(args, "kpt_bbox_conf", 0.3))
        self.kpt_bbox_padding = float(getattr(args, "kpt_bbox_padding", 0.3))
        self.kpt_bbox_padding_v = float(getattr(args, "kpt_bbox_padding_v", 0.4))
        self.kpt_bbox_upper_only = bool(getattr(args, "kpt_bbox_upper_only", True))

        self._panorama_ready = False
        self._panorama_init_lock = threading.Lock()
        self._timing_counter = 0
        self.profile_interval = max(0, int(getattr(args, "profile_interval", 30)))
        self._use_torch_panorama = _CUDA
        self.process_width = max(0, int(getattr(args, "process_width", 0)))
        self._process_size: Optional[Tuple[int, int]] = None
        self._angle_maps_synced = False
        self.direct_slice_remap = bool(getattr(args, "direct_slice_remap", False)) and self.headless
        self.direct_slice_remap_backend = str(getattr(args, "direct_slice_remap_backend", "cpu")).lower()
        self._direct_slice_maps_ready = False
        self._direct_slice_map1: List[np.ndarray] = []
        self._direct_slice_map2: List[np.ndarray] = []
        self._direct_slice_umat_map_x: List[cv2.UMat] = []
        self._direct_slice_umat_map_y: List[cv2.UMat] = []
        self._direct_slice_umat_stacked_map_x: Optional[cv2.UMat] = None
        self._direct_slice_umat_stacked_map_y: Optional[cv2.UMat] = None
        self._direct_slice_umat_stacked_dst: Optional[cv2.UMat] = None
        self._direct_slice_umat_stacked_dsts: List[cv2.UMat] = []
        self._direct_slice_umat_stacked_dst_cursor = 0
        self._direct_slice_stacked_roi_shape: Tuple[int, int] = (0, 0)
        self._direct_slice_stacked_rgb_buffer: Optional[np.ndarray] = None
        self._direct_remap_profile: Optional[Dict] = None
        self._direct_opencl_remap_dst_supported = True
        self._direct_slice_infos: List[dict] = []
        self._direct_slice_decode_metas: List[dict] = []
        self._direct_roi_rects: List[Tuple[int, int, int, int]] = []
        self._direct_slice_buffers: List[np.ndarray] = []
        self._direct_slice_packed_buffer: Optional[np.ndarray] = None
        self._direct_slice_packed_buffers: List[np.ndarray] = []
        self._direct_slice_buffer_cursor = 0
        self._direct_native_merge_failed = False
        self._direct_imgsz = 0
        self._direct_process_shape: Optional[Tuple[int, int, int]] = None

        self.slicer = PanoramaSlicer(
            overlap_ratio=self.slice_overlap,
            iou_threshold=0.2,
            reid_similarity_threshold=0.7,
            dedup_use_reid=bool(getattr(args, "dedup_use_reid", False)),
        )

        boundary_kwargs = dict(
            enable_boundary_matching=True,
            frame_width=3840,
            frame_height=1080,
            boundary_margin=0.04,
            boundary_time_window=90,
            boundary_similarity_thresh=0.3,
            boundary_debug=False,
            enable_top_boundary=False,
            enable_bottom_boundary=False,
            enable_left_boundary=True,
            enable_right_boundary=True,
        )

        # Edge deployment uses HybridSort without OSNet/ReID. That avoids
        # appearance-feature pollution when cropped bodies overlap, while still
        # keeping stable TrackIDs through IoU, motion, and short coasting.
        if self.use_tracker:
            self.tracker = HybridSortTracker(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=getattr(args, "tracker_new_thresh", 0.5),
                new_track_overlap_thresh=getattr(args, "new_track_overlap_thresh", 0.6),
                track_buffer=getattr(args, "track_buffer", 500),
                frame_rate=30,
                match_thresh=getattr(args, "tracker_match_thresh", 0.15),
                inertia=0.1,
                delta_t=3,
                use_byte=getattr(args, "tracker_byte", True),
                tcm_first_step=True,
                tcm_first_step_weight=1,
                tcm_byte_step=True,
                tcm_byte_step_weight=1,
                asso_func="iou",
                min_hits=1,
                cd_thresh=0.5,
                with_reid=False,
                panorama_width=getattr(args, "output_width", 3840),
                panorama_height=getattr(args, "output_height", 1080),
                smooth_bbox=getattr(args, "smooth_bbox", True),
                smooth_bbox_alpha=getattr(args, "smooth_bbox_alpha", 0.5),
                kalman_bbox=False,
                coast_frames=getattr(args, "coast_frames", 0),
                coast_hold=getattr(args, "coast_hold", False),
                **boundary_kwargs,
            )
        else:
            self.tracker = None

    def initialize(self) -> bool:
        """Load models that do not depend on the first frame size."""
        print("初始化边缘部署流水线（YOLO + HybridSort）...")
        os.makedirs(self.args.output_dir, exist_ok=True)
        if not os.path.exists(self.args.model_path):
            print(f"错误: YOLO 模型文件不存在: {self.args.model_path}")
            return False

        self.yolo_detector = YOLOPoseDetector(
            model_path=self.args.model_path,
            conf_threshold=self.args.conf_threshold,
            iou_threshold=self.args.iou_threshold,
            imgsz=getattr(self.args, "imgsz", 864),
            rknn_core_mask=getattr(self.args, "rknn_core_mask", "default"),
            rknn_parallel_slices=getattr(self.args, "rknn_parallel_slices", False),
        )
        if _CUDA:
            print(f"GPU 已就绪: {torch.cuda.get_device_name(0)}")
            model_path_lower = str(self.args.model_path).lower().rstrip("/\\")
            if not model_path_lower.endswith((".engine", "_rknn_model")):
                self.yolo_detector.model.to("cuda")
                print("YOLO 已移至 GPU")
            elif model_path_lower.endswith(".engine"):
                print("YOLO TensorRT 引擎已就绪")
            else:
                print("YOLO RKNN/NPU 后端已就绪")
        else:
            if str(self.args.model_path).lower().rstrip("/\\").endswith("_rknn_model"):
                print("未检测到 CUDA GPU；YOLO 模型使用 RKNN/NPU 后端，其他预处理使用 CPU")
            else:
                print("未检测到 CUDA GPU，使用 CPU 推理")

        if getattr(self.args, "use_osnet", False):
            print("[WebUI] 精简部署版不加载 OSNet；该参数仅为兼容旧命令保留")
        else:
            print("OSNet 特征提取已禁用")

        if bool(getattr(self.args, "direct_slice_remap", False)) and not self.headless:
            print("[direct-slice] WebUI 模式需要完整全景图显示，已忽略 --direct-slice-remap")

        print("模型加载完成！等待第一帧以完成鱼眼展开初始化...")
        return True

    @staticmethod
    def _npz_scalar(data, key: str, default=None):
        if key not in data:
            return default
        value = data[key]
        try:
            return value.item()
        except Exception:
            return value

    @staticmethod
    def _decode_direct_slice_metadata(data) -> Tuple[List[dict], List[dict]]:
        if "slice_infos_array" in data and "letterbox_infos_array" in data:
            slice_array = np.asarray(data["slice_infos_array"])
            letterbox_array = np.asarray(data["letterbox_infos_array"])
            slice_infos = []
            for row in slice_array:
                info = {}
                for key, value in zip(_SLICE_INFO_FIELDS, row):
                    if key == "wrap_around":
                        info[key] = bool(int(value))
                    else:
                        info[key] = int(value)
                slice_infos.append(info)

            letterbox_infos = []
            for row in letterbox_array:
                info = {}
                for key, value in zip(_LETTERBOX_INFO_FIELDS, row):
                    if key == "gain":
                        info[key] = float(value)
                    else:
                        info[key] = int(round(float(value)))
                letterbox_infos.append(info)
            return slice_infos, letterbox_infos

        if "slice_infos_json" in data and "letterbox_infos_json" in data:
            return (
                json.loads(str(data["slice_infos_json"].item())),
                json.loads(str(data["letterbox_infos_json"].item())),
            )

        # Compatibility only for the old local map file; do not rely on this
        # format across machines because dtype=object requires pickle.
        return (
            [dict(item) for item in data["slice_infos"]],
            [dict(item) for item in data["letterbox_infos"]],
        )

    def _load_direct_slice_maps(self, path: str) -> bool:
        if not os.path.exists(path):
            print(f"[direct-slice] 切片映射矩阵不存在: {path}")
            return False
        try:
            data = np.load(path)
            slice_map_x = np.asarray(data["slice_map_x"], dtype=np.float32)
            slice_map_y = np.asarray(data["slice_map_y"], dtype=np.float32)
            if slice_map_x.shape != slice_map_y.shape or slice_map_x.ndim != 3:
                print(f"[direct-slice] map shape 无效: x={slice_map_x.shape}, y={slice_map_y.shape}")
                return False
            if slice_map_x.shape[0] != self.num_slices:
                print(
                    f"[direct-slice] map 切片数={slice_map_x.shape[0]} 与 --num-slices={self.num_slices} 不一致"
                )
                return False

            imgsz = int(self._npz_scalar(data, "imgsz", slice_map_x.shape[1]))
            if self.yolo_detector is not None and imgsz != int(self.yolo_detector.imgsz):
                print(
                    f"[direct-slice] map imgsz={imgsz} 与模型 imgsz={self.yolo_detector.imgsz} 不一致"
                )
                return False

            slice_infos, letterbox_infos = self._decode_direct_slice_metadata(data)
            if len(slice_infos) != self.num_slices or len(letterbox_infos) != self.num_slices:
                print("[direct-slice] slice_infos/letterbox_infos 数量不一致")
                return False

            self._direct_slice_map1 = []
            self._direct_slice_map2 = []
            self._direct_slice_umat_map_x = []
            self._direct_slice_umat_map_y = []
            self._direct_slice_umat_stacked_map_x = None
            self._direct_slice_umat_stacked_map_y = None
            self._direct_slice_umat_stacked_dst = None
            self._direct_slice_umat_stacked_dsts = []
            self._direct_slice_umat_stacked_dst_cursor = 0
            self._direct_slice_stacked_roi_shape = (0, 0)
            self._direct_slice_stacked_rgb_buffer = None
            self._direct_remap_profile = None
            for mx, my in zip(slice_map_x, slice_map_y):
                map1, map2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
                self._direct_slice_map1.append(map1)
                self._direct_slice_map2.append(map2)
                self._direct_slice_umat_map_x.append(cv2.UMat(np.ascontiguousarray(mx)))
                self._direct_slice_umat_map_y.append(cv2.UMat(np.ascontiguousarray(my)))
            roi_h, roi_w = int(slice_map_x.shape[1]), int(slice_map_x.shape[2])
            stacked_x = np.ascontiguousarray(slice_map_x.reshape(self.num_slices * roi_h, roi_w))
            stacked_y = np.ascontiguousarray(slice_map_y.reshape(self.num_slices * roi_h, roi_w))
            self._direct_slice_umat_stacked_map_x = cv2.UMat(stacked_x)
            self._direct_slice_umat_stacked_map_y = cv2.UMat(stacked_y)
            self._direct_slice_stacked_roi_shape = (roi_h, roi_w)

            self._direct_slice_infos = slice_infos
            self._direct_slice_decode_metas = []
            self._direct_roi_rects = []
            for info, lb in zip(slice_infos, letterbox_infos):
                left = int(lb.get("left", 0))
                top = int(lb.get("top", 0))
                new_w = int(lb.get("new_width", imgsz))
                new_h = int(lb.get("new_height", imgsz))
                self._direct_slice_decode_metas.append({
                    "slice_shape": (int(info["slice_height"]), int(info["slice_width"])),
                    "gain": float(lb["gain"]),
                    "pad": (left, top),
                })
                self._direct_roi_rects.append((left, top, new_w, new_h))

            process_w = int(self._npz_scalar(data, "process_width", slice_infos[0]["original_width"]))
            process_h = int(self._npz_scalar(data, "process_height", slice_infos[0]["original_height"]))
            self._process_size = (process_w, process_h)
            self._direct_process_shape = (process_h, process_w, 3)
            self._direct_imgsz = imgsz
            self._direct_slice_packed_buffer = np.full(
                (self.num_slices, imgsz, imgsz, 3),
                114,
                dtype=np.uint8,
            )
            self._direct_slice_packed_buffers = [
                self._direct_slice_packed_buffer,
                np.full((self.num_slices, imgsz, imgsz, 3), 114, dtype=np.uint8),
            ]
            self._direct_slice_buffer_cursor = 0
            self._direct_slice_buffers = [self._direct_slice_packed_buffer[i] for i in range(self.num_slices)]

            self.panorama_processor = type("DirectSlicePanoramaState", (), {})()
            self.panorama_processor.center = tuple(int(v) for v in data["center"]) if "center" in data else None
            self.panorama_processor.radius = int(self._npz_scalar(data, "radius", 0) or 0)
            self.panorama_processor.img_width = int(self._npz_scalar(data, "img_width", 0) or 0)
            self.panorama_processor.img_height = int(self._npz_scalar(data, "img_height", 0) or 0)
            self.panorama_processor.output_width = int(self._npz_scalar(data, "base_output_width", process_w))
            self.panorama_processor.output_height = int(self._npz_scalar(data, "base_output_height", process_h))

            base_map_file = str(self._npz_scalar(data, "base_map_file", "") or "")
            self.panorama_processor.map_x = None
            self.panorama_processor.map_y = None
            if base_map_file and os.path.exists(base_map_file):
                base = np.load(base_map_file)
                if "map_x" in base and "map_y" in base:
                    self.panorama_processor.map_x = np.asarray(base["map_x"], dtype=np.float32)
                    self.panorama_processor.map_y = np.asarray(base["map_y"], dtype=np.float32)

            if self.panorama_processor.map_x is None or self.panorama_processor.map_y is None:
                print("[direct-slice] 找不到 base_map_file，角度映射将使用标定公式兜底")

            self._direct_slice_maps_ready = True
            backend = self.direct_slice_remap_backend
            if backend == "opencl":
                cv2.ocl.setUseOpenCL(True)
                if not cv2.ocl.haveOpenCL():
                    print("[direct-slice] OpenCL 不可用，回退 CPU remap")
                    self.direct_slice_remap_backend = "cpu"
                else:
                    print(
                        f"[direct-slice] OpenCL remap 已启用: "
                        f"useOpenCL={cv2.ocl.useOpenCL()}"
                    )
                    print(
                        f"[direct-slice] OpenCL stacked remap 已启用: "
                        f"stacked_roi={roi_w}x{roi_h * self.num_slices}"
                    )
            print(
                f"[direct-slice] 已加载切片映射: {path} "
                f"slices={slice_map_x.shape[0]} roi={slice_map_x.shape[2]}x{slice_map_x.shape[1]} "
                f"imgsz={imgsz} process={process_w}x{process_h} backend={self.direct_slice_remap_backend}"
            )
            return True
        except Exception as exc:
            print(f"[direct-slice] 加载切片映射失败: {type(exc).__name__}: {exc}")
            return False

    def _ensure_direct_slice_buffers(self) -> List[np.ndarray]:
        imgsz = int(self._direct_imgsz)
        expected_shape = (imgsz, imgsz, 3)
        packed_shape = (self.num_slices, imgsz, imgsz, 3)
        buffers_ok = (
            len(self._direct_slice_packed_buffers) >= 2
            and all(buf.shape == packed_shape and buf.dtype == np.uint8 for buf in self._direct_slice_packed_buffers)
        )
        if (
            imgsz <= 0
            or self._direct_slice_packed_buffer is None
            or self._direct_slice_packed_buffer.shape != packed_shape
            or self._direct_slice_packed_buffer.dtype != np.uint8
            or not buffers_ok
            or len(self._direct_slice_buffers) != self.num_slices
            or any(buf.shape != expected_shape or buf.dtype != np.uint8 for buf in self._direct_slice_buffers)
        ):
            self._direct_slice_packed_buffer = np.full(packed_shape, 114, dtype=np.uint8)
            self._direct_slice_packed_buffers = [
                self._direct_slice_packed_buffer,
                np.full(packed_shape, 114, dtype=np.uint8),
            ]
            self._direct_slice_buffer_cursor = 0
            self._direct_slice_buffers = [self._direct_slice_packed_buffer[i] for i in range(self.num_slices)]
        return self._direct_slice_buffers

    def _next_direct_slice_buffers(self) -> List[np.ndarray]:
        imgsz = int(self._direct_imgsz)
        packed_shape = (self.num_slices, imgsz, imgsz, 3)
        if (
            imgsz <= 0
            or len(self._direct_slice_packed_buffers) < 2
            or any(buf.shape != packed_shape or buf.dtype != np.uint8 for buf in self._direct_slice_packed_buffers)
        ):
            self._direct_slice_packed_buffer = np.full(packed_shape, 114, dtype=np.uint8)
            self._direct_slice_packed_buffers = [
                self._direct_slice_packed_buffer,
                np.full(packed_shape, 114, dtype=np.uint8),
            ]
            self._direct_slice_buffer_cursor = 0
        else:
            self._direct_slice_buffer_cursor = (self._direct_slice_buffer_cursor + 1) % len(self._direct_slice_packed_buffers)
            self._direct_slice_packed_buffer = self._direct_slice_packed_buffers[self._direct_slice_buffer_cursor]
        self._direct_slice_packed_buffer.fill(114)
        self._direct_slice_buffers = [self._direct_slice_packed_buffer[i] for i in range(self.num_slices)]
        return self._direct_slice_buffers

    def _next_stacked_umat_dst(self, shape: Tuple[int, int, int]) -> cv2.UMat:
        buffers_ok = (
            len(self._direct_slice_umat_stacked_dsts) >= 2
            and self._direct_slice_umat_stacked_dst is not None
        )
        if not buffers_ok:
            self._direct_slice_umat_stacked_dsts = [
                cv2.UMat(np.empty(shape, dtype=np.uint8)),
                cv2.UMat(np.empty(shape, dtype=np.uint8)),
            ]
            self._direct_slice_umat_stacked_dst_cursor = 0
        else:
            self._direct_slice_umat_stacked_dst_cursor = (
                self._direct_slice_umat_stacked_dst_cursor + 1
            ) % len(self._direct_slice_umat_stacked_dsts)
        self._direct_slice_umat_stacked_dst = self._direct_slice_umat_stacked_dsts[
            self._direct_slice_umat_stacked_dst_cursor
        ]
        return self._direct_slice_umat_stacked_dst

    @staticmethod
    def _write_rgb_roi(dst: np.ndarray, roi_bgr: np.ndarray, width: int, height: int) -> None:
        src = roi_bgr[:height, :width]
        if dst.flags.c_contiguous:
            cv2.cvtColor(src, cv2.COLOR_BGR2RGB, dst=dst)
        else:
            dst[:] = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

    def _init_panorama_from_frame(self, frame: np.ndarray) -> bool:
        """
        Lazily initialize panorama maps from the first frame.

        The input camera resolution is only known at runtime, so the GPU remap
        grid and angle calculator are created after the first frame arrives.
        """
        with self._panorama_init_lock:
            if self._panorama_ready:
                return True
            h, w = frame.shape[:2]
            print(f"初始化全景处理器，输入分辨率: {w}x{h}")
            if self.direct_slice_remap:
                map_file = str(getattr(self.args, "direct_slice_map_file", ""))
                if not self._load_direct_slice_maps(map_file):
                    return False
                meta_w = int(getattr(self.panorama_processor, "img_width", 0) or 0)
                meta_h = int(getattr(self.panorama_processor, "img_height", 0) or 0)
                if (meta_w and meta_w != w) or (meta_h and meta_h != h):
                    print(
                        f"[direct-slice] 输入分辨率 {w}x{h} 与 map 元数据 {meta_w}x{meta_h} 不一致"
                    )
                    return False

                process_w, process_h = self._process_size or (self.args.output_width, self.args.output_height)
                model_name = os.path.basename(str(getattr(self.args, "model_path", ""))).lower()
                face_kpt = bool(getattr(self.args, "face_kpt", False)) or "face" in model_name
                self.angle_calculator = AngleCalculator(
                    process_w,
                    process_h,
                    self.args.vertical_fov,
                    fit_degree=getattr(self.args, "fit_degree", 4),
                    yaml_file=getattr(self.args, "calib_yaml", None),
                    feature_point_mode="mouth" if face_kpt else "nose",
                )
                if face_kpt:
                    print("人脸关键点模型：角度特征点使用嘴巴中心")
                if self.tracker is not None:
                    self.tracker.set_boundary_frame_size(process_w, process_h)
                self._panorama_ready = True
                self._sync_angle_calculator(frame)
                print("[direct-slice] headless 快速路径已启用：跳过完整全景展开/切片/letterbox")
                return True

            panorama_cls = FisheyePanoramaGPU if self._use_torch_panorama else FisheyePanorama
            self.panorama_processor = panorama_cls(
                cam_width=w,
                cam_height=h,
                output_width=self.args.output_width,
                output_height=self.args.output_height,
                vertical_fov=self.args.vertical_fov,
                map_file=self.args.map_file,
                cam_index=None,
            )
            if not self._use_torch_panorama:
                print("RK3588/CPU 路径：鱼眼展开使用 OpenCV remap，避免 PyTorch CPU grid_sample 大图耗时")
            if not self.panorama_processor.init_from_frame(frame):
                print("全景映射矩阵初始化失败")
                return False

            out_w = self.panorama_processor.output_width
            out_h = self.panorama_processor.output_height
            print(f"全景处理器就绪，全景输出: {out_w}x{out_h}")
            crop_h = out_h // self.args.crop_divisor if self.args.crop_divisor > 0 else 0
            process_h = out_h - crop_h
            process_w = self.process_width if self.process_width > 0 else out_w
            self._process_size = (int(process_w), int(process_h))
            if self.process_width > 0 and self.process_width != out_w:
                print(f"处理全景缩放启用: {out_w}x{process_h} -> {process_w}x{process_h}")

            model_name = os.path.basename(str(getattr(self.args, "model_path", ""))).lower()
            face_kpt = bool(getattr(self.args, "face_kpt", False)) or "face" in model_name
            # yolov8n-face has 5 facial keypoints. For meeting azimuth/elevation
            # display, the mouth center is more stable than the nose on this
            # fisheye setup, with fallbacks handled by AngleCalculator.
            self.angle_calculator = AngleCalculator(
                process_w,
                process_h,
                self.args.vertical_fov,
                fit_degree=getattr(self.args, "fit_degree", 4),
                yaml_file=getattr(self.args, "calib_yaml", None),
                feature_point_mode="mouth" if face_kpt else "nose",
            )
            if face_kpt:
                print("人脸关键点模型：角度特征点使用嘴巴中心")
            if self.tracker is not None:
                self.tracker.set_boundary_frame_size(process_w, process_h)
            self._panorama_ready = True
            return True

    def _sync_angle_calculator(self, frame: np.ndarray) -> None:
        """Copy fish-eye center/radius/maps into AngleCalculator after remap init."""
        if not (
            self.angle_calculator
            and hasattr(self.panorama_processor, "center")
            and self.panorama_processor.center is not None
        ):
            return
        fp = self.panorama_processor
        if (
            self.angle_calculator.fisheye_center != fp.center
            or self.angle_calculator.fisheye_radius != fp.radius
            or not self._angle_maps_synced
        ):
            self.angle_calculator.set_fisheye_mapping(
                fp.center,
                fp.radius,
                getattr(fp, "img_width", frame.shape[1]),
                getattr(fp, "img_height", frame.shape[0]),
            )
            if fp.map_x is not None and fp.map_y is not None:
                crop_h = 0
                map_h = int(fp.map_x.shape[0])
                if self.args.crop_divisor > 0:
                    crop_h = int(getattr(fp, "output_height", map_h)) // self.args.crop_divisor
                crop_h = max(0, min(crop_h, map_h))
                crop_end = int(getattr(fp, "output_height", map_h)) or map_h
                crop_end = max(crop_h, min(crop_end, map_h))

                map_x = fp.map_x[crop_h:crop_end]
                map_y = fp.map_y[crop_h:crop_end]
                if self._process_size is not None:
                    target_w, target_h = self._process_size
                    if (map_x.shape[1], map_x.shape[0]) != (target_w, target_h):
                        map_x = cv2.resize(
                            map_x,
                            (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        map_y = cv2.resize(
                            map_y,
                            (target_w, target_h),
                            interpolation=cv2.INTER_LINEAR,
                        )

                self.angle_calculator.set_panorama_maps(
                    np.ascontiguousarray(map_x.astype(np.float32, copy=False)),
                    np.ascontiguousarray(map_y.astype(np.float32, copy=False)),
                )
                # The injected maps already represent the cropped/processed
                # panorama coordinates used by detection and drawing.
                self.angle_calculator.set_crop_offset(0)
                self._angle_maps_synced = True

    def _postprocess_detections(
        self,
        filtered: List[Dict],
        mark: Callable[[str], None],
    ) -> Tuple[List[Dict], Optional[Dict], Optional[Dict]]:
        if self.kpt_track:
            for det in filtered:
                det["bbox"] = compute_stable_bbox_from_keypoints(
                    det.get("keypoints", []),
                    conf_thresh=self.kpt_bbox_conf,
                    padding=self.kpt_bbox_padding,
                    padding_v=self.kpt_bbox_padding_v,
                    fallback_bbox=det["bbox"],
                    upper_body_only=self.kpt_bbox_upper_only,
                )
        if self.use_tracker:
            tracked = self.tracker.update(filtered)
        else:
            tracked = filtered
            for i, det in enumerate(tracked):
                det["track_id"] = i + 1
        mark("跟踪")

        angle_info = None
        if tracked and self.angle_calculator:
            kpts_list = []
            for det in tracked:
                kp = det.get("keypoints")
                if kp:
                    kpts_list.append(np.array(kp))
                else:
                    x1, y1, x2, y2 = det["bbox"]
                    synth = np.zeros((17, 3), dtype=np.float32)
                    synth[0] = [(x1 + x2) / 2.0, y1 + 0.12 * (y2 - y1), 1.0]
                    kpts_list.append(synth)
            if kpts_list:
                angle_info = self.angle_calculator.calculate_angles_from_keypoints(
                    kpts_list
                )
                persons = angle_info.get("persons", [])
                for idx, det in enumerate(tracked):
                    current = persons[idx] if idx < len(persons) else None
                    if current is not None:
                        continue
                    x1, y1, x2, y2 = det["bbox"]
                    fallback = self.angle_calculator.calculate_angle_from_point(
                        (x1 + x2) / 2.0,
                        y1 + 0.12 * (y2 - y1),
                        person_id=idx,
                    )
                    if idx < len(persons):
                        persons[idx] = fallback
                    else:
                        persons.append(fallback)
        mark("角度")

        sectors = None
        show_sectors = self.show_sectors and not self.headless
        for det in tracked or []:
            det.pop("_sector_rep", None)
        if (getattr(self.args, "sector_output", False) or show_sectors) and tracked:
            sectors, rep_indices = aggregate_sectors(
                tracked,
                angle_info,
                getattr(self.args, "num_sectors", 8),
            )
            if getattr(self.args, "sector_output", False):
                for idx in rep_indices:
                    tracked[idx]["_sector_rep"] = True
        mark("扇区")
        return tracked, angle_info, sectors

    def detect_direct_slice_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Run direct-slice remap, RKNN detection, and slice merge only."""
        if not self.direct_slice_remap:
            raise RuntimeError("detect_direct_slice_frame requires --direct-slice-remap")
        if not self._panorama_ready and not self._init_panorama_from_frame(frame):
            return None

        t = []

        def mark(name: str) -> None:
            t.append((name, time.perf_counter()))

        mark("start")
        self._sync_angle_calculator(frame)
        mark("角度映射")

        slices = self._direct_remap_slices(frame)
        mark("直接切片展开")
        return self._detect_direct_slice_remapped(slices, t, self._direct_remap_profile)

    def _detect_direct_slice_remapped(
        self,
        slices: np.ndarray,
        t: List[Tuple[str, float]],
        remap_profile: Optional[Dict] = None,
    ) -> Dict:
        """Run RKNN detection and merge for already prepared YOLO slices."""
        def mark(name: str) -> None:
            t.append((name, time.perf_counter()))

        native_merge_profile = None
        with torch.no_grad():
            merged = None
            detect_merged = getattr(self.yolo_detector, "detect_preletterboxed_merged", None)
            if detect_merged is not None and not self._direct_native_merge_failed:
                try:
                    merged = detect_merged(
                        slices,
                        self._direct_slice_decode_metas,
                        self._direct_slice_infos,
                        input_format="rgb",
                        overlap_ratio=getattr(self.slicer, "overlap_ratio", self.slice_overlap),
                        merge_iou_threshold=getattr(self.slicer, "iou_threshold", None),
                        nms_iou_thresh=getattr(self.slicer, "nms_iou_thresh", 0.5),
                    )
                    native_merge_profile = getattr(self.yolo_detector, "last_native_merge_profile", None)
                except Exception as exc:
                    print(f"[direct-slice] native fused merge 不可用，回退原 merge-fast: {type(exc).__name__}: {exc}")
                    self._direct_native_merge_failed = True
                    merged = None
            if merged is None:
                all_yolo_results = self.yolo_detector.detect_preletterboxed_batch(
                    slices,
                    self._direct_slice_decode_metas,
                    input_format="rgb",
                )
        mark("YOLO")

        process_h, process_w = self._direct_process_shape[:2]
        if merged is None:
            merged = self.slicer.merge_detections(
                all_yolo_results,
                self._direct_slice_infos,
                slice_images=None,
                slice_tensors=None,
                feature_extractor=None,
            )
            merge_profile = getattr(self.slicer, "last_merge_profile", None)
        else:
            merge_profile = native_merge_profile
        filtered = filter_cross_boundary_detections(merged, (process_h, process_w, 3))
        filtered = self.slicer.filter_wide_detections(filtered, process_w)
        mark("合并过滤")

        rknn_profile = {
            "pre": getattr(self.yolo_detector, "last_rknn_pre_ms", 0.0),
            "infer": getattr(self.yolo_detector, "last_rknn_infer_ms", 0.0),
            "run": getattr(self.yolo_detector, "last_rknn_run_ms", 0.0),
            "out": getattr(self.yolo_detector, "last_rknn_output_ms", 0.0),
            "decode": getattr(self.yolo_detector, "last_rknn_decode_ms", 0.0),
            "post": getattr(self.yolo_detector, "last_rknn_post_ms", 0.0),
        } if getattr(self.yolo_detector, "direct_rknn", False) else None
        return {
            "filtered": filtered,
            "timing": t,
            "n_slices": len(slices),
            "panorama_shape": (process_h, process_w, 3),
            "yolo_batch_ms": getattr(self.yolo_detector, "last_batch_ms", None),
            "slice_shapes": getattr(self.yolo_detector, "last_batch_shapes", None),
            "merge_profile": merge_profile,
            "rknn_profile": rknn_profile,
            "remap_profile": remap_profile if remap_profile is not None else self._direct_remap_profile,
        }

    def start_direct_slice_frame_remap(self, frame: np.ndarray) -> Optional[Dict]:
        """Submit OpenCL direct-slice remap work and return a pending remap token."""
        if not self.direct_slice_remap:
            raise RuntimeError("start_direct_slice_frame_remap requires --direct-slice-remap")
        if not self._panorama_ready and not self._init_panorama_from_frame(frame):
            return None
        t = []

        def mark(name: str) -> None:
            t.append((name, time.perf_counter()))

        mark("start")
        self._sync_angle_calculator(frame)
        mark("角度映射")
        remap_state = self._direct_remap_slices_start(frame)
        return {
            "timing": t,
            "remap_state": remap_state,
        }

    def finish_direct_slice_frame_remap(self, pending: Dict) -> Optional[Dict]:
        """Finish a pending remap token and return slices ready for RKNN."""
        if pending is None:
            return None
        t = list(pending.get("timing") or [])
        slices = self._direct_remap_slices_finish(pending["remap_state"])
        finish_t = time.perf_counter()
        remap_profile = self._direct_remap_profile
        active_ms = float((remap_profile or {}).get("total", 0.0) or 0.0)
        wall_ms = float((remap_profile or {}).get("wall", active_ms) or active_ms)
        if t and wall_ms > active_ms + 0.1 and active_ms > 0.0:
            active_start = finish_t - active_ms / 1000.0
            if active_start > t[-1][1]:
                t.append(("remap预提交等待", active_start))
            t.append(("直接切片展开", finish_t))
        else:
            t.append(("直接切片展开", finish_t))
        return {
            "slices": slices,
            "timing": t,
            "remap_profile": remap_profile,
        }

    def detect_direct_slice_remapped(self, remapped: Dict) -> Optional[Dict]:
        """Run detection for slices returned by finish_direct_slice_frame_remap."""
        if remapped is None:
            return None
        return self._detect_direct_slice_remapped(
            remapped["slices"],
            list(remapped.get("timing") or []),
            remapped.get("remap_profile"),
        )

    def finish_direct_slice_detection(
        self,
        detection_result: Dict,
        print_profile: bool = False,
        extra_timing: Optional[List[Tuple[str, float]]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[List], Optional[Dict]]:
        """Run tracker/angle/sector for an already merged direct-slice result."""
        filtered = detection_result["filtered"]
        t = list(detection_result.get("timing") or [])

        def mark(name: str) -> None:
            t.append((name, time.perf_counter()))

        if extra_timing:
            t.extend(extra_timing)

        tracked, angle_info, _sectors = self._postprocess_detections(filtered, mark)
        process_h, process_w = detection_result["panorama_shape"][:2]
        annotated = None
        panorama = np.empty((process_h, process_w, 3), dtype=np.uint8)

        self._timing_counter += 1
        if print_profile:
            tracker_profile = (
                getattr(self.tracker, "last_profile", None)
                if self.use_tracker and self.tracker is not None
                else None
            )
            self._print_timing(
                t,
                len(filtered),
                int(detection_result.get("n_slices", 0)),
                detection_result["panorama_shape"],
                detection_result.get("yolo_batch_ms"),
                detection_result.get("slice_shapes"),
                detection_result.get("merge_profile"),
                detection_result.get("rknn_profile"),
                tracker_profile,
                detection_result.get("remap_profile"),
            )
        return panorama, None, annotated, tracked, angle_info

    def _direct_remap_slices(self, frame: np.ndarray) -> np.ndarray:
        if not self._direct_slice_maps_ready:
            raise RuntimeError("direct slice maps are not loaded")
        if self.direct_slice_remap_backend == "opencl":
            return self._direct_remap_slices_finish(self._direct_remap_slices_opencl_start(frame))
        return self._direct_remap_slices_finish(self._direct_remap_slices_cpu_start(frame))

    def _direct_remap_slices_start(self, frame: np.ndarray) -> Dict:
        if not self._direct_slice_maps_ready:
            raise RuntimeError("direct slice maps are not loaded")
        if self.direct_slice_remap_backend == "opencl":
            return self._direct_remap_slices_opencl_start(frame)
        return self._direct_remap_slices_cpu_start(frame)

    def _direct_remap_slices_finish(self, state: Dict) -> np.ndarray:
        mode = state.get("mode")
        if mode == "opencl_stacked":
            return self._direct_remap_slices_opencl_stacked_finish(state)
        if mode == "opencl":
            return self._direct_remap_slices_opencl_finish(state)
        if mode == "cpu":
            self._direct_remap_profile = state.get("profile")
            return state["slices"]
        raise RuntimeError(f"unknown direct remap state: {mode}")

    def _direct_remap_slices_cpu_start(self, frame: np.ndarray) -> Dict:
        profile_start = time.perf_counter()
        buffers = self._next_direct_slice_buffers()
        remap_ms = 0.0
        cvt_copy_ms = 0.0
        for idx, (map1, map2, roi) in enumerate(zip(self._direct_slice_map1, self._direct_slice_map2, self._direct_roi_rects)):
            left, top, new_w, new_h = roi
            image = buffers[idx]
            step_start = time.perf_counter()
            roi_img = cv2.remap(
                frame,
                map1,
                map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )
            remap_ms += (time.perf_counter() - step_start) * 1000
            step_start = time.perf_counter()
            self._write_rgb_roi(image[top:top + new_h, left:left + new_w], roi_img, new_w, new_h)
            cvt_copy_ms += (time.perf_counter() - step_start) * 1000
        profile = {
            "mode": "cpu",
            "remap": remap_ms,
            "cvt_copy": cvt_copy_ms,
            "total": (time.perf_counter() - profile_start) * 1000,
        }
        self._direct_remap_profile = profile
        return {
            "mode": "cpu",
            "slices": self._direct_slice_packed_buffer,
            "profile": profile,
        }

    def _direct_remap_slices_opencl_start(self, frame: np.ndarray) -> Dict:
        if (
            self._direct_slice_umat_stacked_map_x is not None
            and self._direct_slice_umat_stacked_map_y is not None
            and self._direct_slice_stacked_roi_shape[0] > 0
            and self._direct_slice_stacked_roi_shape[1] > 0
        ):
            return self._direct_remap_slices_opencl_stacked_start(frame)

        profile_start = time.perf_counter()
        frame_umat = cv2.UMat(frame)
        upload_ms = (time.perf_counter() - profile_start) * 1000
        remap_ms = 0.0
        roi_umats = []
        for idx, (map_x, map_y, roi) in enumerate(zip(
            self._direct_slice_umat_map_x,
            self._direct_slice_umat_map_y,
            self._direct_roi_rects,
        )):
            step_start = time.perf_counter()
            roi_umat = cv2.remap(
                frame_umat,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(114, 114, 114),
            )
            remap_ms += (time.perf_counter() - step_start) * 1000
            roi_umats.append(roi_umat)
        return {
            "mode": "opencl",
            "profile_start": profile_start,
            "upload_ms": upload_ms,
            "remap_ms": remap_ms,
            "roi_umats": roi_umats,
        }

    def _direct_remap_slices_opencl_finish(self, state: Dict) -> np.ndarray:
        profile_start = state["profile_start"]
        upload_ms = state.get("upload_ms", 0.0)
        remap_ms = state.get("remap_ms", 0.0)
        get_ms = 0.0
        cvt_copy_ms = 0.0
        buffers = self._next_direct_slice_buffers()
        for idx, (roi_umat, roi) in enumerate(zip(state["roi_umats"], self._direct_roi_rects)):
            left, top, new_w, new_h = roi
            image = buffers[idx]
            step_start = time.perf_counter()
            roi_img = roi_umat.get()
            get_ms += (time.perf_counter() - step_start) * 1000
            step_start = time.perf_counter()
            self._write_rgb_roi(image[top:top + new_h, left:left + new_w], roi_img, new_w, new_h)
            cvt_copy_ms += (time.perf_counter() - step_start) * 1000
        active_total = upload_ms + remap_ms + get_ms + cvt_copy_ms
        wall_total = (time.perf_counter() - profile_start) * 1000
        self._direct_remap_profile = {
            "mode": "opencl",
            "upload": upload_ms,
            "remap": remap_ms,
            "get": get_ms,
            "cvt_copy": cvt_copy_ms,
            "total": active_total,
            "wall": wall_total,
            "prefetch_wait": max(0.0, wall_total - active_total),
        }
        return self._direct_slice_packed_buffer

    def _direct_remap_slices_opencl_stacked_start(self, frame: np.ndarray) -> Dict:
        profile_start = time.perf_counter()
        frame_umat = cv2.UMat(frame)
        upload_ms = (time.perf_counter() - profile_start) * 1000

        roi_h, roi_w = self._direct_slice_stacked_roi_shape
        stacked_shape = (roi_h * self.num_slices, roi_w, 3)
        prep_start = time.perf_counter()
        if (
            self._direct_slice_stacked_rgb_buffer is None
            or self._direct_slice_stacked_rgb_buffer.shape != stacked_shape
            or self._direct_slice_stacked_rgb_buffer.dtype != np.uint8
        ):
            self._direct_slice_stacked_rgb_buffer = np.empty(stacked_shape, dtype=np.uint8)
        dst_umat = self._next_stacked_umat_dst(stacked_shape)
        prep_ms = (time.perf_counter() - prep_start) * 1000

        remap_start = time.perf_counter()
        if self._direct_opencl_remap_dst_supported:
            try:
                stacked_umat = cv2.remap(
                    frame_umat,
                    self._direct_slice_umat_stacked_map_x,
                    self._direct_slice_umat_stacked_map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(114, 114, 114),
                    dst=dst_umat,
                )
                if stacked_umat is None:
                    stacked_umat = dst_umat
            except (TypeError, cv2.error):
                self._direct_opencl_remap_dst_supported = False
                stacked_umat = cv2.remap(
                    frame_umat,
                    self._direct_slice_umat_stacked_map_x,
                    self._direct_slice_umat_stacked_map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(114, 114, 114),
                )
        else:
            stacked_umat = cv2.remap(
                frame_umat,
                self._direct_slice_umat_stacked_map_x,
                self._direct_slice_umat_stacked_map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(114, 114, 114),
                )
        if stacked_umat is None:
            stacked_umat = dst_umat
        remap_ms = (time.perf_counter() - remap_start) * 1000
        return {
            "mode": "opencl_stacked",
            "profile_start": profile_start,
            "upload_ms": upload_ms,
            "prep_ms": prep_ms,
            "remap_ms": remap_ms,
            "stacked_umat": stacked_umat,
            "roi_h": roi_h,
        }

    def _direct_remap_slices_opencl_stacked_finish(self, state: Dict) -> np.ndarray:
        profile_start = state["profile_start"]
        upload_ms = state.get("upload_ms", 0.0)
        prep_ms = state.get("prep_ms", 0.0)
        remap_ms = state.get("remap_ms", 0.0)
        stacked_umat = state["stacked_umat"]
        roi_h = int(state["roi_h"])
        get_start = time.perf_counter()
        stacked_bgr = stacked_umat.get()
        get_ms = (time.perf_counter() - get_start) * 1000

        cvt_start = time.perf_counter()
        stacked_rgb = self._direct_slice_stacked_rgb_buffer
        cv2.cvtColor(stacked_bgr, cv2.COLOR_BGR2RGB, dst=stacked_rgb)
        cvt_ms = (time.perf_counter() - cvt_start) * 1000

        copy_start = time.perf_counter()
        buffers = self._next_direct_slice_buffers()
        for idx, (left, top, new_w, new_h) in enumerate(self._direct_roi_rects):
            src_top = idx * roi_h
            src = stacked_rgb[src_top:src_top + new_h, :new_w]
            buffers[idx][top:top + new_h, left:left + new_w] = src
        copy_ms = (time.perf_counter() - copy_start) * 1000
        active_total = upload_ms + prep_ms + remap_ms + get_ms + cvt_ms + copy_ms
        wall_total = (time.perf_counter() - profile_start) * 1000
        self._direct_remap_profile = {
            "mode": "opencl_stacked",
            "upload": upload_ms,
            "prep": prep_ms,
            "remap": remap_ms,
            "get": get_ms,
            "cvt": cvt_ms,
            "copy": copy_ms,
            "total": active_total,
            "wall": wall_total,
            "prefetch_wait": max(0.0, wall_total - active_total),
        }
        return self._direct_slice_packed_buffer

    def process_panorama_slices(
        self,
        frame: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
               Optional[List], Optional[Dict]]:
        """
        Process one decoded BGR frame.

        Return format matches the original WebUI processor:
          (panorama, yolo_only, annotated, tracked, angle_info)
        where yolo_only is always None in this clean deployment path.
        """
        if not self._panorama_ready and not self._init_panorama_from_frame(frame):
            return None, None, None, None, None

        t = []

        def mark(name: str) -> None:
            t.append((name, time.perf_counter()))

        # Synchronizing CUDA every frame hurts latency. Only sync on sampled
        # timing frames; CPU transfers already provide the necessary ordering.
        should_profile = (
            self.profile_interval > 0
            and self._timing_counter % self.profile_interval == 0
        )
        will_time = _CUDA and should_profile
        sync = (lambda: torch.cuda.synchronize()) if will_time else (lambda: None)

        mark("start")
        if self.direct_slice_remap:
            detection_result = self.detect_direct_slice_frame(frame)
            if detection_result is None:
                return None, None, None, None, None
            return self.finish_direct_slice_detection(
                detection_result,
                print_profile=should_profile,
            )

        crop_h = 0
        crop_y_end = None
        if self.args.crop_divisor > 0:
            crop_h = int(getattr(self.panorama_processor, "output_height", 0)) // self.args.crop_divisor
            crop_y_end = int(getattr(self.panorama_processor, "output_height", 0)) or None

        if self._use_torch_panorama:
            # 1. CPU BGR frame -> GPU float tensor.
            frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)
            mark("输入转GPU")

            # 2. GPU fish-eye unwrap. The loaded map converts circular fisheye
            #    to a 3840x1080 panorama before the top crop is applied.
            panorama_tensor = self.panorama_processor.apply_panorama_gpu(frame_tensor)
            sync()
            mark("鱼眼展开")

            # 3. YOLO and tracking work on CPU numpy images after unwrap.
            panorama = (
                (panorama_tensor * 255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            mark("GPU到CPU")
        else:
            # RK3588 has no CUDA. OpenCV remap is significantly cheaper than
            # PyTorch CPU grid_sample for the full 3840x1080 panorama.
            panorama = self.panorama_processor.apply_panorama(
                frame,
                y_start=crop_h,
                y_end=crop_y_end,
            )
            mark("OpenCV鱼眼展开")

        # 4. Optional top crop then processing-width resize. The CPU remap can
        #    crop by map rows directly; the CUDA path crops after full unwrap.
        if self._use_torch_panorama and self.args.crop_divisor > 0:
            panorama = panorama[crop_h:crop_y_end, :]
            mark("裁剪全景")
        if (
            self._process_size is not None
            and (panorama.shape[1], panorama.shape[0]) != self._process_size
        ):
            panorama = cv2.resize(
                panorama,
                self._process_size,
                interpolation=cv2.INTER_AREA,
            )
            mark("处理缩放")
        self._sync_angle_calculator(frame)
        mark("角度映射")
        if self.angle_calculator:
            self.angle_calculator.set_crop_offset(0 if self._angle_maps_synced else crop_h)
        slices, slice_infos = self.slicer.slice_panorama(panorama, num_slices=self.num_slices)
        mark("裁剪切片")

        # 5. Batched slice inference. One YOLO call sees all slices, which is
        #    faster than running slice-by-slice and keeps GPU overhead lower.
        with torch.no_grad():
            all_yolo_results = self.yolo_detector.detect_batch(slices)
        sync()
        mark("YOLO")

        # 6. Convert slice-local boxes/keypoints back to panorama coordinates
        #    and remove duplicate detections in overlap and wrap-around regions.
        merged = self.slicer.merge_detections(
            all_yolo_results,
            slice_infos,
            slice_images=slices,
            slice_tensors=None,
            feature_extractor=None,
        )
        filtered = filter_cross_boundary_detections(merged, panorama.shape)
        filtered = self.slicer.filter_wide_detections(filtered, panorama.shape[1])
        mark("合并过滤")

        tracked, angle_info, sectors = self._postprocess_detections(filtered, mark)

        annotated = None
        if not self.headless:
            # 10. Draw TrackID/score labels for WebUI viewing/recording.
            annotated = draw_detections(
                panorama,
                tracked,
                self.tracker,
                show_id=self.show_id,
                show_conf=self.show_conf,
                use_kpt_bbox=self.kpt_display,
                kpt_bbox_conf=self.kpt_bbox_conf,
                kpt_bbox_padding=self.kpt_bbox_padding,
                kpt_bbox_padding_v=self.kpt_bbox_padding_v,
                kpt_bbox_upper_only=self.kpt_bbox_upper_only,
                draw_kpt=self.show_kpt,
            )
            mark("绘框")

            # 11. Optional angle and sector overlays for calibration/debug viewing.
            if angle_info is not None and (self.show_angle or self.show_arrow):
                annotated = self.angle_calculator.draw_angles_on_image(
                    annotated,
                    angle_info,
                    show_angle=self.show_angle,
                    show_arrow=self.show_arrow,
                )
            if self.show_sectors:
                annotated = draw_sector_grid(
                    annotated,
                    getattr(self.args, "num_sectors", 8),
                    sectors,
                    inplace=True,
                )
            mark("叠加绘制")

        self._timing_counter += 1
        if should_profile:
            yolo_batch_ms = getattr(self.yolo_detector, "last_batch_ms", None)
            slice_shapes = getattr(self.yolo_detector, "last_batch_shapes", None)
            merge_profile = getattr(self.slicer, "last_merge_profile", None)
            rknn_profile = {
                "pre": getattr(self.yolo_detector, "last_rknn_pre_ms", 0.0),
                "infer": getattr(self.yolo_detector, "last_rknn_infer_ms", 0.0),
                "run": getattr(self.yolo_detector, "last_rknn_run_ms", 0.0),
                "out": getattr(self.yolo_detector, "last_rknn_output_ms", 0.0),
                "decode": getattr(self.yolo_detector, "last_rknn_decode_ms", 0.0),
                "post": getattr(self.yolo_detector, "last_rknn_post_ms", 0.0),
            } if getattr(self.yolo_detector, "direct_rknn", False) else None
            tracker_profile = (
                getattr(self.tracker, "last_profile", None)
                if self.use_tracker and self.tracker is not None
                else None
            )
            self._print_timing(
                t,
                len(filtered),
                len(slices),
                panorama.shape,
                yolo_batch_ms,
                slice_shapes,
                merge_profile,
                rknn_profile,
                tracker_profile,
            )
        return panorama, None, annotated, tracked, angle_info

    @staticmethod
    def _print_timing(
        t: list,
        n_det: int,
        n_slices: int,
        panorama_shape: tuple,
        yolo_batch_ms: Optional[float] = None,
        slice_shapes: Optional[list] = None,
        merge_profile: Optional[dict] = None,
        rknn_profile: Optional[dict] = None,
        tracker_profile: Optional[dict] = None,
        remap_profile: Optional[dict] = None,
    ) -> None:
        if len(t) < 2:
            return
        parts = []
        durations = []
        for (prev_name, prev_t), (name, cur_t) in zip(t, t[1:]):
            duration_ms = (cur_t - prev_t) * 1000
            parts.append(f"{name}={duration_ms:.1f}ms")
            durations.append((duration_ms, name))
        total = (t[-1][1] - t[0][1]) * 1000
        slow = sorted(durations, reverse=True)[:3]
        slow_text = ", ".join(f"{name}:{duration_ms:.1f}ms" for duration_ms, name in slow)
        h, w = panorama_shape[:2]
        yolo_text = ""
        if yolo_batch_ms is not None:
            yolo_text = f" yolo_predict={float(yolo_batch_ms):.1f}ms"
        shapes_text = ""
        if slice_shapes:
            unique_shapes = sorted(set(tuple(shape) for shape in slice_shapes))
            shapes_text = " slice_shapes=" + ",".join(f"{sw}x{sh}" for sh, sw in unique_shapes)
        remap_text = ""
        if remap_profile:
            mode = remap_profile.get("mode", "")
            detail_parts = [f"mode:{mode}"] if mode else []
            for key in (
                "upload",
                "prep",
                "remap",
                "get",
                "cvt",
                "copy",
                "cvt_copy",
                "total",
                "wall",
                "prefetch_wait",
            ):
                if key in remap_profile:
                    detail_parts.append(f"{key}:{float(remap_profile.get(key, 0.0)):.1f}")
            remap_text = " remap=" + " ".join(detail_parts)
        merge_text = ""
        if merge_profile:
            mode_text = " mode:fused" if merge_profile.get("fused") else ""
            merge_text = (
                " merge="
                f"raw:{merge_profile.get('raw', 0)} "
                f"nms:{merge_profile.get('nms_kept', merge_profile.get('kept', 0))} "
                f"kept:{merge_profile.get('kept', 0)} "
                f"extract:{merge_profile.get('extract', 0.0):.1f} "
                f"coords:{merge_profile.get('coords', 0.0):.1f} "
                f"dedup:{merge_profile.get('dedup', 0.0):.1f} "
                f"nms_ms:{merge_profile.get('nms', 0.0):.1f} "
                f"final:{merge_profile.get('final_dedup', 0.0):.1f} "
                f"total:{merge_profile.get('total', 0.0):.1f}"
                f"{mode_text}"
            )
        rknn_text = ""
        if rknn_profile:
            rknn_text = (
                " rknn="
                f"pre:{rknn_profile.get('pre', 0.0):.1f} "
                f"infer:{rknn_profile.get('infer', 0.0):.1f} "
                f"run:{rknn_profile.get('run', 0.0):.1f} "
                f"out:{rknn_profile.get('out', 0.0):.1f} "
                f"decode:{rknn_profile.get('decode', 0.0):.1f} "
                f"post:{rknn_profile.get('post', 0.0):.1f}"
            )
        tracker_text = ""
        if tracker_profile:
            assignment_shapes = tracker_profile.get("assignment_shapes") or []
            shape_text = ""
            if assignment_shapes:
                shape_text = " shapes:" + ",".join(f"{r}x{c}" for r, c in assignment_shapes[:3])
            tracker_text = (
                " tracker="
                f"total:{tracker_profile.get('total', 0.0):.1f} "
                f"inner:{tracker_profile.get('inner_total', tracker_profile.get('inner_update', 0.0)):.1f} "
                f"pred:{tracker_profile.get('inner_predict', 0.0):.1f} "
                f"vec:{tracker_profile.get('inner_state_vectors', 0.0):.1f} "
                f"a1:{tracker_profile.get('inner_assoc_first', 0.0):.1f} "
                f"upd1:{tracker_profile.get('inner_update_first', 0.0):.1f} "
                f"byte:{tracker_profile.get('inner_assoc_byte', 0.0):.1f} "
                f"ocr:{tracker_profile.get('inner_assoc_ocr', 0.0):.1f} "
                f"miss:{tracker_profile.get('inner_mark_unmatched', 0.0):.1f} "
                f"newtrk:{tracker_profile.get('inner_new_tracks', 0.0):.1f} "
                f"collect:{tracker_profile.get('inner_collect', 0.0):.1f} "
                f"lap:{tracker_profile.get('assignment_ms', 0.0):.1f}/{tracker_profile.get('assignment_calls', 0)} "
                f"rev:{tracker_profile.get('reverse_iou', 0.0):.1f} "
                f"lost:{tracker_profile.get('lost_maintenance', 0.0):.1f} "
                f"build:{tracker_profile.get('output_build', 0.0):.1f} "
                f"smooth:{tracker_profile.get('bbox_smoothing', 0.0):.1f} "
                f"boundary:{tracker_profile.get('boundary_pre', 0.0) + tracker_profile.get('boundary_new', 0.0) + tracker_profile.get('boundary_post', 0.0):.1f} "
                f"state:d{tracker_profile.get('detections', 0)}/o{tracker_profile.get('output', 0)}/"
                f"trk{tracker_profile.get('inner_trackers', 0)}/lost{tracker_profile.get('lost_ids', 0)}/"
                f"new{tracker_profile.get('new_ids', 0)}/pool{tracker_profile.get('public_map', 0)}"
                f"{shape_text}"
            )
        print(
            f"[pipeline profile] total={total:.1f}ms det={n_det} "
            f"slices={n_slices} pano={w}x{h}{shapes_text}{yolo_text} | "
            + " ".join(parts)
            + f"{remap_text}{merge_text}{rknn_text}{tracker_text} | slowest={slow_text}"
        )

    def cleanup(self) -> None:
        if self.yolo_detector is not None:
            self.yolo_detector.release()
        if self.use_tracker:
            print_assignment_stats()
