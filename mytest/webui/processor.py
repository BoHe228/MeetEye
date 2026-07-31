"""
FisheyePanoramaYOLOPose — 核心 GPU 推理处理类（WebUI 版）
包含逐步耗时打印（每 30 帧一次），方便定位性能瓶颈。

与 main.py 的差异（刻意保留的，非遗漏）：
  · 不初始化摄像头（摄像头由 camera_client.py 远程推流）
  · 不调用 draw_yolo_only（WebUI 不需要纯检测流，节省 ~3ms/帧）
  · 假设 CUDA 始终可用（实时 GPU 服务器）

跟踪参数与 main.py 保持对齐，需修改时两处同步更新。
"""
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import config as _config

import cv2
import numpy as np
import torch

from core import (
    FisheyePanoramaGPU,
    YOLOPoseDetector,
    PanoramaSlicer,
    AngleCalculator,
    BoT_SORTTracker,
    HybridSortTracker,
    print_assignment_stats,
)
from utils import (
    DisplayManager,
    draw_detections,
    filter_cross_boundary_detections,
    compute_stable_bbox_from_keypoints,
    box_iou,
)
from utils.feature_extractor import FeatureExtractor
from utils.sector import aggregate_sectors, draw_sector_grid
from utils.display_id_manager import DisplayIdManager

try:
    from face_rec.face_rec_manager import FaceRecManager
    _FACE_REC_AVAILABLE = True
except Exception as _e:
    _FACE_REC_AVAILABLE = False

_CUDA = torch.cuda.is_available()


class FisheyePanoramaYOLOPose:
    """鱼眼全景/广角整图 YOLO 姿态检测 — GPU 版，几何参数从第一帧懒初始化"""

    def __init__(self, args):
        self.args = args
        self.projection_mode = str(
            getattr(args, 'projection_mode', 'fisheye-panorama') or 'fisheye-panorama'
        ).replace('_', '-')
        self.is_wide_angle = self.projection_mode in {'wide-angle', 'wideangle', 'wide'}
        self._active_model_path = self._resolve_active_model_path()
        self._active_yolo_imgsz = self._resolve_active_yolo_imgsz()
        self.panorama_processor: Optional[FisheyePanoramaGPU] = None
        self.yolo_detector: Optional[YOLOPoseDetector] = None
        self.recall_detector: Optional[YOLOPoseDetector] = None   # 补漏检测模型（--recall-boost）
        self.display_manager = None
        self.angle_calculator: Optional[AngleCalculator] = None
        self.feature_extractor: Optional[FeatureExtractor] = None

        self.show_angles = True
        self.show_angle_overview = False
        self.show_id: bool = getattr(args, 'show_id', True)
        self.show_conf: bool = getattr(args, 'show_conf', True)
        self.show_kpt: bool = getattr(args, 'show_kpt', False)
        self.show_angle: bool = getattr(args, 'show_angle', True)
        self.show_arrow: bool = getattr(args, 'show_arrow', True)
        self.kpt_display: bool = getattr(args, 'kpt_display', False)
        self.kpt_track: bool = getattr(args, 'kpt_track', False)
        self.kpt_bbox_conf: float = getattr(args, 'kpt_bbox_conf', 0.3)
        self.kpt_bbox_padding: float = getattr(args, 'kpt_bbox_padding', 0.15)
        self.kpt_bbox_upper_only: bool = getattr(args, 'kpt_bbox_upper_only', True)

        # 人脸识别
        self.face_rec: Optional[FaceRecManager] = None
        self._face_name_map: Dict[int, str] = {}
        self._prev_track_ids: set = set()
        self.display_id_manager = DisplayIdManager(
            max_count=getattr(args, 'display_id_max_count', 8),
            binding_ttl_frames=getattr(args, 'display_id_binding_ttl', 60),
            reuse_max_frames=getattr(args, 'display_id_reuse_max_frames', 900),
            fallback_min_frames=getattr(args, 'display_id_reuse_fallback_min_frames', 150),
            reuse_center_thresh=getattr(args, 'display_id_reuse_center_thresh', 2.0),
            reuse_size_ratio=getattr(args, 'display_id_reuse_size_ratio', 0.4),
        )

        self.num_slices: int = getattr(args, 'num_slices', 3)
        self.slice_overlap: float = getattr(args, 'slice_overlap', 0.05)
        self.use_tracker: bool = (args.tracker != 'none')
        self.no_display: bool = args.no_display

        self._panorama_ready = False
        self._panorama_init_lock = threading.Lock()
        self._timing_counter = 0

        self.slicer = PanoramaSlicer(
            overlap_ratio=self.slice_overlap,
            iou_threshold=0.2,
            reid_similarity_threshold=0.7,
            dedup_use_reid=getattr(args, 'dedup_use_reid', False),
        )

        # ── 边界匹配公共参数（与 main.py 保持对齐）──────────────────────
        _boundary_kwargs = dict(
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

        # 无 OSNet 特征时禁用 ReID 关联：零特征会让 embedding_distance 的余弦距离变 NaN，
        # 污染关联代价矩阵，导致高置信度目标也无法关联确认而被丢弃（见 HybridSORT association.py）。
        _reid_ok = getattr(args, 'use_osnet', True)

        if args.tracker == 'botsort':
            self.tracker = BoT_SORTTracker(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.5,
                track_buffer=500,
                frame_rate=30,
                match_thresh=args.botsort_match_thresh,
                appearance_thresh=args.appearance_thresh,
                reid_lost_thresh=args.reid_lost_thresh,
                with_reid=_reid_ok,
                use_hungarian=args.use_hungarian,
                kalman_bbox=getattr(args, 'kalman_bbox', False),
                coast_frames=getattr(args, 'coast_frames', 0),
                coast_hold=getattr(args, 'coast_hold', False),
                **_boundary_kwargs,
            )
        elif args.tracker == 'hybridsort':
            self.tracker = HybridSortTracker(
                # ── 置信度阈值（与 main.py 对齐）──────────────────────────
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.5,
                # ── 轨迹生命周期 ──────────────────────────────────────────
                track_buffer=500,          # WebUI 长时运行，保留更多历史
                frame_rate=30,
                # ── 关联阈值（与 main.py 对齐）────────────────────────────
                match_thresh=0.15,
                # ── Hybrid-SORT 专有参数（与 main.py 对齐）───────────────
                inertia=0.1,
                delta_t=3,
                use_byte=True,
                tcm_first_step=True,
                tcm_first_step_weight=1,
                tcm_byte_step=True,
                tcm_byte_step_weight=1,
                asso_func="iou",
                min_hits=1,
                # Round 3.5 兜底
                cd_thresh=0.5,
                # ── ReID ──────────────────────────────────────────────────
                with_reid=(args.use_reid and _reid_ok),
                reid_emb_weight_high=args.reid_emb_weight_high,
                reid_emb_weight_low=args.reid_emb_weight_low,
                new_track_overlap_thresh=getattr(args, 'new_track_overlap_thresh', 0.6),
                # ── 全景图尺寸 ─────────────────────────────────────────────
                panorama_width=3840,
                panorama_height=1080,
                # ── 框平滑 ─────────────────────────────────────────────────
                smooth_bbox=getattr(args, 'smooth_bbox', True),
                smooth_bbox_alpha=getattr(args, 'smooth_bbox_alpha', 0.5),
                # ── Kalman 轨迹框 ───────────────────────────────────────────
                kalman_bbox=getattr(args, 'kalman_bbox', False),
                coast_frames=getattr(args, 'coast_frames', 0),
                coast_hold=getattr(args, 'coast_hold', False),
                **_boundary_kwargs,
            )
        else:
            self.tracker = None

        _osnet_path = _config.OSNET_WEIGHT_MAP.get(args.osnet_model, '')
        self._osnet_model_name = _config.OSNET_ARCH_MAP.get(args.osnet_model, args.osnet_model)
        self._osnet_model_path = _osnet_path if os.path.exists(_osnet_path) else None

    def _resolve_active_model_path(self) -> str:
        if self.is_wide_angle:
            wide_model = str(getattr(self.args, 'wide_model_path', '') or '').strip()
            if wide_model:
                return wide_model
        return str(getattr(self.args, 'model_path', '') or '')

    def _resolve_active_yolo_imgsz(self) -> int:
        if self.is_wide_angle:
            wide_imgsz = int(getattr(self.args, 'wide_yolo_imgsz', 0) or 0)
            if wide_imgsz > 0:
                return wide_imgsz
        return int(getattr(self.args, 'yolo_imgsz', 864) or 864)

    # ──────────────────────────────────────────────────────────────────
    def initialize(self) -> bool:
        """加载 YOLO 和 OSNet 模型（不初始化摄像头，摄像头由远端推流）"""
        print("初始化模型（YOLO + OSNet）...")
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)

        if not os.path.exists(self._active_model_path):
            print(f"错误: YOLO 模型文件不存在: {self._active_model_path}")
            return False

        print(
            f"YOLO 投影模式: {self.projection_mode}, "
            f"模型: {self._active_model_path}, imgsz={self._active_yolo_imgsz}"
        )
        self.yolo_detector = YOLOPoseDetector(
            model_path=self._active_model_path,
            conf_threshold=self.args.conf_threshold,
            iou_threshold=self.args.iou_threshold,
            imgsz=self._active_yolo_imgsz,
        )
        if _CUDA:
            print(f"GPU 已就绪: {torch.cuda.get_device_name(0)}")
            if not self._active_model_path.endswith('.engine'):
                self.yolo_detector.model.to('cuda')
                print("YOLO 已移至 GPU（FP16 由推理时 half=True 控制）")
            else:
                print("YOLO TensorRT 引擎已就绪（GPU 绑定在导出时完成）")
        else:
            print("未检测到 CUDA GPU，使用 CPU 推理")

        # ── 补漏检测模型（--recall-boost）─────────────────────────────
        # 默认关闭，关闭时不加载、不推理，对实时性零影响。
        self.recall_detector = None
        if getattr(self.args, 'recall_boost', False):
            if not os.path.exists(self.args.recall_model):
                print(f"警告: 补漏模型 {self.args.recall_model} 不存在，补漏检测已禁用")
            elif os.path.abspath(self.args.recall_model) == os.path.abspath(self._active_model_path):
                print("警告: 补漏模型与主模型相同，补漏无意义，已禁用补漏检测")
            else:
                _recall_conf = getattr(self.args, 'recall_conf_threshold', 0.4)
                self.recall_detector = YOLOPoseDetector(
                    model_path=self.args.recall_model,
                    conf_threshold=_recall_conf,
                    iou_threshold=self.args.iou_threshold,
                    imgsz=getattr(self.args, 'yolo_imgsz', 864),
                )
                if _CUDA and not self.args.recall_model.endswith('.engine'):
                    self.recall_detector.model.to('cuda')
                print(f"补漏检测已启用: {self.args.recall_model}"
                      f"（置信度阈值={_recall_conf}, IoU 关联阈值={self.args.recall_match_iou}）")

        self.display_manager = DisplayManager(
            use_dual_windows=self.args.use_dual_windows,
            no_display=self.no_display,
        )

        if getattr(self.args, 'use_osnet', True):
            print(f"初始化 OSNet 特征提取器 ({self._osnet_model_name})...")
            try:
                self.feature_extractor = FeatureExtractor(
                    model_name=self._osnet_model_name,
                    model_path=self._osnet_model_path,
                )
                if _CUDA:
                    self.feature_extractor.extractor.model.to('cuda')
                    self.feature_extractor.device = 'cuda'
                    print("OSNet 已移至 GPU（FP32，BatchNorm 不支持 FP16）")
            except Exception as e:
                print(f"OSNet 初始化失败: {e}，将不使用 ReID 特征")
                self.feature_extractor = None
        else:
            print("OSNet 特征提取已禁用（--no-use-osnet）")
            self.feature_extractor = None

        if getattr(self.args, 'use_face_rec', False):
            if not _FACE_REC_AVAILABLE:
                print("警告: FaceRecManager 不可用，跳过人脸识别初始化")
            else:
                try:
                    self.face_rec = FaceRecManager(
                        model_path=self.args.face_rec_model,
                        library_dir=self.args.face_library_dir,
                        threshold=getattr(self.args, 'face_rec_threshold', 0.35),
                        frontal_yaw_thresh=getattr(self.args, 'face_frontal_threshold', 0.35),
                        cooldown_frames=getattr(self.args, 'face_rec_cooldown', 30),
                        device='cuda' if _CUDA else 'cpu',
                    )
                except Exception as e:
                    print(f"警告: 人脸识别初始化失败: {e}")
                    self.face_rec = None

        print("模型加载完成！等待第一帧以完成全景处理器初始化...")
        return True

    # ──────────────────────────────────────────────────────────────────
    def _init_panorama_from_frame(self, frame: np.ndarray) -> bool:
        """使用第一帧图像懒初始化 GPU 全景处理器"""
        with self._panorama_init_lock:
            if self._panorama_ready:
                return True

            if self.is_wide_angle:
                return self._init_wide_angle_from_frame(frame)

            h, w = frame.shape[:2]
            print(f"初始化全景处理器，输入分辨率: {w}×{h}")

            self.panorama_processor = FisheyePanoramaGPU(
                cam_width=w, cam_height=h,
                output_width=self.args.output_width,
                output_height=self.args.output_height,
                vertical_fov=self.args.vertical_fov,
                map_file=self.args.map_file,
                cam_index=None,
            )
            if not self.panorama_processor.init_from_frame(frame):
                print("GPU 映射矩阵初始化失败")
                return False

            out_w = self.panorama_processor.output_width
            out_h = self.panorama_processor.output_height
            print(f"全景处理器就绪，全景输出: {out_w}×{out_h}")

            # 人脸关键点模型（5 点）→ 角度特征点用嘴巴中心；模型名含 face 时自动开启
            _mp = str(self._active_model_path).lower()
            _face_kpt = getattr(self.args, 'face_kpt', False) or ('face' in os.path.basename(_mp))
            self.angle_calculator = AngleCalculator(
                out_w, out_h, self.args.vertical_fov,
                fit_degree=getattr(self.args, 'fit_degree', 5),
                yaml_file=getattr(self.args, 'calib_yaml', None),
                feature_point_mode='mouth' if _face_kpt else 'nose',
            )
            if _face_kpt:
                print("人脸关键点模型：角度特征点使用嘴巴中心（左右嘴角中点）")

            if (self.tracker is not None
                    and self.tracker.enable_boundary_matching):
                self.tracker.set_boundary_frame_size(out_w, out_h)

            self._panorama_ready = True
            return True

    def _init_wide_angle_from_frame(self, frame: np.ndarray) -> bool:
        """广角模式：不做鱼眼展开，按原图尺寸初始化角度和边界尺度。"""
        h, w = frame.shape[:2]
        print(f"初始化广角整图处理，输入分辨率: {w}×{h}")

        _mp = str(self._active_model_path).lower()
        _face_kpt = getattr(self.args, 'face_kpt', False) or ('face' in os.path.basename(_mp))
        wide_fx = getattr(self.args, 'wide_fx', 0.0)
        wide_fy = getattr(self.args, 'wide_fy', 0.0)
        wide_cx = getattr(self.args, 'wide_cx', -1.0)
        wide_cy = getattr(self.args, 'wide_cy', -1.0)
        self.angle_calculator = AngleCalculator(
            w, h, self.args.vertical_fov,
            fit_degree=getattr(self.args, 'fit_degree', 5),
            yaml_file=getattr(self.args, 'calib_yaml', None),
            feature_point_mode='mouth' if _face_kpt else 'nose',
            projection_mode='wide-angle',
            wide_fx=None if wide_fx <= 0 else wide_fx,
            wide_fy=None if wide_fy <= 0 else wide_fy,
            wide_cx=None if wide_cx < 0 else wide_cx,
            wide_cy=None if wide_cy < 0 else wide_cy,
            wide_horizontal_fov=getattr(self.args, 'wide_horizontal_fov', 90.0),
        )
        print(
            "广角角度模式："
            f"fx={self.angle_calculator.wide_fx:.2f}, "
            f"fy={self.angle_calculator.wide_fy:.2f}, "
            f"cx={self.angle_calculator.wide_cx:.2f}, "
            f"cy={self.angle_calculator.wide_cy:.2f}"
        )
        if _face_kpt:
            print("人脸关键点模型：角度特征点使用嘴巴中心（左右嘴角中点）")

        if (self.tracker is not None
                and self.tracker.enable_boundary_matching):
            self.tracker.set_boundary_frame_size(w, h)
            for attr, value in (('panorama_width', w), ('panorama_height', h)):
                if hasattr(self.tracker, attr):
                    setattr(self.tracker, attr, value)

        self._panorama_ready = True
        return True

    # ──────────────────────────────────────────────────────────────────
    def process_panorama_slices(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[List], Optional[Dict]]:
        """
        完整推理流水线（每步均附有设备标注）：
          CPU→GPU  →  GPU 鱼眼展开  →  GPU→CPU
          →  CPU 裁剪切片  →  GPU YOLO批量推理
          →  CPU 合并过滤  →  GPU OSNet特征提取
          →  CPU 跟踪绘制  →  CPU 角度计算
        第二返回值固定 None（WebUI 不需要纯检测流）。
        """
        if not self._panorama_ready:
            if not self._init_panorama_from_frame(frame):
                return None, None, None, None, None

        if self.is_wide_angle:
            return self._process_wide_angle_frame(frame)

        t = {}
        # sync 仅为分阶段计时准确而存在；只有"本帧会打印计时"时才真正同步，
        # 其余帧 no-op，避免每帧强制 GPU 全同步增加延迟。（.cpu()/merge 自带必要同步，
        # 不影响正确性。）打印条件：自增后 %30==1 → 即当前 _timing_counter%30==0
        _will_time = _CUDA and (self._timing_counter % 30 == 0)
        sync = (lambda: torch.cuda.synchronize()) if _will_time else (lambda: None)

        # ① CPU → GPU  [CPU→GPU]
        t[0] = time.perf_counter()
        frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC → CHW
        t[1] = time.perf_counter()

        # ② GPU 鱼眼展开  [GPU]
        panorama_tensor = self.panorama_processor.apply_panorama_gpu(frame_tensor)
        sync()
        t[2] = time.perf_counter()

        # ③ GPU → CPU  [GPU→CPU]
        panorama = (
            (panorama_tensor * 255.0)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .cpu()
            .numpy()
        )
        t[3] = time.perf_counter()

        # ④ 裁剪 + 切片  [CPU]
        original_h = panorama.shape[0]
        self._sync_angle_calculator(frame)
        crop_h = 0
        if self.args.crop_divisor > 0:
            crop_h = original_h // self.args.crop_divisor
            panorama = panorama[crop_h:, :]
        if self.angle_calculator:
            self.angle_calculator.set_crop_offset(crop_h)
        slices, slice_infos = self.slicer.slice_panorama(panorama, num_slices=self.num_slices)

        slice_tensors_gpu = None
        if self.feature_extractor and _CUDA:
            pano_rgb = panorama_tensor[[2, 1, 0], crop_h:, :]
            pw = pano_rgb.shape[2]
            slice_tensors_gpu = []
            for info in slice_infos:
                sx, ex = info['start_x'], info['end_x']
                if info['wrap_around']:
                    if sx < 0:
                        st = torch.cat([pano_rgb[:, :, sx:], pano_rgb[:, :, :ex]], dim=2)
                    else:
                        st = torch.cat([pano_rgb[:, :, sx:pw], pano_rgb[:, :, :ex - pw]], dim=2)
                else:
                    st = pano_rgb[:, :, sx:ex]
                slice_tensors_gpu.append(st)
        t[4] = time.perf_counter()

        # ⑤ 批量 YOLO 推理  [GPU]
        with torch.no_grad():
            all_yolo_results = self.yolo_detector.detect_batch(slices)
            # 补漏路（--recall-boost）：同样的切片再跑一次纯检测模型；关闭时为 None，无开销
            recall_yolo_results = None
            if self.recall_detector is not None:
                recall_yolo_results = self.recall_detector.detect_batch(slices)
        sync()
        t[5] = time.perf_counter()

        # ⑥ 合并 + 过滤  [CPU + GPU]
        merged = self.slicer.merge_detections(
            all_yolo_results, slice_infos,
            slice_images=slices,
            slice_tensors=slice_tensors_gpu,
            feature_extractor=self.feature_extractor,
            recall_yolo_results=recall_yolo_results,
            recall_match_iou=getattr(self.args, 'recall_match_iou', 0.4),
        )
        filtered = filter_cross_boundary_detections(merged, panorama.shape)
        filtered = self.slicer.filter_wide_detections(filtered, panorama.shape[1])
        t[6] = time.perf_counter()

        panorama, annotated, tracked, angle_info = self._postprocess_detections(
            panorama, filtered, t
        )

        self._timing_counter += 1
        if self._timing_counter % 30 == 1:
            self._print_timing(t, len(filtered))

        return panorama, None, annotated, tracked, angle_info

    def _process_wide_angle_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
               Optional[np.ndarray], Optional[List], Optional[Dict]]:
        """广角模式：整图直接送入 YOLO，后续复用同一套跟踪/人脸/角度/绘制。"""
        t = {}
        _will_time = _CUDA and (self._timing_counter % 30 == 0)
        sync = (lambda: torch.cuda.synchronize()) if _will_time else (lambda: None)

        # ① 输入准备：广角模式不做鱼眼展开，直接使用原始解码图。
        t[0] = time.perf_counter()
        panorama = frame.copy()
        t[1] = time.perf_counter()
        t[2] = t[1]
        t[3] = t[1]

        # ② 广角整图输入准备。
        if self.angle_calculator:
            self.angle_calculator.set_crop_offset(0)
        t[4] = time.perf_counter()

        # ③ 整图 YOLO 推理。
        with torch.no_grad():
            yolo_results = self._detect_wide_full_frame(self.yolo_detector, panorama)
            recall_yolo_results = None
            if self.recall_detector is not None:
                recall_yolo_results = self._detect_wide_full_frame(self.recall_detector, panorama)
        sync()
        t[5] = time.perf_counter()

        # ④ 广角整图后处理。这里不走鱼眼切片合并/环绕过滤，只做整图框过滤、
        # 同帧重复框去重、可选补漏融合，以及 OSNet 特征批量提取。
        filtered = self._extract_wide_detections(yolo_results)
        filtered = self._dedup_wide_same_frame(filtered)
        filtered = self._merge_wide_recall_detections(filtered, recall_yolo_results)
        filtered = self._dedup_wide_same_frame(filtered)
        self._attach_wide_features(panorama, filtered)
        t[6] = time.perf_counter()

        panorama, annotated, tracked, angle_info = self._postprocess_detections(
            panorama, filtered, t
        )

        self._timing_counter += 1
        if self._timing_counter % 30 == 1:
            self._print_wide_timing(t, len(filtered))

        return panorama, None, annotated, tracked, angle_info

    def _detect_wide_full_frame(self, detector: YOLOPoseDetector, image: np.ndarray):
        return detector.model.predict(
            image,
            imgsz=self._active_yolo_imgsz,
            conf=self.args.conf_threshold,
            iou=self.args.iou_threshold,
            verbose=False,
            half=_CUDA,
            device=0 if _CUDA else 'cpu',
        )

    def _extract_wide_detections(self, yolo_results) -> List[Dict]:
        detections: List[Dict] = []
        for result in yolo_results or []:
            boxes = result.boxes
            if boxes is None:
                continue
            boxes_data = boxes.xyxy.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            class_ids = boxes.cls.cpu().numpy().astype(int)
            keypoints_data = []
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                keypoints_data = result.keypoints.data.cpu().numpy()
            names = getattr(result, 'names', {}) or {}
            multiclass = len(names) > 1
            for i, (box, conf, cls_id) in enumerate(zip(boxes_data, confidences, class_ids)):
                if isinstance(names, dict):
                    class_name = names.get(int(cls_id), str(cls_id))
                elif isinstance(names, (list, tuple)) and int(cls_id) < len(names):
                    class_name = names[int(cls_id)]
                else:
                    class_name = str(cls_id)
                if multiclass and class_name != 'person':
                    continue
                x1, y1, x2, y2 = map(float, box.tolist())
                if self._wide_bbox_too_small(x1, y1, x2, y2):
                    continue
                det = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(cls_id),
                    'class_name': class_name,
                }
                if i < len(keypoints_data):
                    det['keypoints'] = keypoints_data[i].tolist()
                detections.append(det)
        return detections

    def _wide_bbox_too_small(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        min_w = float(getattr(self.args, 'wide_min_det_width', 20) or 0)
        min_h = float(getattr(self.args, 'wide_min_det_height', 20) or 0)
        if min_w <= 0 and min_h <= 0:
            return False
        w, h = x2 - x1, y2 - y1
        return (min_w > 0 and w < min_w) or (min_h > 0 and h < min_h)

    @staticmethod
    def _box_min_iou(box1, box2) -> float:
        x1 = max(float(box1[0]), float(box2[0]))
        y1 = max(float(box1[1]), float(box2[1]))
        x2 = min(float(box1[2]), float(box2[2]))
        y2 = min(float(box1[3]), float(box2[3]))
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
        area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))
        return inter / (min(area1, area2) + 1e-6)

    def _dedup_wide_same_frame(self, detections: List[Dict]) -> List[Dict]:
        if len(detections) < 2:
            return detections

        min_iou_thresh = float(getattr(self.args, 'wide_dedup_min_iou', 0.7) or 0.0)
        iou_thresh = float(getattr(self.args, 'wide_dedup_iou', 0.95) or 0.0)
        if min_iou_thresh <= 0 and iou_thresh <= 0:
            return detections

        def _area(det: Dict) -> float:
            x1, y1, x2, y2 = det['bbox']
            return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))

        order = sorted(
            range(len(detections)),
            key=lambda idx: (float(detections[idx].get('confidence', 0.0)), _area(detections[idx])),
            reverse=True,
        )
        keep: List[int] = []
        for idx in order:
            box = detections[idx]['bbox']
            duplicate = False
            for kept_idx in keep:
                kept_box = detections[kept_idx]['bbox']
                same_by_iou = iou_thresh > 0 and box_iou(box, kept_box) >= iou_thresh
                same_by_min_iou = min_iou_thresh > 0 and self._box_min_iou(box, kept_box) >= min_iou_thresh
                if same_by_iou or same_by_min_iou:
                    duplicate = True
                    break
            if not duplicate:
                keep.append(idx)

        if len(keep) == len(detections):
            return detections
        keep.sort()
        return [detections[idx] for idx in keep]

    def _merge_wide_recall_detections(self, main_dets: List[Dict], recall_results) -> List[Dict]:
        if recall_results is None:
            return main_dets
        merged = list(main_dets)
        recall_dets = self._extract_wide_detections(recall_results)
        match_iou = float(getattr(self.args, 'recall_match_iou', 0.3))
        for rdet in recall_dets:
            if any(box_iou(rdet['bbox'], det['bbox']) >= match_iou for det in merged):
                continue
            rdet['from_recall'] = True
            merged.append(rdet)
        return merged

    def _attach_wide_features(self, image: np.ndarray, detections: List[Dict]) -> None:
        if self.feature_extractor is None or not detections:
            return
        crops = []
        refs = []
        h, w = image.shape[:2]
        for idx, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
                refs.append(idx)
        if not crops:
            return
        try:
            feats = self.feature_extractor.extract_batch_arrays(crops)
            for det_idx, feat in zip(refs, feats):
                detections[det_idx]['feature'] = feat.numpy()
        except Exception as exc:
            print(f"OSNet 特征提取失败: {exc}")

    def _postprocess_detections(
        self, panorama: np.ndarray, filtered: List[Dict], t: Dict[int, float]
    ) -> Tuple[np.ndarray, np.ndarray, List, Optional[Dict]]:
        # ⑦ 跟踪 + 绘制  [CPU]
        if self.kpt_track:
            for det in filtered:
                det['bbox'] = compute_stable_bbox_from_keypoints(
                    det.get('keypoints', []),
                    conf_thresh=self.kpt_bbox_conf,
                    padding=self.kpt_bbox_padding,
                    fallback_bbox=det['bbox'],
                    upper_body_only=self.kpt_bbox_upper_only,
                )
        if self.use_tracker:
            tracked = self.tracker.update(filtered)
        else:
            tracked = filtered
            for i, d in enumerate(tracked):
                d['track_id'] = i + 1

        # ⑦-b 人脸识别
        current_ids = {d['track_id'] for d in tracked}
        new_ids = current_ids - self._prev_track_ids
        if self.face_rec is not None:
            if hasattr(self.face_rec, 'process_frame'):
                self.face_rec.process_frame(
                    panorama, tracked, new_ids, self._face_name_map,
                    frame_id=self._timing_counter,
                )
            else:
                for det in tracked:
                    tid = det['track_id']
                    self.face_rec.process_detection(
                        panorama, det.get('keypoints', []), tid,
                        is_new_track=(tid in new_ids),
                        face_name_map=self._face_name_map,
                        frame_id=self._timing_counter,
                        bbox=det.get('bbox'),
                        confidence=det.get('confidence'),
                    )
            for gone in ((self._prev_track_ids | set(self._face_name_map)) - current_ids):
                self._face_name_map.pop(gone, None)
                self.face_rec.cleanup_track(gone)
        self._prev_track_ids = current_ids

        tracked = self.display_id_manager.apply(tracked, self._timing_counter)

        # ⑧ 角度数据计算（仅算数据，画框/画角度在后面）  [CPU]
        # 有真实关键点的（pose）直接用；无关键点的（补漏框）用「框顶中心」合成一个鼻子点：
        #   x 取框中心 → 水平角准确；y 取框顶往下 recall_head_ratio×框高 → 近似头部，
        #   使俯仰角与 pose 口径一致。这样 angle_info['persons'] 与 tracked 严格 1:1 对齐
        #   （每个目标一条），下游 JSON 直接按下标取角度。
        angle_info = None
        _recall_pts = []   # 补漏框的合成鼻子点，仅用于可视化标记（不写入 det['keypoints']）
        if tracked and self.angle_calculator:
            _head_ratio = getattr(self.args, 'recall_head_ratio', 0.12)
            kpts_list = []
            for d in tracked:
                kp = d.get('keypoints')
                if kp:
                    kpts_list.append(np.array(kp))
                else:
                    x1, y1, x2, y2 = d['bbox']
                    nx, ny = (x1 + x2) / 2.0, y1 + _head_ratio * (y2 - y1)
                    synth = np.zeros((17, 3), dtype=np.float32)
                    synth[0] = [nx, ny, 1.0]
                    kpts_list.append(synth)
                    _recall_pts.append((int(nx), int(ny)))
            if kpts_list:
                angle_info = self.angle_calculator.calculate_angles_from_keypoints(
                    np.array(kpts_list)
                )

        # ⑧-b 扇区代表标记（--sector-output）：在画框之前标好哪些目标是扇区代表，
        #     由 draw_detections 把代表框直接画成红色（单框，不再另叠加），与
        #     main_GPU_webui 的 JSON 用同一份 aggregate_sectors 判定，保证一致。
        _sectors = None
        _show_sectors = getattr(self.args, 'show_sectors', False)
        if (getattr(self.args, 'sector_output', False) or _show_sectors) and tracked:
            _sectors, rep_indices = aggregate_sectors(
                tracked, angle_info, getattr(self.args, 'num_sectors', 8)
            )
            # 只有 --sector-output 才把代表框画红；--show-sectors 单独开时不改框色
            if getattr(self.args, 'sector_output', False):
                for i in rep_indices:
                    tracked[i]['_sector_rep'] = True

        # ⑨ 画检测框（扇区代表为红框，其余按置信度色）
        annotated = draw_detections(panorama, tracked, self.tracker,
                                    show_id=self.show_id, show_conf=self.show_conf,
                                    face_name_map=self._face_name_map,
                                    use_kpt_bbox=self.kpt_display,
                                    kpt_bbox_conf=self.kpt_bbox_conf,
                                    kpt_bbox_padding=self.kpt_bbox_padding,
                                    kpt_bbox_upper_only=self.kpt_bbox_upper_only,
                                    draw_kpt=self.show_kpt)
        t[7] = time.perf_counter()

        # ⑩ 角度绘制（鼻子圈/箭头 + 补漏框合成点标记）
        if angle_info is not None:
            if self.show_angle_overview:
                annotated = self.angle_calculator.draw_angle_overview(annotated, angle_info)
            else:
                annotated = self.angle_calculator.draw_angles_on_image(
                    annotated, angle_info,
                    show_angle=self.show_angle,
                    show_arrow=self.show_arrow,
                )
            # 补漏框合成点用洋红实心圈+白边标记，区别于 pose 的黄/橙鼻子圈
            for (px, py) in _recall_pts:
                cv2.circle(annotated, (px, py), 6, (255, 0, 255), -1)
                cv2.circle(annotated, (px, py), 8, (255, 255, 255), 2)

        # ⑪ 扇区范围可视化（--show-sectors）：均匀竖线 + 编号/角度区间，
        #    有目标的扇区顶部色带高亮（沿用 aggregate_sectors 的判定）
        if _show_sectors:
            annotated = draw_sector_grid(
                annotated, getattr(self.args, 'num_sectors', 8), _sectors, inplace=True)
        t[8] = time.perf_counter()

        return panorama, annotated, tracked, angle_info

    # ──────────────────────────────────────────────────────────────────
    def _sync_angle_calculator(self, frame: np.ndarray) -> None:
        if not (self.angle_calculator
                and hasattr(self.panorama_processor, 'center')
                and self.panorama_processor.center is not None):
            return
        fp = self.panorama_processor
        if (self.angle_calculator.fisheye_center != fp.center
                or self.angle_calculator.fisheye_radius != fp.radius):
            self.angle_calculator.set_fisheye_mapping(
                fp.center, fp.radius,
                getattr(fp, 'img_width', frame.shape[1]),
                getattr(fp, 'img_height', frame.shape[0]),
            )
            if fp.map_x is not None and fp.map_y is not None:
                self.angle_calculator.set_panorama_maps(fp.map_x, fp.map_y)

    @staticmethod
    def _print_timing(t: dict, n_det: int) -> None:
        ms = lambda a, b: (t[b] - t[a]) * 1000
        steps = [
            ("①CPU→GPU",  "CPU→GPU", 0, 1),
            ("②鱼眼展开", "GPU",     1, 2),
            ("③GPU→CPU",  "GPU→CPU", 2, 3),
            ("④裁剪切片", "CPU",     3, 4),
            ("⑤YOLO推理", "GPU",     4, 5),
            ("⑥合并过滤", "CPU+GPU", 5, 6),
            ("⑦跟踪绘制", "CPU",     6, 7),
            ("⑧角度计算", "CPU",     7, 8),
        ]
        total = (t[8] - t[0]) * 1000
        parts = "  ".join(f"{name}[{dev}]={ms(a,b):.1f}ms"
                          for name, dev, a, b in steps)
        print(f"[总耗时 {total:.1f}ms | 检测 {n_det} 人]  {parts}")

    @staticmethod
    def _print_wide_timing(t: dict, n_det: int) -> None:
        ms = lambda a, b: (t[b] - t[a]) * 1000
        steps = [
            ("①输入准备", "CPU",     0, 1),
            ("②广角直通", "CPU",     1, 3),
            ("③整图输入", "CPU",     3, 4),
            ("④YOLO推理", "GPU",     4, 5),
            ("⑤合并过滤", "CPU+GPU", 5, 6),
            ("⑥跟踪绘制", "CPU",     6, 7),
            ("⑦角度计算", "CPU",     7, 8),
        ]
        total = (t[8] - t[0]) * 1000
        parts = "  ".join(f"{name}[{dev}]={ms(a,b):.1f}ms"
                          for name, dev, a, b in steps)
        print(f"[广角总耗时 {total:.1f}ms | 检测 {n_det} 人]  {parts}")

    # ──────────────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        print("\n清理资源...")
        if self.display_manager:
            self.display_manager.destroy_windows()
        if self.use_tracker:
            print_assignment_stats()
