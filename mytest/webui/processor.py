"""
FisheyePanoramaYOLOPose — 核心 GPU 推理处理类
包含逐步耗时打印（每 30 帧一次），方便定位性能瓶颈。
"""
import os
import threading
import time
from typing import Dict, List, Optional, Tuple

import config as _config

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
)
from utils.feature_extractor import FeatureExtractor

_CUDA = torch.cuda.is_available()


class FisheyePanoramaYOLOPose:
    """鱼眼全景 YOLO 姿态检测 — GPU 版，全景处理器从第一帧懒初始化"""

    def __init__(self, args):
        self.args = args
        self.panorama_processor: Optional[FisheyePanoramaGPU] = None
        self.yolo_detector: Optional[YOLOPoseDetector] = None
        self.display_manager = None
        self.angle_calculator: Optional[AngleCalculator] = None
        self.feature_extractor: Optional[FeatureExtractor] = None

        self.show_angles = True
        self.show_angle_overview = False
        self.num_slices: int = getattr(args, 'num_slices', 3)
        self.slice_overlap: float = getattr(args, 'slice_overlap', 0.05)
        self.use_tracker: bool = (args.tracker != 'none')
        self.no_display: bool = args.no_display

        self._panorama_ready = False
        self._panorama_init_lock = threading.Lock()
        self._timing_counter = 0  # 控制耗时日志频率

        self.slicer = PanoramaSlicer(
            overlap_ratio=self.slice_overlap,
            iou_threshold=0.2,
            reid_similarity_threshold=0.7,
        )

        _boundary_kwargs = dict(
            enable_boundary_matching=True,
            frame_width=3840,
            frame_height=1080,
            boundary_margin=0.1,
            boundary_time_window=90,
            boundary_similarity_thresh=0.3,
            boundary_debug=False,
            enable_top_boundary=False,
            enable_bottom_boundary=False,
            enable_left_boundary=True,
            enable_right_boundary=True,
        )

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
                with_reid=True,
                use_hungarian=args.use_hungarian,
                **_boundary_kwargs,
            )
        elif args.tracker == 'hybridsort':
            self.tracker = HybridSortTracker(
                track_high_thresh=0.5,
                track_low_thresh=0.1,
                new_track_thresh=0.5,
                track_buffer=500,
                frame_rate=30,
                match_thresh=0.4,
                inertia=0.4,
                delta_t=3,
                use_byte=True,
                tcm_first_step=True,
                tcm_first_step_weight=1.0,
                tcm_byte_step=True,
                tcm_byte_step_weight=1.0,
                asso_func="iou",
                min_hits=1,
                with_reid=args.use_reid,
                reid_emb_weight_high=args.reid_emb_weight_high,
                reid_emb_weight_low=args.reid_emb_weight_low,
                panorama_width=3840,
                panorama_height=1080,
                **_boundary_kwargs,
            )
        else:
            self.tracker = None

        _osnet_path = _config.OSNET_WEIGHT_MAP.get(args.osnet_model, '')
        self._osnet_model_name = args.osnet_model
        self._osnet_model_path = _osnet_path if os.path.exists(_osnet_path) else None

    # ──────────────────────────────────────────────────────────────────
    def initialize(self) -> bool:
        """加载 YOLO 和 OSNet 模型"""
        print("初始化模型（YOLO + OSNet）...")
        os.makedirs(self.args.output_dir, exist_ok=True)
        os.makedirs("screenshots", exist_ok=True)

        if not os.path.exists(self.args.model_path):
            print(f"错误: YOLO 模型文件不存在: {self.args.model_path}")
            return False

        self.yolo_detector = YOLOPoseDetector(
            model_path=self.args.model_path,
            conf_threshold=self.args.conf_threshold,
            iou_threshold=self.args.iou_threshold,
        )

        if _CUDA:
            print(f"GPU 已就绪: {torch.cuda.get_device_name(0)}")
            # TensorRT .engine 文件在导出时已绑定 GPU，不支持 .to()；
            # PyTorch .pt 文件需要手动移至 GPU。
            if not self.args.model_path.endswith('.engine'):
                self.yolo_detector.model.to('cuda')
                print("YOLO 已移至 GPU（FP16 由推理时 half=True 控制）")
            else:
                print("YOLO TensorRT 引擎已就绪（GPU 绑定在导出时完成）")
        else:
            print("未检测到 CUDA GPU，使用 CPU 推理")

        self.display_manager = DisplayManager(
            use_dual_windows=self.args.use_dual_windows,
            no_display=self.no_display,
        )

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

        print("模型加载完成！等待第一帧以完成全景处理器初始化...")
        return True

    # ──────────────────────────────────────────────────────────────────
    def _init_panorama_from_frame(self, frame: np.ndarray) -> bool:
        """使用第一帧图像懒初始化 GPU 全景处理器"""
        with self._panorama_init_lock:
            if self._panorama_ready:
                return True

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

            self.angle_calculator = AngleCalculator(
                out_w, out_h, self.args.vertical_fov,
                fit_degree=getattr(self.args, 'fit_degree', 5),
                yaml_file=getattr(self.args, 'calib_yaml', None),
            )

            if (self.tracker is not None
                    and self.tracker.enable_boundary_matching):
                self.tracker.set_boundary_frame_size(out_w, out_h)

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
        """
        if not self._panorama_ready:
            if not self._init_panorama_from_frame(frame):
                return None, None, None, None, None

        t = {}  # 时间戳字典，用于打印各步耗时
        sync = (lambda: torch.cuda.synchronize()) if _CUDA else (lambda: None)

        # ① CPU → GPU  [CPU→GPU]
        t[0] = time.perf_counter()
        frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1)  # HWC → CHW
        t[1] = time.perf_counter()

        # ② GPU 鱼眼展开  [GPU]
        panorama_tensor = self.panorama_processor.apply_panorama_gpu(frame_tensor)
        sync()
        t[2] = time.perf_counter()

        # ③ GPU → CPU  [GPU→CPU] — 先在GPU上转uint8（12MB vs 50MB），再传输
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

        # panorama_tensor 仍在 GPU（步骤③只做了 numpy 拷贝未释放张量）
        # 构建 RGB GPU 切片张量供 OSNet 直接在 GPU 上裁切，跳过 numpy→PIL→transform
        slice_tensors_gpu = None
        if self.feature_extractor and _CUDA:
            # BGR→RGB channel flip + crop_h 裁剪（都是内存视图/小量运算，<0.1ms）
            pano_rgb = panorama_tensor[[2, 1, 0], crop_h:, :]  # [3, H-crop_h, W] RGB float 0-1
            pw = pano_rgb.shape[2]
            slice_tensors_gpu = []
            for info in slice_infos:
                sx, ex = info['start_x'], info['end_x']
                if info['wrap_around']:
                    if sx < 0:   # 左侧越界
                        st = torch.cat([pano_rgb[:, :, sx:], pano_rgb[:, :, :ex]], dim=2)
                    else:        # 右侧越界
                        st = torch.cat([pano_rgb[:, :, sx:pw], pano_rgb[:, :, :ex - pw]], dim=2)
                else:
                    st = pano_rgb[:, :, sx:ex]
                slice_tensors_gpu.append(st)
        t[4] = time.perf_counter()

        # ⑤ 批量 YOLO 推理  [GPU]
        with torch.no_grad():
            all_yolo_results = self.yolo_detector.detect_batch(slices)
        sync()
        t[5] = time.perf_counter()

        # ⑥ 合并 + 过滤  [CPU + GPU]
        # slice_tensors_gpu 存在时走 GPU crop 路径，跳过 CPU numpy→PIL→transform（省 ~5ms）
        merged = self.slicer.merge_detections(
            all_yolo_results, slice_infos,
            slice_images=slices,
            slice_tensors=slice_tensors_gpu,
            feature_extractor=self.feature_extractor,
        )
        filtered = filter_cross_boundary_detections(merged, panorama.shape)
        filtered = self.slicer.filter_wide_detections(filtered, panorama.shape[1])
        t[6] = time.perf_counter()

        # ⑦ OSNet 特征已在步骤⑥ merge_detections 内批量提取（切片坐标 crop），
        #    此处直接复用，不重复提取，节省 ~15ms
        dets_with_feat = filtered  # 特征已挂在 det['feature'] 上（numpy [1,feat_dim] 或 None）
        t[7] = time.perf_counter()

        # ⑧ 跟踪 + 绘制  [CPU]
        if self.use_tracker:
            tracked = self.tracker.update(dets_with_feat)
        else:
            tracked = filtered
            for i, d in enumerate(tracked):
                d['track_id'] = i + 1

        annotated = draw_detections(panorama, tracked, self.tracker)
        t[8] = time.perf_counter()

        # ⑨ 角度计算  [CPU]
        angle_info = None
        if tracked and self.angle_calculator:
            kpts_list = [np.array(d['keypoints']) for d in tracked if d.get('keypoints')]
            if kpts_list:
                angle_info = self.angle_calculator.calculate_angles_from_keypoints(
                    np.array(kpts_list)
                )
                draw_fn = (self.angle_calculator.draw_angle_overview if self.show_angle_overview
                           else self.angle_calculator.draw_angles_on_image)
                annotated = draw_fn(annotated, angle_info)
        t[9] = time.perf_counter()

        # ─── 每 30 帧打印一次各步耗时 ────────────────────────────────
        self._timing_counter += 1
        if self._timing_counter % 30 == 1:
            self._print_timing(t, len(filtered))

        return panorama, None, annotated, tracked, angle_info

    # ──────────────────────────────────────────────────────────────────
    def _sync_angle_calculator(self, frame: np.ndarray) -> None:
        """同步 angle_calculator 的鱼眼映射参数（仅首次或参数变化时）"""
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
        """打印各步耗时及设备标注"""
        ms = lambda a, b: (t[b] - t[a]) * 1000
        steps = [
            ("①CPU→GPU",   "CPU→GPU", 0, 1),
            ("②鱼眼展开",  "GPU",     1, 2),
            ("③GPU→CPU",   "GPU→CPU", 2, 3),
            ("④裁剪切片",  "CPU",     3, 4),
            ("⑤YOLO推理",  "GPU",     4, 5),
            ("⑥合并过滤",  "CPU",     5, 6),
            ("⑦特征复用",   "CPU",     6, 7),
            ("⑧跟踪绘制",  "CPU",     7, 8),
            ("⑨角度计算",  "CPU",     8, 9),
        ]
        total = (t[9] - t[0]) * 1000
        parts = "  ".join(f"{name}[{dev}]={ms(a,b):.1f}ms"
                          for name, dev, a, b in steps)
        print(f"[总耗时 {total:.1f}ms | 检测 {n_det} 人]  {parts}")

    # ──────────────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        print("\n清理资源...")
        if self.display_manager:
            self.display_manager.destroy_windows()
        if self.use_tracker:
            print_assignment_stats()
