"""
鱼眼全景YOLO姿态检测主程序 — GPU 优化版
整合了 webui/processor.py 的 GPU 推理流水线：
  ① CPU→GPU 帧上传
  ② GPU 鱼眼展开（FisheyePanoramaGPU / grid_sample）
  ③ GPU→CPU 优先转 uint8 再拷贝（减少带宽约4倍）
  ④ 裁剪切片 + 构建 GPU 切片张量（供 OSNet 直接在 GPU crop）
  ⑤ torch.no_grad() YOLO 批量推理
  ⑥ merge_detections 传入 slice_tensors_gpu（跳过 CPU numpy→PIL→transform）
  ⑦ OSNet / YOLO .pt 模型初始化时移至 GPU
  ⑧ 每 200 帧调用 torch.cuda.empty_cache() 清理显存碎片
  ⑨ 每 30 帧打印各步耗时日志，方便定位性能瓶颈
"""
import cv2
import numpy as np
import threading
import time
import os
from typing import List, Tuple, Optional, Dict

import torch

# 导入配置
import config

# 导入核心模块
from core import (
    FisheyePanoramaGPU,
    YOLOPoseDetector,
    PanoramaSlicer,
    AngleCalculator,
    BoT_SORTTracker,
    HybridSortTracker,
    print_assignment_stats,
)

# 导入工具模块
from utils import (
    DisplayManager,
    draw_yolo_only,
    draw_detections,
    filter_cross_boundary_detections,
    compute_stable_bbox_from_keypoints,
)

# 导入特征提取器
from utils.feature_extractor import FeatureExtractor

# 导入人脸识别（可选）
try:
    from face_rec.face_rec_manager import FaceRecManager
    _FACE_REC_AVAILABLE = True
except Exception as _e:
    _FACE_REC_AVAILABLE = False
    print(f"[FaceRec] 导入失败，人脸识别不可用: {_e}")

# 导入说话检测（可选，需 pip install mediapipe）
try:
    from utils.talking_detector import TalkingDetector, _MP_AVAILABLE as _TALKING_AVAILABLE
except Exception as _e:
    _TALKING_AVAILABLE = False
    print(f"[TalkingDetector] 导入失败，说话检测不可用: {_e}")

# ── 全局 CUDA 标志（导入时确定一次）─────────────────────────────────────
_CUDA = torch.cuda.is_available()


def _append_wrap_strips(frame: np.ndarray, overlap_width: int) -> np.ndarray:
    """
    在显示帧左右各拼接一条环绕重叠区域副本（调试用）。
      左侧条带 = 全景图右端 overlap_width 像素（slice0 左侧延伸的内容来源）
      右侧条带 = 全景图左端 overlap_width 像素（slice2 右侧延伸的内容来源）
    两条带均叠加半透明青色遮罩，并在内侧边缘画竖线标记边界。
    """
    h, w = frame.shape[:2]
    if overlap_width <= 0 or overlap_width >= w:
        return frame

    left_strip  = frame[:, w - overlap_width : w].copy()   # 右端内容 → 放到最左边
    right_strip = frame[:, 0 : overlap_width].copy()        # 左端内容 → 放到最右边

    tint = np.full_like(left_strip, (200, 200, 0), dtype=np.uint8)  # 青色 (BGR)
    cv2.addWeighted(left_strip,  0.72, tint, 0.28, 0, left_strip)
    cv2.addWeighted(right_strip, 0.72, tint, 0.28, 0, right_strip)

    # 内侧边界分隔线
    cv2.line(left_strip,  (overlap_width - 1, 0), (overlap_width - 1, h - 1), (0, 255, 255), 2)
    cv2.line(right_strip, (0, 0),                 (0, h - 1),                 (0, 255, 255), 2)

    # 标注文字
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(left_strip,  'R-edge', (2, 18), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(right_strip, 'L-edge', (2, 18), font, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    return np.concatenate([left_strip, frame, right_strip], axis=1)


class FisheyePanoramaYOLOPose:
    """鱼眼全景YOLO姿态检测主类（GPU 优化版）"""

    def __init__(self, args):
        self.args = args
        self.camera = None
        self.panorama_processor: Optional[FisheyePanoramaGPU] = None
        self.yolo_detector = None
        self.display_manager = None
        self.last_detection_result = None
        self.angle_calculator = None
        self.show_angles = True
        self.show_angle_overview = False
        self.show_id: bool = getattr(args, 'show_id', True)
        self.show_conf: bool = getattr(args, 'show_conf', True)
        self.show_angle: bool = getattr(args, 'show_angle', False)
        self.show_arrow: bool = getattr(args, 'show_arrow', False)
        self.kpt_display: bool = getattr(args, 'kpt_display', False)
        self.kpt_track: bool = getattr(args, 'kpt_track', False)
        self.kpt_bbox_conf: float = getattr(args, 'kpt_bbox_conf', 0.3)
        self.kpt_bbox_padding: float = getattr(args, 'kpt_bbox_padding', 0.15)
        self.kpt_bbox_padding_v: float = getattr(args, 'kpt_bbox_padding_v', 0.3)
        self.kpt_bbox_upper_only: bool = getattr(args, 'kpt_bbox_upper_only', True)

        # 人脸识别
        self.face_rec: Optional[FaceRecManager] = None
        self._face_name_map: Dict[int, str] = {}   # track_id → person_name
        self._prev_track_ids: set = set()

        # 说话检测
        self.talking_detector: Optional['TalkingDetector'] = None
        self.num_slices = getattr(args, 'num_slices', 3)
        self.slice_overlap = getattr(args, 'slice_overlap', 0.05)
        self.use_tracker = (args.tracker != 'none')
        self.image_files = []
        self.current_image_index = 0
        self.no_display = args.no_display

        # 全景处理器懒初始化（第一帧时触发）
        self._panorama_ready = False
        self._panorama_init_lock = threading.Lock()

        # 耗时日志计数器
        self._timing_counter = 0

        if not hasattr(self, 'slicer'):
            self.slicer = PanoramaSlicer(
                overlap_ratio=self.slice_overlap,
                iou_threshold=0.2,
                reid_similarity_threshold=0.7,
                dedup_use_reid=getattr(args, 'dedup_use_reid', False),
                verbose=False,
            )

        # 边界匹配公共参数（两种 tracker 共用）
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
                kalman_bbox=getattr(args, 'kalman_bbox', False),
                **_boundary_kwargs,
            )
        elif args.tracker == 'hybridsort':
            self.tracker = HybridSortTracker(
                # ── 置信度阈值 ──────────────────────────────────────────────
                track_high_thresh=0.4,
                track_low_thresh=0.1,
                # 新增 ID 严苛化①：置信度门控。仅 conf >= 0.6 的未匹配高分检测才生成候选轨迹
                # （> det_thresh=0.5，过滤掉刚过线的低质量检测）。
                new_track_thresh=0.5,
                # ── 轨迹生命周期 ────────────────────────────────────────────
                track_buffer=500,
                frame_rate=30,
                # ── 关联阈值 ────────────────────────────────────────────────
                # 帧间 IoU 关联门限调低（0.2→0.15）：检测框/人体小幅晃动时更容易关联回
                # 原轨迹，减少"旧轨迹还在却没接上→另起新号"的碎片化（代价：略增误配风险）。
                match_thresh=0.15,
                # ── Hybrid-SORT 专有参数 ─────────────────────────────────────
                inertia=0.1,
                delta_t=4,
                use_byte=True,
                tcm_first_step=True,
                tcm_first_step_weight=0.5,
                tcm_byte_step=True,
                tcm_byte_step_weight=0.5,
                asso_func="iou",
                # 新增 ID 严苛化②：连续命中门控。候选轨迹需连续命中 3 帧才确认并分配 ID，
                # 未确认期间漏检即删除、且不占用全局 ID 计数器（见 hybrid_sort 延迟分配逻辑）。
                min_hits=3,
                # Round 3.5 兜底：IoU < match_thresh 但中心距离 < cd_thresh 倍框高时强制关联
                # 应对遮挡后框大小变化导致 IoU 不足进而重新生成 ID 的问题
                cd_thresh=0.5,
                # ── ReID ────────────────────────────────────────────────────
                with_reid=args.use_reid,
                reid_emb_weight_high=args.reid_emb_weight_high,
                reid_emb_weight_low=args.reid_emb_weight_low,
                # ── 全景图尺寸 ───────────────────────────────────────────────
                panorama_width=3840,
                panorama_height=1080,
                # ── 框平滑 ──────────────────────────────────────────────────
                smooth_bbox=getattr(args, 'smooth_bbox', False),
                smooth_bbox_alpha=getattr(args, 'smooth_bbox_alpha', 0.5),
                # ── Kalman 轨迹框 ────────────────────────────────────────────
                kalman_bbox=getattr(args, 'kalman_bbox', False),
                **_boundary_kwargs,
            )
        else:
            self.tracker = None

        self.feature_extractor = None
        _osnet_path = config.OSNET_WEIGHT_MAP.get(args.osnet_model, '')
        # OSNET_ARCH_MAP 将用户选项（如 osnet_ain_x1_0_D）映射到 torchreid 实际识别的架构名
        # 不同数据集预训练的变体底层架构相同，直接传用户 key 会导致 torchreid 报错
        self.osnet_model_name = config.OSNET_ARCH_MAP.get(args.osnet_model, args.osnet_model)
        self.osnet_model_path = _osnet_path if os.path.exists(_osnet_path) else None

    # ──────────────────────────────────────────────────────────────────
    def initialize(self) -> bool:
        """初始化所有组件"""
        print("初始化鱼眼展开和YOLO姿态检测系统...")

        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.args.folder_path:
            if not os.path.isdir(self.args.folder_path):
                print(f"错误: 文件夹 {self.args.folder_path} 不存在")
                return False
            self.image_files = self._get_image_files(self.args.folder_path)
            if not self.image_files:
                print(f"错误: 文件夹 {self.args.folder_path} 中没有找到图片文件")
                return False
            print(f"输入源: 图片文件夹 ({len(self.image_files)} 张图片)")
        else:
            from core import CameraProcessor
            self.camera = CameraProcessor(
                cam_index=self.args.cam_index,
                video_path=self.args.video_path,
                width=self.args.cam_width,
                height=self.args.cam_height
            )
            if not self.camera.initialize():
                return False
            source_type = "视频文件" if self.args.video_path else "摄像头"
            print(f"输入源: {source_type}")

        # 全景处理器在第一帧时懒初始化，此处不提前创建
        print("全景处理器将在第一帧时完成 GPU 初始化...")

        # ── YOLO 模型加载 ────────────────────────────────────────────
        if not os.path.exists(self.args.model_path):
            print(f"错误: YOLO模型文件 {self.args.model_path} 不存在")
            return False

        self.yolo_detector = YOLOPoseDetector(
            model_path=self.args.model_path,
            conf_threshold=self.args.conf_threshold,
            iou_threshold=self.args.iou_threshold
        )

        if _CUDA:
            print(f"GPU 已就绪: {torch.cuda.get_device_name(0)}")
            # TensorRT .engine 在导出时已绑定 GPU，无需也不能调用 .to()
            if not self.args.model_path.endswith('.engine'):
                self.yolo_detector.model.to('cuda')
                print("YOLO 已移至 GPU（FP16 由推理时 half=True 控制）")
            else:
                print("YOLO TensorRT 引擎已就绪（GPU 绑定在导出时完成）")
        else:
            print("未检测到 CUDA GPU，使用 CPU 推理")

        # ── 补漏检测模型（--recall-boost）─────────────────────────────
        # 用一个纯检测模型捞回 pose 漏检的遮挡/背身目标，作为无关键点框补入。
        self.recall_detector = None
        if getattr(self.args, 'recall_boost', False):
            if not os.path.exists(self.args.recall_model):
                print(f"警告: 补漏模型 {self.args.recall_model} 不存在，补漏检测已禁用")
            else:
                _recall_conf = getattr(self.args, 'recall_conf_threshold', 0.4)
                self.recall_detector = YOLOPoseDetector(
                    model_path=self.args.recall_model,
                    conf_threshold=_recall_conf,
                    iou_threshold=self.args.iou_threshold,
                )
                if _CUDA and not self.args.recall_model.endswith('.engine'):
                    self.recall_detector.model.to('cuda')
                print(f"补漏检测已启用: {self.args.recall_model}"
                      f"（置信度阈值={_recall_conf}, IoU 关联阈值={self.args.recall_match_iou}）")

        self.display_manager = DisplayManager(
            use_dual_windows=self.args.use_dual_windows,
            no_display=self.no_display
        )

        # ── OSNet 特征提取器 ─────────────────────────────────────────
        if getattr(self.args, 'use_osnet', True):
            print(f"初始化OSNet特征提取器 ({self.osnet_model_name})...")
            try:
                self.feature_extractor = FeatureExtractor(
                    model_name=self.osnet_model_name,
                    model_path=self.osnet_model_path
                )
                if _CUDA:
                    self.feature_extractor.extractor.model.to('cuda')
                    self.feature_extractor.device = 'cuda'
                    print("OSNet 已移至 GPU（FP32，BatchNorm 不支持 FP16）")
                else:
                    print("OSNet 特征提取器初始化成功（CPU 模式）")
            except Exception as e:
                print(f"警告: OSNet特征提取器初始化失败: {e}")
                print("将不使用ReID特征进行跟踪")
                self.feature_extractor = None
        else:
            print("OSNet 特征提取已禁用（--no-use-osnet）")
            self.feature_extractor = None

        # ── 人脸识别 ─────────────────────────────────────────────────────
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

        # ── 说话检测 ─────────────────────────────────────────────────────
        if getattr(self.args, 'talking_detection', False):
            if not _TALKING_AVAILABLE:
                print("警告: mediapipe 未安装，说话检测不可用。请运行: pip install mediapipe")
            else:
                try:
                    self.talking_detector = TalkingDetector(
                        mar_threshold=getattr(self.args, 'talking_mar_threshold', 0.035),
                        detect_interval=getattr(self.args, 'talking_detect_interval', 3),
                    )
                    print("说话检测已启用（MediaPipe FaceMesh MAR）")
                except Exception as e:
                    print(f"警告: 说话检测初始化失败: {e}")
                    self.talking_detector = None

        print("初始化完成！")
        return True

    # ──────────────────────────────────────────────────────────────────
    def _init_panorama_from_frame(self, frame: np.ndarray) -> bool:
        """使用第一帧懒初始化 GPU 全景处理器（线程安全）"""
        with self._panorama_init_lock:
            if self._panorama_ready:
                return True

            h, w = frame.shape[:2]
            print(f"初始化 GPU 全景处理器，输入分辨率: {w}×{h}")

            self.panorama_processor = FisheyePanoramaGPU(
                cam_width=w, cam_height=h,
                output_width=self.args.output_width,
                output_height=self.args.output_height,
                vertical_fov=self.args.vertical_fov,
                map_file=self.args.map_file,
                cam_index=self.args.cam_index,
            )
            if not self.panorama_processor.init_from_frame(frame):
                print("GPU 映射矩阵初始化失败")
                return False

            out_w = self.panorama_processor.output_width
            out_h = self.panorama_processor.output_height
            print(f"GPU 全景处理器就绪，全景输出: {out_w}×{out_h}")

            self.angle_calculator = AngleCalculator(
                out_w, out_h, self.args.vertical_fov,
                fit_degree=getattr(self.args, 'fit_degree', 5),
                yaml_file=getattr(self.args, 'calib_yaml', None)
            )

            if (self.tracker is not None
                    and self.tracker.enable_boundary_matching):
                self.tracker.set_boundary_frame_size(out_w, out_h)
                print(f"边界匹配器初始化：画面={out_w}×{out_h}")

            self._panorama_ready = True
            return True

    # ──────────────────────────────────────────────────────────────────
    def process_panorama_slices(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List], Optional[Dict]]:
        """
        完整 GPU 推理流水线（每步均附有设备标注）：
          ① CPU→GPU 帧上传
          ② GPU 鱼眼展开（grid_sample）
          ③ GPU→CPU（先在 GPU 转 uint8，带宽约为 float32 的 1/4）
          ④ 裁剪 + 切片 + 构建 GPU 切片张量（OSNet 直接在 GPU crop）
          ⑤ torch.no_grad() YOLO 批量推理
          ⑥ merge_detections（slice_tensors_gpu 路径跳过 numpy→PIL→transform）
          ⑦ 跟踪 + 绘制
          ⑧ 角度计算
        """
        if not self._panorama_ready:
            if not self._init_panorama_from_frame(frame):
                return frame, frame, frame, None, None

        t: dict = {}  # 各步时间戳
        sync = (lambda: torch.cuda.synchronize()) if _CUDA else (lambda: None)

        # ① CPU → GPU  [CPU→GPU]
        t[0] = time.perf_counter()
        if _CUDA:
            frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0
            frame_tensor = frame_tensor.permute(2, 0, 1)   # HWC → CHW
        t[1] = time.perf_counter()

        # ② GPU 鱼眼展开  [GPU]
        if _CUDA:
            panorama_tensor = self.panorama_processor.apply_panorama_gpu(frame_tensor)
            sync()
        t[2] = time.perf_counter()

        # ③ GPU → CPU  [GPU→CPU] — 先在 GPU 转 uint8（4× 节省带宽），再传输
        if _CUDA:
            panorama = (
                (panorama_tensor * 255.0)
                .clamp_(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
        else:
            panorama = self.panorama_processor.apply_panorama(frame)
        t[3] = time.perf_counter()

        # ④ 裁剪 + 切片  [CPU] + 构建 GPU 切片张量
        original_panorama_height = panorama.shape[0]
        self._sync_angle_calculator(frame)

        crop_height = 0
        if self.args.crop_divisor > 0:
            crop_height = original_panorama_height // self.args.crop_divisor
            panorama = panorama[crop_height:, :]
        if self.angle_calculator:
            self.angle_calculator.set_crop_offset(crop_height)

        slices, slice_infos = self.slicer.slice_panorama(panorama, num_slices=self.num_slices)

        # 构建 RGB GPU 切片张量，供 OSNet 直接在 GPU 上 crop，跳过 numpy→PIL→transform（省 ~5ms）
        slice_tensors_gpu = None
        if self.feature_extractor and _CUDA:
            # BGR→RGB 通道翻转 + crop_h 裁剪（均为内存视图/轻量运算，<0.1ms）
            pano_rgb = panorama_tensor[[2, 1, 0], crop_height:, :]  # [3, H-crop_h, W] RGB float 0-1
            pw = pano_rgb.shape[2]
            slice_tensors_gpu = []
            for info in slice_infos:
                sx, ex = info['start_x'], info['end_x']
                if info.get('wrap_around', False):
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
            # 补漏路（--recall-boost）：同样的切片再跑一次纯检测模型
            recall_yolo_results = None
            if getattr(self, 'recall_detector', None) is not None:
                recall_yolo_results = self.recall_detector.detect_batch(slices)
        sync()
        t[5] = time.perf_counter()

        # ⑥ 合并 + 过滤  [CPU + GPU]
        # slice_tensors_gpu 存在时走 GPU crop 路径，跳过 CPU numpy→PIL→transform（省 ~5ms）
        merged_detections = self.slicer.merge_detections(
            all_yolo_results,
            slice_infos,
            slice_images=slices,
            slice_tensors=slice_tensors_gpu,
            feature_extractor=self.feature_extractor,
            recall_yolo_results=recall_yolo_results,
            recall_match_iou=getattr(self.args, 'recall_match_iou', 0.4),
        )
        filtered_detections = filter_cross_boundary_detections(merged_detections, panorama.shape)
        filtered_detections = self.slicer.filter_wide_detections(filtered_detections, panorama.shape[1])
        t[6] = time.perf_counter()

        # ⑦ 跟踪 + 绘制  [CPU]
        if self.kpt_track:
            for det in filtered_detections:
                det['bbox'] = compute_stable_bbox_from_keypoints(
                    det.get('keypoints', []),
                    conf_thresh=self.kpt_bbox_conf,
                    padding=self.kpt_bbox_padding,
                    fallback_bbox=det['bbox'],
                    upper_body_only=self.kpt_bbox_upper_only,
                    padding_v=self.kpt_bbox_padding_v,
                )
        if self.use_tracker:
            tracked_detections = self.tracker.update(filtered_detections)
        else:
            tracked_detections = filtered_detections
            for i, det in enumerate(tracked_detections):
                det['track_id'] = i + 1

        yolo_only_image = draw_yolo_only(panorama, filtered_detections)

        # ⑦-b 人脸识别  [CPU/GPU]
        current_ids = {d['track_id'] for d in tracked_detections}
        new_ids = current_ids - self._prev_track_ids
        if self.face_rec is not None:
            for det in tracked_detections:
                tid = det['track_id']
                kpts = det.get('keypoints', [])
                self.face_rec.process_detection(
                    panorama, kpts, tid,
                    is_new_track=(tid in new_ids),
                    face_name_map=self._face_name_map,
                    frame_id=self._timing_counter,
                )
            for gone in (set(self._face_name_map) - current_ids):
                self._face_name_map.pop(gone, None)
                self.face_rec.cleanup_track(gone)
        self._prev_track_ids = current_ids

        # ⑦-c 说话检测  [CPU, MediaPipe]
        if self.talking_detector is not None:
            gone_ids = set(self._mar_prev_ids) - current_ids if hasattr(self, '_mar_prev_ids') else set()
            for gone in gone_ids:
                self.talking_detector.cleanup_track(gone)
            self._mar_prev_ids = current_ids
            for det in tracked_detections:
                kpts = det.get('keypoints', [])
                det['talking'] = self.talking_detector.detect(panorama, kpts, det['track_id'])

        annotated_panorama = draw_detections(panorama, tracked_detections, self.tracker,
                                             show_id=self.show_id, show_conf=self.show_conf,
                                             face_name_map=self._face_name_map,
                                             use_kpt_bbox=self.kpt_display,
                                             kpt_bbox_conf=self.kpt_bbox_conf,
                                             kpt_bbox_padding=self.kpt_bbox_padding,
                                             kpt_bbox_upper_only=self.kpt_bbox_upper_only,
                                             kpt_bbox_padding_v=self.kpt_bbox_padding_v)
        t[7] = time.perf_counter()

        # ⑧ 角度计算  [CPU]
        angle_info = None
        if tracked_detections and self.angle_calculator:
            # 有真实关键点的（pose）直接用；无关键点的（补漏框）用「框顶中心」合成一个鼻子点：
            #   x 取框中心 → 水平角准确；y 取框顶往下 recall_head_ratio×框高 → 近似头部，
            #   使俯仰角与 pose 口径一致。其余点留零（仅鼻子 COCO 索引 0 参与角度计算），
            #   补够 17×3 以满足 calculate_angles_from_keypoints 的 >=5 点校验。
            _head_ratio = getattr(self.args, 'recall_head_ratio', 0.12)
            kpts_list = []
            _recall_pts = []   # 补漏框的合成鼻子点，仅用于可视化标记（不写入 det['keypoints']）
            for d in tracked_detections:
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
                if self.show_angle_overview:
                    annotated_panorama = self.angle_calculator.draw_angle_overview(
                        annotated_panorama, angle_info)
                else:
                    annotated_panorama = self.angle_calculator.draw_angles_on_image(
                        annotated_panorama, angle_info,
                        show_angle=self.show_angle,
                        show_arrow=self.show_arrow,
                    )
                # 补漏框合成点用洋红实心圈+白边标记，区别于 pose 的黄/橙鼻子圈，便于确认角度生效
                for (px, py) in _recall_pts:
                    cv2.circle(annotated_panorama, (px, py), 6, (255, 0, 255), -1)
                    cv2.circle(annotated_panorama, (px, py), 8, (255, 255, 255), 2)
        t[8] = time.perf_counter()

        # ── 每 30 帧打印一次各步耗时 ─────────────────────────────────
        self._timing_counter += 1
        if self._timing_counter % 30 == 1:
            self._print_timing(t, len(filtered_detections))

        # ── 每 200 帧清理一次显存碎片 ────────────────────────────────
        if self._timing_counter % 200 == 0 and _CUDA:
            torch.cuda.empty_cache()

        # ── 环绕重叠可视化（--show-wrap-overlap）────────────────────
        if getattr(self.args, 'show_wrap_overlap', False):
            _ow = int((panorama.shape[1] // self.num_slices) * self.slice_overlap)
            annotated_panorama = _append_wrap_strips(annotated_panorama, _ow)

        return panorama, yolo_only_image, annotated_panorama, tracked_detections, angle_info

    # ──────────────────────────────────────────────────────────────────
    def _sync_angle_calculator(self, frame: np.ndarray) -> None:
        """同步 angle_calculator 的鱼眼映射参数（仅首次或参数变化时更新）"""
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
        """打印各步耗时及设备标注（每 30 帧一次）"""
        def ms(a, b):
            return (t[b] - t[a]) * 1000
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

    # ──────────────────────────────────────────────────────────────────
    def _get_image_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有图片文件"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        return sorted(image_files)

    def _process_single_image(self, image_path: str, crops_dir: str = None):
        """处理单张图片"""
        print(f"处理图片: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"警告: 无法读取图片 {image_path}")
            return

        panorama, yolo_only_frame, annotated_frame, detection_results, angle_info = self.process_panorama_slices(frame)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = self.args.output_dir

        cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.jpg'), frame)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_panorama.jpg'), panorama)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_detection.jpg'), annotated_frame)

        if self.args.save_crops and detection_results and crops_dir:
            for crop_idx, det in enumerate(detection_results, 1):
                x1, y1, x2, y2 = det['bbox']
                h, w = panorama.shape[:2]
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    crop_img = panorama[y1:y2, x1:x2]
                    crop_filename = f'{base_name}_{crop_idx}.jpg'
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop_img)
                    print(f"  保存抠图: {crop_filename}")

        print(f"  检测到 {len(detection_results) if detection_results else 0} 个人")

    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_video_path(output_dir: str, name: Optional[str],
                            default_stem: str, timestamp: str) -> str:
        """解析视频输出路径：指定了 --video-name 则用它（自动补 .mp4），否则用带时间戳的默认名"""
        if name:
            if not name.endswith('.mp4'):
                name += '.mp4'
            return os.path.join(output_dir, name)
        return os.path.join(output_dir, f'{default_stem}_{timestamp}.mp4')

    def _create_single_video_writer(self, output_dir: str, timestamp: str,
                                    sample: np.ndarray) -> cv2.VideoWriter:
        """根据首帧显示图像尺寸创建单窗口 VideoWriter（懒初始化）"""
        path = self._resolve_video_path(output_dir, self.args.video_name,
                                        'detection_result', timestamp)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, self.args.video_fps,
                                 (sample.shape[1], sample.shape[0]))
        print(f"正在保存视频到: {path}")
        return writer

    def _create_dual_video_writers(self, output_dir: str, timestamp: str,
                                   yolo_sample: np.ndarray,
                                   final_sample: np.ndarray):
        """根据首帧显示图像尺寸创建双窗口 VideoWriter（懒初始化），返回 (yolo_writer, final_writer)"""
        yolo_path = self._resolve_video_path(output_dir, self.args.yolo_video_name,
                                             'yolo_detection', timestamp)
        final_path = self._resolve_video_path(output_dir, self.args.video_name,
                                              'final_result', timestamp)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        yolo_writer = cv2.VideoWriter(yolo_path, fourcc, self.args.video_fps,
                                      (yolo_sample.shape[1], yolo_sample.shape[0]))
        final_writer = cv2.VideoWriter(final_path, fourcc, self.args.video_fps,
                                       (final_sample.shape[1], final_sample.shape[0]))
        print(f"正在保存视频到:\n  {yolo_path}\n  {final_path}")
        return yolo_writer, final_writer

    # ──────────────────────────────────────────────────────────────────
    def run(self):
        """运行主循环"""
        if self.args.folder_path:
            source_type = "图片文件夹"
        elif self.args.video_path:
            source_type = "视频文件"
        else:
            source_type = "摄像头"

        print("开始鱼眼展开 + YOLO姿态检测...")

        if self.args.folder_path:
            print("文件夹处理模式")
        else:
            print("按 'q' 键退出")
            print("按 's' 键保存当前帧")
            print("按 'i' 键切换置信度阈值 (0.3/0.5)")
            print("按 'o' 键切换IOU阈值 (0.3/0.45)")
            print("按 'p' 键显示性能统计")
            print("按 'a' 键切换角度显示模式")
            if self.args.video_path:
                print("按 'r' 键重新播放，按 'q' 键退出")
            if self.args.save_video:
                print(f"视频保存已启用，帧率: {self.args.video_fps} FPS")

        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        crops_dir = None
        if self.args.save_crops:
            crops_dir = os.path.join(output_dir, 'crops')
            if not os.path.exists(crops_dir):
                os.makedirs(crops_dir)

        if self.args.folder_path:
            print(f"开始处理文件夹中的 {len(self.image_files)} 张图片...")
            for img_path in self.image_files:
                self._process_single_image(img_path, crops_dir)
            print(f"\n处理完成！共处理 {len(self.image_files)} 张图片")
            print(f"结果保存在: {output_dir}")
            return

        is_video = self.args.video_path is not None
        frame_count = 0
        video_writer = None
        yolo_video_writer = None

        # VideoWriter 在主循环首帧根据实际显示尺寸懒初始化。
        # 早期实现会先抓一帧 test_frame 跑完整 process_panorama_slices() 只为拿尺寸，
        # 再 set(POS_FRAMES, 0) 回退——但那次调用已经把首帧喂进 tracker.update / 人脸识别，
        # 回退后首帧又被处理一遍，导致首帧轨迹初始化重复、track_id 偏移。改为懒初始化后
        # 首帧只处理一次，建 writer 所需尺寸直接取自该帧的显示图像。
        timestamp = None
        if self.args.save_video:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        while True:
            ret, frame = self.camera.get_frame()
            if not ret:
                if is_video:
                    print("视频播放完毕")
                    if hasattr(self, 'no_display') and self.no_display:
                        print("无界面模式：视频处理完成，程序退出。")
                        return
                    print("按 'r' 键重新播放，按 'q' 键退出")
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('r'):
                            self.camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_count = 0
                            ret, frame = self.camera.get_frame()
                            if ret:
                                break
                        elif key == ord('q'):
                            return
                else:
                    print("无法从摄像头接收帧")
                    break

            panorama, yolo_only_frame, annotated_frame, detection_results, _ = self.process_panorama_slices(frame)
            useful_area = self.panorama_processor.get_useful_area(frame)
            fps = self.display_manager.update_fps()

            info_text = f"YOLO Pose Detection (Conf: {self.args.conf_threshold}, IOU: {self.args.iou_threshold})"

            if self.args.save_frames:
                frame_count += 1
                self.display_manager.save_frame(frame, f'original_{frame_count:06d}.jpg', output_dir)
                self.display_manager.save_frame(panorama, f'panorama_{frame_count:06d}.jpg', output_dir)
                self.display_manager.save_frame(annotated_frame, f'detection_{frame_count:06d}.jpg', output_dir)
                if frame_count % 10 == 0:
                    print(f"已保存 {frame_count} 帧...")

            if self.args.save_crops and detection_results:
                if not self.args.save_frames:
                    frame_count += 1
                for crop_idx, det in enumerate(detection_results, 1):
                    x1, y1, x2, y2 = det['bbox']
                    h, w = panorama.shape[:2]
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    if x2 > x1 and y2 > y1:
                        crop_img = panorama[y1:y2, x1:x2]
                        crop_filename = f'frame_{frame_count:06d}_{crop_idx}.jpg'
                        crop_path = os.path.join(crops_dir, crop_filename)
                        cv2.imwrite(crop_path, crop_img)

            perf_stats = self.yolo_detector.get_performance_stats()
            perf_text = f"FPS: {fps}, Inference: {perf_stats['avg_inference_time_ms']:.1f}ms"
            num_people = len(detection_results) if detection_results else 0
            count_text = f"Detected Persons: {num_people}"

            if self.show_angles:
                mode_text = " | Angles ON"
                if self.show_angle_overview:
                    mode_text += " (Overview)"
                else:
                    mode_text += " (Detail)"
                info_text += mode_text

            if self.args.use_dual_windows:
                yolo_display = self.display_manager.add_info_overlay(
                    yolo_only_frame, "YOLO Detection Only", perf_text, count_text
                )
                final_display = self.display_manager.create_layout(
                    frame, useful_area, annotated_frame, self.args.display_scale
                )
                final_display = self.display_manager.add_info_overlay(
                    final_display, info_text, perf_text, count_text
                )
                self.display_manager.show_dual(yolo_display, final_display)

                if self.args.save_video:
                    if video_writer is None:
                        yolo_video_writer, video_writer = self._create_dual_video_writers(
                            output_dir, timestamp, yolo_display, final_display
                        )
                    yolo_video_writer.write(yolo_display)
                    video_writer.write(final_display)
            else:
                display_image = self.display_manager.create_layout(
                    frame, useful_area, annotated_frame, self.args.display_scale
                )
                display_image = self.display_manager.add_info_overlay(
                    display_image, info_text, perf_text, count_text
                )
                self.display_manager.show(display_image)

                if self.args.save_video:
                    if video_writer is None:
                        video_writer = self._create_single_video_writer(
                            output_dir, timestamp, display_image
                        )
                    video_writer.write(display_image)

            key = cv2.waitKey(1) & 0xFF
            if self.handle_keyboard(key, frame, panorama, annotated_frame, output_dir):
                break

        if self.args.save_video:
            if video_writer:
                video_writer.release()
                print("视频已保存完成！")
            if yolo_video_writer:
                yolo_video_writer.release()

    # ──────────────────────────────────────────────────────────────────
    def handle_keyboard(self, key: int, original_frame: np.ndarray,
                       panorama: np.ndarray, annotated_frame: np.ndarray,
                       output_dir: str) -> bool:
        """处理键盘输入，返回True表示退出"""
        if key == ord('q'):
            return True
        elif key == ord('s'):
            timestamp = int(time.time())
            self.display_manager.save_frame(original_frame, f'original_{timestamp}.jpg', output_dir)
            self.display_manager.save_frame(panorama, f'panorama_{timestamp}.jpg', output_dir)
            self.display_manager.save_frame(annotated_frame, f'detection_{timestamp}.jpg', output_dir)
            print(f"已保存3张图片到 {output_dir}/")
        elif key == ord('i'):
            new_conf = 0.3 if self.args.conf_threshold >= 0.5 else 0.5
            self.args.conf_threshold = new_conf
            self.yolo_detector.update_thresholds(conf_threshold=new_conf)
            print(f"置信度阈值切换为: {self.args.conf_threshold}")
        elif key == ord('o'):
            new_iou = 0.3 if self.args.iou_threshold >= 0.45 else 0.45
            self.args.iou_threshold = new_iou
            self.yolo_detector.update_thresholds(iou_threshold=new_iou)
            print(f"IOU阈值切换为: {self.args.iou_threshold}")
        elif key == ord('a'):
            if not self.show_angles:
                self.show_angles = True
                self.show_angle_overview = False
                print("角度显示已开启 (详细模式)")
            elif not self.show_angle_overview:
                self.show_angle_overview = True
                print("角度显示已切换为概览模式")
            else:
                self.show_angles = False
                self.show_angle_overview = False
                print("角度显示已关闭")
        return False

    # ──────────────────────────────────────────────────────────────────
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        if self.camera:
            self.camera.release()
        if self.display_manager:
            self.display_manager.destroy_windows()
        if self.use_tracker:
            print_assignment_stats()


# ── 主函数 ─────────────────────────────────────────────────────────────
def main():
    args = config.parse_args()
    processor = FisheyePanoramaYOLOPose(args)

    if not processor.initialize():
        print("初始化失败，程序退出")
        return

    try:
        processor.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        processor.cleanup()
        print("程序结束")


if __name__ == "__main__":
    main()
