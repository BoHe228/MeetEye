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
    print_assignment_stats,
)

# 导入工具模块
from utils import (
    DisplayManager,
    draw_yolo_only,
    draw_detections,
    filter_cross_boundary_detections,
)

# 导入特征提取器
from utils.feature_extractor import FeatureExtractor

# ── 全局 CUDA 标志（导入时确定一次）─────────────────────────────────────
_CUDA = torch.cuda.is_available()


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
        self.num_slices = getattr(args, 'num_slices', 3)
        self.slice_overlap = getattr(args, 'slice_overlap', 0.05)
        self.use_deep_sort = args.use_deep_sort
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
                reid_similarity_threshold=0.7
            )

        self.deep_sort_tracker = BoT_SORTTracker(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.5,
            track_buffer=500,
            match_thresh=0.3,
            proximity_thresh=0.4,
            appearance_thresh=args.appearance_thresh,
            reid_lost_thresh=args.reid_lost_thresh,
            frame_rate=30,
            feat_history=500,
            with_reid=True,
            use_hungarian=args.use_hungarian,
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
            enable_right_boundary=True
        )

        self.feature_extractor = None
        self.model_path = "imagenet.pyth/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        if not os.path.exists(self.model_path):
            self.model_path = None

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

        self.display_manager = DisplayManager(
            use_dual_windows=self.args.use_dual_windows,
            no_display=self.no_display
        )

        # ── OSNet 特征提取器 ─────────────────────────────────────────
        print("初始化OSNet特征提取器...")
        try:
            self.feature_extractor = FeatureExtractor(
                model_name='osnet_x0_25',
                model_path=self.model_path
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

            if self.deep_sort_tracker.enable_boundary_matching:
                self.deep_sort_tracker.set_boundary_frame_size(out_w, out_h)
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
        sync()
        t[5] = time.perf_counter()

        # ⑥ 合并 + 过滤  [CPU + GPU]
        # slice_tensors_gpu 存在时走 GPU crop 路径，跳过 CPU numpy→PIL→transform（省 ~5ms）
        merged_detections = self.slicer.merge_detections(
            all_yolo_results,
            slice_infos,
            slice_images=slices,
            slice_tensors=slice_tensors_gpu,
            feature_extractor=self.feature_extractor
        )
        filtered_detections = filter_cross_boundary_detections(merged_detections, panorama.shape)
        filtered_detections = self.slicer.filter_wide_detections(filtered_detections, panorama.shape[1])
        t[6] = time.perf_counter()

        # ⑦ 跟踪 + 绘制  [CPU]
        if self.use_deep_sort:
            tracked_detections = self.deep_sort_tracker.update(filtered_detections)
        else:
            tracked_detections = filtered_detections
            for i, det in enumerate(tracked_detections):
                det['track_id'] = i + 1

        yolo_only_image = draw_yolo_only(panorama, filtered_detections)
        annotated_panorama = draw_detections(panorama, tracked_detections, self.deep_sort_tracker)
        t[7] = time.perf_counter()

        # ⑧ 角度计算  [CPU]
        angle_info = None
        if tracked_detections and self.angle_calculator:
            kpts_list = [np.array(d['keypoints']) for d in tracked_detections if d.get('keypoints')]
            if kpts_list:
                angle_info = self.angle_calculator.calculate_angles_from_keypoints(
                    np.array(kpts_list)
                )
                draw_fn = (self.angle_calculator.draw_angle_overview if self.show_angle_overview
                           else self.angle_calculator.draw_angles_on_image)
                annotated_panorama = draw_fn(annotated_panorama, angle_info)
        t[8] = time.perf_counter()

        # ── 每 30 帧打印一次各步耗时 ─────────────────────────────────
        self._timing_counter += 1
        if self._timing_counter % 30 == 1:
            self._print_timing(t, len(filtered_detections))

        # ── 每 200 帧清理一次显存碎片 ────────────────────────────────
        if self._timing_counter % 200 == 0 and _CUDA:
            torch.cuda.empty_cache()

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

        if self.args.save_video:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            ret, test_frame = self.camera.get_frame()
            if ret:
                # 第一帧：触发 GPU 全景处理器懒初始化
                panorama, yolo_only_frame, annotated_frame, _, _ = self.process_panorama_slices(test_frame)
                useful_area = self.panorama_processor.get_useful_area(test_frame)

                if self.args.use_dual_windows:
                    yolo_display = self.display_manager.add_info_overlay(
                        yolo_only_frame, "YOLO Detection Only", "", ""
                    )
                    final_display = self.display_manager.create_layout(
                        test_frame, useful_area, annotated_frame, self.args.display_scale
                    )
                    if self.args.yolo_video_name:
                        yolo_video_filename = self.args.yolo_video_name
                        if not yolo_video_filename.endswith('.mp4'):
                            yolo_video_filename += '.mp4'
                        yolo_video_path = os.path.join(output_dir, yolo_video_filename)
                    else:
                        yolo_video_path = os.path.join(output_dir, f'yolo_detection_{timestamp}.mp4')

                    if self.args.video_name:
                        final_video_filename = self.args.video_name
                        if not final_video_filename.endswith('.mp4'):
                            final_video_filename += '.mp4'
                        final_video_path = os.path.join(output_dir, final_video_filename)
                    else:
                        final_video_path = os.path.join(output_dir, f'final_result_{timestamp}.mp4')

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    yolo_video_writer = cv2.VideoWriter(
                        yolo_video_path, fourcc, self.args.video_fps,
                        (yolo_display.shape[1], yolo_display.shape[0])
                    )
                    video_writer = cv2.VideoWriter(
                        final_video_path, fourcc, self.args.video_fps,
                        (final_display.shape[1], final_display.shape[0])
                    )
                    print(f"正在保存视频到:\n  {yolo_video_path}\n  {final_video_path}")
                else:
                    display_image = self.display_manager.create_layout(
                        test_frame, useful_area, annotated_frame, self.args.display_scale
                    )
                    if self.args.video_name:
                        video_filename = self.args.video_name
                        if not video_filename.endswith('.mp4'):
                            video_filename += '.mp4'
                        video_path = os.path.join(output_dir, video_filename)
                    else:
                        video_path = os.path.join(output_dir, f'detection_result_{timestamp}.mp4')
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        video_path, fourcc, self.args.video_fps,
                        (display_image.shape[1], display_image.shape[0])
                    )
                    print(f"正在保存视频到: {video_path}")

                self.camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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

                if self.args.save_video and video_writer and yolo_video_writer:
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

                if self.args.save_video and video_writer:
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
        if self.use_deep_sort:
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
