"""
鱼眼全景YOLO姿态检测主程序 - 重构版
整合了原 main.py 的功能，使用新的目录结构
"""
import cv2
import numpy as np
import time
import os
from typing import List, Tuple, Optional, Dict

# 导入配置
import config

# 导入核心模块
from core import (
    CameraProcessor,
    FisheyePanorama,
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


class FisheyePanoramaYOLOPose:
    """鱼眼全景YOLO姿态检测主类"""

    def __init__(self, args):
        """
        初始化
        """
        self.args = args
        self.camera = None
        self.panorama_processor = None
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
            match_thresh=0.6,
            proximity_thresh=0.4,
            appearance_thresh=args.appearance_thresh,
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
        self.seg_masker = None   # SegMasker，initialize() 中按模型文件是否存在决定是否加载
        self.model_path = "imagenet.pyth/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
        if not os.path.exists(self.model_path):
            self.model_path = None

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
            first_img = cv2.imread(self.image_files[0])
            if first_img is None:
                print(f"错误: 无法读取图片 {self.image_files[0]}")
                return False
            actual_width, actual_height = first_img.shape[1], first_img.shape[0]
        else:
            self.camera = CameraProcessor(
                cam_index=self.args.cam_index,
                video_path=self.args.video_path,
                width=self.args.cam_width,
                height=self.args.cam_height
            )

            if not self.camera.initialize():
                return False

            camera_info = self.camera.get_camera_info()
            source_type = "视频文件" if self.args.video_path else "摄像头"
            print(f"输入源: {source_type}")
            actual_width, actual_height = camera_info['width'], camera_info['height']

        print(f"使用实际分辨率: {actual_width}x{actual_height} 初始化全景处理器")

        self.panorama_processor = FisheyePanorama(
            cam_width=actual_width,
            cam_height=actual_height,
            output_width=self.args.output_width,
            output_height=self.args.output_height,
            vertical_fov=self.args.vertical_fov,
            map_file=self.args.map_file,
            cam_index=self.args.cam_index
        )

        if self.panorama_processor:
            output_width = self.panorama_processor.output_width
            output_height = self.panorama_processor.output_height
            self.angle_calculator = AngleCalculator(
                output_width,
                output_height,
                self.args.vertical_fov,
                fit_degree=getattr(self.args, 'fit_degree', 5),
                yaml_file=getattr(self.args, 'calib_yaml', None)
            )

            if self.deep_sort_tracker.enable_boundary_matching:
                self.deep_sort_tracker.set_boundary_frame_size(output_width, output_height)
                print(f"边界匹配器初始化：画面={output_width}x{output_height}")

        if not os.path.exists(self.args.model_path):
            print(f"错误: YOLO模型文件 {self.args.model_path} 不存在")
            print("请确保模型文件在当前目录，或指定正确的路径")
            return False

        self.yolo_detector = YOLOPoseDetector(
            model_path=self.args.model_path,
            conf_threshold=self.args.conf_threshold,
            iou_threshold=self.args.iou_threshold
        )

        self.display_manager = DisplayManager(use_dual_windows=self.args.use_dual_windows,no_display= self.no_display)

        print("初始化OSNet特征提取器...")
        try:
            self.feature_extractor = FeatureExtractor(
                model_name='osnet_x0_25',
                model_path=self.model_path
            )
            print("OSNet特征提取器初始化成功！")
        except Exception as e:
            print(f"警告: OSNet特征提取器初始化失败: {e}")
            print("将不使用ReID特征进行跟踪")
            self.feature_extractor = None

        # Seg-guided ReID: 加载分割模型（需 --use-seg-reid 开关启用）
        if getattr(self.args, 'use_seg_reid', False):
            import torch as _torch
            seg_path = getattr(self.args, 'seg_model_path', './yolo26n-seg.pt')
            if os.path.exists(seg_path):
                try:
                    from core.seg_masker import SegMasker
                    self.seg_masker = SegMasker(
                        seg_path,
                        device='cuda' if _torch.cuda.is_available() else 'cpu',
                    )
                except Exception as e:
                    print(f"[SegMasker] 加载失败: {e}，将跳过 Seg 步骤")
                    self.seg_masker = None
            else:
                print(f"[SegMasker] 模型文件不存在: {seg_path}，将跳过 Seg 步骤")
        else:
            print("[SegMasker] --use-seg-reid 未启用，跳过分割模型加载")

        print("初始化完成！")
        return True

    def process_panorama_slices(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List], Optional[Dict]]:
        """
        多切片处理
        返回: panorama, yolo_only_image, final_image, tracked_detections, angle_info
        """
        panorama = self.panorama_processor.apply_panorama(frame)
        original_panorama_height = panorama.shape[0]

        if self.angle_calculator and self.panorama_processor.center is not None:
            fp = self.panorama_processor
            if (self.angle_calculator.fisheye_center != fp.center or
                self.angle_calculator.fisheye_radius != fp.radius):
                self.angle_calculator.set_fisheye_mapping(
                    fp.center,
                    fp.radius,
                    getattr(fp, 'img_width', frame.shape[1]),
                    getattr(fp, 'img_height', frame.shape[0])
                )
                if fp.map_x is not None and fp.map_y is not None:
                    self.angle_calculator.set_panorama_maps(fp.map_x, fp.map_y)

        crop_height = 0
        if self.args.crop_divisor > 0:
            crop_height = original_panorama_height // self.args.crop_divisor
            panorama = panorama[crop_height:, :]

        if self.angle_calculator:
            self.angle_calculator.set_crop_offset(crop_height)

        slices, slice_infos = self.slicer.slice_panorama(panorama, num_slices=self.num_slices)

        all_yolo_results = []
        for slice_img in slices:
            yolo_result = self.yolo_detector.detect(slice_img, use_tracking=False)
            all_yolo_results.append(yolo_result)

        merged_detections = self.slicer.merge_detections(
            all_yolo_results,
            slice_infos,
            slice_images=slices,
            feature_extractor=self.feature_extractor
        )

        filtered_detections = filter_cross_boundary_detections(merged_detections, panorama.shape)
        filtered_detections = self.slicer.filter_wide_detections(filtered_detections, panorama.shape[1])

        # Seg-guided ReID：用分割掩码消除背景后重提特征
        if self.seg_masker is not None and self.feature_extractor is not None:
            self.seg_masker.reextract_features(panorama, filtered_detections, self.feature_extractor)

        detections_with_features = []
        for det in filtered_detections:
            det_with_feat = det.copy()
            if 'feature' not in det_with_feat:
                det_with_feat['feature'] = None
            detections_with_features.append(det_with_feat)

        if self.use_deep_sort:
            tracked_detections = self.deep_sort_tracker.update(detections_with_features)
        else:
            tracked_detections = filtered_detections
            for i, det in enumerate(tracked_detections):
                det['track_id'] = i + 1

        yolo_only_image = draw_yolo_only(panorama, filtered_detections)
        annotated_panorama = draw_detections(panorama, tracked_detections, self.deep_sort_tracker)
        if self.seg_masker is not None:
            self.seg_masker.draw_last_masks(annotated_panorama)

        angle_info = None
        if tracked_detections and self.angle_calculator:
            keypoints_list = []
            for detection in tracked_detections:
                if 'keypoints' in detection and detection['keypoints']:
                    kpts = np.array(detection['keypoints'])
                    keypoints_list.append(kpts)

            if keypoints_list:
                keypoints_array = np.array(keypoints_list)
                angle_info = self.angle_calculator.calculate_angles_from_keypoints(keypoints_array)

                if self.show_angle_overview:
                    annotated_panorama = self.angle_calculator.draw_angle_overview(annotated_panorama, angle_info)
                else:
                    annotated_panorama = self.angle_calculator.draw_angles_on_image(annotated_panorama, angle_info)

        return panorama, yolo_only_image, annotated_panorama, tracked_detections, angle_info

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
                     # 关键修改：如果是无界面模式，直接退出，不等待键盘
                    if hasattr(self, 'no_display') and self.no_display:
                        print("无界面模式：视频处理完成，程序退出。")
                        return  # 直接结束程序
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

    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")

        if self.camera:
            self.camera.release()
        if self.display_manager:
            self.display_manager.destroy_windows()

        if self.use_deep_sort:
            print_assignment_stats()


def main():
    """主函数"""
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
