import cv2
import numpy as np
import time
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional, Dict

# 导入自定义模块
import config
from camera_processor import CameraProcessor
from panorama_processor import PanoramaProcessor
from yolo_pose import YOLOPoseDetector
from display_manager import DisplayManager
from PanoramaSlicer import PanoramaSlicer
from AngleCalculator import AngleCalculator
from BoT_SORT import BoT_SORTTracker, print_assignment_stats

# 导入特征提取器和相似度计算
from db import FeatureExtractorManager, cosine_similarity_numpy


class SimilarityTracker:
    """
    特征相似度追踪器 - 混合策略

    策略：
    1. 保存整个视频中出现的第一个目标的特征
    2. 保存整个视频中置信度最高(>0.85)的目标特征
    3. 后续每一帧的所有目标都与这两个全局特征计算相似度
    """

    def __init__(self, high_conf_threshold=0.85):
        self.high_conf_threshold = high_conf_threshold
        # 存储第一个出现的目标的特征
        self.first_ever_feature = None  # {'feature': np.array, 'frame_id': int, 'track_id': int, 'confidence': float}
        # 存储整个视频中置信度最高的目标特征
        self.highest_conf_feature = None  # {'feature': np.array, 'frame_id': int, 'track_id': int, 'confidence': float}
        # 记录每一帧的相似度数据
        self.frame_records = []

    def update_and_calculate(self, frame_id: int, tracked_detections: List[Dict], panorama_img: np.ndarray,
                             feature_extractor) -> List[Dict]:
        """
        更新特征并计算相似度

        参数:
            frame_id: 帧编号
            tracked_detections: 追踪结果列表，每个元素包含 'track_id', 'feature', 'bbox', 'confidence'
            panorama_img: 全景图像（用于抠图）
            feature_extractor: 特征提取器

        返回:
            该帧的记录列表
        """
        frame_results = []

        for det in tracked_detections:
            track_id = det.get('track_id', -1)
            if track_id == -1:
                continue

            confidence = det.get('confidence', 0.0)
            feature = det.get('feature', None)

            # 如果检测结果中没有特征，尝试提取
            if feature is None and feature_extractor is not None:
                bbox = det.get('bbox', None)
                if bbox is not None:
                    x1, y1, x2, y2 = [int(x) for x in bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(panorama_img.shape[1], x2), min(panorama_img.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        crop_img = panorama_img[y1:y2, x1:x2]
                        try:
                            feature_tensor = feature_extractor.extract_features_from_image_array(crop_img)
                            feature = feature_tensor.numpy().flatten()
                            det['feature'] = feature  # 保存回检测结果
                        except Exception as e:
                            print(f"  提取特征失败 (ID={track_id}): {e}")
                            feature = None

            if feature is None:
                continue

            # 1. 保存第一个出现的目标的特征（如果还没有保存）
            if self.first_ever_feature is None:
                self.first_ever_feature = {
                    'feature': feature.copy(),
                    'frame_id': frame_id,
                    'track_id': track_id,
                    'confidence': confidence
                }
                print(f"  [特征] 首个目标特征已保存 (帧={frame_id}, ID={track_id}, 置信度={confidence:.2f})")

            # 2. 更新整个视频中置信度最高的特征
            if confidence >= self.high_conf_threshold:
                if (self.highest_conf_feature is None or
                    confidence > self.highest_conf_feature['confidence']):
                    self.highest_conf_feature = {
                        'feature': feature.copy(),
                        'frame_id': frame_id,
                        'track_id': track_id,
                        'confidence': confidence
                    }
                    print(f"  [特征] 最高置信度特征已更新 (帧={frame_id}, ID={track_id}, 置信度={confidence:.2f})")

            # 3. 计算相似度
            sim_first_ever = 0.0
            sim_highest_conf = 0.0

            # 与第一个出现的目标特征的相似度
            if self.first_ever_feature is not None:
                sim_first_ever = cosine_similarity_numpy(feature, self.first_ever_feature['feature'])

            # 与整个视频最高置信度特征的相似度
            if self.highest_conf_feature is not None:
                sim_highest_conf = cosine_similarity_numpy(feature, self.highest_conf_feature['feature'])

            # 记录结果
            result = {
                'frame_id': frame_id,
                'track_id': track_id,
                'confidence': confidence,
                'bbox': str(det.get('bbox', [])),
                'similarity_with_first_ever': sim_first_ever,
                'first_ever_feature_frame_id': self.first_ever_feature['frame_id'] if self.first_ever_feature is not None else -1,
                'first_ever_feature_track_id': self.first_ever_feature['track_id'] if self.first_ever_feature is not None else -1,
                'similarity_with_highest_conf': sim_highest_conf,
                'highest_conf_feature_frame_id': self.highest_conf_feature['frame_id'] if self.highest_conf_feature is not None else -1,
                'highest_conf_feature_track_id': self.highest_conf_feature['track_id'] if self.highest_conf_feature is not None else -1,
                'highest_conf_feature_confidence': self.highest_conf_feature['confidence'] if self.highest_conf_feature is not None else 0.0
            }

            frame_results.append(result)
            self.frame_records.append(result)

        return frame_results

    def export_to_excel(self, output_path: str):
        """导出结果到Excel"""
        if not self.frame_records:
            print("  没有相似度记录可导出")
            return None

        df = pd.DataFrame(self.frame_records)

        # 重新排列列顺序
        columns_order = [
            'frame_id', 'track_id', 'confidence', 'bbox',
            'similarity_with_first_ever', 'first_ever_feature_frame_id', 'first_ever_feature_track_id',
            'similarity_with_highest_conf', 'highest_conf_feature_frame_id', 'highest_conf_feature_track_id', 'highest_conf_feature_confidence'
        ]

        # 确保所有列都存在
        for col in columns_order:
            if col not in df.columns:
                df[col] = -1

        df = df[columns_order]

        # 保存到Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Frame_Level_Similarity', index=False)

            # 添加摘要页
            summary_data = []

            # 首个目标信息
            if self.first_ever_feature is not None:
                summary_data.append({
                    'item': 'First_Ever_Target',
                    'track_id': self.first_ever_feature['track_id'],
                    'frame_id': self.first_ever_feature['frame_id'],
                    'confidence': self.first_ever_feature['confidence']
                })

            # 最高置信度目标信息
            if self.highest_conf_feature is not None:
                summary_data.append({
                    'item': 'Highest_Conf_Target',
                    'track_id': self.highest_conf_feature['track_id'],
                    'frame_id': self.highest_conf_feature['frame_id'],
                    'confidence': self.highest_conf_feature['confidence']
                })

            # 每个ID的统计
            all_track_ids = sorted(set(r['track_id'] for r in self.frame_records))
            for track_id in all_track_ids:
                # 获取该ID的所有记录
                id_records = [r for r in self.frame_records if r['track_id'] == track_id]
                if not id_records:
                    continue

                sim_first_ever_list = [r['similarity_with_first_ever'] for r in id_records]
                sim_highest_conf_list = [r['similarity_with_highest_conf'] for r in id_records if r['similarity_with_highest_conf'] > 0]

                summary_data.append({
                    'item': f'Track_ID_{track_id}',
                    'track_id': track_id,
                    'total_frames': len(id_records),
                    'avg_sim_with_first_ever': np.mean(sim_first_ever_list) if sim_first_ever_list else 0,
                    'min_sim_with_first_ever': np.min(sim_first_ever_list) if sim_first_ever_list else 0,
                    'max_sim_with_first_ever': np.max(sim_first_ever_list) if sim_first_ever_list else 0,
                    'avg_sim_with_highest_conf': np.mean(sim_highest_conf_list) if sim_highest_conf_list else 0,
                    'min_sim_with_highest_conf': np.min(sim_highest_conf_list) if sim_highest_conf_list else 0,
                    'max_sim_with_highest_conf': np.max(sim_highest_conf_list) if sim_highest_conf_list else 0
                })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"  相似度结果已导出到: {output_path}")
        return df


class DetectionLogger:
    """检测日志记录器"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, f'detection_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        self.excel_path = os.path.join(output_dir, f'detection_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        self.data = {
            'yolo_model': '',
            'osnet_model': '',
            'frames': [],
            'summary': {}
        }
        self.total_frames = 0
        self.frames_with_yolo = 0
        self.frames_with_final = 0
        self.similarity_tracker = SimilarityTracker(high_conf_threshold=0.85)

    def set_models(self, yolo_model: str, osnet_model: str):
        """设置模型名称"""
        self.data['yolo_model'] = yolo_model
        self.data['osnet_model'] = osnet_model

    def log_frame(self, frame_id: int, yolo_ids: List[int], final_ids: List[int],
                  tracked_detections: List[Dict] = None, panorama_img: np.ndarray = None,
                  feature_extractor = None):
        """记录单帧检测结果"""
        self.total_frames += 1

        if yolo_ids:
            self.frames_with_yolo += 1
        if final_ids:
            self.frames_with_final += 1

        # 记录相似度
        sim_results = []
        if tracked_detections is not None and panorama_img is not None:
            sim_results = self.similarity_tracker.update_and_calculate(
                frame_id, tracked_detections, panorama_img, feature_extractor
            )

        self.data['frames'].append({
            'frame_id': frame_id,
            'yolo_detection_ids': yolo_ids,
            'final_detection_ids': final_ids,
            'similarity_results': sim_results
        })

    def save(self):
        """保存日志到文件并计算统计"""
        # 计算统计
        yolo_ratio = self.frames_with_yolo / self.total_frames if self.total_frames > 0 else 0
        final_ratio = self.frames_with_final / self.total_frames if self.total_frames > 0 else 0

        self.data['summary'] = {
            'total_frames': self.total_frames,
            'frames_with_yolo': self.frames_with_yolo,
            'frames_with_final': self.frames_with_final,
            'yolo_detection_ratio': yolo_ratio,
            'final_detection_ratio': final_ratio
        }

        # 保存JSON
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        # 保存相似度Excel
        excel_path = os.path.join(self.output_dir, f'similarity_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        self.similarity_tracker.export_to_excel(excel_path)

        # 打印统计
        print(f"\n{'='*60}")
        print(f"检测统计:")
        print(f"  YOLO模型: {self.data['yolo_model']}")
        print(f"  OSNet模型: {self.data['osnet_model']}")
        print(f"  总帧数: {self.total_frames}")
        print(f"  YOLO检测到目标的帧数: {self.frames_with_yolo} ({yolo_ratio:.2%})")
        print(f"  最终检测到目标的帧数: {self.frames_with_final} ({final_ratio:.2%})")
        if self.similarity_tracker.first_ever_feature is not None:
            print(f"  首个目标: ID={self.similarity_tracker.first_ever_feature['track_id']}, 帧={self.similarity_tracker.first_ever_feature['frame_id']}, 置信度={self.similarity_tracker.first_ever_feature['confidence']:.2f}")
        if self.similarity_tracker.highest_conf_feature is not None:
            print(f"  最高置信度: ID={self.similarity_tracker.highest_conf_feature['track_id']}, 帧={self.similarity_tracker.highest_conf_feature['frame_id']}, 置信度={self.similarity_tracker.highest_conf_feature['confidence']:.2f}")
        print(f"{'='*60}")
        print(f"JSON日志已保存到: {self.log_file}")


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
        # 添加角度计算器
        self.angle_calculator = None
        # 添加角度显示控制属性
        self.show_angles = True
        self.show_angle_overview = False
        # 添加3切片相关的参数
        self.num_slices = 3  # 默认使用3切片
        self.slice_overlap = 0.1  # 重叠比例
        self.use_deep_sort = args.use_deep_sort  # 使用DeepSORT（结合运动和外观特征）
        # 文件夹处理模式相关
        self.image_files = []
        self.current_image_index = 0

        if not hasattr(self, 'slicer'):
            self.slicer = PanoramaSlicer(
                overlap_ratio=self.slice_overlap,
                iou_threshold=0.3,
                confidence_threshold=0.5,
                reid_similarity_threshold=0.7
            )

        # 初始化BoT-SORT跟踪器（结合运动和外观特征）
        # 注意：画面尺寸会在initialize()中确定后更新
        self.deep_sort_tracker = BoT_SORTTracker(
            track_high_thresh=0.3,                        # 高置信度检测阈值
            track_low_thresh=0.1,                         # 低置信度检测阈值
            new_track_thresh=0.3,                         # 新轨迹阈值
            track_buffer=30,                               # 轨迹缓存帧数
            match_thresh=args.deep_sort_match_thresh,     # 融合匹配阈值
            proximity_thresh=0.5,                           # IoU阈值
            appearance_thresh=args.appearance_thresh,     # 外观特征匹配阈值
            frame_rate=30,
            feat_history=50,                              # 特征历史长度
            with_reid=True,                                # 启用ReID特征融合
            use_hungarian=args.use_hungarian,              # 是否使用匈牙利算法
            enable_boundary_matching=True,                 # 启用边界ID穿越匹配
            frame_width=3840,                              # 默认值，会在initialize()中更新
            frame_height=1080,                             # 默认值，会在initialize()中更新
            boundary_margin=0.1,                           # 边界区域10%
            boundary_time_window=30,                       # 30帧时间窗口
            boundary_similarity_thresh=0.3,                # 特征相似度阈值
            boundary_debug=False,                            # 启用调试输出
            enable_top_boundary=False,                      # 禁用顶部边界
            enable_bottom_boundary=False,                     # 禁用底部边界
            enable_left_boundary=True,                      # 启用左侧边界
            enable_right_boundary=True                      # 启用右侧边界
        )

        # 初始化OSNet特征提取器
        self.feature_extractor = None
        self.model_path = r"imagenet.pyth\osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
        if not os.path.exists(self.model_path):
            self.model_path = None

        # 初始化日志记录器
        self.logger = None

    def initialize(self) -> bool:
        """初始化所有组件"""
        print("初始化鱼眼展开和YOLO姿态检测系统...")

        # 初始化日志记录器
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.logger = DetectionLogger(output_dir)

        # 判断输入源类型
        if self.args.folder_path:
            # 文件夹模式
            if not os.path.isdir(self.args.folder_path):
                print(f"错误: 文件夹 {self.args.folder_path} 不存在")
                return False
            # 获取文件夹中的所有图片文件
            self.image_files = self._get_image_files(self.args.folder_path)
            if not self.image_files:
                print(f"错误: 文件夹 {self.args.folder_path} 中没有找到图片文件")
                return False
            print(f"输入源: 图片文件夹 ({len(self.image_files)} 张图片)")
            # 读取第一张图片来获取分辨率
            first_img = cv2.imread(self.image_files[0])
            if first_img is None:
                print(f"错误: 无法读取图片 {self.image_files[0]}")
                return False
            actual_width, actual_height = first_img.shape[1], first_img.shape[0]
        else:
            # 摄像头或视频模式
            self.camera = CameraProcessor(
                cam_index=self.args.cam_index,
                video_path=self.args.video_path,  # 新增参数
                width=self.args.cam_width,
                height=self.args.cam_height
            )

            if not self.camera.initialize():
                return False

            # 获取相机信息
            camera_info = self.camera.get_camera_info()

            # 打印输入源类型
            source_type = "视频文件" if self.args.video_path else "摄像头"
            print(f"输入源: {source_type}")
            actual_width, actual_height = camera_info['width'], camera_info['height']

        # 3. 使用实际分辨率初始化全景处理器

        print(f"使用实际分辨率: {actual_width}x{actual_height} 初始化全景处理器")

        self.panorama_processor = PanoramaProcessor(
            cam_width=actual_width,
            cam_height=actual_height,
            output_width=self.args.output_width,
            output_height=self.args.output_height,
            vertical_fov=self.args.vertical_fov,
            map_file=self.args.map_file,
            cam_index=self.args.cam_index  # 传递相机索引
        )
        if self.panorama_processor:
            output_width = self.panorama_processor.output_width
            output_height = self.panorama_processor.output_height
            self.angle_calculator = AngleCalculator(output_width, output_height, self.args.vertical_fov)

            # === 更新边界匹配器的画面尺寸 ===
            if self.deep_sort_tracker.enable_boundary_matching:
                self.deep_sort_tracker.set_boundary_frame_size(output_width, output_height)
                print(f"边界匹配器画面尺寸已设置: {output_width}x{output_height}")

        # 3. 检查YOLO模型文件
        if not os.path.exists(self.args.model_path):
            print(f"错误: YOLO模型文件 {self.args.model_path} 不存在")
            print("请确保模型文件在当前目录，或指定正确的路径")
            print("您可以从以下地址下载预训练模型:")
            print("https://github.com/ultralytics/ultralytics")
            return False

        # 4. 初始化YOLO检测器
        self.yolo_detector = YOLOPoseDetector(
            model_path=self.args.model_path,
            conf_threshold=self.args.conf_threshold,
            iou_threshold=self.args.iou_threshold
        )

        # 5. 初始化显示管理器
        self.display_manager = DisplayManager(use_dual_windows=self.args.use_dual_windows)

        # 6. 初始化OSNet特征提取器
        print("初始化OSNet特征提取器...")
        try:
            self.feature_extractor = FeatureExtractorManager(
                model_name='osnet_ain_x1_0',
                model_path=self.model_path
            )
            print("OSNet特征提取器初始化成功！")
        except Exception as e:
            print(f"警告: OSNet特征提取器初始化失败: {e}")
            print("将不使用ReID特征进行跟踪")
            self.feature_extractor = None

        print("初始化完成！")

        # 记录模型名称
        yolo_model_name = os.path.basename(self.args.model_path)
        osnet_model_name = os.path.basename(self.model_path) if self.model_path else 'None'
        self.logger.set_models(yolo_model_name, osnet_model_name)

        return True

    def process_frame_three_slices_optimized(self, frame: np.ndarray, frame_id: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List], Optional[Dict]]:
        """
        3切片处理
        方案：切片检测(高精度) -> 合并坐标
        返回: panorama, yolo_only_image, final_image, tracked_detections, angle_info
        """
        # 1. 全景展开
        panorama = self.panorama_processor.apply_panorama(frame)
        original_panorama_height = panorama.shape[0]
        original_panorama_width = panorama.shape[1]

        # --- 裁剪处理并记录偏移量 ---
        crop_height = 0
        if self.args.crop_divisor > 0:
            crop_height = original_panorama_height // self.args.crop_divisor
            panorama = panorama[crop_height:, :]

        # 设置角度计算器的裁剪偏移量
        if self.angle_calculator:
            self.angle_calculator.set_crop_offset(crop_height)

        # 2. 使用 PanoramaSlicer 切为3片
        slices, slice_infos = self.slicer.slice_panorama(panorama, num_slices=self.num_slices)

        # 3. 对每个切片进行检测（不使用跟踪，避免切片内ID冲突）
        all_yolo_results = []
        slice_detections_count = []
        for i, slice_img in enumerate(slices):
            yolo_result = self.yolo_detector.detect(slice_img, use_tracking=False)
            all_yolo_results.append(yolo_result)
            # 统计每个切片的检测数
            dets = self.slicer.extract_detections_from_yolo_results(yolo_result)
            slice_detections_count.append(len(dets))

        # 4. 合并检测结果 - 使用OSNet特征提取器进行智能去重
        merged_detections = self.slicer.merge_detections(
            all_yolo_results,
            slice_infos,
            slice_images=slices,  # 传入切片图像用于特征提取
            feature_extractor=self.feature_extractor  # 传入OSNet特征提取器
        )

        # 5. 过滤检测结果
        filtered_detections = self._filter_cross_boundary_detections(merged_detections, panorama.shape)

        # 5.1. 新增：过滤横跨全景边界的超宽无效检测框
        filtered_detections = self.slicer.filter_wide_detections(filtered_detections, panorama.shape[1])

        # 6. 准备带特征的检测结果（注意：特征已在PanoramaSlicer中提取，直接使用即可）
        detections_with_features = []
        feature_count = 0
        for det in filtered_detections:
            # 确保 'feature' 字段存在（PanoramaSlicer 中已提取）
            det_with_feat = det.copy()
            if 'feature' not in det_with_feat:
                det_with_feat['feature'] = None
            else:
                if det_with_feat['feature'] is not None:
                    feature_count += 1
            detections_with_features.append(det_with_feat)

        # 7. 使用DeepSORT跟踪器进行全局跟踪
        if self.use_deep_sort:
            # 使用DeepSORT风格跟踪器（结合运动和外观特征）
            tracked_detections = self.deep_sort_tracker.update(detections_with_features)

            # === 调试：打印边界匹配统计 ===
            if self.deep_sort_tracker.enable_boundary_matching and self.deep_sort_tracker.frame_id % 30 == 0:
                boundary_stats = self.deep_sort_tracker.get_boundary_stats()
                print(f"[边界匹配统计] 帧={self.deep_sort_tracker.frame_id}: {boundary_stats}")

            # === 检查是否有边界匹配的ID ===
            matched_count = sum(1 for det in tracked_detections if det.get('_boundary_matched', False))
            if matched_count > 0:
                print(f"  边界匹配成功: {matched_count} 个ID被复用")
        else:
            # 不使用跟踪器，直接用过滤后的检测结果（手动分配临时ID）
            tracked_detections = filtered_detections
            for i, det in enumerate(tracked_detections):
                det['track_id'] = i + 1  # 分配临时ID

        # 8. 绘制纯YOLO检测结果（不含跟踪ID和角度）
        yolo_only_image = self._draw_yolo_only(panorama, filtered_detections)

        # 9. 绘制检测结果（使用BoT-SORT提供的track_id）
        annotated_panorama = self._draw_detections(panorama, tracked_detections)

        # 9. 计算并绘制角度
        angle_info = None
        if tracked_detections and self.angle_calculator:
            # 从检测结果中提取关键点
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

        # 记录检测日志和相似度
        if self.logger:
            yolo_ids = [i+1 for i in range(len(filtered_detections))]  # YOLO检测用临时ID
            final_ids = [det.get('track_id', -1) for det in tracked_detections if det.get('track_id', -1) != -1]
            self.logger.log_frame(frame_id, yolo_ids, final_ids, tracked_detections, panorama, self.feature_extractor)

        return panorama, yolo_only_image, annotated_panorama, tracked_detections, angle_info

    def _get_image_files(self, folder_path: str) -> List[str]:
        """获取文件夹中的所有图片文件"""
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        image_files = []
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                image_files.append(os.path.join(folder_path, filename))
        return sorted(image_files)

    def _process_single_image(self, image_path: str, crops_dir: str = None, frame_id: int = 0):
        """处理单张图片"""
        print(f"处理图片: {os.path.basename(image_path)}")
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"警告: 无法读取图片 {image_path}")
            return

        # 处理帧
        panorama, yolo_only_frame, annotated_frame, detection_results, angle_info = self.process_frame_three_slices_optimized(frame, frame_id)

        # 保存处理后的图片
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = self.args.output_dir

        # 保存原图、全景图、检测结果图
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_original.jpg'), frame)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_panorama.jpg'), panorama)
        cv2.imwrite(os.path.join(output_dir, f'{base_name}_detection.jpg'), annotated_frame)

        # 保存检测框抠图（如果 --save-crops 参数启用）
        if self.args.save_crops and detection_results and crops_dir:
            for crop_idx, det in enumerate(detection_results, 1):
                x1, y1, x2, y2 = det['bbox']
                # 确保坐标在图像范围内
                h, w = panorama.shape[:2]
                x1 = max(0, int(x1))
                y1 = max(0, int(y1))
                x2 = min(w, int(x2))
                y2 = min(h, int(y2))
                # 裁剪图像
                if x2 > x1 and y2 > y1:
                    crop_img = panorama[y1:y2, x1:x2]
                    # 保存抠图，命名为原图名称加上_1,_2等
                    crop_filename = f'{base_name}_{crop_idx}.jpg'
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, crop_img)
                    print(f"  保存抠图: {crop_filename}")

        print(f"  检测到 {len(detection_results) if detection_results else 0} 个人")

    def run(self):
        """
        运行主循环
        """
        # 判断输入源类型
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

        # 创建输出目录
        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 如果保存抠图，创建crops子目录
        crops_dir = None
        if self.args.save_crops:
            crops_dir = os.path.join(output_dir, 'crops')
            if not os.path.exists(crops_dir):
                os.makedirs(crops_dir)

        # 文件夹处理模式
        if self.args.folder_path:
            print(f"开始处理文件夹中的 {len(self.image_files)} 张图片...")
            for frame_idx, img_path in enumerate(self.image_files, 1):
                self._process_single_image(img_path, crops_dir, frame_idx)
            print(f"\n处理完成！共处理 {len(self.image_files)} 张图片")
            print(f"结果保存在: {output_dir}")
            return

        # 摄像头或视频模式
        # 如果是视频，添加重新播放选项
        is_video = self.args.video_path is not None

        # 帧计数（用于自动保存）
        frame_count = 0

        # 视频写入器初始化
        video_writer = None
        yolo_video_writer = None
        if self.args.save_video:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 先处理一帧获取显示尺寸
            ret, test_frame = self.camera.get_frame()
            if ret:
                panorama, yolo_only_frame, annotated_frame, _, _ = self.process_frame_three_slices_optimized(test_frame, 0)
                useful_area = self.panorama_processor.get_useful_area(test_frame)

                if self.args.use_dual_windows:
                    # 双窗口模式：保存两个视频
                    yolo_display = self.display_manager.add_info_overlay(
                        yolo_only_frame, "YOLO Detection Only", "", ""
                    )
                    final_display = self.display_manager.create_layout(
                        test_frame, useful_area, annotated_frame, self.args.display_scale
                    )
                    # 使用用户指定的文件名或自动生成
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
                    # 单窗口模式：保存一个视频
                    display_image = self.display_manager.create_layout(
                        test_frame, useful_area, annotated_frame, self.args.display_scale
                    )
                    # 使用用户指定的文件名或自动生成
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

                # 将测试帧重置
                self.camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            # 获取帧
            ret, frame = self.camera.get_frame()
            if not ret:
                if is_video:
                    # 视频播放完毕，询问是否重新播放
                    print("视频播放完毕")
                    print("按 'r' 键重新播放，按 'q' 键退出")
                    while True:
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('r'):
                            # 重新播放视频
                            self.camera.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_count = 0
                            ret, frame = self.camera.get_frame()
                            if ret:
                                break
                        elif key == ord('q'):
                            return
                else:
                    # 摄像头错误
                    print("无法从摄像头接收帧")
                    break

            frame_count += 1

            # 处理帧
            panorama, yolo_only_frame, annotated_frame, detection_results, _ = self.process_frame_three_slices_optimized(frame, frame_count)

            # 获取有效区域
            useful_area = self.panorama_processor.get_useful_area(frame)

            # 更新FPS
            fps = self.display_manager.update_fps()

            # 准备信息文本
            info_text = f"YOLO Pose Detection (Conf: {self.args.conf_threshold}, IOU: {self.args.iou_threshold})"

            # 自动保存帧（如果 --save-frames 参数启用）
            if self.args.save_frames:
                # 使用帧编号而不是时间戳，确保顺序正确
                self.display_manager.save_frame(frame, f'original_{frame_count:06d}.jpg', output_dir)
                self.display_manager.save_frame(panorama, f'panorama_{frame_count:06d}.jpg', output_dir)
                self.display_manager.save_frame(annotated_frame, f'detection_{frame_count:06d}.jpg', output_dir)
                if frame_count % 10 == 0:
                    print(f"已保存 {frame_count} 帧...")

            # 保存检测框抠图（如果 --save-crops 参数启用）
            if self.args.save_crops and detection_results:
                # 保存每个检测框的抠图
                for crop_idx, det in enumerate(detection_results, 1):
                    x1, y1, x2, y2 = det['bbox']
                    # 确保坐标在图像范围内
                    h, w = panorama.shape[:2]
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(w, int(x2))
                    y2 = min(h, int(y2))
                    # 裁剪图像
                    if x2 > x1 and y2 > y1:
                        crop_img = panorama[y1:y2, x1:x2]
                        # 保存抠图，命名为 frame_{帧编号}_{序号}.jpg
                        crop_filename = f'frame_{frame_count:06d}_{crop_idx}.jpg'
                        crop_path = os.path.join(crops_dir, crop_filename)
                        cv2.imwrite(crop_path, crop_img)

            # 获取性能统计
            perf_stats = self.yolo_detector.get_performance_stats()
            perf_text = f"FPS: {fps}, Inference: {perf_stats['avg_inference_time_ms']:.1f}ms"

            # 检测计数
            num_people = len(detection_results) if detection_results else 0
            count_text = f"Detected Persons: {num_people}"

            # 添加显示模式信息
            if self.show_angles:
                mode_text = " | Angles ON"
                if self.show_angle_overview:
                    mode_text += " (Overview)"
                else:
                    mode_text += " (Detail)"
                info_text += mode_text

            # 根据配置选择显示方式
            if self.args.use_dual_windows:
                # 双窗口模式：一个窗口只显示YOLO检测结果，另一个显示最终结果
                # 为YOLO窗口添加信息覆盖层
                yolo_display = self.display_manager.add_info_overlay(
                    yolo_only_frame, "YOLO Detection Only", perf_text, count_text
                )
                # 为最终窗口创建布局并添加信息覆盖层
                final_display = self.display_manager.create_layout(
                    test_frame, useful_area, annotated_frame, self.args.display_scale
                )
                final_display = self.display_manager.add_info_overlay(
                    final_display, info_text, perf_text, count_text
                )
                # 双窗口显示
                self.display_manager.show_dual(yolo_display, final_display)

                # 写入视频帧
                if self.args.save_video and video_writer and yolo_video_writer:
                    yolo_video_writer.write(yolo_display)
                    video_writer.write(final_display)
            else:
                # 单窗口模式：创建组合显示布局
                display_image = self.display_manager.create_layout(
                    frame, useful_area, annotated_frame, self.args.display_scale
                )
                # 添加信息覆盖层
                display_image = self.display_manager.add_info_overlay(
                    display_image, info_text, perf_text, count_text
                )
                # 单窗口显示
                self.display_manager.show(display_image)

                # 写入视频帧
                if self.args.save_video and video_writer:
                    video_writer.write(display_image)

            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if self.handle_keyboard(key, frame, panorama, annotated_frame, output_dir):
                break

            # 键盘输入已在 handle_keyboard 中统一处理

        # 释放视频写入器
        if self.args.save_video:
            if video_writer:
                video_writer.release()
                print("视频已保存完成！")
            if yolo_video_writer:
                yolo_video_writer.release()

    def handle_keyboard(self, key: int, original_frame: np.ndarray,
                  panorama: np.ndarray, annotated_frame: np.ndarray,
                  output_dir: str) -> bool:
        """
        处理键盘输入
        返回: True表示退出，False表示继续
        """
        if key == ord('q'):
            return True
        elif key == ord('s'):
            # 保存帧
            timestamp = int(time.time())
            self.display_manager.save_frame(original_frame, f'original_{timestamp}.jpg', output_dir)
            self.display_manager.save_frame(panorama, f'panorama_{timestamp}.jpg', output_dir)
            self.display_manager.save_frame(annotated_frame, f'detection_{timestamp}.jpg', output_dir)
            print(f"已保存3张图片到 {output_dir}/")
        elif key == ord('i'):
            # 切换置信度阈值
            new_conf = 0.3 if self.args.conf_threshold >= 0.5 else 0.5
            self.args.conf_threshold = new_conf
            self.yolo_detector.update_thresholds(conf_threshold=new_conf)
            print(f"置信度阈值切换为: {self.args.conf_threshold}")
        elif key == ord('o'):
            # 切换IOU阈值
            new_iou = 0.3 if self.args.iou_threshold >= 0.45 else 0.45
            self.args.iou_threshold = new_iou
            self.yolo_detector.update_thresholds(iou_threshold=new_iou)
            print(f"IOU阈值切换为: {self.args.iou_threshold}")
        elif key == ord('p'):
            # 显示性能统计
            stats = self.yolo_detector.get_performance_stats()
            print(f"\n性能统计:")
            print(f"  总帧数: {stats['total_frames']}")
            print(f"  平均推理时间: {stats['avg_inference_time_ms']:.1f}ms")
            print(f"  最近FPS: {stats['recent_fps']:.1f}")
        elif key == ord('a'):
            # 切换角度显示模式
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

        # 保存日志
        if self.logger:
            self.logger.save()

        if self.camera:
            self.camera.release()
        if self.display_manager:
            self.display_manager.destroy_windows()

        # 打印最终统计
        if self.yolo_detector and self.yolo_detector.frame_count > 0:
            stats = self.yolo_detector.get_performance_stats()
            print(f"\n处理完成！")
            print(f"  总帧数: {stats['total_frames']}")
            print(f"  平均推理时间: {stats['avg_inference_time_ms']:.1f}ms")

        # 打印线性分配算法统计
        if self.use_deep_sort:
            print_assignment_stats()


    def _filter_cross_boundary_detections(self, detections: List[Dict], image_shape: Tuple[int, int]) -> List[Dict]:
        """
        过滤跨边界的检测框，处理坐标转换异常

        Args:
            detections: 检测结果列表
            image_shape: 图像形状

        Returns:
            过滤后的检测结果
        """
        if not detections:
            return []

        filtered = []
        height, width = image_shape[:2]

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox

            # 检查坐标是否有效
            if x1 >= width or x2 <= 0 or y1 >= height or y2 <= 0:
                # 完全在图像外的框
                continue

            if x2 < x1:
                # 坐标异常的框，跳过
                continue

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            # 确保宽度和高度为正
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            # 更新坐标
            det['bbox'] = [x1, y1, x2, y2]
            filtered.append(det)

        return filtered

    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制检测结果（包括关键点）
        """

        # === 绘制边界区域（调试用）===
        if self.use_deep_sort and self.deep_sort_tracker.enable_boundary_matching:
            annotated = self.deep_sort_tracker.draw_boundary_regions(image)
        else:
            annotated = image.copy()

        # 先绘制边界框
        annotated = self._draw_bounding_boxes(annotated, detections)

        # 再绘制关键点和骨架
        annotated = self._draw_keypoints(annotated, detections)

        return annotated

    def _draw_yolo_only(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制纯YOLO检测结果（不含跟踪ID、ReID等额外信息）
        只显示边界框、置信度和类别
        """
        annotated = image.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det['confidence']
            class_name = det['class_name']

            # 根据置信度设置颜色
            if confidence > 0.8:
                color = (0, 255, 0)  # 高置信度：绿色
            elif confidence > 0.6:
                color = (0, 200, 255)  # 中置信度：黄色
            else:
                color = (0, 165, 255)  # 低置信度：橙色

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 构建标签：只显示类别和置信度
            label = f"{class_name}: {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # 绘制标签背景
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # 绘制标签文字
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # 字体大小
                (0, 0, 0),  # 黑色文字
                2
            )

        return annotated

    def _draw_bounding_boxes(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制边界框（包含ReID ID）
        """
        annotated = image.copy()

        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det['confidence']
            class_name = det['class_name']

            # 获取跟踪ID
            track_id = det.get('track_id', -1)

            # 根据置信度设置颜色
            if confidence > 0.8:
                color = (0, 255, 0)  # 高置信度：绿色
            elif confidence > 0.6:
                color = (0, 200, 255)  # 中置信度：黄色
            else:
                color = (0, 165, 255)  # 低置信度：橙色

            # 绘制边界框
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 构建标签：显示Track ID
            label_parts = []
            if track_id != -1:
                label_parts.append(f"ID:{track_id}")
            label_parts.append(f"{class_name}: {confidence:.2f}")
            label = " ".join(label_parts)

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            # 绘制标签背景
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # 绘制标签文字
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,  # 字体大小
                (0, 0, 0),  # 黑色文字
                2
            )

        return annotated

    # 绘制关键点和骨架
    def _draw_keypoints(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制关键点和骨架
        """
        from config import KEYPOINT_COLORS, SKELETON_CONNECTIONS

        annotated = image.copy()

        for det in detections:
            # 检查是否有关键点信息
            if 'keypoints' not in det:
                continue

            keypoints = det['keypoints']

            # 绘制关键点
            for i, kp in enumerate(keypoints):
                if len(kp) >= 3:  # 确保有关键点坐标和置信度
                    x, y, conf = kp
                    if conf > 0.3:  # 关键点置信度阈值
                        # 绘制关键点
                        cv2.circle(annotated, (int(x), int(y)), 4, KEYPOINT_COLORS[i], -1)

            # 绘制骨架连接
            for connection in SKELETON_CONNECTIONS:
                start_idx, end_idx = connection
                if (len(keypoints) > max(start_idx, end_idx) and
                    len(keypoints[start_idx]) >= 3 and
                    len(keypoints[end_idx]) >= 3):

                    x1, y1, conf1 = keypoints[start_idx]
                    x2, y2, conf2 = keypoints[end_idx]

                    if conf1 > 0.3 and conf2 > 0.3:  # 只绘制置信度高的连接
                        # 绘制线条
                        cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                                (0, 255, 0), 2)  # 绿色骨架

        return annotated


def main():
    """主函数"""
    # 解析参数
    args = config.parse_args()

    # 创建主处理器
    processor = FisheyePanoramaYOLOPose(args)

    # 初始化
    if not processor.initialize():
        print("初始化失败，程序退出")
        return

    # 运行
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
