"""
YOLO姿态检测器
"""
import cv2
import numpy as np
import time
import torch
from typing import List, Dict, Optional
import config

_USE_HALF = torch.cuda.is_available()  # FP16 only on GPU; set once at import time


class YOLOPoseDetector:
    """YOLO姿态检测器"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        初始化YOLO姿态检测器
        """
        print(f"加载YOLO姿态估计模型: {model_path}")
        from ultralytics import YOLO
        # .engine 不带 task 元数据时 ultralytics 默认按 detect 解析：pose/face 模型的
        # 关键点通道会被误当作类别，argmax 落到无效类别 → KeyError。按文件名显式指定 task：
        #   名含 pose/face → 'pose'（17 点 / 人脸 5 点）；其余（如 yolo26n.engine）→ 'detect'
        _p = str(model_path).lower()
        if _p.endswith('.engine'):
            _task = 'pose' if ('pose' in _p or 'face' in _p) else 'detect'
            self.model = YOLO(model_path, task=_task)
            print(f"  引擎任务类型推断为: {_task}")
        else:
            self.model = YOLO(model_path)  # .pt/.onnx 自带 task 元数据
        # self.model = self.model.half()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 性能统计
        self.inference_times = []
        self.frame_count = 0
        self.total_inference_time = 0

    def detect_batch(self, images: List[np.ndarray]) -> List[List]:
        """
        批量推理多张图像（一次 GPU forward pass 处理所有切片）。
        返回格式与多次调用 detect() 一致：List[List[Result]]
        """
        start_time = time.time()
        results = self.model.predict(
            images,
            imgsz=864,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
            half=_USE_HALF,
            device=0 if _USE_HALF else 'cpu',
        )
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        self.total_inference_time += inference_time
        # 每个元素包成单元素列表，与 merge_detections 期望的接口一致
        return [[r] for r in results]

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
                imgsz=864,
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
