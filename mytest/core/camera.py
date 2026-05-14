"""
相机处理器，负责相机初始化和帧捕获
"""
import cv2
import time
import numpy as np
from typing import Tuple, Optional

class CameraProcessor:
    """相机处理器类，支持摄像头和视频文件"""
    
    def __init__(self, cam_index: int = 0, video_path: str = None, 
                 width: int = 1920, height: int = 1080):
        """
        初始化相机或视频
        参数:
            cam_index: 摄像头索引
            video_path: 视频文件路径，如果提供则使用视频文件
            width: 期望的宽度（摄像头模式下）
            height: 期望的高度（摄像头模式下）
        """
        self.cam_index = cam_index
        self.video_path = video_path
        self.cam_width = width
        self.cam_height = height
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.is_video = video_path is not None
        
    def initialize(self) -> bool:
        """
        初始化相机或视频
        返回: 是否初始化成功
        """
        if self.is_video:
            # 从视频文件初始化
            print(f"初始化视频文件: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                print(f"错误: 无法打开视频文件 {self.video_path}")
                return False
                
            # 获取视频的实际分辨率
            self.cam_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.cam_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"视频信息: 分辨率 {self.cam_width}x{self.cam_height}, "
                  f"FPS: {video_fps:.2f}, 总帧数: {total_frames}")
        else:
            # 从摄像头初始化（原有逻辑）
            print(f"初始化相机 (索引: {self.cam_index}, 分辨率: {self.cam_width}x{self.cam_height})")
            self.cap = cv2.VideoCapture(self.cam_index)
            
            if not self.cap.isOpened():
                print(f"错误: 无法打开摄像头 {self.cam_index}")
                return False
            
            # 设置相机参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 获取实际分辨率
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"实际分辨率: {actual_width}x{actual_height}")
            
            self.cam_width = actual_width
            self.cam_height = actual_height
        
        return True
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        获取一帧图像
        返回: (是否成功, 图像)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret and not self.is_video:
            # 仅对摄像头计算FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_time = current_time
        elif not ret and self.is_video:
            # 视频播放完毕
            print("视频播放完毕")
        
        return ret, frame
    
    def get_camera_info(self) -> dict:
        """
        获取相机/视频信息
        """
        if self.cap is None:
            return {}
        
        if self.is_video:
            info = {
                'width': self.cam_width,
                'height': self.cam_height,
                'fps': self.cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'current_frame': int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)),
                'source_type': 'video_file',
                'video_path': self.video_path
            }
        else:
            info = {
                'width': self.cam_width,
                'height': self.cam_height,
                'fps': self.fps,  # 使用计算出的实时FPS
                'format': self.cap.get(cv2.CAP_PROP_FORMAT),
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'source_type': 'camera',
                'cam_index': self.cam_index
            }
        return info
    
    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            source_type = "视频" if self.is_video else "相机"
            print(f"{source_type}资源已释放")
    
    def __del__(self):
        """析构函数"""
        self.release()