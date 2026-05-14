"""
全景展开处理器，负责鱼眼图像的展开
"""
import os
import cv2
import numpy as np
from typing import Tuple, Optional
from fisheye_panorama import FisheyePanoramaRealtimeWithClick


class PanoramaProcessor:
    """全景展开处理器"""

    def __init__(self, cam_width: int, cam_height: int,
                 output_width: int = 1920, output_height: int = 540,
                 vertical_fov: float = 150.0, map_file: str = None,
                 cam_index: int = 1):
        """
        初始化全景展开处理器
        """
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.output_width = output_width
        self.output_height = output_height
        self.vertical_fov = vertical_fov

        # 计算合适的显示尺寸
        self.display_scale = 0.5
        self.display_width = int(cam_width * self.display_scale)
        self.display_height = int(cam_height * self.display_scale)

        # 初始化鱼眼展开处理器 - 不初始化摄像头
        self.fisheye_processor = FisheyePanoramaRealtimeWithClick(
            cam_index=cam_index,
            output_width=output_width,
            output_height=output_height,
            desired_width=cam_width,
            desired_height=cam_height,
            map_file=map_file,
            force_recompute=False,
            vertical_fov_deg=vertical_fov,
            init_camera=False  # 关键：不初始化摄像头
        )

        # 注意：我们会在第一次处理帧时初始化映射

    def apply_panorama(self, frame: np.ndarray) -> np.ndarray:
        """
        应用全景展开
        参数:
            frame: 原始鱼眼图像
        返回: 展开后的全景图像
        """
        # 如果映射矩阵未初始化，先用这一帧初始化
        if (self.fisheye_processor.map_x is None or
            self.fisheye_processor.map_y is None):

            # 设置图像尺寸
            self.fisheye_processor.img_height, self.fisheye_processor.img_width = frame.shape[:2]
            self.fisheye_processor.frame = frame

            # 尝试加载或计算映射矩阵
            map_file = self.fisheye_processor.map_file
            if map_file and os.path.exists(map_file):
                print(f"尝试从文件加载映射矩阵: {map_file}")
                if not self.fisheye_processor.load_maps(map_file):
                    print("映射矩阵加载失败，将重新计算")
                    self.fisheye_processor.compute_and_save_maps()
            else:
                # 重新计算映射矩阵
                self.fisheye_processor.compute_and_save_maps()

        # 直接使用原类的 apply_panorama 方法
        panorama = self.fisheye_processor.apply_panorama(frame)
        return panorama

    def get_useful_area(self, frame: np.ndarray) -> np.ndarray:
        """
        获取有效区域
        参数:
            frame: 原始鱼眼图像
        返回: 有效区域图像
        """
        return self.fisheye_processor.get_useful_area(frame)

    def resize_for_display(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像大小用于显示
        参数:
            image: 输入图像
        返回: 调整大小后的图像
        """
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        # 如果图像大小与显示大小不匹配，则调整
        if w != self.display_width or h != self.display_height:
            return cv2.resize(image, (self.display_width, self.display_height))
        return image

    def get_panorama_size(self) -> Tuple[int, int]:
        """
        获取全景图尺寸
        返回: (宽度, 高度)
        """
        return self.output_width, self.output_height

    def get_processor_info(self) -> dict:
        """
        获取处理器信息
        返回: 信息字典
        """
        return {
            'img_width': getattr(self.fisheye_processor, 'img_width', 0),
            'img_height': getattr(self.fisheye_processor, 'img_height', 0),
            'output_width': self.fisheye_processor.output_width,
            'output_height': self.fisheye_processor.output_height,
            'vertical_fov_deg': getattr(self.fisheye_processor, 'vertical_fov_deg', 150.0)
        }
