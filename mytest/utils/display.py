"""
显示管理器，负责图像显示和UI交互
"""
import cv2
import numpy as np
import time
from typing import Optional, Tuple, Callable

class DisplayManager:
    """显示管理器"""

    def __init__(self, window_name: str = "Fisheye Panorama with YOLO Pose Detection", use_dual_windows: bool = False, no_display: bool = False):
        """
        初始化显示管理器
        """
        self.window_name = window_name
        self.use_dual_windows = use_dual_windows
        self.no_display = no_display
        self.fps_counter = 0
        self.fps_time = time.time()
        self.frame_count = 0

        # 只有在非无头模式下才创建显示窗口
        if not self.no_display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # 如果使用双窗口模式，创建第二个窗口
            if self.use_dual_windows:
                self.yolo_window_name = "YOLO Detection Only"
                cv2.namedWindow(self.yolo_window_name, cv2.WINDOW_NORMAL)
        
    def create_layout(self, original_frame: np.ndarray, useful_area: np.ndarray, 
                     panorama: np.ndarray, display_scale: float = 0.5) -> np.ndarray:
        """
        创建显示布局
        参数:
            original_frame: 原始图像
            useful_area: 有效区域
            panorama: 全景图像
            display_scale: 显示缩放比例
        返回: 组合后的显示图像
        """
        # 调整大小
        display_original = cv2.resize(original_frame, 
                                     (int(original_frame.shape[1] * display_scale), 
                                      int(original_frame.shape[0] * display_scale)))
        
        display_useful = cv2.resize(useful_area, 
                                   (int(useful_area.shape[1] * display_scale), 
                                    int(useful_area.shape[0] * display_scale)))
        
        # 调整全景图大小
        top_row = np.hstack((display_original, display_useful))
        panorama_resized = cv2.resize(panorama, (top_row.shape[1], panorama.shape[0]))
        
        # 组合显示
        combined_display = np.vstack((top_row, panorama_resized))
        
        return combined_display
    
    def add_info_overlay(self, image: np.ndarray, info_text: str, 
                        perf_text: str, count_text: str = "") -> np.ndarray:
        """
        添加信息覆盖层
        参数:
            image: 输入图像
            info_text: 信息文本
            perf_text: 性能文本
            count_text: 计数文本
        返回: 添加了信息的图像
        """
        # 添加信息标签
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(image, perf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if count_text:
            cv2.putText(image, count_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return image
    
    def update_fps(self) -> int:
        """
        更新FPS计数
        返回: 当前FPS
        """
        self.fps_counter += 1
        self.frame_count += 1
        current_time = time.time()
        
        fps = 0
        if current_time - self.fps_time >= 1.0:
            fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
        
        return fps
    
    def set_mouse_callback(self, callback_func: Callable):
        """
        设置鼠标回调函数
        参数:
            callback_func: 回调函数
        """
        if not self.no_display:
            cv2.setMouseCallback(self.window_name, callback_func)

    def show(self, image: np.ndarray):
        """显示图像"""
        if not self.no_display:
            cv2.imshow(self.window_name, image)

    def show_dual(self, yolo_image: np.ndarray, final_image: np.ndarray):
        """
        双窗口显示模式：一个窗口显示YOLO检测结果，另一个显示最终结果
        参数:
            yolo_image: 只包含YOLO检测结果的图像
            final_image: 包含所有处理的最终结果图像
        """
        if not self.no_display:
            if self.use_dual_windows:
                cv2.imshow(self.yolo_window_name, yolo_image)
                cv2.imshow(self.window_name, final_image)
            else:
                # 如果不是双窗口模式，只显示最终结果
                cv2.imshow(self.window_name, final_image)

    def destroy_windows(self):
        """销毁所有窗口"""
        if not self.no_display:
            cv2.destroyAllWindows()
    
    def save_frame(self, image: np.ndarray, filename: str, output_dir: str = "output"):
        """
        保存帧图像
        参数:
            image: 要保存的图像
            filename: 文件名
            output_dir: 输出目录
        """
        import os
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        cv2.imwrite(f'{output_dir}/{filename}', image)
        print(f"已保存图片: {output_dir}/{filename}")