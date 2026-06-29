"""
全景展开模块 - 负责鱼眼图像的展开
整合了 fisheye_panorama.py 和 panorama_processor.py
"""
import os
import cv2
import numpy as np
import time
from typing import Tuple, Optional

# 尝试导入 torch（用于 GPU 版本）
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 未安装，GPU 版本将不可用")


class FisheyePanorama:
    """鱼眼全景展开类（CPU 版本）"""
    
    def __init__(self, cam_width: int, cam_height: int,
                 output_width: int = 3840, output_height: int = 1080,
                 vertical_fov: float = 100.0, map_file: str = None,
                 cam_index: int = 1):
        """
        初始化全景展开器
        """
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.output_width = output_width
        self.output_height = output_height
        self.vertical_fov = vertical_fov
        self.map_file = map_file
        self.cam_index = cam_index
        
        # 映射相关属性
        self.map_x = None
        self.map_y = None
        self.map1 = None
        self.map2 = None
        self.center = None
        self.radius = None
        self.img_width = cam_width
        self.img_height = cam_height
        
        # 鼠标交互相关（仅独立运行时使用）
        self.window_name = None
        self.last_click_info = None
        self.click_history = []
    
    def init_from_frame(self, frame: np.ndarray) -> bool:
        """从图像帧初始化映射"""
        self.img_height, self.img_width = frame.shape[:2]
        
        # 尝试加载或计算映射矩阵
        if self.map_file and os.path.exists(self.map_file):
            print(f"尝试从文件加载映射矩阵: {self.map_file}")
            if self.load_maps(self.map_file):
                print("映射矩阵加载成功")
                return True
        
        print("重新计算映射矩阵...")
        return self.compute_and_save_maps(frame)
    
    def compute_and_save_maps(self, frame: np.ndarray) -> bool:
        """计算并保存映射矩阵"""
        # 检测鱼眼有效区域
        self.center, self.radius = self.detect_fisheye_region_improved(frame)
        print(f"检测到圆心: {self.center}, 半径: {self.radius}")
        
        # 计算映射矩阵
        self.map_x, self.map_y = self.create_panorama_map()
        self._prepare_cpu_maps()
        
        # 如果指定了映射文件，保存映射矩阵
        if self.map_file:
            self.save_maps(self.map_file)
        
        return True

    def _prepare_cpu_maps(self) -> None:
        """Prepare fixed-point OpenCV remap maps for faster ARM CPU remap."""
        if self.map_x is None or self.map_y is None:
            self.map1 = None
            self.map2 = None
            return
        try:
            self.map1, self.map2 = cv2.convertMaps(
                self.map_x,
                self.map_y,
                cv2.CV_16SC2,
            )
        except Exception as e:
            print(f"OpenCV 定点映射转换失败，回退 float maps: {e}")
            self.map1 = None
            self.map2 = None
    
    def create_panorama_map(self) -> Tuple[np.ndarray, np.ndarray]:
        """创建横向展开的映射矩阵（CPU 版本）"""
        h, w = self.img_height, self.img_width
        center_x, center_y = self.center
        
        # 初始化映射矩阵
        map_x = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        map_y = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        
        # 将圆形鱼眼图按经纬度展开为矩形全景图
        for y_out in range(self.output_height):
            # 从 0 开始到 1，覆盖从圆心到边缘的完整范围
            r_ratio = y_out / (self.output_height - 1)
            
            for x_out in range(self.output_width):
                angle = 2 * np.pi * x_out / self.output_width
                
                x_in = center_x + r_ratio * self.radius * np.cos(angle)
                y_in = center_y + r_ratio * self.radius * np.sin(angle)
                
                x_in = np.clip(x_in, 0, w - 1)
                y_in = np.clip(y_in, 0, h - 1)
                
                map_x[y_out, x_out] = x_in
                map_y[y_out, x_out] = y_in
        
        print(f"映射矩阵计算完成，展开尺寸: {self.output_width}x{self.output_height}")
        return map_x, map_y
    
    def detect_fisheye_region_improved(self, img: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """改进的鱼眼有效区域检测方法"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        h, w = gray.shape[:2]
        
        # 默认策略：图像中心
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 2
        
        # 尝试使用霍夫圆变换
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            circles = cv2.HoughCircles(
                edges,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=min(w, h) // 4,
                param1=50,
                param2=30,
                minRadius=min(w, h) // 4,
                maxRadius=min(w, h) // 2
            )
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                detected_center = (circles[0][0], circles[0][1])
                detected_radius = circles[0][2]
                print(f"霍夫圆变换检测到: 圆心{detected_center}, 半径{detected_radius}")
                
                if self._is_valid_circle(detected_center, detected_radius, w, h):
                    center_x, center_y = detected_center
                    radius = detected_radius
                else:
                    print("霍夫圆变换结果不合理，使用默认值")
            else:
                print("霍夫圆变换未检测到圆，使用阈值化方法")
                # 尝试阈值化方法
                best_radius = 0
                for thresh_val in [20, 30, 40, 50]:
                    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        (x, y), r = cv2.minEnclosingCircle(largest_contour)
                        if r > best_radius and self._is_valid_circle((int(x), int(y)), int(r), w, h):
                            center_x, center_y = int(x), int(y)
                            radius = int(r)
                            best_radius = int(r)
                
                if best_radius > 0:
                    print(f"阈值化方法检测到: 圆心({center_x}, {center_y}), 半径{radius}")
        
        except Exception as e:
            print(f"高级检测方法失败: {e}, 使用默认值")
        
        # 确保圆心和半径合理
        max_center_offset = min(w, h) // 4
        center_x = np.clip(center_x, w // 2 - max_center_offset, w // 2 + max_center_offset)
        center_y = np.clip(center_y, h // 2 - max_center_offset, h // 2 + max_center_offset)
        
        min_radius = min(w, h) // 3
        max_radius = min(w, h) // 2 + 50
        radius = np.clip(radius, min_radius, max_radius)
        
        final_center = (int(center_x), int(center_y))
        final_radius = int(radius)
        
        print(f"最终鱼眼区域: 圆心{final_center}, 半径{final_radius}")
        return final_center, final_radius
    
    def _is_valid_circle(self, center: Tuple[int, int], radius: int,
                       img_w: int, img_h: int) -> bool:
        """验证检测到的圆是否合理"""
        max_offset = min(img_w, img_h) // 3
        center_x, center_y = center
        if abs(center_x - img_w // 2) > max_offset or abs(center_y - img_h // 2) > max_offset:
            return False
        
        min_radius = min(img_w, img_h) // 4
        max_radius = min(img_w, img_h) // 2 + 50
        if not (min_radius <= radius <= max_radius):
            return False
        
        return True
    
    def save_maps(self, filepath: str) -> bool:
        """保存映射矩阵到文件"""
        try:
            np.savez_compressed(
                filepath,
                map_x=self.map_x,
                map_y=self.map_y,
                center=self.center,
                radius=self.radius,
                img_width=self.img_width,
                img_height=self.img_height,
                output_width=self.output_width,
                output_height=self.output_height,
                camera_index=self.cam_index,
                vertical_fov_deg=self.vertical_fov
            )
            
            print(f"映射矩阵已保存到: {filepath}")
            return True
        except Exception as e:
            print(f"保存映射矩阵失败: {e}")
            return False
    
    def load_maps(self, filepath: str) -> bool:
        """从文件加载映射矩阵"""
        try:
            data = np.load(filepath, allow_pickle=True)
            self.map_x = data['map_x']
            self.map_y = data['map_y']
            self.center = tuple(data['center'])
            self.radius = int(data['radius'])
            
            if 'output_width' in data and 'output_height' in data:
                self.output_width = int(data['output_width'])
                self.output_height = int(data['output_height'])
            
            if 'vertical_fov_deg' in data:
                self.vertical_fov = float(data['vertical_fov_deg'])
            
            if 'img_width' in data and 'img_height' in data:
                self.img_width = int(data['img_width'])
                self.img_height = int(data['img_height'])
            self._prepare_cpu_maps()
            
            print(f"映射矩阵加载成功")
            print(f"  圆心: {self.center}, 半径: {self.radius}")
            print(f"  输出尺寸: {self.output_width}x{self.output_height}")
            return True
        except Exception as e:
            print(f"加载映射矩阵失败: {e}")
            return False
    
    def apply_panorama(
        self,
        frame: np.ndarray,
        y_start: int = 0,
        y_end: Optional[int] = None,
    ) -> np.ndarray:
        """应用全景展开（CPU 版本）"""
        if self.map_x is None or self.map_y is None:
            if not self.init_from_frame(frame):
                print("警告: 映射矩阵初始化失败，返回原图")
                return frame

        h = self.map_x.shape[0]
        y0 = max(0, min(int(y_start), h))
        y1 = h if y_end is None else max(y0, min(int(y_end), h))
        if self.map1 is not None and self.map2 is not None:
            map1 = self.map1[y0:y1]
            map2 = self.map2[y0:y1]
        else:
            map1 = self.map_x[y0:y1]
            map2 = self.map_y[y0:y1]

        panorama = cv2.remap(
            frame,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return panorama
    
    def get_useful_area(self, img: np.ndarray) -> np.ndarray:
        """提取圆形有效区域"""
        if self.center is None or self.radius is None:
            self.center, self.radius = self.detect_fisheye_region_improved(img)
        
        mask = np.zeros_like(img)
        cv2.circle(mask, self.center, self.radius, (255, 255, 255), -1)
        return cv2.bitwise_and(img, mask)
    
    def get_size(self) -> Tuple[int, int]:
        """获取全景图尺寸"""
        return self.output_width, self.output_height


# ==================== GPU 版本 ====================
class FisheyePanoramaGPU:
    """鱼眼全景展开类（GPU 版本）"""
    
    def __init__(self, cam_width: int, cam_height: int,
                 output_width: int = 3840, output_height: int = 1080,
                 vertical_fov: float = 100.0, map_file: str = None,
                 cam_index: int = 1):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch 未安装，无法使用 GPU 版本")
        
        self.cam_width = cam_width
        self.cam_height = cam_height
        self.output_width = output_width
        self.output_height = output_height
        self.vertical_fov = vertical_fov
        self.map_file = map_file
        self.cam_index = cam_index
        
        # GPU 映射相关属性
        self.map_x = None
        self.map_y = None
        self.map_x_gpu = None
        self.map_y_gpu = None
        self.grid_gpu = None
        self.center = None
        self.radius = None
        self.img_width = cam_width
        self.img_height = cam_height
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 GPU 鱼眼展开器初始化，使用设备: {self.device}")
    
    def init_from_frame(self, frame: np.ndarray) -> bool:
        """从图像帧初始化映射（GPU 版本）"""
        self.img_height, self.img_width = frame.shape[:2]
        
        # 尝试加载或计算映射矩阵
        if self.map_file and os.path.exists(self.map_file):
            print(f"尝试从文件加载映射矩阵: {self.map_file}")
            if self.load_maps(self.map_file):
                print("✅ 映射矩阵加载成功")
                return True
        
        print("重新计算映射矩阵...")
        return self.compute_and_save_maps(frame)
    
    def compute_and_save_maps(self, frame: np.ndarray) -> bool:
        """计算并保存映射矩阵（GPU 版本）"""
        # 检测鱼眼有效区域（使用 CPU 版本的方法）
        cpu_version = FisheyePanorama(
            self.cam_width, self.cam_height,
            self.output_width, self.output_height,
            self.vertical_fov, self.map_file, self.cam_index
        )
        self.center, self.radius = cpu_version.detect_fisheye_region_improved(frame)
        print(f"检测到圆心: {self.center}, 半径: {self.radius}")
        
        # 计算映射矩阵（使用 CPU 版本的方法）
        self.map_x, self.map_y = cpu_version.create_panorama_map()
        
        # ✅ 关键：将映射矩阵上传到 GPU
        self._upload_maps_to_gpu()
        
        # 如果指定了映射文件，保存映射矩阵
        if self.map_file:
            cpu_version.map_x = self.map_x
            cpu_version.map_y = self.map_y
            cpu_version.center = self.center
            cpu_version.radius = self.radius
            cpu_version.save_maps(self.map_file)
        
        return True
    
    def _upload_maps_to_gpu(self):
        """将映射矩阵上传到 GPU 并创建归一化网格"""
        if self.map_x is None or self.map_y is None:
            raise ValueError("映射矩阵未计算")
        
        # 转换为 PyTorch 张量并上传到 GPU
        self.map_x_gpu = torch.from_numpy(self.map_x).to(self.device)
        self.map_y_gpu = torch.from_numpy(self.map_y).to(self.device)
        
        # 创建归一化网格（用于 grid_sample）
        # map_x/map_y 存储的是源帧（摄像头）的像素坐标，范围 0~img_width/height-1
        # grid_sample 要求 [-1,1]，分母必须用 INPUT 图像尺寸，而非输出地图尺寸
        out_H, out_W = self.map_x_gpu.shape          # 输出全景尺寸
        grid_x = self.map_x_gpu / (self.img_width  - 1) * 2 - 1   # 按输入宽度归一化
        grid_y = self.map_y_gpu / (self.img_height - 1) * 2 - 1   # 按输入高度归一化
        self.grid_gpu = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        print(f"✅ GPU 映射矩阵上传成功: 输出={out_W}x{out_H}, 输入={self.img_width}x{self.img_height}")
        print(f"   grid_x ∈ [{grid_x.min():.3f}, {grid_x.max():.3f}]  "
              f"grid_y ∈ [{grid_y.min():.3f}, {grid_y.max():.3f}]")
    
    def load_maps(self, filepath: str) -> bool:
        """从文件加载映射矩阵（GPU 版本）"""
        try:
            data = np.load(filepath, allow_pickle=True)
            self.map_x = data['map_x']
            self.map_y = data['map_y']
            self.center = tuple(data['center'])
            self.radius = int(data['radius'])
            
            if 'output_width' in data and 'output_height' in data:
                self.output_width = int(data['output_width'])
                self.output_height = int(data['output_height'])
            
            if 'vertical_fov_deg' in data:
                self.vertical_fov = float(data['vertical_fov_deg'])
            
            if 'img_width' in data and 'img_height' in data:
                self.img_width = int(data['img_width'])
                self.img_height = int(data['img_height'])
            
            # ✅ 关键：上传到 GPU
            self._upload_maps_to_gpu()
            
            print(f"✅ GPU 映射矩阵加载成功")
            print(f"  圆心: {self.center}, 半径: {self.radius}")
            print(f"  输出尺寸: {self.output_width}x{self.output_height}")
            return True
        except Exception as e:
            print(f"❌ 加载映射矩阵失败: {e}")
            return False
    
    def apply_panorama_gpu(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        应用全景展开（GPU 版本）
        
        参数:
            frame_tensor: GPU Tensor [C, H, W] (BGR 格式)
        
        返回:
            GPU Tensor [C, output_H, output_W]
        """
        if self.grid_gpu is None:
            raise ValueError("GPU 映射矩阵未初始化，请先调用 init_from_frame()")
        
        # 确保输入在正确的设备上
        if not frame_tensor.is_cuda:
            frame_tensor = frame_tensor.to(self.device)
        
        # 添加批次维度 [1, C, H, W]
        frame_batch = frame_tensor.unsqueeze(0)
        
        # 使用 grid_sample 进行 GPU 鱼眼展开
        panorama = torch.nn.functional.grid_sample(
            frame_batch, 
            self.grid_gpu, 
            mode='bilinear', 
            align_corners=True
        )
        
        # 移除批次维度 [C, H_out, W_out]
        return panorama.squeeze(0)

    def get_useful_area(self, img: np.ndarray) -> np.ndarray:
        """提取圆形有效区域（与 CPU 版本接口一致，供显示布局使用）"""
        if self.center is None or self.radius is None:
            return img
        mask = np.zeros_like(img)
        cv2.circle(mask, self.center, self.radius, (255, 255, 255), -1)
        return cv2.bitwise_and(img, mask)
