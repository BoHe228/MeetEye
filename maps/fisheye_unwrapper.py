"""
鱼眼相机实时展开模块 - 工程化版本

功能特性:
- 支持摄像头和视频文件输入，一键切换
- 支持静态图片计算映射矩阵
- 支持保存/加载映射矩阵
- 支持多种摄像头后端 (DirectShow/Media Foundation)
- 支持切开点位置调整 (左/右)
- 支持单帧/多帧平均计算映射矩阵
"""

import cv2
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
from enum import Enum


class CutPosition(Enum):
    """展开切开点位置"""
    RIGHT = "right"  # 正右方
    LEFT = "left"    # 正左方


class InputSourceType(Enum):
    """输入源类型"""
    CAMERA = "camera"  # 摄像头
    VIDEO = "video"    # 视频文件


@dataclass
class InputSource:
    """输入源配置"""
    source_type: InputSourceType
    source: Union[int, str]  # 摄像头索引或视频路径
    width: int = 1920
    height: int = 1080
    fps: int = 30

    @classmethod
    def from_camera(cls, index: int = 0, width: int = 1920, height: int = 1080) -> 'InputSource':
        """从摄像头创建输入源"""
        return cls(InputSourceType.CAMERA, index, width, height)

    @classmethod
    def from_video(cls, path: str) -> 'InputSource':
        """从视频文件创建输入源"""
        return cls(InputSourceType.VIDEO, path)


@dataclass
class OutputConfig:
    """输出配置"""
    width: int = 3840
    height: int = 1080


@dataclass
class MappingData:
    """映射矩阵数据"""
    map_x: np.ndarray
    map_y: np.ndarray
    center: Tuple[int, int]
    radius: int
    img_width: int
    img_height: int
    output_width: int
    output_height: int
    cut_position: str


class CameraBackend(Enum):
    """摄像头后端类型"""
    DIRECTSHOW = cv2.CAP_DSHOW
    MEDIA_FOUNDATION = cv2.CAP_MSMF
    AUTO = cv2.CAP_ANY


class FisheyeUnwrapper:
    """鱼眼图像展开器 - 核心类"""

    def __init__(
        self,
        input_source: Optional[InputSource] = None,
        output_config: Optional[OutputConfig] = None,
        map_file: Optional[str] = None,
        cut_position: CutPosition = CutPosition.RIGHT,
        display_scale: float = 0.3,
    ):
        """
        初始化鱼眼展开器

        Args:
            input_source: 输入源配置 (摄像头或视频文件)
            output_config: 输出配置
            map_file: 映射矩阵文件路径
            cut_position: 切开点位置
            display_scale: 显示缩放比例 (默认: 0.3)
        """
        self.input_source = input_source or InputSource.from_camera(0)
        self.output_config = output_config or OutputConfig()
        self.map_file = map_file
        self.cut_position = cut_position
        self.display_scale = display_scale

        self._cap: Optional[cv2.VideoCapture] = None
        self._mapping: Optional[MappingData] = None
        self._frame: Optional[np.ndarray] = None

    # =========================================================================
    # 便捷创建方法
    # =========================================================================

    @classmethod
    def with_camera(
        cls,
        camera_index: int = 0,
        map_file: Optional[str] = None,
        output_width: int = 3840,
        output_height: int = 1080,
        cut_position: str = "right",
        display_scale: float = 0.3,
    ) -> 'FisheyeUnwrapper':
        """
        便捷创建：使用摄像头

        Args:
            camera_index: 摄像头索引
            map_file: 映射矩阵文件
            output_width: 输出宽度
            output_height: 输出高度
            cut_position: 切开点位置
            display_scale: 显示缩放比例 (默认: 0.3)

        Returns:
            FisheyeUnwrapper 实例
        """
        source = InputSource.from_camera(camera_index)
        output = OutputConfig(width=output_width, height=output_height)
        cut = CutPosition.LEFT if cut_position.lower() == "left" else CutPosition.RIGHT
        return cls(input_source=source, output_config=output, map_file=map_file, cut_position=cut, display_scale=display_scale)

    @classmethod
    def with_video(
        cls,
        video_path: str,
        map_file: Optional[str] = None,
        output_width: int = 3840,
        output_height: int = 1080,
        cut_position: str = "right",
        display_scale: float = 0.3,
    ) -> 'FisheyeUnwrapper':
        """
        便捷创建：使用视频文件

        Args:
            video_path: 视频文件路径
            map_file: 映射矩阵文件
            output_width: 输出宽度
            output_height: 输出高度
            cut_position: 切开点位置
            display_scale: 显示缩放比例 (默认: 0.3)

        Returns:
            FisheyeUnwrapper 实例
        """
        source = InputSource.from_video(video_path)
        output = OutputConfig(width=output_width, height=output_height)
        cut = CutPosition.LEFT if cut_position.lower() == "left" else CutPosition.RIGHT
        return cls(input_source=source, output_config=output, map_file=map_file, cut_position=cut, display_scale=display_scale)

    # =========================================================================
    # 状态查询
    # =========================================================================

    @property
    def is_opened(self) -> bool:
        """检查输入源是否已打开"""
        return self._cap is not None and self._cap.isOpened()

    @property
    def is_ready(self) -> bool:
        """检查是否准备好展开图像"""
        return self._mapping is not None

    @property
    def source_type(self) -> InputSourceType:
        """获取当前输入源类型"""
        return self.input_source.source_type

    # =========================================================================
    # 输入源管理
    # =========================================================================

    def use_camera(self, index: int = 0, width: int = 1920, height: int = 1080):
        """
        切换到摄像头输入

        Args:
            index: 摄像头索引
            width: 期望宽度
            height: 期望高度
        """
        self.input_source = InputSource.from_camera(index, width, height)
        if self.is_opened:
            self._close_source()

    def use_video(self, path: str):
        """
        切换到视频文件输入

        Args:
            path: 视频文件路径
        """
        self.input_source = InputSource.from_video(path)
        if self.is_opened:
            self._close_source()

    def open(self, auto_load_mapping: bool = True, auto_compute: bool = True) -> bool:
        """
        打开输入源（自动识别是摄像头还是视频）

        Args:
            auto_load_mapping: 是否自动加载映射矩阵
            auto_compute: 如果加载失败是否自动计算

        Returns:
            是否成功打开
        """
        if self.input_source.source_type == InputSourceType.CAMERA:
            return self._open_camera(auto_load_mapping, auto_compute)
        else:
            return self._open_video(auto_load_mapping, auto_compute)

    def _open_camera(self, auto_load_mapping: bool = True, auto_compute: bool = True) -> bool:
        """打开摄像头"""
        self._close_source()

        backends = [
            CameraBackend.DIRECTSHOW,
            CameraBackend.MEDIA_FOUNDATION,
            CameraBackend.AUTO,
        ]

        for backend in backends:
            if self._try_open_camera(backend):
                if auto_load_mapping:
                    self._try_load_or_compute(auto_compute)
                return True

        print(f"❌ 无法打开摄像头 {self.input_source.source}")
        return False

    def _open_video(self, auto_load_mapping: bool = True, auto_compute: bool = True) -> bool:
        """打开视频文件"""
        path = self.input_source.source
        if not os.path.exists(path):
            print(f"❌ 视频文件不存在: {path}")
            return False

        self._close_source()

        print(f"打开视频文件: {path}")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频文件")
            return False

        ret, frame = cap.read()
        if not ret:
            cap.release()
            print(f"❌ 无法读取视频帧")
            return False

        self._cap = cap
        self._frame = frame
        self.input_source.width = frame.shape[1]
        self.input_source.height = frame.shape[0]
        print(f"✓ 视频打开成功: {frame.shape[1]}x{frame.shape[0]}")

        if auto_load_mapping:
            self._try_load_or_compute(auto_compute)

        return True

    def _try_open_camera(self, backend: CameraBackend) -> bool:
        """尝试使用指定后端打开摄像头"""
        idx = self.input_source.source
        print(f"尝试使用 {backend.name} 后端打开摄像头 {idx}...")

        cap = cv2.VideoCapture(idx, backend.value)
        if not cap.isOpened():
            cap.release()
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.input_source.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.input_source.height)
        cap.set(cv2.CAP_PROP_FPS, self.input_source.fps)
        cv2.waitKey(100)

        ret, frame = cap.read()
        if not ret:
            cap.release()
            print(f"   {backend.name} 能打开但无法读取帧")
            return False

        self._cap = cap
        self._frame = frame
        print(f"✓ 使用 {backend.name} 后端成功打开摄像头")
        print(f"  实际分辨率: {frame.shape[1]}x{frame.shape[0]}")
        return True

    def _close_source(self):
        """关闭输入源"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _try_load_or_compute(self, auto_compute: bool = True):
        """尝试加载或计算映射矩阵"""
        if self.map_file and self.load_mapping():
            return
        if auto_compute:
            self.compute_mapping(num_frames=30)

    # =========================================================================
    # 映射矩阵管理
    # =========================================================================

    def load_mapping(self, filepath: Optional[str] = None) -> bool:
        """加载映射矩阵"""
        filepath = filepath or self.map_file
        if not filepath or not os.path.exists(filepath):
            return False

        try:
            data = np.load(filepath, allow_pickle=True)
            self._mapping = MappingData(
                map_x=data['map_x'],
                map_y=data['map_y'],
                center=tuple(data['center']),
                radius=int(data['radius']),
                img_width=int(data['img_width']),
                img_height=int(data['img_height']),
                output_width=int(data['output_width']),
                output_height=int(data['output_height']),
                cut_position=str(data.get('cut_position', 'right')),
            )
            print(f"✓ 映射矩阵加载成功")
            return True
        except Exception as e:
            print(f"映射矩阵加载失败: {e}")
            return False

    def save_mapping(self, filepath: Optional[str] = None) -> bool:
        """保存映射矩阵"""
        if self._mapping is None:
            return False

        filepath = filepath or self.map_file
        if not filepath:
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            np.savez_compressed(
                filepath,
                map_x=self._mapping.map_x,
                map_y=self._mapping.map_y,
                center=self._mapping.center,
                radius=self._mapping.radius,
                img_width=self._mapping.img_width,
                img_height=self._mapping.img_height,
                output_width=self._mapping.output_width,
                output_height=self._mapping.output_height,
                cut_position=self._mapping.cut_position,
            )
            print(f"✓ 映射矩阵已保存: {filepath}")
            return True
        except Exception as e:
            print(f"保存映射矩阵失败: {e}")
            return False

    def compute_mapping(self, num_frames: int = 30, show_progress: bool = True) -> bool:
        """
        计算映射矩阵（自动从当前输入源读取）

        Args:
            num_frames: 使用的帧数
            show_progress: 是否显示进度

        Returns:
            是否成功计算
        """
        if not self.is_opened and self._frame is None:
            print("请先打开输入源")
            return False

        centers = []
        radii = []

        source_name = "摄像头" if self.source_type == InputSourceType.CAMERA else "视频"
        print(f"从{source_name}使用 {num_frames} 帧计算映射矩阵...")

        # 如果已打开，从源读取帧
        if self.is_opened:
            for i in range(num_frames):
                ret, frame = self._cap.read()
                if not ret:
                    # 视频可能结束了，循环播放
                    if self.source_type == InputSourceType.VIDEO:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    break

                center, radius = self._detect_fisheye_region(frame)
                if self._validate_circle(center, radius, frame.shape[1], frame.shape[0]):
                    centers.append(center)
                    radii.append(radius)

                if show_progress:
                    self._show_calibration_frame(frame, center, radius, i + 1, num_frames)
        else:
            # 只有单帧
            frame = self._frame
            center, radius = self._detect_fisheye_region(frame)
            centers.append(center)
            radii.append(radius)

        if len(centers) == 0:
            print("无法检测到鱼眼区域")
            return False

        center = (
            int(np.median([c[0] for c in centers])),
            int(np.median([c[1] for c in centers])),
        )
        radius = int(np.median(radii))

        # 使用当前帧尺寸
        ref_frame = self._frame
        if self.is_opened:
            ret, ref_frame = self._cap.read()
            if not ret:
                ref_frame = self._frame

        self._create_mapping(center, radius, ref_frame.shape[1], ref_frame.shape[0])
        return True

    def compute_mapping_from_images(self, image_paths: List[str]) -> bool:
        """从静态图片计算映射矩阵"""
        centers = []
        radii = []
        valid_images = []

        print(f"从 {len(image_paths)} 张图片计算映射矩阵...")

        for i, path in enumerate(image_paths):
            if not os.path.exists(path):
                continue

            img = cv2.imread(path)
            if img is None:
                continue

            valid_images.append(img)
            center, radius = self._detect_fisheye_region(img)
            if self._validate_circle(center, radius, img.shape[1], img.shape[0]):
                centers.append(center)
                radii.append(radius)

            self._show_calibration_frame(img, center, radius, i + 1, len(image_paths))

        if not valid_images:
            print("没有有效的图片")
            return False

        center = (
            int(np.median([c[0] for c in centers])),
            int(np.median([c[1] for c in centers])),
        )
        radius = int(np.median(radii))

        self._create_mapping(center, radius, valid_images[0].shape[1], valid_images[0].shape[0])
        return True

    def compute_mapping_from_frame(self, frame: np.ndarray) -> bool:
        """从单帧图片计算映射矩阵"""
        self._frame = frame
        center, radius = self._detect_fisheye_region(frame)
        self._create_mapping(center, radius, frame.shape[1], frame.shape[0])
        return True

    def _create_mapping(self, center: Tuple[int, int], radius: int, img_width: int, img_height: int):
        """创建映射矩阵"""
        print(f"检测到圆心: {center}, 半径: {radius}")

        map_x = np.zeros((self.output_config.height, self.output_config.width), dtype=np.float32)
        map_y = np.zeros((self.output_config.height, self.output_config.width), dtype=np.float32)

        angle_offset = np.pi if self.cut_position == CutPosition.LEFT else 0

        for y_out in range(self.output_config.height):
            r_ratio = y_out / (self.output_config.height - 1)
            for x_out in range(self.output_config.width):
                angle = -(2 * np.pi * x_out / self.output_config.width) + angle_offset
                x_in = center[0] + r_ratio * radius * np.cos(angle)
                y_in = center[1] + r_ratio * radius * np.sin(angle)
                map_x[y_out, x_out] = np.clip(x_in, 0, img_width - 1)
                map_y[y_out, x_out] = np.clip(y_in, 0, img_height - 1)

        self._mapping = MappingData(
            map_x=map_x,
            map_y=map_y,
            center=center,
            radius=radius,
            img_width=img_width,
            img_height=img_height,
            output_width=self.output_config.width,
            output_height=self.output_config.height,
            cut_position=self.cut_position.value,
        )
        print(f"✓ 映射矩阵创建完成")

    # =========================================================================
    # 鱼眼区域检测
    # =========================================================================

    def _detect_fisheye_region(self, img: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """检测鱼眼有效区域"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        h, w = gray.shape[:2]
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 2

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
                maxRadius=min(w, h) // 2,
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                detected_center = (circles[0][0], circles[0][1])
                detected_radius = circles[0][2]

                if self._validate_circle(detected_center, detected_radius, w, h):
                    center_x, center_y = detected_center
                    radius = detected_radius
        except Exception:
            pass

        return (int(center_x), int(center_y)), int(radius)

    def _validate_circle(
        self,
        center: Tuple[int, int],
        radius: int,
        img_w: int,
        img_h: int,
    ) -> bool:
        """验证检测到的圆是否合理"""
        max_offset = min(img_w, img_h) // 3
        if abs(center[0] - img_w // 2) > max_offset or abs(center[1] - img_h // 2) > max_offset:
            return False

        min_radius = min(img_w, img_h) // 4
        max_radius = min(img_w, img_h) // 2 + 50
        if not (min_radius <= radius <= max_radius):
            return False

        return True

    def _show_calibration_frame(self, frame, center, radius, current, total):
        """显示标定进度帧"""
        display = frame.copy()
        cv2.circle(display, center, radius, (0, 255, 0), 2)
        cv2.circle(display, center, 5, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"Calibrating {current}/{total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        display = cv2.resize(display, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.imshow("Calibration", display)
        cv2.waitKey(50)

    # =========================================================================
    # 图像展开
    # =========================================================================

    def unwrap(self, img: np.ndarray) -> np.ndarray:
        """展开鱼眼图像"""
        if self._mapping is None:
            raise RuntimeError("映射矩阵未初始化，请先计算或加载映射矩阵")

        return cv2.remap(
            img,
            self._mapping.map_x,
            self._mapping.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    def unwrap_file(self, input_path: str, output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """展开图片文件"""
        img = cv2.imread(input_path)
        if img is None:
            print(f"无法读取图片: {input_path}")
            return None

        result = self.unwrap(img)

        if output_path:
            cv2.imwrite(output_path, result)
            print(f"已保存: {output_path}")

        return result

    def get_fisheye_mask(self, img: np.ndarray) -> np.ndarray:
        """获取鱼眼有效区域掩码"""
        if self._mapping is None:
            raise RuntimeError("映射矩阵未初始化")

        mask = np.zeros_like(img)
        cv2.circle(mask, self._mapping.center, self._mapping.radius, (255, 255, 255), -1)
        return cv2.bitwise_and(img, mask)

    # =========================================================================
    # 实时处理
    # =========================================================================

    def run(self, window_name: str = "Fisheye Panorama", show_original: bool = True):
        """
        运行实时/视频展开

        Args:
            window_name: 窗口名称
            show_original: 是否显示原图

        按键说明:
            q: 退出
            s: 保存当前帧
            m: 保存映射矩阵
            (视频模式) 空格: 暂停/继续
        """
        if not self.is_opened:
            print("输入源未打开，正在尝试打开...")
            if not self.open():
                return

        if not self.is_ready:
            print("映射矩阵未就绪，正在计算...")
            self.compute_mapping()

        source_name = "摄像头" if self.source_type == InputSourceType.CAMERA else "视频"
        print("=" * 60)
        print(f"{source_name}展开模式")
        print("  q: 退出")
        print("  s: 保存当前帧")
        print("  m: 保存映射矩阵")
        if self.source_type == InputSourceType.VIDEO:
            print("  空格: 暂停/继续")
        print("=" * 60)

        paused = False
        try:
            while True:
                if not paused:
                    ret, frame = self._cap.read()
                    if not ret:
                        if self.source_type == InputSourceType.VIDEO:
                            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        break
                    self._frame = frame

                display = self._create_display_frame(self._frame, show_original, paused)
                cv2.imshow(window_name, display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self._save_current_frame(self._frame)
                elif key == ord('m'):
                    self.save_mapping()
                elif key == ord(' ') and self.source_type == InputSourceType.VIDEO:
                    paused = not paused

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            cv2.destroyAllWindows()

    def _create_display_frame(self, frame: np.ndarray, show_original: bool, paused: bool = False) -> np.ndarray:
        """创建显示帧"""
        panorama = self.unwrap(frame)

        if not show_original:
            if paused:
                cv2.putText(panorama, "PAUSED", (panorama.shape[1] // 2 - 50, panorama.shape[0] // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            return cv2.resize(panorama, (int(panorama.shape[1] * self.display_scale), int(panorama.shape[0] * self.display_scale)))

        orig_small = cv2.resize(frame, (
            int(frame.shape[1] * self.display_scale),
            int(frame.shape[0] * self.display_scale),
        ))

        if self._mapping:
            center = (
                int(self._mapping.center[0] * self.display_scale),
                int(self._mapping.center[1] * self.display_scale),
            )
            radius = int(self._mapping.radius * self.display_scale)
            cv2.circle(orig_small, center, radius, (0, 0, 255), 2)
            cv2.circle(orig_small, center, 3, (0, 0, 255), -1)

        mask_small = self.get_fisheye_mask(frame)
        mask_small = cv2.resize(mask_small, (
            int(frame.shape[1] * self.display_scale),
            int(frame.shape[0] * self.display_scale),
        ))

        top_row = np.hstack((orig_small, mask_small))
        panorama_scaled_height = int(panorama.shape[0] * self.display_scale)
        panorama_resized = cv2.resize(panorama, (top_row.shape[1], panorama_scaled_height))

        font_scale = max(0.4, self.display_scale * 1.2)
        cv2.putText(top_row, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        cv2.putText(top_row, "Fisheye Area", (orig_small.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
        cv2.putText(panorama_resized, f"Panorama {self.output_config.width}x{self.output_config.height}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

        if paused:
            cv2.putText(panorama_resized, "PAUSED", (panorama_resized.shape[1] // 2 - 80, panorama_resized.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 2, (0, 0, 255), 3)

        return np.vstack((top_row, panorama_resized))

    def _save_current_frame(self, frame: np.ndarray):
        """保存当前帧"""
        import time
        timestamp = int(time.time())
        panorama = self.unwrap(frame)
        filename = f"panorama_{timestamp}.jpg"
        cv2.imwrite(filename, panorama)
        print(f"已保存: {filename}")

    # =========================================================================
    # 资源管理
    # =========================================================================

    def release(self):
        """释放所有资源"""
        self._close_source()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# =============================================================================
# 便捷函数
# =============================================================================

def run_camera(
    camera_index: int = 0,
    map_file: Optional[str] = None,
    output_width: int = 3840,
    output_height: int = 1080,
    cut_position: str = "right",
):
    """
    便捷函数：直接运行摄像头展开

    Args:
        camera_index: 摄像头索引
        map_file: 映射矩阵文件
        output_width: 输出宽度
        output_height: 输出高度
        cut_position: 切开点位置
    """
    unwrapper = FisheyeUnwrapper.with_camera(
        camera_index=camera_index,
        map_file=map_file,
        output_width=output_width,
        output_height=output_height,
        cut_position=cut_position,
    )
    unwrapper.open()
    unwrapper.run()
    unwrapper.release()


def run_video(
    video_path: str,
    map_file: Optional[str] = None,
    output_width: int = 3840,
    output_height: int = 1080,
    cut_position: str = "right",
):
    """
    便捷函数：直接运行视频文件展开

    Args:
        video_path: 视频文件路径
        map_file: 映射矩阵文件
        output_width: 输出宽度
        output_height: 输出高度
        cut_position: 切开点位置
    """
    unwrapper = FisheyeUnwrapper.with_video(
        video_path=video_path,
        map_file=map_file,
        output_width=output_width,
        output_height=output_height,
        cut_position=cut_position,
    )
    unwrapper.open()
    unwrapper.run()
    unwrapper.release()


def unwrap_single_image(
    image_path: str,
    map_file: str,
    output_path: Optional[str] = None,
) -> Optional[np.ndarray]:
    """便捷函数：展开单张图片"""
    unwrapper = FisheyeUnwrapper(map_file=map_file)
    if not unwrapper.load_mapping():
        img = cv2.imread(image_path)
        if img is None:
            return None
        unwrapper.compute_mapping_from_frame(img)

    return unwrapper.unwrap_file(image_path, output_path)
