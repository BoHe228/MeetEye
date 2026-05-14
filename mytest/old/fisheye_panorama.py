import cv2
import numpy as np
import os
import time


class FisheyePanoramaRealtimeWithClick:
    def __init__(self, cam_index=1, output_width=1920, output_height=540,
                 desired_width=1920, desired_height=1080,
                 map_file=None, force_recompute=False, vertical_fov_deg=150.0,
                 init_camera=True):
        """
        初始化实时鱼眼横向展开类（带点击获取角度功能）
        Args:
            cam_index: 摄像头索引
            output_width: 展开后全景图的宽度
            output_height: 展开后全景图的高度
            desired_width: 期望的相机宽度
            desired_height: 期望的相机高度
            map_file: 映射矩阵文件路径，如果提供则尝试加载
            force_recompute: 强制重新计算映射矩阵，即使存在文件
            vertical_fov_deg: 垂直视场角（度），用于计算俯仰角
            init_camera: 是否初始化摄像头（设为False时仅作为映射器使用）
        """
        self.cam_index = cam_index
        self.output_width = output_width
        self.output_height = output_height
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.map_file = map_file
        self.vertical_fov_deg = vertical_fov_deg

        # 初始化属性
        self.cap = None
        self.map_x = None
        self.map_y = None
        self.center = None
        self.radius = None
        self.img_width = None
        self.img_height = None
        self.frame = None
        self.window_name = None
        self.last_click_info = None
        self.click_history = []

        if init_camera:
            self._init_camera_and_window()
        else:
            # 不初始化摄像头，仅设置基本属性
            self.img_width = desired_width
            self.img_height = desired_height

        # 尝试加载或计算映射矩阵（仅在有图像尺寸时）
        if self.img_width is not None and self.img_height is not None:
            if not force_recompute and map_file and os.path.exists(map_file):
                print(f"尝试从文件加载映射矩阵: {map_file}")
                if self.load_maps(map_file):
                    print("映射矩阵加载成功")
                else:
                    print("映射矩阵加载失败，将重新计算")
                    # 需要先有frame才能计算
                    if self.frame is not None:
                        self.compute_and_save_maps()
            elif self.frame is not None:
                # 重新计算映射矩阵
                self.compute_and_save_maps()

    def _init_camera_and_window(self):
        """初始化摄像头和显示窗口（仅在需要时调用）"""
        # 初始化摄像头
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 {self.cam_index}")
            return

        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.desired_height)

        cv2.waitKey(100)

        # 获取一帧以确定图像尺寸
        ret, self.frame = self.cap.read()
        if not ret:
            print("错误：无法从摄像头读取画面")
            return

        self.img_height, self.img_width = self.frame.shape[:2]
        print(f"实际分辨率: {self.img_width}x{self.img_height}")
        print(f"垂直视场角: {self.vertical_fov_deg}度")

        # 创建显示窗口
        self.window_name = 'Fisheye Panorama - Click for Angles'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # 设置鼠标回调函数
        cv2.setMouseCallback(self.window_name, self.on_mouse_click)

    def init_from_frame(self, frame):
        """
        从图像帧初始化（用于不使用摄像头的情况）
        Args:
            frame: 输入图像帧
        """
        self.frame = frame
        self.img_height, self.img_width = frame.shape[:2]

        # 尝试加载或计算映射矩阵
        if self.map_file and os.path.exists(self.map_file):
            print(f"尝试从文件加载映射矩阵: {self.map_file}")
            if not self.load_maps(self.map_file):
                print("映射矩阵加载失败，将重新计算")
                self.compute_and_save_maps()
        else:
            self.compute_and_save_maps()

    def compute_and_save_maps(self):
        """
        计算并保存映射矩阵
        基于【文档内容】中横向展开法的几何原理
        """
        if self.frame is None:
            print("错误：没有可用的帧来计算映射矩阵")
            return

        # 检测鱼眼有效区域的圆心和半径（使用改进的方法）
        self.center, self.radius = self.detect_fisheye_region_improved(self.frame)
        print(f"检测到圆心: {self.center}, 半径: {self.radius}")

        # 计算映射矩阵
        self.map_x, self.map_y = self.create_panorama_map()

        # 如果指定了映射文件，保存映射矩阵
        if self.map_file:
            self.save_maps(self.map_file)

    def create_panorama_map(self):
        """
        创建横向展开的映射矩阵
        基于【文档内容】中横向展开法的几何原理
        """
        h, w = self.img_height, self.img_width
        center_x, center_y = self.center

        # 初始化映射矩阵
        map_x = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        map_y = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        # 【文档内容】原理：将圆形鱼眼图按经纬度展开为矩形全景图
        for y_out in range(self.output_height):
            # 修改：从 0 开始到 1，覆盖从圆心到边缘的完整范围
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

    def on_mouse_click(self, event, x, y, flags, param):
        """
        鼠标点击回调函数
        在全景图上点击获取水平方位角和俯仰角
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # 计算鼠标点击在全景图中的位置
            # 注意：由于显示图像可能被调整大小，我们需要计算实际点击在全景图区域的位置

            # 假设显示布局：
            # 上半部分：原图和有效区域（各占一半宽度）
            # 下半部分：全景图

            # 先计算原图和有效区域显示的高度
            preview_height = int(self.img_height * 0.5)  # 假设显示比例为0.5

            # 检查点击是否在全景图区域
            if y > preview_height:
                # 点击在全景图区域
                panorama_y = y - preview_height

                # 计算全景图在显示中的实际高度
                panorama_display_height = self.output_height

                # 如果全景图被缩放显示，需要计算缩放比例
                # 这里假设全景图高度被缩放以适应显示宽度
                # 实际情况需要根据显示代码调整

                # 计算水平方位角 (0-360度)
                azimuth_deg = 360.0 * x / self.output_width

                # 计算俯仰角
                # 假设：全景图顶部 (y=0) 对应 +90°
                #       全景图底部 (y=output_height) 对应 -10°
                #       垂直视场角共 100°
                elevation_deg = 90.0 - (self.vertical_fov_deg * panorama_y / panorama_display_height)

                # 记录点击信息
                self.last_click_info = {
                    'x': x,
                    'y': y,
                    'panorama_x': x,
                    'panorama_y': panorama_y,
                    'azimuth_deg': azimuth_deg,
                    'elevation_deg': elevation_deg,
                    'time': time.time()
                }

                # 添加到历史记录
                self.click_history.append(self.last_click_info.copy())
                if len(self.click_history) > 10:  # 只保留最近的10次点击
                    self.click_history.pop(0)

                print("\n" + "="*50)
                print(f"点击位置: 屏幕({x}, {y}), 全景图({x}, {panorama_y})")
                print(f"水平方位角: {azimuth_deg:.2f}°")
                print(f"俯仰角: {elevation_deg:.2f}°")
                print(f"视角方向: 方位{azimuth_deg:.1f}°, 俯仰{elevation_deg:.1f}°")
                print("="*50)

    def save_maps(self, filepath):
        """
        保存映射矩阵到文件
        根据【文档内容】，映射矩阵可以保存并重复使用
        """
        try:
            # 保存映射矩阵和相关的相机参数
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
                vertical_fov_deg=self.vertical_fov_deg
            )

            print(f"映射矩阵已保存到: {filepath}")
            print(f"保存的参数: 相机{self.cam_index}, 分辨率{self.img_width}x{self.img_height}")
            print(f"          圆心{self.center}, 半径{self.radius}")
            print(f"          垂直FOV: {self.vertical_fov_deg}度")
            return True

        except Exception as e:
            print(f"保存映射矩阵失败: {e}")
            return False

    def load_maps(self, filepath):
        """
        从文件加载映射矩阵
        根据【文档内容】，映射矩阵可以重复使用
        """
        try:
            # 加载映射矩阵
            data = np.load(filepath, allow_pickle=True)

            # 检查相机参数是否匹配（仅当我们有当前图像尺寸时）
            if self.img_width is not None and self.img_height is not None:
                if ('img_width' in data and 'img_height' in data and
                    'camera_index' in data):
                    if (data['img_width'] != self.img_width or
                        data['img_height'] != self.img_height or
                        data['camera_index'] != self.cam_index):
                        print(f"警告: 映射矩阵参数不匹配")
                        print(f"  文件中的: 相机{data['camera_index']}, 分辨率{data['img_width']}x{data['img_height']}")
                        print(f"  当前的: 相机{self.cam_index}, 分辨率{self.img_width}x{self.img_height}")
                        # 不询问用户，直接继续使用（因为可能作为映射器运行）

            # 加载映射矩阵
            self.map_x = data['map_x']
            self.map_y = data['map_y']
            self.center = tuple(data['center'])
            self.radius = int(data['radius'])

            # 更新输出尺寸（如果文件中的尺寸与当前不同）
            if 'output_width' in data and 'output_height' in data:
                self.output_width = int(data['output_width'])
                self.output_height = int(data['output_height'])

            # 加载垂直FOV
            if 'vertical_fov_deg' in data:
                self.vertical_fov_deg = float(data['vertical_fov_deg'])
                print(f"加载垂直FOV: {self.vertical_fov_deg}度")

            # 更新图像尺寸
            if 'img_width' in data and 'img_height' in data:
                self.img_width = int(data['img_width'])
                self.img_height = int(data['img_height'])

            print(f"映射矩阵加载成功")
            print(f"  圆心: {self.center}, 半径: {self.radius}")
            print(f"  输出尺寸: {self.output_width}x{self.output_height}")
            return True

        except Exception as e:
            print(f"加载映射矩阵失败: {e}")
            return False

    def detect_fisheye_region(self, img):
        """自动检测鱼眼图像的有效圆形区域"""
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius * 1.0)  # 不再缩小半径，使用完整检测到的区域
        else:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2  # 不减边距

        return center, radius

    def detect_fisheye_region_improved(self, img):
        """
        改进的鱼眼有效区域检测方法
        使用多种策略确保检测的准确性
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape[:2]

        # 策略1：假设图像中心就是圆心，半径是图像短边的一半
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 2

        # 策略2：尝试使用边缘检测找圆
        try:
            # 使用高斯模糊减少噪声
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 使用Canny边缘检测
            edges = cv2.Canny(blurred, 50, 150)

            # 使用霍夫圆变换检测圆
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
                # 取第一个检测到的圆（通常是最明显的）
                detected_center = (circles[0][0], circles[0][1])
                detected_radius = circles[0][2]
                print(f"霍夫圆变换检测到: 圆心{detected_center}, 半径{detected_radius}")

                # 验证检测结果的合理性
                if self._is_valid_circle(detected_center, detected_radius, w, h):
                    center_x, center_y = detected_center
                    radius = detected_radius
                else:
                    print("霍夫圆变换结果不合理，使用默认值")
            else:
                print("霍夫圆变换未检测到圆，使用阈值化方法")

                # 策略3：使用自适应阈值化和轮廓检测
                # 使用多种阈值尝试
                best_contour = None
                best_radius = 0

                for thresh_val in [20, 30, 40, 50]:
                    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        (x, y), r = cv2.minEnclosingCircle(largest_contour)
                        if r > best_radius and self._is_valid_circle((int(x), int(y)), int(r), w, h):
                            best_contour = largest_contour
                            best_radius = int(r)

                if best_contour is not None:
                    (x, y), r = cv2.minEnclosingCircle(best_contour)
                    center_x, center_y = int(x), int(y)
                    radius = int(r)
                    print(f"阈值化方法检测到: 圆心({center_x}, {center_y}), 半径{radius}")
                else:
                    print("阈值化方法也失败，使用默认值")

        except Exception as e:
            print(f"高级检测方法失败: {e}, 使用默认值")

        # 最终验证和优化
        # 确保圆心不会太偏离图像中心
        max_center_offset = min(w, h) // 4
        center_x = np.clip(center_x, w // 2 - max_center_offset, w // 2 + max_center_offset)
        center_y = np.clip(center_y, h // 2 - max_center_offset, h // 2 + max_center_offset)

        # 确保半径合理
        min_radius = min(w, h) // 3
        max_radius = min(w, h) // 2 + 50
        radius = np.clip(radius, min_radius, max_radius)

        final_center = (int(center_x), int(center_y))
        final_radius = int(radius)

        print(f"最终鱼眼区域: 圆心{final_center}, 半径{final_radius}")
        return final_center, final_radius

    def _is_valid_circle(self, center, radius, img_w, img_h):
        """验证检测到的圆是否合理"""
        # 圆心应该在图像中心附近
        max_offset = min(img_w, img_h) // 3
        center_x, center_y = center
        if abs(center_x - img_w // 2) > max_offset or abs(center_y - img_h // 2) > max_offset:
            return False

        # 半径应该在合理范围内
        min_radius = min(img_w, img_h) // 4
        max_radius = min(img_w, img_h) // 2 + 50
        if not (min_radius <= radius <= max_radius):
            return False

        return True

    def get_useful_area(self, img):
        """提取圆形有效区域"""
        if self.center is None or self.radius is None:
            # 如果还没有检测圆心和半径，先检测
            self.center, self.radius = self.detect_fisheye_region(img)

        mask = np.zeros_like(img)
        cv2.circle(mask, self.center, self.radius, (255, 255, 255), -1)
        return cv2.bitwise_and(img, mask)

    def apply_panorama(self, img):
        """应用横向展开变换"""
        # 如果映射矩阵未初始化且有frame，先初始化
        if self.map_x is None or self.map_y is None:
            if self.frame is None:
                self.frame = img
            if self.img_width is None or self.img_height is None:
                self.img_height, self.img_width = img.shape[:2]
            self.compute_and_save_maps()

        panorama = cv2.remap(img, self.map_x, self.map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
        return panorama

    def draw_click_markers(self, panorama_img):
        """
        在全景图上绘制点击标记
        """
        img_with_markers = panorama_img.copy()

        # 绘制最近一次点击的标记
        if self.last_click_info:
            x = int(self.last_click_info['panorama_x'])
            y = int(self.last_click_info['panorama_y'])

            # 绘制十字标记
            cv2.drawMarker(img_with_markers, (x, y), (0, 0, 255),
                          markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # 绘制圆形标记
            cv2.circle(img_with_markers, (x, y), 5, (0, 255, 255), 2)

            # 添加角度文本
            azimuth = self.last_click_info['azimuth_deg']
            elevation = self.last_click_info['elevation_deg']
            text = f"Az:{azimuth:.1f}°, El:{elevation:.1f}°"
            cv2.putText(img_with_markers, text, (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 绘制历史点击标记（较小的标记）
        for i, click in enumerate(self.click_history[-5:]):  # 只显示最近的5次
            if click != self.last_click_info:  # 避免重复绘制
                x = int(click['panorama_x'])
                y = int(click['panorama_y'])
                cv2.circle(img_with_markers, (x, y), 3, (0, 200, 0), -1)

        return img_with_markers

    def run(self):
        """主循环"""
        if self.cap is None:
            print("错误：摄像头未初始化，无法运行主循环")
            return

        print("开始实时鱼眼横向展开...")
        print("在全景图区域点击鼠标可获取该点的水平方位角和俯仰角")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        print("按 'm' 键保存当前映射矩阵")
        print("按 'r' 键重新计算映射矩阵")
        print("按 'f' 键调整垂直FOV参数")
        print("按 'c' 键清空点击标记")

        fps_counter = 0
        fps_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            fps_counter += 1
            current_time = time.time()
            if current_time - fps_time >= 1.0:
                print(f"FPS: {fps_counter}")
                fps_counter = 0
                fps_time = current_time

            # 应用横向展开
            panorama = self.apply_panorama(frame)

            # 获取有效区域
            useful_area = self.get_useful_area(frame)

            # 添加点击标记
            panorama_with_markers = self.draw_click_markers(panorama)

            # 显示
            display_scale = 0.5
            display_original = cv2.resize(frame,
                                         (int(self.img_width * display_scale),
                                          int(self.img_height * display_scale)))
            display_useful = cv2.resize(useful_area,
                                       (int(self.img_width * display_scale),
                                        int(self.img_height * display_scale)))

            # 上半部分：原图和有效区域
            top_row = np.hstack((display_original, display_useful))

            # 调整全景图大小以匹配上半部分的宽度
            panorama_resized = cv2.resize(panorama_with_markers, (top_row.shape[1], self.output_height))

            # 组合显示
            combined_display = np.vstack((top_row, panorama_resized))

            # 添加标签
            cv2.putText(combined_display, f"Original ({self.img_width}x{self.img_height})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined_display, f"Useful Area", (display_original.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 在全景图区域添加说明
            info_text = f"Panorama ({self.output_width}x{self.output_height}) - Click for angles (FOV: {self.vertical_fov_deg}°)"
            cv2.putText(combined_display, info_text, (10, top_row.shape[0] + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 如果有点击历史，显示最近点击信息
            if self.click_history:
                last_click = self.click_history[-1]
                click_info = f"Last click: Az={last_click['azimuth_deg']:.1f}°, El={last_click['elevation_deg']:.1f}°"
                cv2.putText(combined_display, click_info, (10, top_row.shape[0] + 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow(self.window_name, combined_display)

            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # 保存当前帧
                timestamp = int(time.time())
                cv2.imwrite(f'panorama_{timestamp}.jpg', panorama)
                print(f"已保存全景图: panorama_{timestamp}.jpg")
            elif key == ord('m'):
                # 保存映射矩阵
                if not self.map_file:
                    self.map_file = f"panorama_maps_{int(time.time())}.npz"
                self.save_maps(self.map_file)
            elif key == ord('r'):
                # 重新计算映射矩阵
                print("重新计算映射矩阵...")
                self.compute_and_save_maps()
            elif key == ord('f'):
                # 调整垂直FOV参数 - 仅在交互式模式下
                print("FOV调整仅在独立运行模式下可用")
            elif key == ord('c'):
                # 清空点击标记
                self.click_history = []
                self.last_click_info = None
                print("点击标记已清空")

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        if self.window_name is not None:
            cv2.destroyAllWindows()


# 主程序入口
if __name__ == "__main__":
    # 参数说明：
    # vertical_fov_deg: 垂直视场角，用于计算俯仰角
    # 常见的鱼眼相机垂直FOV通常在120-180度之间

    processor = FisheyePanoramaRealtimeWithClick(
        cam_index=1,
        output_width=1920,
        output_height=540,
        desired_width=1920,
        desired_height=1080,
        map_file="my_fisheye_maps.npz",
        force_recompute=False,
        vertical_fov_deg=150.0,
        init_camera=True
    )

    try:
        processor.run()
    except KeyboardInterrupt:
        pass
    finally:
        processor.release()
