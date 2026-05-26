import cv2
import numpy as np
import os


class CeilingFisheyePanorama:
    def __init__(self, video_source, output_width=3840, output_height=1080,
                 map_file=None, force_recompute=False, num_frames=10, view_type="bottom", cut_position="right"):
        """
        初始化吸顶鱼眼摄像头横向展开类
        Args:
            video_source: 视频源路径或摄像头索引
            output_width: 展开后全景图的宽度
            output_height: 展开后全景图的高度
            map_file: 映射矩阵文件路径，如果提供则尝试加载
            force_recompute: 强制重新计算映射矩阵，即使存在文件
            num_frames: 用于计算映射矩阵的帧数（默认10帧）
            view_type: 视角类型 - "top"(顶部视角) 或 "bottom"(底部视角)
            cut_position: 切开位置 - "right"(正右方) 或 "left"(正左方)
        """
        self.video_source = video_source
        self.output_width = output_width
        self.output_height = output_height
        self.map_file = map_file
        self.num_frames = num_frames
        self.view_type = view_type.lower()
        self.cut_position = cut_position.lower()

        # 打开视频源
        if isinstance(video_source, str) and os.path.exists(video_source):
            print(f"从视频文件打开: {video_source}")
            self.cap = cv2.VideoCapture(video_source)
        else:
            print(f"从摄像头打开: {video_source}")
            self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            print(f"错误：无法打开视频源 {video_source}")
            return

        # 读取第一帧确定图像尺寸
        ret, self.frame = self.cap.read()
        if not ret:
            print("错误：无法从视频源读取画面")
            return

        self.img_height, self.img_width = self.frame.shape[:2]
        print(f"图像分辨率: {self.img_width}x{self.img_height}")

        # 尝试加载或计算映射矩阵
        if not force_recompute and map_file and os.path.exists(map_file):
            print(f"尝试从文件加载映射矩阵: {map_file}")
            if self.load_maps(map_file):
                print("映射矩阵加载成功")
            else:
                print("映射矩阵加载失败，将重新计算")
                self.compute_and_save_maps()
        else:
            self.compute_and_save_maps()

        # 创建显示窗口
        cv2.namedWindow('Ceiling Fisheye Panorama', cv2.WINDOW_NORMAL)

    def compute_and_save_maps(self):
        """
        计算并保存映射矩阵
        使用前num_frames帧进行多帧平均，提高检测稳定性
        """
        # 使用多帧检测鱼眼有效区域的圆心和半径
        print(f"使用前 {self.num_frames} 帧计算映射矩阵...")
        self.center, self.radius = self.detect_fisheye_region_multiframe()

        print(f"最终检测结果: 圆心{self.center}, 半径{self.radius}")

        # 计算映射矩阵（吸顶鱼眼专用）
        self.map_x, self.map_y = self.create_ceiling_panorama_map()

        # 保存映射矩阵
        if self.map_file:
            self.save_maps(self.map_file)

    def detect_fisheye_region_multiframe(self):
        """
        从视频前num_frames帧中检测鱼眼有效区域
        通过多帧平均获得更稳定的圆心和半径
        """
        centers = []
        radii = []

        # 重新定位到视频开头
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        valid_frames = 0
        for i in range(self.num_frames):
            ret, frame = self.cap.read()
            if not ret:
                print(f"  警告：无法读取第 {i+1} 帧")
                continue

            # 对每帧进行检测
            center, radius = self.detect_fisheye_region(frame)

            # 验证检测结果的合理性
            if self._is_valid_circle(center, radius, frame.shape[1], frame.shape[0]):
                centers.append(center)
                radii.append(radius)
                valid_frames += 1
                print(f"  帧 {i+1}/{self.num_frames}: 圆心{center}, 半径{radius}")
            else:
                print(f"  帧 {i+1}/{self.num_frames}: 检测结果不合理，跳过")

            # 显示检测过程
            display_frame = frame.copy()
            cv2.circle(display_frame, center, radius, (0, 255, 0), 2)
            cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, f"Calibration Frame {i+1}/{self.num_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            display_scale = 0.5
            display_frame = cv2.resize(display_frame,
                                       (int(frame.shape[1] * display_scale),
                                        int(frame.shape[0] * display_scale)))
            cv2.imshow('Calibration in progress...', display_frame)
            cv2.waitKey(100)

        cv2.destroyWindow('Calibration in progress...')

        if valid_frames < max(1, self.num_frames // 2):
            print("警告：有效帧数太少，使用单帧检测结果")
            return self.detect_fisheye_region(self.frame)

        # 使用中位数代替平均值，更稳健
        centers = np.array(centers)
        radii = np.array(radii)

        final_center_x = int(np.median(centers[:, 0]))
        final_center_y = int(np.median(centers[:, 1]))
        final_radius = int(np.median(radii))

        print(f"多帧检测完成:")
        print(f"  有效帧数: {valid_frames}/{self.num_frames}")
        print(f"  圆心范围: ({np.min(centers[:,0])}, {np.min(centers[:,1])}) ~ ({np.max(centers[:,0])}, {np.max(centers[:,1])})")
        print(f"  半径范围: {np.min(radii)} ~ {np.max(radii)}")

        return (final_center_x, final_center_y), final_radius

    def create_ceiling_panorama_map(self):
        """
        创建吸顶鱼眼摄像头的横向展开映射矩阵
        【关键】针对吸顶安装进行优化
        """
        h, w = self.img_height, self.img_width
        center_x, center_y = self.center

        # 初始化映射矩阵
        map_x = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        map_y = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        print(f"计算吸顶鱼眼横向展开映射矩阵，视角: {self.view_type}...")

        # 【吸顶鱼眼展开原理】
        # 吸顶摄像头朝下，展开时将圆周方向展为水平方向，半径方向展为垂直方向
        # view_type="top": 输出图顶部对应图像中心（下方），输出图底部对应图像边缘（上方）
        # view_type="bottom": 输出图底部对应图像中心（下方），输出图顶部对应图像边缘（上方）

        for y_out in range(self.output_height):
            if self.view_type == "bottom":
                # 底部视角：输出底部 = 鱼眼中心，输出顶部 = 鱼眼边缘
                r_ratio = 1 - (y_out / (self.output_height - 1))
            else:
                # 顶部视角（默认）：输出顶部 = 鱼眼中心，输出底部 = 鱼眼边缘
                r_ratio = y_out / (self.output_height - 1)

            for x_out in range(self.output_width):
                # angle: 0 → 2π，从右开始逆时针旋转
                angle = 2 * np.pi * x_out / self.output_width

                # 计算输入图像中的坐标
                x_in = center_x + r_ratio * self.radius * np.cos(angle)
                y_in = center_y + r_ratio * self.radius * np.sin(angle)

                # 确保坐标在有效范围内
                x_in = np.clip(x_in, 0, w - 1)
                y_in = np.clip(y_in, 0, h - 1)

                map_x[y_out, x_out] = x_in
                map_y[y_out, x_out] = y_in

        print(f"映射矩阵计算完成，展开尺寸: {self.output_width}x{self.output_height}")
        return map_x, map_y

    def detect_fisheye_region(self, img):
        """
        检测鱼眼图像的有效圆形区域
        针对吸顶鱼眼优化
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        h, w = gray.shape[:2]

        # 默认值：假设图像中心就是圆心，半径是图像短边的一半
        center_x = w // 2
        center_y = h // 2
        radius = min(w, h) // 2

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
                detected_center = (circles[0][0], circles[0][1])
                detected_radius = circles[0][2]
                print(f"霍夫圆变换检测到: 圆心{detected_center}, 半径{detected_radius}")

                if self._is_valid_circle(detected_center, detected_radius, w, h):
                    center_x, center_y = detected_center
                    radius = detected_radius
                else:
                    print("霍夫圆变换结果不合理，使用阈值化方法")
                    # 使用自适应阈值化找最大轮廓
                    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        (x, y), r = cv2.minEnclosingCircle(largest_contour)
                        if self._is_valid_circle((int(x), int(y)), int(r), w, h):
                            center_x, center_y = int(x), int(y)
                            radius = int(r)
                            print(f"阈值化方法检测到: 圆心({center_x}, {center_y}), 半径{radius}")
            else:
                print("霍夫圆变换未检测到圆，使用阈值化方法")
                _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    (x, y), r = cv2.minEnclosingCircle(largest_contour)
                    if self._is_valid_circle((int(x), int(y)), int(r), w, h):
                        center_x, center_y = int(x), int(y)
                        radius = int(r)
                        print(f"阈值化方法检测到: 圆心({center_x}, {center_y}), 半径{radius}")

        except Exception as e:
            print(f"检测方法失败: {e}, 使用默认值")

        # 确保圆心不会太偏离图像中心
        max_center_offset = min(w, h) // 4
        center_x = np.clip(center_x, w // 2 - max_center_offset, w // 2 + max_center_offset)
        center_y = np.clip(center_y, h // 2 - max_center_offset, h // 2 + max_center_offset)

        # 确保半径合理
        min_radius = min(w, h) // 3
        max_radius = min(w, h) // 2
        radius = np.clip(radius, min_radius, max_radius)

        return (int(center_x), int(center_y)), int(radius)

    def _is_valid_circle(self, center, radius, img_w, img_h):
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

    def save_maps(self, filepath):
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
                video_source=str(self.video_source),
                view_type=self.view_type,
                cut_position=self.cut_position
            )
            print(f"映射矩阵已保存到: {filepath}")
            print(f"  圆心: {self.center}, 半径: {self.radius}, 视角: {self.view_type}, 切开位置: {self.cut_position}")
            return True
        except Exception as e:
            print(f"保存映射矩阵失败: {e}")
            return False

    def load_maps(self, filepath):
        """从文件加载映射矩阵"""
        try:
            data = np.load(filepath, allow_pickle=True)

            if ('img_width' in data and 'img_height' in data):
                if (data['img_width'] != self.img_width or
                    data['img_height'] != self.img_height):
                    print(f"警告: 图像分辨率不匹配")
                    print(f"  文件中的: {data['img_width']}x{data['img_height']}")
                    print(f"  当前的: {self.img_width}x{self.img_height}")
                    choice = input("图像分辨率不匹配，是否继续使用? (y/n): ").strip().lower()
                    if choice != 'y':
                        return False

            self.map_x = data['map_x']
            self.map_y = data['map_y']
            self.center = tuple(data['center'])
            self.radius = int(data['radius'])

            # 加载视角和切开位置设置
            if 'view_type' in data:
                self.view_type = str(data['view_type'])
            if 'cut_position' in data:
                self.cut_position = str(data['cut_position'])

            if 'output_width' in data and 'output_height' in data:
                if (data['output_width'] != self.output_width or
                    data['output_height'] != self.output_height):
                    print(f"警告: 输出尺寸不匹配，使用文件中的尺寸")
                    self.output_width = int(data['output_width'])
                    self.output_height = int(data['output_height'])

            print(f"映射矩阵加载成功")
            print(f"  圆心: {self.center}, 半径: {self.radius}, 视角: {self.view_type}")
            print(f"  输出尺寸: {self.output_width}x{self.output_height}")
            return True

        except Exception as e:
            print(f"加载映射矩阵失败: {e}")
            return False

    def set_cut_position(self, cut_position):
        """动态设置切开位置，无需重新计算映射矩阵"""
        self.cut_position = cut_position.lower()
        print(f"切开位置已切换为: {self.cut_position}")

    def apply_panorama(self, img):
        """应用横向展开变换"""
        panorama = cv2.remap(img, self.map_x, self.map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))

        # 根据切开位置进行循环滚动
        if self.cut_position == "left":
            # 向左切开 = 将图像右半部分滚动到左边
            shift = self.output_width // 2
            panorama = np.roll(panorama, shift, axis=1)

        return panorama

    def get_useful_area(self, img):
        """提取圆形有效区域"""
        mask = np.zeros_like(img)
        cv2.circle(mask, self.center, self.radius, (255, 255, 255), -1)
        return cv2.bitwise_and(img, mask)

    def process_single_image(self, img, save_path=None):
        """处理单张图片"""
        if isinstance(img, str):
            frame = cv2.imread(img)
            if frame is None:
                print(f"错误：无法读取图片 {img}")
                return None, None, None
        else:
            frame = img.copy()

        panorama = self.apply_panorama(frame)
        useful_area = self.get_useful_area(frame)

        if save_path:
            cv2.imwrite(save_path, panorama)
            print(f"已保存全景图: {save_path}")

        return frame, useful_area, panorama

    def run(self):
        """主循环"""
        print("开始吸顶鱼眼横向展开...")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        print("按 'm' 键保存当前映射矩阵")
        print("按 'c' 键切换切开位置 (right/left)")

        # 重新定位到视频开头
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                # 如果是视频文件，循环播放
                if isinstance(self.video_source, str):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            self._display_frame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                import time
                timestamp = int(time.time())
                panorama = self.apply_panorama(frame)
                cv2.imwrite(f'xiding_panorama_{timestamp}.jpg', panorama)
                print(f"已保存全景图: xiding_panorama_{timestamp}.jpg")
            elif key == ord('m'):
                import time
                if not self.map_file:
                    self.map_file = f"xiding_maps_{int(time.time())}.npz"
                self.save_maps(self.map_file)
            elif key == ord('c'):
                # 切换切开位置
                new_cut = "left" if self.cut_position == "right" else "right"
                self.set_cut_position(new_cut)

    def _display_frame(self, frame):
        """显示单帧画面"""
        panorama = self.apply_panorama(frame)
        useful_area = self.get_useful_area(frame)

        display_scale = 0.5
        display_original = cv2.resize(frame,
                                     (int(self.img_width * display_scale),
                                      int(self.img_height * display_scale)))
        display_useful = cv2.resize(useful_area,
                                   (int(self.img_width * display_scale),
                                    int(self.img_height * display_scale)))

        top_row = np.hstack((display_original, display_useful))
        panorama_resized = cv2.resize(panorama, (top_row.shape[1], self.output_height))

        cv2.putText(top_row, f"Original ({self.img_width}x{self.img_height})", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(top_row, "Useful Area", (display_original.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(panorama_resized, f"Ceiling Panorama ({self.output_width}x{self.output_height}) | Cut: {self.cut_position}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        display_original_with_circle = display_original.copy()
        scaled_center = (int(self.center[0] * display_scale), int(self.center[1] * display_scale))
        scaled_radius = int(self.radius * display_scale)
        cv2.circle(display_original_with_circle, scaled_center, scaled_radius, (0, 0, 255), 2)
        cv2.circle(display_original_with_circle, scaled_center, 3, (0, 0, 255), -1)
        top_row = np.hstack((display_original_with_circle, display_useful))

        combined_display = np.vstack((top_row, panorama_resized))
        cv2.imshow('Ceiling Fisheye Panorama', combined_display)

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 使用示例1：从视频文件处理 - 底部视角
    processor = CeilingFisheyePanorama(
        video_source=r"C:\Users\bo.he_sx\Pictures\小会议室_吸顶.avi",  # 替换为您的视频路径
        output_width=3840,
        output_height=1080,
        map_file=r"map\xiding_maps_top.npz",
        force_recompute=False,
        num_frames=10,  # 使用前10帧计算映射矩阵
        view_type="top"  # 底部视角：输出底部 = 鱼眼中心
    )


    # 使用示例2：从摄像头处理 - 顶部视角
    # processor = CeilingFisheyePanorama(
    #     video_source=0,  # 摄像头索引
    #     output_width=3840,
    #     output_height=1080,
    #     map_file="xiding_maps.npz",
    #     force_recompute=False,
    #     num_frames=10,
    #     view_type="top"  # 顶部视角：输出顶部 = 鱼眼中心
    # )

    try:
        processor.run()
    except KeyboardInterrupt:
        pass
    finally:
        processor.release()
