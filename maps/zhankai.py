import cv2
import numpy as np
import os
import pickle


class FisheyePanoramaRealtime:
    def __init__(self, cam_index=1, output_width=1920, output_height=540,
                 desired_width=1920, desired_height=1080,
                 map_file=None, force_recompute=False):
        """
        初始化实时鱼眼横向展开类
        Args:
            cam_index: 摄像头索引
            output_width: 展开后全景图的宽度
            output_height: 展开后全景图的高度
            desired_width: 期望的相机宽度
            desired_height: 期望的相机高度
            map_file: 映射矩阵文件路径，如果提供则尝试加载
            force_recompute: 强制重新计算映射矩阵，即使存在文件
        """
        self.cam_index = cam_index
        self.output_width = output_width
        self.output_height = output_height
        self.desired_width = desired_width
        self.desired_height = desired_height
        self.map_file = map_file

        # 初始化摄像头
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            print(f"错误：无法打开摄像头 {cam_index}")
            return

        # 设置摄像头分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

        cv2.waitKey(100)

        # 获取一帧以确定图像尺寸
        ret, self.frame = self.cap.read()
        if not ret:
            print("错误：无法从摄像头读取画面")
            return

        self.img_height, self.img_width = self.frame.shape[:2]
        print(f"实际分辨率: {self.img_width}x{self.img_height}")

        # 核心：尝试加载或计算映射矩阵
        if not force_recompute and map_file and os.path.exists(map_file):
            # 尝试从文件加载映射矩阵
            print(f"尝试从文件加载映射矩阵: {map_file}")
            if self.load_maps(map_file):
                print("映射矩阵加载成功")
            else:
                print("映射矩阵加载失败，将重新计算")
                self.compute_and_save_maps()
        else:
            # 重新计算映射矩阵
            self.compute_and_save_maps()

        # 创建显示窗口
        cv2.namedWindow('Fisheye Panorama - Real-time', cv2.WINDOW_NORMAL)

    def compute_and_save_maps(self):
        """
        计算并保存映射矩阵
        基于横向展开法的几何原理
        """
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
        基于横向展开法的几何原理
        注意：这是反向映射（输出→输入），正是 cv2.remap() 需要的
        """
        h, w = self.img_height, self.img_width
        center_x, center_y = self.center

        # 初始化映射矩阵
        map_x = np.zeros((self.output_height, self.output_width), dtype=np.float32)
        map_y = np.zeros((self.output_height, self.output_width), dtype=np.float32)

        # 原理：将圆形鱼眼图按经纬度展开为矩形全景图
        # 这是反向映射：对于输出图像的每个像素，计算它在输入图像中的位置
        for y_out in range(self.output_height):
            # r_ratio 从 0 到 1，表示从圆心到边缘的距离比例
            r_ratio = y_out / (self.output_height - 1)

            for x_out in range(self.output_width):
                # 角度从 0 到 2π
                angle = 2 * np.pi * x_out / self.output_width

                # 计算输入图像中的坐标
                x_in = center_x + r_ratio * self.radius * np.cos(angle)
                y_in = center_y + r_ratio * self.radius * np.sin(angle)

                # 确保坐标在有效范围内
                x_in = np.clip(x_in, 0, w - 1)
                y_in = np.clip(y_in, 0, h - 1)

                # 存储映射关系（这正是 cv2.remap() 需要的反向映射）
                map_x[y_out, x_out] = x_in
                map_y[y_out, x_out] = y_in

        print(f"映射矩阵计算完成，展开尺寸: {self.output_width}x{self.output_height}")
        return map_x, map_y

    def save_maps(self, filepath):
        """
        保存映射矩阵到文件
        映射矩阵可以保存并重复使用
        """
        try:
            # 使用NumPy的npz格式保存
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
                camera_index=self.cam_index
            )

            print(f"映射矩阵已保存到: {filepath}")
            print(f"保存的参数: 相机{self.cam_index}, 分辨率{self.img_width}x{self.img_height}")
            print(f"          圆心{self.center}, 半径{self.radius}")
            return True

        except Exception as e:
            print(f"保存映射矩阵失败: {e}")
            return False

    def load_maps(self, filepath):
        """
        从文件加载映射矩阵
        映射矩阵可以重复使用
        """
        try:
            # 加载映射矩阵
            data = np.load(filepath, allow_pickle=True)

            # 检查相机参数是否匹配
            if ('img_width' in data and 'img_height' in data and
                'camera_index' in data):
                if (data['img_width'] != self.img_width or
                    data['img_height'] != self.img_height or
                    data['camera_index'] != self.cam_index):
                    print(f"警告: 映射矩阵参数不匹配")
                    print(f"  文件中的: 相机{data['camera_index']}, 分辨率{data['img_width']}x{data['img_height']}")
                    print(f"  当前的: 相机{self.cam_index}, 分辨率{self.img_width}x{self.img_height}")

                    # 询问是否继续使用
                    choice = input("映射矩阵参数不匹配，是否继续使用? (y/n): ").strip().lower()
                    if choice != 'y':
                        return False

            # 加载映射矩阵
            self.map_x = data['map_x']
            self.map_y = data['map_y']
            self.center = tuple(data['center'])
            self.radius = int(data['radius'])

            # 更新输出尺寸（如果文件中的尺寸与当前不同）
            if 'output_width' in data and 'output_height' in data:
                if (data['output_width'] != self.output_width or
                    data['output_height'] != self.output_height):
                    print(f"警告: 输出尺寸不匹配，将使用文件中的尺寸")
                    print(f"  文件中的: {data['output_width']}x{data['output_height']}")
                    print(f"  指定的: {self.output_width}x{self.output_height}")
                    self.output_width = int(data['output_width'])
                    self.output_height = int(data['output_height'])

            print(f"映射矩阵加载成功")
            print(f"  圆心: {self.center}, 半径: {self.radius}")
            print(f"  输出尺寸: {self.output_width}x{self.output_height}")
            return True

        except Exception as e:
            print(f"加载映射矩阵失败: {e}")
            return False

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
        max_radius = min(w, h) // 2
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

    def detect_fisheye_region(self, img):
        """
        保留旧方法作为后备（简单阈值化）
        """
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
            radius = int(radius * 1.0)
        else:
            h, w = gray.shape[:2]
            center = (w // 2, h // 2)
            radius = min(w, h) // 2

        return center, radius

    def get_useful_area(self, img):
        """提取圆形有效区域"""
        mask = np.zeros_like(img)
        cv2.circle(mask, self.center, self.radius, (255, 255, 255), -1)
        return cv2.bitwise_and(img, mask)

    def apply_panorama(self, img):
        """应用横向展开变换"""
        # 使用线性插值和边界填充
        panorama = cv2.remap(img, self.map_x, self.map_y,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0))
        return panorama

    def run(self):
        """主循环"""
        print("开始实时鱼眼横向展开...")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        print("按 'm' 键保存当前映射矩阵")
        print("按 'r' 键重新计算映射矩阵")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # 应用横向展开
            panorama = self.apply_panorama(frame)

            # 获取有效区域
            useful_area = self.get_useful_area(frame)

            # 显示
            display_scale = 0.5
            display_original = cv2.resize(frame,
                                         (int(self.img_width * display_scale),
                                          int(self.img_height * display_scale)))
            display_useful = cv2.resize(useful_area,
                                       (int(self.img_width * display_scale),
                                        int(self.img_height * display_scale)))

            top_row = np.hstack((display_original, display_useful))
            panorama_resized = cv2.resize(panorama, (top_row.shape[1], self.output_height))

            # 添加标签
            cv2.putText(top_row, f"Original ({self.img_width}x{self.img_height})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(top_row, f"Useful Area", (display_original.shape[1] + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(panorama_resized, f"Panorama ({self.output_width}x{self.output_height})",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 在原图上画出检测到的圆
            display_original_with_circle = display_original.copy()
            scaled_center = (int(self.center[0] * display_scale), int(self.center[1] * display_scale))
            scaled_radius = int(self.radius * display_scale)
            cv2.circle(display_original_with_circle, scaled_center, scaled_radius, (0, 0, 255), 2)
            cv2.circle(display_original_with_circle, scaled_center, 3, (0, 0, 255), -1)
            top_row = np.hstack((display_original_with_circle, display_useful))

            combined_display = np.vstack((top_row, panorama_resized))
            cv2.imshow('Fisheye Panorama - Real-time', combined_display)

            # 键盘交互
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                import time
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

    def release(self):
        """释放资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()


# 主程序入口
if __name__ == "__main__":
    # 参数说明：
    # map_file: 映射矩阵文件路径，如果存在则加载，不存在则计算并保存
    # force_recompute: 强制重新计算映射矩阵

    processor = FisheyePanoramaRealtime(
        cam_index=1,
        output_width=3840,
        output_height=1080,
        desired_width=1920,
        desired_height=1080,
        map_file="maps\3840_fisheye_maps.npz",
        force_recompute=False
    )

    try:
        processor.run()
    except KeyboardInterrupt:
        pass
    finally:
        processor.release()
