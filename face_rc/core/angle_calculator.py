"""
角度计算模块
将关键点坐标转换为角度（俯仰角、水平角）并显示
"""
import numpy as np
from typing import List, Dict, Optional, Tuple

# 可选导入cv2，用于绘图功能
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class FisheyeConverter:
    """鱼眼相机角度转换器（从 use_calib.py 复制）"""

    def __init__(self, yaml_file=None, fit_degree=None):
        """
        初始化转换器

        参数:
            yaml_file: YAML文件路径
            fit_degree: 使用的多项式次数，4 或 5。如果不指定则使用YAML中的默认值
        """
        self.fit_degree = fit_degree
        self.cx = 922.0
        self.cy = 564.0

        if yaml_file:
            self._load_from_yaml(yaml_file)
        else:
            self._use_default_coeffs()

    def _load_from_yaml(self, yaml_file):
        """从YAML文件加载参数"""
        try:
            import yaml
            with open(yaml_file, 'r', encoding='utf-8') as f:
                calib = yaml.load(f, Loader=yaml.FullLoader)

            # 读取圆心
            self.cx = calib['center']['x']
            self.cy = calib['center']['y']

            # 确定使用的次数
            if self.fit_degree is None:
                self.fit_degree = calib.get('default_fit', 5)

            # 加载对应次数的系数
            if self.fit_degree == 4:
                self.coeffs = calib['fit_4']['coefficients']
                self.r2_score = calib['fit_4']['r2_score']
            elif self.fit_degree == 5:
                self.coeffs = calib['fit_5']['coefficients']
                self.r2_score = calib['fit_5']['r2_score']
            else:
                raise ValueError(f"不支持的多项式次数: {self.fit_degree}，请使用 4 或 5")

            self.degree = self.fit_degree
            print(f"已从 {yaml_file} 加载标定参数")
        except Exception as e:
            print(f"加载YAML失败: {e}，使用默认系数")
            self._use_default_coeffs()

    def _use_default_coeffs(self):
        """使用默认的系数"""
        if self.fit_degree is None:
            self.fit_degree = 5

        if self.fit_degree == 4:
            self.coeffs = [
                -2.405707485718247e-09,
                1.9929171857454457e-06,
                -0.0005816463691548895,
                -0.1157626905427259,
                89.28693676949699
            ]
            self.r2_score = 0.999418
        elif self.fit_degree == 5:
            self.coeffs = [
                -1.1658012518547278e-11,
                1.1878284840943126e-08,
                -4.2396349229203516e-06,
                0.0005668408282723097,
                -0.19620578806477496,
                90.56394245932009
            ]
            self.r2_score = 0.999585
        else:
            raise ValueError(f"不支持的多项式次数: {self.fit_degree}，请使用 4 或 5")

        self.degree = self.fit_degree
        print(f"使用内置的{self.degree}次多项式系数")

    def set_center(self, cx, cy):
        """设置圆心坐标"""
        self.cx = cx
        self.cy = cy

    def set_fit_degree(self, degree):
        """切换多项式次数 (4或5)"""
        if degree not in (4, 5):
            raise ValueError("只支持4次或5次多项式")
        self.fit_degree = degree
        self._use_default_coeffs()

    def pixel_to_radius(self, px, py):
        """像素坐标转半径"""
        return np.sqrt((px - self.cx)**2 + (py - self.cy)**2)

    def radius_to_angle(self, r):
        """半径转角度"""
        return np.polyval(self.coeffs, r)

    def pixel_to_angle(self, px, py):
        """
        像素坐标直接转角度

        参数:
            px, py: 像素坐标

        返回:
            角度（度）
        """
        r = self.pixel_to_radius(px, py)
        return self.radius_to_angle(r)


class AngleCalculator:
    """角度计算器"""

    def __init__(self, panorama_width: int, panorama_height: int, vertical_fov: float = 200.0,
                 fisheye_center: Optional[Tuple[float, float]] = None, fit_degree: int = 5,
                 yaml_file: Optional[str] = None, feature_point_mode: str = 'nose'):
        """
        初始化角度计算器

        参数:
            panorama_width: 全景图宽度
            panorama_height: 全景图高度
            vertical_fov: 垂直视场角（度）
            fisheye_center: 鱼眼图像圆心坐标 (cx, cy)
            fit_degree: 使用的多项式次数，4 或 5
            yaml_file: 标定文件路径
        """
        self.panorama_width = panorama_width
        self.panorama_height = panorama_height
        self.vertical_fov = vertical_fov
        self.crop_top_offset = 0  # 顶部裁剪偏移量（像素）

        # 鱼眼转换相关参数
        self.fisheye_center = fisheye_center or (922.0, 564.0)
        self.fisheye_radius = None
        self.panorama_to_fisheye_map = None

        # 预计算映射表（来自 FisheyePanorama/FisheyePanoramaGPU）
        self.panorama_map_x: Optional[np.ndarray] = None
        self.panorama_map_y: Optional[np.ndarray] = None

        # 初始化鱼眼角度转换器
        self.fisheye_converter = FisheyeConverter(yaml_file=yaml_file, fit_degree=fit_degree)
        if fisheye_center:
            self.fisheye_converter.set_center(fisheye_center[0], fisheye_center[1])

        # 关键点索引定义（与config.py保持一致）
        self.NOSE_INDEX = 0
        self.LEFT_EYE_INDEX = 1
        self.RIGHT_EYE_INDEX = 2
        self.LEFT_EAR_INDEX = 3
        self.RIGHT_EAR_INDEX = 4

        # 角度特征点来源：
        #   'nose'  —— COCO pose（17 点），用鼻子(idx 0)
        #   'mouth' —— 人脸模型 yolov8n-face（5 点：左眼/右眼/鼻子/左嘴角/右嘴角），
        #              用左右嘴角(idx 3,4)中点。两嘴角无效时回退到人脸鼻子(idx 2)，
        #              再回退到 idx 0（兼容补漏框的合成点）。
        self.feature_point_mode = feature_point_mode
        self.FACE_NOSE_INDEX = 2
        self.FACE_LEFT_MOUTH_INDEX = 3
        self.FACE_RIGHT_MOUTH_INDEX = 4

    def set_crop_offset(self, crop_top_offset: int):
        """
        设置顶部裁剪偏移量

        Args:
            crop_top_offset: 从顶部裁剪掉的像素数
        """
        self.crop_top_offset = crop_top_offset

    @staticmethod
    def _kp_valid(kp) -> bool:
        """关键点有效：存在、非全零、置信度>0（缺置信度列时按存在算）。"""
        if kp is None:
            return False
        arr = np.asarray(kp, dtype=float)
        if arr.size < 2 or (arr[:2] == 0).all():
            return False
        if arr.size >= 3 and arr[2] <= 0:
            return False
        return True

    def _pick_feature_point(self, person_kpts) -> Optional[Tuple[float, float]]:
        """
        返回用于算角度的特征点 (x, y)，无可用点返回 None。

        mouth 模式（人脸模型）：左右嘴角中点 → 人脸鼻子(idx2) → idx0（合成点）回退。
        nose  模式（pose）：鼻子(idx0)。
        """
        n = len(person_kpts)
        if self.feature_point_mode == 'mouth':
            lm = person_kpts[self.FACE_LEFT_MOUTH_INDEX] if n > self.FACE_LEFT_MOUTH_INDEX else None
            rm = person_kpts[self.FACE_RIGHT_MOUTH_INDEX] if n > self.FACE_RIGHT_MOUTH_INDEX else None
            if self._kp_valid(lm) and self._kp_valid(rm):
                return (float(lm[0]) + float(rm[0])) / 2.0, (float(lm[1]) + float(rm[1])) / 2.0
            # 一个嘴角可用就用它，避免半遮挡时丢点
            if self._kp_valid(lm):
                return float(lm[0]), float(lm[1])
            if self._kp_valid(rm):
                return float(rm[0]), float(rm[1])
            # 回退：人脸鼻子(idx2)
            fn = person_kpts[self.FACE_NOSE_INDEX] if n > self.FACE_NOSE_INDEX else None
            if self._kp_valid(fn):
                return float(fn[0]), float(fn[1])
            # 再回退：idx0（补漏框合成点）
            z = person_kpts[self.NOSE_INDEX] if n > self.NOSE_INDEX else None
            if self._kp_valid(z):
                return float(z[0]), float(z[1])
            return None

        # nose 模式
        nose_kp = person_kpts[self.NOSE_INDEX] if n > self.NOSE_INDEX else None
        if self._kp_valid(nose_kp):
            return float(nose_kp[0]), float(nose_kp[1])
        return None

    def set_panorama_maps(self, map_x: np.ndarray, map_y: np.ndarray) -> None:
        """
        注入来自展开器的预计算映射表，供 panorama_to_fisheye() 直接查表使用。
        map_x[y, x] / map_y[y, x] 存储全景像素 (x, y) 对应的原始鱼眼像素坐标。
        """
        self.panorama_map_x = map_x
        self.panorama_map_y = map_y

    def set_fisheye_mapping(self, center: Tuple[float, float], radius: float,
                           original_width: int, original_height: int):
        """
        设置鱼眼映射参数，用于将全景坐标映射回原始鱼眼坐标

        Args:
            center: 鱼眼圆心 (cx, cy)
            radius: 鱼眼半径
            original_width: 原始鱼眼图像宽度
            original_height: 原始鱼眼图像高度
        """
        self.fisheye_center = center
        self.fisheye_radius = radius
        self.fisheye_converter.set_center(center[0], center[1])
        print(f"设置鱼眼映射: 圆心{center}, 半径{radius}")

    def panorama_to_fisheye(self, x_panorama: float, y_panorama: float) -> Tuple[float, float]:
        """
        将全景图坐标映射回原始鱼眼图像坐标

        Args:
            x_panorama: 全景图中的x坐标
            y_panorama: 全景图中的y坐标

        Returns:
            (x_fisheye, y_fisheye): 原始鱼眼图像中的坐标
        """
        # 如果有裁剪，先加回偏移（还原为完整全景中的行索引）
        y_original = y_panorama + self.crop_top_offset

        # 优先：直接查预计算映射表，避免重复三角运算
        if self.panorama_map_x is not None and self.panorama_map_y is not None:
            xi = int(np.clip(x_panorama, 0, self.panorama_map_x.shape[1] - 1))
            yi = int(np.clip(y_original, 0, self.panorama_map_x.shape[0] - 1))
            return float(self.panorama_map_x[yi, xi]), float(self.panorama_map_y[yi, xi])

        # 降级：映射表未注入时，用三角函数重新计算
        angle = 2 * np.pi * x_panorama / self.panorama_width
        r_ratio = np.clip(y_original / (self.panorama_height - 1), 0.0, 1.0)
        radius = self.fisheye_radius if self.fisheye_radius is not None else 500.0
        cx, cy = self.fisheye_center
        x_fisheye = cx + r_ratio * radius * np.cos(angle)
        y_fisheye = cy + r_ratio * radius * np.sin(angle)

        return x_fisheye, y_fisheye
        
    def calculate_angles_from_keypoints(self, keypoints: np.ndarray) -> Dict[str, List[Optional[Dict]]]:
        """
        从关键点计算每个人的角度
        """
        if keypoints is None or len(keypoints) == 0:
            return {'persons': []}
        
        persons_angles = []
        
        for person_idx, person_kpts in enumerate(keypoints):
            person_kpts = np.asarray(person_kpts, dtype=np.float32)
            # 确保关键点是二维数组
            if person_kpts.ndim == 1:
                person_kpts = person_kpts.reshape(-1, 3)
            
            # 检查关键点数量
            if person_kpts.shape[0] < 5:  # 至少需要5个关键点（鼻子、双眼、双耳）
                print(f"警告：人员 {person_idx} 的关键点数量不足: {person_kpts.shape[0]}")
                persons_angles.append(None)
                continue
            
            # 选取角度特征点：pose→鼻子；人脸模型→嘴巴中心（带回退）
            feat = self._pick_feature_point(person_kpts)
            if feat is None:
                persons_angles.append(None)
                continue

            try:
                nose_x, nose_y = feat

                # 计算角度
                azimuth_deg, elevation_deg = self._pixel_to_angle(nose_x, nose_y)
                
                
                # 存储角度信息
                angle_info = {
                    'person_id': person_idx,
                    'nose_position': (nose_x, nose_y),
                    'azimuth_deg': azimuth_deg,
                    'elevation_deg': elevation_deg,
                    'visible': True
                }
                
                persons_angles.append(angle_info)
            except (IndexError, ValueError, TypeError) as e:
                print(f"处理人员 {person_idx} 的角度时出错: {e}")
                persons_angles.append(None)
        
        return {'persons': persons_angles}

    def calculate_angle_from_point(self, x: float, y: float, person_id: int = 0) -> Dict:
        """
        从全景图上的单个特征点计算角度。

        用于关键点不可用时的 bbox 兜底，保证检测目标仍能参与扇区聚合。
        """
        px = float(x)
        py = float(y)
        azimuth_deg, elevation_deg = self._pixel_to_angle(px, py)
        return {
            'person_id': person_id,
            'nose_position': (px, py),
            'azimuth_deg': azimuth_deg,
            'elevation_deg': elevation_deg,
            'visible': True,
        }
    
    def _pixel_to_angle(self, x: float, y: float) -> Tuple[float, float]:
        """
        将像素坐标转换为角度（考虑裁剪偏移量）
        水平角保持原方法不变，俯仰角使用标定多项式计算

        参数:
            x: 像素x坐标（相对于裁剪后图像）
            y: 像素y坐标（相对于裁剪后图像）

        返回:
            (azimuth_deg, elevation_deg) 水平角和俯仰角（度）
        """
        # ========== 水平角：保持原有计算方法不变 ==========
        azimuth_deg = 360.0 * (x / self.panorama_width)
        azimuth_deg = azimuth_deg % 360.0

        # ========== 俯仰角：使用标定多项式计算 ==========
        # 步骤1: 将全景图坐标映射回原始鱼眼图像坐标
        x_fisheye, y_fisheye = self.panorama_to_fisheye(x, y)

        # 步骤2: 使用鱼眼转换器计算俯仰角
        elevation_deg = self.fisheye_converter.pixel_to_angle(x_fisheye, y_fisheye)

        return azimuth_deg, elevation_deg
    
    
    def draw_angles_on_image(self, image: np.ndarray, angle_info: Dict,
                             show_angle: bool = True, show_arrow: bool = True) -> np.ndarray:
        """
        在图像上绘制角度信息
        
        参数:
            image: 输入图像
            angle_info: 角度信息字典
        
        返回:
            绘制了角度信息的图像
        """
        annotated_image = image.copy()
        persons_angles = angle_info.get('persons', [])
        
        for angle_data in persons_angles:
            if angle_data is None or not angle_data.get('visible', False):
                continue
            
            person_id = angle_data['person_id']
            nose_x, nose_y = angle_data['nose_position']
            azimuth = angle_data['azimuth_deg']
            elevation = angle_data['elevation_deg']
            head_direction = angle_data.get('head_direction', 'unknown')
            
            # 绘制鼻子位置
            cv2.circle(annotated_image, (int(nose_x), int(nose_y)), 6, (0, 255, 255), -1)  # 黄色
            cv2.circle(annotated_image, (int(nose_x), int(nose_y)), 8, (0, 165, 255), 2)   # 橙色边框

            if show_angle:
                # 创建角度文本
                angle_text = f"P{person_id}: A={azimuth:.1f}, E={elevation:.1f}"

                # 添加头部方向信息
                if head_direction != "unknown":
                    angle_text += f" ({head_direction})"

                # 文本位置（在鼻子下方）
                text_x = int(nose_x)
                text_y = int(nose_y) + 25

                # 绘制文本背景
                (text_width, text_height), baseline = cv2.getTextSize(
                    angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
                )

                # 背景矩形 - 黄色背景
                cv2.rectangle(
                    annotated_image,
                    (text_x - 5, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + 5),
                    (0, 255, 255),  # 黄色背景
                    -1
                )

                # 绘制角度文本 - 黑色文字
                cv2.putText(
                    annotated_image,
                    angle_text,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),  # 黑色文本
                    2
                )

            if show_arrow:
                # 绘制角度指示线（从中心指向鼻子）
                center_x = self.panorama_width // 2
                center_y = self.panorama_height // 2

                cv2.arrowedLine(
                    annotated_image,
                    (center_x, center_y),
                    (int(nose_x), int(nose_y)),
                    (255, 0, 0),  # 蓝色箭头
                    2,
                    tipLength=0.05
                )
        
        return annotated_image
    
    def draw_angle_overview(self, image: np.ndarray, angle_info: Dict) -> np.ndarray:
        """
        在图像上绘制角度概览信息
        
        参数:
            image: 输入图像
            angle_info: 角度信息字典
        
        返回:
            绘制了角度概览的图像
        """
        annotated_image = image.copy()
        persons_angles = angle_info.get('persons', [])
        
        visible_count = sum(1 for p in persons_angles if p is not None and p.get('visible', False))
        
        if visible_count == 0:
            return annotated_image
        
        # 添加概览标题
        overview_title = f"Angle Overview ({visible_count} persons detected)"

        # 绘制标题背景
        (title_width, title_height), _ = cv2.getTextSize(
            overview_title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        cv2.rectangle(
            annotated_image,
            (5, 5),
            (15 + title_width, 35),
            (0, 255, 255),
            -1
        )
        cv2.putText(
            annotated_image,
            overview_title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        # 为每个检测到的人添加角度信息
        y_offset = 60
        for angle_data in persons_angles:
            if angle_data is None or not angle_data.get('visible', False):
                continue

            person_id = angle_data['person_id']
            azimuth = angle_data['azimuth_deg']
            elevation = angle_data['elevation_deg']
            head_direction = angle_data.get('head_direction', 'unknown')

            # 格式化角度信息
            angle_text = f"Person {person_id}: Azimuth={azimuth:6.1f}°, Elevation={elevation:6.1f}°, Direction={head_direction}"

            # 绘制文本背景
            (text_width, text_height), _ = cv2.getTextSize(
                angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
            )
            cv2.rectangle(
                annotated_image,
                (5, y_offset - text_height - 5),
                (15 + text_width, y_offset + 5),
                (0, 255, 255),
                -1
            )

            # 绘制文本
            cv2.putText(
                annotated_image,
                angle_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                2
            )

            y_offset += 35
        
        return annotated_image
    
    def get_angle_statistics(self, angle_info: Dict) -> Dict:
        """
        获取角度统计信息
        
        返回:
            统计信息字典
        """
        persons_angles = angle_info.get('persons', [])
        visible_angles = [p for p in persons_angles if p is not None and p.get('visible', False)]
        
        if not visible_angles:
            return {
                'count': 0,
                'avg_azimuth': 0,
                'avg_elevation': 0,
                'azimuth_range': (0, 0),
                'elevation_range': (0, 0)
            }
        
        azimuths = [p['azimuth_deg'] for p in visible_angles]
        elevations = [p['elevation_deg'] for p in visible_angles]
        
        return {
            'count': len(visible_angles),
            'avg_azimuth': np.mean(azimuths),
            'avg_elevation': np.mean(elevations),
            'azimuth_range': (min(azimuths), max(azimuths)),
            'elevation_range': (min(elevations), max(elevations)),
            'azimuth_std': np.std(azimuths),
            'elevation_std': np.std(elevations)
        }


if __name__ == "__main__":
    """测试角度计算功能"""
    print("=" * 60)
    print("AngleCalculator 角度计算测试")
    print("=" * 60)

    # 模拟参数
    panorama_width = 3840
    panorama_height = 1080
    fisheye_center = (922, 564)  # 鱼眼圆心
    fisheye_radius = 494
    vertical_fov = 100.0

    # 创建角度计算器
    calc = AngleCalculator(
        panorama_width=panorama_width,
        panorama_height=panorama_height,
        vertical_fov=vertical_fov,
        fisheye_center=fisheye_center,
        fit_degree=4
    )

    # 设置鱼眼映射参数
    calc.set_fisheye_mapping(
        center=fisheye_center,
        radius=fisheye_radius,
        original_width=1920,
        original_height=1080
    )

    # 测试几个特殊像素点
    test_points = [
        # (描述, x, y)
        ("全景中心点", panorama_width // 2, panorama_height // 2),
        ("全景左上角", 0, 0),
        ("全景右上角", panorama_width - 1, 0),
        ("全景左下角", 0, panorama_height - 1),
        ("全景右下角", panorama_width - 1, panorama_height - 1),
        ("全景1/4位置", panorama_width // 4, panorama_height // 2),
        ("全景3/4位置", panorama_width * 3 // 4, panorama_height // 2),
        ("全景顶部中点", panorama_width // 2, 0),
        ("全景底部中点", panorama_width // 2, panorama_height - 1),
        ("左侧边缘附近", 100, panorama_height // 2),
        ("右侧边缘附近", panorama_width - 100, panorama_height // 2),
    ]

    print(f"\n全景尺寸: {panorama_width} x {panorama_height}")
    print(f"鱼眼圆心: {fisheye_center}")
    print(f"鱼眼半径: {fisheye_radius}")
    print(f"\n测试点结果:")
    print("-" * 85)
    print(f"{'描述':<20} {'全景坐标':<20} {'鱼眼坐标':<20} {'水平角(°)':<12} {'俯仰角(°)':<12}")
    print("-" * 85)

    for desc, x, y in test_points:
        # 计算角度
        azimuth, elevation = calc._pixel_to_angle(x, y)
        # 获取鱼眼坐标
        fx, fy = calc.panorama_to_fisheye(x, y)

        print(f"{desc:<20} ({x:4d}, {y:4d})    ({fx:6.1f}, {fy:6.1f})    {azimuth:10.1f}    {elevation:12.1f}")
