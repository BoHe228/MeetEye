"""
配置文件，用于管理所有参数和配置
"""
import argparse

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='鱼眼展开与YOLO姿态检测系统')
    
    # 输入源参数
    parser.add_argument('--cam-index', type=int, default=1, help='摄像头索引')
    parser.add_argument('--video-path', type=str, default=None, help='视频文件路径（如果提供则使用视频，否则使用摄像头）')
    parser.add_argument('--folder-path', type=str, default=None, help='图片文件夹路径（如果提供则处理文件夹中的所有图片）')
    
    # 摄像头参数（当使用摄像头时有效）
    parser.add_argument('--cam-width', type=int, default=1920, help='摄像头宽度')
    parser.add_argument('--cam-height', type=int, default=1080, help='摄像头高度')
    
    # 鱼眼展开参数
    parser.add_argument('--output-width', type=int, default=3840, help='输出宽度')
    parser.add_argument('--output-height', type=int, default=1080, help='输出高度')
    parser.add_argument('--vertical-fov', type=float, default=100.0, help='垂直视场角')
    parser.add_argument('--map-file', type=str, default=r'maps/3840_fisheye_maps_2026.5.18.npz', help='映射文件路径')

    # 全景切片参数
    parser.add_argument('--num-slices', type=int, default=3, choices=[2, 3, 4, 5, 6,7],
                        help='全景图切片数量 (默认: 3)')
    parser.add_argument('--slice-overlap', type=float, default=0.1,
                        help='切片重叠比例 (默认: 0.1)')

    # 角度计算参数
    parser.add_argument('--fit-degree', type=int, default=4, choices=[4, 5],
                        help='俯仰角计算使用的多项式次数 (4 或 5, 默认: 5)')
    parser.add_argument('--calib-yaml', type=str, default=r'mytest/fisheye_calib.yaml',
                        help='鱼眼标定YAML文件路径 (如果不提供则使用内置系数)')
    
    # YOLO参数
    parser.add_argument('--model-path', type=str, default='./yolo26n-pose.engine', help='YOLO模型路径（.pt 或 .engine）')
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.99, help='IOU阈值')
    
    # 显示参数
    parser.add_argument('--display-scale', type=float, default=0.5, help='显示缩放比例')
    parser.add_argument('--output-dir', type=str, default='yolo_pose_output', help='输出目录')
    
    # 性能参数
    parser.add_argument('--save-frames', action='store_true', help='保存处理后的帧')
    parser.add_argument('--save-crops', action='store_true', help='保存每个检测框的内容（抠图）')
    parser.add_argument('--show-fps', action='store_true', default=False, help='显示FPS')

    # 显示窗口参数
    parser.add_argument('--use-dual-windows', action='store_true', default=False,
                        help='是否使用双窗口显示：一个窗口只显示YOLO检测结果，另一个显示最终结果 (默认: False)')
    parser.add_argument('--no-display', action='store_true', default=True,
                        help='不显示窗口（Linux无显示器环境使用） (默认: False)')

    # 视频保存参数
    parser.add_argument('--save-video', action='store_true', default=False,
                        help='是否保存检测结果为视频 (默认: False)')
    parser.add_argument('--video-fps', type=float, default=30.0,
                        help='保存视频的帧率 (默认: 30 FPS)')
    parser.add_argument('--video-name', type=str, default=None,
                        help='保存视频的文件名 (默认: 自动生成带时间戳的文件名)')
    parser.add_argument('--yolo-video-name', type=str, default=None,
                        help='双窗口模式下YOLO检测结果视频的文件名 (默认: 自动生成)')

    # 顶部裁剪参数
    parser.add_argument('--crop-divisor', type=int, default=3,
                        help='裁剪全景图正上方区域的分母，0表示不裁剪，3表示裁剪1/3，以此类推 (默认: 0)')

    # BoT-SORT跟踪器参数
    parser.add_argument('--use-deep-sort', action=argparse.BooleanOptionalAction, default=True,
                        help='是否使用BoT-SORT跟踪器（结合运动和外观特征），--no-use-deep-sort 禁用')
    parser.add_argument('--deep-sort-match-thresh', type=float, default=0.3,
                        help='BoT-SORT匹配阈值')
    parser.add_argument('--appearance-thresh', type=float, default=0.4,
                        help='BoT-SORT外观特征匹配阈值（余弦距离，越小越严格，建议 0.35-0.45）')
    parser.add_argument('--use-hungarian', action=argparse.BooleanOptionalAction, default=True,
                        help='是否使用匈牙利算法进行线性分配，--no-use-hungarian 改用贪心算法')

    # 是否使用外部UI
    parser.add_argument('--webui',action='store_true',help='Run with local web UI (browser)')

    return parser.parse_args()

# 关键点颜色映射
KEYPOINT_COLORS = [
    (0, 255, 0),    # 绿色 - 鼻子
    (0, 255, 255),  # 黄色 - 左眼
    (0, 255, 255),  # 黄色 - 右眼
    (0, 0, 255),    # 红色 - 左耳
    (0, 0, 255),    # 红色 - 右耳
    (255, 0, 0),    # 蓝色 - 左肩
    (255, 0, 0),    # 蓝色 - 右肩
    (255, 0, 255),  # 紫色 - 左肘
    (255, 0, 255),  # 紫色 - 右肘
    (128, 0, 128),  # 深紫色 - 左手腕
    (128, 0, 128),  # 深紫色 - 右手腕
    (0, 165, 255),  # 橙色 - 左髋
    (0, 165, 255),  # 橙色 - 右髋
    (0, 128, 128),  # 青色 - 左膝
    (0, 128, 128),  # 青色 - 右膝
    (255, 255, 0),  # 青色 - 左脚踝
    (255, 255, 0)   # 青色 - 右脚踝
]

# 骨架连接
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),  # 鼻子到眼睛
    (1, 3), (2, 4),  # 眼睛到耳朵
    (5, 6),  # 肩膀连接
    (5, 7), (6, 8),  # 肩膀到手肘
    (7, 9), (8, 10),  # 手肘到手腕
    (5, 11), (6, 12),  # 肩膀到髋部
    (11, 12),  # 髋部连接
    (11, 13), (12, 14),  # 髋部到膝盖
    (13, 15), (14, 16)  # 膝盖到脚踝
]