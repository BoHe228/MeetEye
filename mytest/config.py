"""
配置文件，用于管理所有参数和配置
"""
import argparse

OSNET_WEIGHT_MAP = {
    'osnet_ain_x1_0':   'imagenet.pyth/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
    'osnet_x0_25':      'imagenet.pyth/osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth',
    'osnet_ain_x1_0_D': 'imagenet.pyth/osnet_ain_x1_0_dukemtmcreid_256x128_amsgrad_ep90_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth',
}

# torchreid 实际识别的架构名（与 OSNET_WEIGHT_MAP 的 key 一一对应）
# 不同数据集预训练的变体（如 osnet_ain_x1_0_D）底层架构相同，只是权重文件不同
OSNET_ARCH_MAP = {
    'osnet_ain_x1_0':   'osnet_ain_x1_0',
    'osnet_x0_25':      'osnet_x0_25',
    'osnet_ain_x1_0_D': 'osnet_ain_x1_0',   # DukeMTMC 权重，同 osnet_ain_x1_0 架构
}


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
    parser.add_argument('--num-slices', type=int, default=3, choices=[2, 3, 4, 5, 6, 7],
                        help='全景图切片数量 (默认: 3)')
    parser.add_argument('--slice-overlap', type=float, default=0.1,
                        help='切片重叠比例 (默认: 0.1)')
    parser.add_argument('--dedup-use-reid', action=argparse.BooleanOptionalAction, default=False,
                        help='跨切片去重时是否用 ReID 特征辅助判断；'
                             '--no-dedup-use-reid（默认）只用空间 IoU，避免切片边缘裁图质量差导致去重失败 (默认: False)')

    # 角度计算参数
    parser.add_argument('--fit-degree', type=int, default=4, choices=[4, 5],
                        help='俯仰角计算使用的多项式次数 (4 或 5, 默认: 5)')
    parser.add_argument('--calib-yaml', type=str, default=r'mytest/fisheye_calib.yaml',
                        help='鱼眼标定YAML文件路径 (如果不提供则使用内置系数)')

    # YOLO参数
    parser.add_argument('--model-path', type=str, default='./yolo_model/yolo26n-pose.engine', help='YOLO模型路径（.pt 或 .engine）')
    parser.add_argument('--conf-threshold', type=float, default=0.1, help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.99, help='IOU阈值')

    # OSNet 开关与模型选择
    parser.add_argument('--use-osnet', action=argparse.BooleanOptionalAction, default=True,
                        help='是否启用 OSNet 特征提取；--no-use-osnet 完全跳过，节省计算 (默认: True)')
    parser.add_argument('--osnet-model', type=str, default='osnet_ain_x1_0',
                        choices=list(OSNET_WEIGHT_MAP.keys()),
                        help='OSNet ReID 模型选型（默认: osnet_ain_x1_0）')

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

    # 环绕重叠可视化（调试用）
    parser.add_argument('--show-wrap-overlap', action='store_true', default=False,
                        help='在最终显示帧左右各拼接环绕重叠区域副本，验证 slice0/slice2 环绕切片效果')

    # 人脸识别开关
    parser.add_argument('--use-face-rec', action=argparse.BooleanOptionalAction, default=False,
                        help='是否启用人脸识别（AdaFace IR-18，默认: False）')
    parser.add_argument('--face-library-dir', type=str, default='face_library',
                        help='人脸特征库目录，每个 .npy 文件对应一人，文件名即为人名 (默认: face_library)')
    parser.add_argument('--face-rec-model', type=str,
                        default='face_rec_model/adaface_ir18_vgg2.ckpt',
                        help='AdaFace IR-18 模型权重路径')
    parser.add_argument('--face-rec-threshold', type=float, default=0.35,
                        help='人脸识别余弦相似度阈值，低于此值视为未知（默认: 0.35）')
    parser.add_argument('--face-frontal-threshold', type=float, default=0.65,
                        help='正面人脸判断阈值（鼻子水平偏移/眼距），越小越严格（默认: 0.35）')
    parser.add_argument('--face-rec-cooldown', type=int, default=30,
                        help='未识别目标重试间隔帧数，避免每帧触发推理（默认: 30）')

    # 说话检测开关
    parser.add_argument('--talking-detection', action=argparse.BooleanOptionalAction, default=False,
                        help='是否启用说话检测（基于 MediaPipe FaceMesh MAR，需 pip install mediapipe，默认: False）')
    parser.add_argument('--talking-mar-threshold', type=float, default=0.035,
                        help='嘴巴纵横比（MAR）阈值，超过则判定为说话（默认: 0.035，可按场景在 0.03-0.06 间调整）')
    parser.add_argument('--talking-detect-interval', type=int, default=15,
                        help='说话检测跳帧间隔：每隔 N 帧才跑一次 MediaPipe，中间帧复用上次结果（默认: 3）')

    # 画面标注显示开关
    parser.add_argument('--show-id', action=argparse.BooleanOptionalAction, default=True,
                        help='是否在检测框上显示 Track ID (默认: True)')
    parser.add_argument('--show-conf', action=argparse.BooleanOptionalAction, default=True,
                        help='是否在检测框上显示置信度 (默认: True)')
    parser.add_argument('--show-angle', action=argparse.BooleanOptionalAction, default=False,
                        help='是否在画面上显示角度文字标注 (默认: False)')
    parser.add_argument('--show-arrow', action=argparse.BooleanOptionalAction, default=False,
                        help='是否在画面上绘制方向箭头 (默认: False)')

    # 检测框策略
    # --kalman-bbox   : 跟踪输出改用 Kalman 状态框（非 YOLO 原始框），
    #                   且在目标暂时未被检测到时继续用 Kalman 预测框保持显示（灰色细线）
    # --kpt-track     : 跟踪输入（送进 tracker 的 bbox）改用关键点推导框，
    #                   减少大框之间的假性重叠，降低 freeze_feat 误触发
    # --kpt-display   : 仅画面绘制时改用关键点推导框，不影响跟踪逻辑
    parser.add_argument('--kalman-bbox', action='store_true', default=False,
                        help='用 Kalman 状态框替代 YOLO 原始框输出；目标丢失时继续显示预测框（灰色）(默认: False)')
    parser.add_argument('--kpt-track', action='store_true', default=False,
                        help='跟踪层（IoU 匹配 + Kalman 初始化）使用关键点推导框，减少大框重叠误判 (默认: False)')
    parser.add_argument('--kpt-display', action='store_true', default=False,
                        help='显示层用关键点推导框替代原始框绘制，不影响跟踪逻辑 (默认: False)')

    # 关键点框公共参数（--kpt-track 和 --kpt-display 共用）
    parser.add_argument('--kpt-bbox-conf', type=float, default=0.3,
                        help='关键点可见性阈值，低于此值的关键点不参与框推导（默认: 0.3）')
    parser.add_argument('--kpt-bbox-padding', type=float, default=0.3,
                        help='关键点框左右(水平)扩展比例，相对于关键点跨度（默认: 0.2）')
    parser.add_argument('--kpt-bbox-padding-v', type=float, default=0.4,
                        help='关键点框上下(垂直)扩展比例，相对于关键点跨度（默认: 0.3，比左右多扩以包住头顶/下颌）')
    parser.add_argument('--kpt-bbox-upper-only', action=argparse.BooleanOptionalAction, default=True,
                        help='仅用头肩关键点（0-6：鼻/眼/耳/肩）推导框，排除手肘/腕/腿（默认: True）')

    # 跟踪器选择
    parser.add_argument('--tracker', type=str, default='hybridsort',
                        choices=['none', 'botsort', 'hybridsort'],
                        help='跟踪器类型：none=不使用跟踪，botsort=BoT-SORT，hybridsort=Hybrid-SORT (默认: hybridsort)')

    # 检测框平滑（对两种跟踪器均有效）
    parser.add_argument('--smooth-bbox', action='store_true', default=True,
                        help='对输出框做全框 EMA 平滑（中心坐标 + 宽高），减少 YOLO 帧间抖动')
    parser.add_argument('--smooth-bbox-alpha', type=float, default=0.5,
                        help='框宽高平滑的 EMA 系数：0=完全用当前帧，1=完全用历史 (默认: 0.5)')

    # BoT-SORT 跟踪器参数（--tracker botsort 时生效）
    parser.add_argument('--botsort-match-thresh', type=float, default=0.3,
                        help='BoT-SORT匹配阈值')
    parser.add_argument('--appearance-thresh', type=float, default=0.2,
                        help='BoT-SORT外观特征匹配阈值（余弦距离，第一阶段+IoU门控，越小越严格）')
    parser.add_argument('--reid-lost-thresh', type=float, default=0.05,
                        help='纯ReID恢复Lost轨迹的阈值（第三阶段，无IoU门控，须比appearance-thresh更严格）')
    parser.add_argument('--use-hungarian', action=argparse.BooleanOptionalAction, default=True,
                        help='是否使用匈牙利算法进行线性分配，--no-use-hungarian 改用贪心算法')

    # ReID 参数（--tracker hybridsort 时生效）
    parser.add_argument('--use-reid', action=argparse.BooleanOptionalAction, default=True,
                        help='HybridSort 是否启用 ReID 外观特征参与关联，--no-use-reid 禁用 (默认: True)')
    parser.add_argument('--reid-emb-weight-high', type=float, default=0.1,
                        help='HybridSort ReID 第一轮关联外观代价权重（0=纯IoU+VDC，越大越依赖外观，默认: 0.3）')
    parser.add_argument('--reid-emb-weight-low', type=float, default=0.0,
                        help='HybridSort ReID BYTE第二轮关联外观代价权重（默认: 0.0）')

    # JSON 结果保存参数（仅 WebUI 模式生效）
    parser.add_argument('--save-json', action='store_true', default=False,
                        help='将每帧推理结果追加保存为 JSONL 文件（每行一个 JSON，含角度/距离/特征）')
    parser.add_argument('--json-output', type=str, default=None,
                        help='JSONL 输出文件路径（默认：output_dir/视频名或camera_时间戳.jsonl）')

    # 是否使用外部UI
    parser.add_argument('--webui', action='store_true', help='Run with local web UI (browser)')

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
