import argparse


def add_bool_arg(parser, name: str, default: bool, help: str = None):
    """Python 3.8 compatible --flag / --no-flag boolean argument."""
    dest = name.lstrip("-").replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(name, dest=dest, action="store_true", help=help)
    group.add_argument(f"--no-{name[2:]}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


KEYPOINT_COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (0, 255, 255),
    (0, 0, 255),
    (0, 0, 255),
    (255, 0, 0),
    (255, 0, 0),
    (255, 0, 255),
    (255, 0, 255),
    (128, 0, 128),
    (128, 0, 128),
    (0, 165, 255),
    (0, 165, 255),
    (0, 128, 128),
    (0, 128, 128),
    (255, 255, 0),
    (255, 255, 0),
]

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (5, 11),
    (6, 12), (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Edge WebUI runtime: fisheye panorama + YOLO face + HybridSort + sector output"
    )

    parser.add_argument("--webui", action="store_true", default=False,
                        help="启动本地 WebUI 服务；默认关闭，走 headless JSON 输出")
    parser.add_argument("--video-path", type=str, default=None,
                        help="测试视频路径；WebUI 模式不传则默认读取板端本地摄像头")
    parser.add_argument("--camera-device", type=str, default="/dev/video0",
                        help="WebUI 模式本地摄像头设备，默认 /dev/video0；传 none 则等待 /ws/camera")
    parser.add_argument("--camera-width", type=int, default=1920,
                        help="本地摄像头采集宽度")
    parser.add_argument("--camera-height", type=int, default=1080,
                        help="本地摄像头采集高度")
    parser.add_argument("--camera-fps", type=float, default=30.0,
                        help="本地摄像头采集帧率")
    parser.add_argument("--camera-format", type=str, default="mjpeg",
                        choices=["mjpeg", "yuyv", "any"],
                        help="本地摄像头输入格式；mjpeg 优先走 FFmpeg 零重编码")
    parser.add_argument("--output-jsonl", type=str, default=None,
                        help="headless 模式逐帧 JSONL 输出路径；默认写入 output-dir")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="headless 模式最多处理帧数；0 表示处理完整视频")
    parser.add_argument("--cpu-affinity", type=str, default="none",
                        help="主进程 CPU 亲和性；默认关闭，传 4-7 可绑定 RK3588 大核")

    parser.add_argument("--model-path", type=str,
                        default="face_rc/yolo_model/RK3588/yolov8n-face_608_b1_int8_split_rknn_model",
                        help="YOLO face pose 模型路径")
    parser.add_argument("--imgsz", type=int, default=608,
                        help="YOLO/RKNN 推理输入尺寸；608 RKNN 模型应传 608")
    parser.add_argument("--rknn-core-mask", type=str, default="default",
                        choices=["default", "auto", "core0", "core1", "core2", "core01", "core012", "all"],
                        help="RKNNLite NPU core mask；default 不传参，all 使用 NPU_CORE_ALL，core012 强制 0/1/2")
    parser.add_argument("--rknn-parallel-slices", action="store_true", default=False,
                        help="使用 batch=1 RKNN 模型创建 3 个 RKNNLite 实例，分别绑定 core0/core1/core2 并行跑 3 个切片")
    parser.add_argument("--conf-threshold", type=float, default=0.1)
    parser.add_argument("--iou-threshold", type=float, default=0.99)

    parser.add_argument("--output-width", type=int, default=2560)
    parser.add_argument("--output-height", type=int, default=720)
    parser.add_argument("--process-width", type=int, default=0,
                        help="裁剪后全景处理宽度；0 表示不额外缩放")
    parser.add_argument("--vertical-fov", type=float, default=100.0)
    parser.add_argument("--map-file", type=str,
                        default="face_rc/maps/6.22_2560.npz")
    parser.add_argument("--direct-slice-remap", action="store_true", default=False,
                        help="headless 模式直接从鱼眼 remap 到 YOLO 输入切片，跳过完整全景展开/切片/letterbox")
    parser.add_argument("--direct-slice-remap-backend", type=str, default="cpu",
                        choices=["cpu", "opencl"],
                        help="--direct-slice-remap 的 remap 后端；opencl 使用 OpenCV UMat/OpenCL，默认 cpu")
    parser.add_argument("--direct-slice-map-file", type=str,
                        default="face_rc/maps/6.22_2560_yolo_slices_608.npz",
                        help="--direct-slice-remap 使用的切片映射矩阵 npz")
    parser.add_argument("--headless-pipeline-parallel", action="store_true", default=False,
                        help="headless + direct-slice 模式下并行重叠 detection 与上一帧 tracker")
    parser.add_argument("--headless-pipeline-queue-size", type=int, default=2,
                        help="--headless-pipeline-parallel 的 detection 结果队列深度；1 接近旧的单 pending 行为")
    parser.add_argument("--headless-remap-prefetch", action="store_true", default=False,
                        help="headless pipeline 中提前提交下一帧 OpenCL direct-slice remap，与当前帧 YOLO 重叠")
    parser.add_argument("--fit-degree", type=int, default=4, choices=[4, 5])
    parser.add_argument("--calib-yaml", type=str, default="face_rc/fisheye_calib.yaml")
    parser.add_argument("--crop-divisor", type=int, default=3,
                        help="裁剪全景图顶部区域的分母；3 表示裁掉上方 1/3，0 表示不裁剪")

    parser.add_argument("--num-slices", type=int, default=3, choices=[2, 3, 4, 5, 6, 7])
    parser.add_argument("--slice-overlap", type=float, default=0.1)
    add_bool_arg(parser, "--dedup-use-reid", default=False)

    parser.add_argument("--tracker", type=str, default="hybridsort",
                        choices=["none", "hybridsort"])
    parser.add_argument("--track-buffer", type=int, default=500,
                        help="HybridSort 轨迹保留帧数；越大越稳但轨迹池越大、Kalman predict 越慢")
    parser.add_argument("--tracker-match-thresh", type=float, default=0.15,
                        help="HybridSort IoU 匹配阈值")
    parser.add_argument("--tracker-new-thresh", type=float, default=0.5,
                        help="HybridSort 新轨迹置信度阈值")
    parser.add_argument("--new-track-overlap-thresh", type=float, default=0.6,
                        help="新轨迹与现有轨迹 IoU 超过该值时不新建；1.0 表示关闭")
    add_bool_arg(parser, "--tracker-byte", default=True,
                 help="启用 HybridSort BYTE 低分检测二阶段关联")
    parser.add_argument("--coast-frames", type=int, default=0)
    parser.add_argument("--coast-hold", action="store_true", default=False)
    add_bool_arg(parser, "--smooth-bbox", default=True)
    parser.add_argument("--smooth-bbox-alpha", type=float, default=0.5)

    add_bool_arg(parser, "--use-osnet", default=False,
                 help="兼容旧命令；精简部署版不加载 OSNet")

    parser.add_argument("--sector-output", action="store_true", default=False)
    parser.add_argument("--num-sectors", type=int, default=8)
    parser.add_argument("--show-sectors", action="store_true", default=False)

    add_bool_arg(parser, "--show-id", default=True)
    add_bool_arg(parser, "--show-conf", default=True)
    add_bool_arg(parser, "--show-kpt", default=False)
    add_bool_arg(parser, "--show-angle", default=False)
    add_bool_arg(parser, "--show-arrow", default=False)
    parser.add_argument("--face-kpt", action="store_true", default=False)
    parser.add_argument("--kpt-track", action="store_true", default=False)
    parser.add_argument("--kpt-display", action="store_true", default=False)
    parser.add_argument("--kpt-bbox-conf", type=float, default=0.3)
    parser.add_argument("--kpt-bbox-padding", type=float, default=0.3)
    parser.add_argument("--kpt-bbox-padding-v", type=float, default=0.4)
    add_bool_arg(parser, "--kpt-bbox-upper-only", default=True)

    parser.add_argument("--webrtc-fps", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default="face_rc_output")
    parser.add_argument("--save-video", action="store_true", default=False)
    add_bool_arg(parser, "--save-original-video", default=False)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--video-name", type=str, default=None)
    parser.add_argument("--profile-interval", type=int, default=30,
                        help="每 N 帧打印一次分阶段耗时；设为 1 每帧打印，0 关闭")

    return parser.parse_args()
