"""
全局共享状态 — 锁、帧缓冲、性能数据、推理执行器
"""
import asyncio
import threading
import time
from typing import Optional, Dict, Any, Callable, Set
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# ── GPU 推理互斥锁（防止并发 CUDA 调用）
camera_lock = threading.Lock()

# ── 帧缓冲锁（inference 线程写，MJPEG 协程读）
frame_lock = threading.Lock()

# ── 性能数据锁（RLock 避免同线程重入死锁）
perf_lock = threading.RLock()

# ── 单线程推理池，确保 GPU 串行使用
inference_executor = ThreadPoolExecutor(max_workers=1)

# ── 最新帧缓冲（由 inference_and_encode 更新）
latest_original_frame: Optional[np.ndarray] = None
latest_annotated_frame: Optional[np.ndarray] = None

# ── 原始 JPEG 字节（camera_ws.drain_recv 更新，/video/original MJPEG 端点读取）
latest_original_jpeg: Optional[bytes] = None

# ── 全局处理器实例（由 main() 赋值）
processor = None

# ── 推理回调（由 main() 赋值，供 routes.py 的 WebSocket handler 调用）
inference_fn: Optional[Callable] = None

# ── 定期 GPU 显存清理计数器
gpu_clean_counter: int = 0

# ── 预转换好的 WebRTC 帧（av.VideoFrame yuv420p），由推理线程在 inference_executor 中准备好。
# recv() 直接取用，无需在 asyncio 事件循环中做 BGR→YUV 转换，消除事件循环阻塞。
latest_webrtc_frame = None   # type: Optional[Any]  (av.VideoFrame，避免此处 import av)
webrtc_fps: int = 30

# ── 推理结果 JSON 缓冲（inference 线程写，/inference/latest 和 /ws/inference 读）
# 存储为 bytes（json.dumps 后 encode），方便 WebSocket 直接 send_bytes，避免重复序列化。
latest_inference_result: Optional[bytes] = None

# ── asyncio 事件循环引用（startup 时赋值，供推理线程 call_soon_threadsafe 使用）
_frame_event_loop: Optional[asyncio.AbstractEventLoop] = None

# ── 推理结果推流：inference 线程写完后通知各 /ws/inference 协程
_inference_ready_waiters: Set[asyncio.Event] = set()
_inference_waiters_lock = threading.Lock()

# ── 视频录制状态（由 /record/start、/record/stop 接口控制）
record_lock = threading.Lock()
is_recording: bool = False
_video_writer_original = None   # cv2.VideoWriter | None
_video_writer_annotated = None  # cv2.VideoWriter | None
record_filenames: dict = {}     # {'original': str, 'annotated': str}

# ── 性能指标（由 gpu_info.update_perf 更新，由 /performance 接口读取）
performance_data: Dict[str, Any] = {
    "fps": 0.0,
    "_frame_count": 0,
    "_last_fps_time": time.time(),
    "gpu_usage": "N/A",
    "gpu_memory": "N/A",
    "gpu_temp": "N/A",
    "inference_time_ms": 0.0,
    "detected_persons": 0,
    "tracking_ids": [],
    "system_cpu": 0.0,
    "system_memory": 0.0,
    "system_memory_avail_mb": 0.0,   # 系统可用内存 MB，录制期间低于阈值时触发急停
    "connected_clients": 0,
    "theoretical_fps": 0.0,     # 理论最高 FPS = 1000/帧处理耗时，用于判断瓶颈在推理还是摄像头
    # 注：网络耗时由浏览器端 Date.now()-T0 展示，服务端不重复计算
}
