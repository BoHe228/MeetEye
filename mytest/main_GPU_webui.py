"""
鱼眼全景 YOLO 姿态检测 — GPU 版 WebUI
架构: 本地摄像头 → WebSocket → GPU推理 → MJPEG 展示

子模块:
  webui/state.py      全局状态（锁、帧缓冲、性能数据）
  webui/gpu_info.py   GPU 信息查询 + 性能更新
  webui/processor.py  FisheyePanoramaYOLOPose（推理流水线，含逐步耗时）
  webui/routes.py     FastAPI 路由（WebSocket / MJPEG / REST / 首页）
  webui/html_page.py  前端页面 HTML
"""
import json
import logging
import socket
import threading
import time
from typing import Optional

import av
import cv2
import numpy as np
import torch
import uvicorn


class _FilterInvalidHTTP(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return 'Invalid HTTP request' not in record.getMessage()

logging.getLogger('uvicorn.error').addFilter(_FilterInvalidHTTP())

import config
import webui.state as ws
from webui.gpu_info import update_perf, start_gpu_monitor
from webui.processor import FisheyePanoramaYOLOPose
from webui.routes import app


# ── WebRTC 预转换（在推理线程完成，不在 asyncio 事件循环中执行）────────
_WEBRTC_MAX_W = 3840


def _make_webrtc_frame(bgr: np.ndarray) -> av.VideoFrame:
    """BGR ndarray → YUV420P av.VideoFrame，耗时操作移出事件循环。"""
    h, w = bgr.shape[:2]
    if w > _WEBRTC_MAX_W:
        bgr = cv2.resize(bgr, (_WEBRTC_MAX_W, int(h * _WEBRTC_MAX_W / w)),
                         interpolation=cv2.INTER_AREA)
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return av.VideoFrame.from_ndarray(yuv, format='yuv420p')


# ── 推理结果序列化（在 inference_executor 单线程中调用，天然无竞争）────
_frame_id_counter = 0


def _build_inference_json(tracked: list, angle_info: dict) -> bytes:
    """
    将推理结果序列化为 JSON bytes。
    返回值直接存入 state，可被 /inference/latest 和 /ws/inference 读取。

    track_id ↔ angle_info 对齐方式：
      angle_info['persons'][j] 对应 tracked 中第 j 个有 keypoints 的目标，
      因此用过滤后的下标重建 (tracked_index → angle_data) 映射。
    """
    global _frame_id_counter
    _frame_id_counter += 1

    # 建立 tracked 下标 → 角度数据 的映射
    angle_by_tracked_idx: dict = {}
    if angle_info:
        persons = angle_info.get('persons', [])
        kpt_indices = [i for i, d in enumerate(tracked or []) if d.get('keypoints')]
        for j, ti in enumerate(kpt_indices):
            if j < len(persons) and persons[j] is not None:
                angle_by_tracked_idx[ti] = persons[j]

    targets: dict = {}
    for i, det in enumerate(tracked or []):
        tid = int(det.get('track_id', i + 1))
        angle = angle_by_tracked_idx.get(i)

        feat = det.get('feature')
        if feat is not None:
            # feature shape: [1, feat_dim] (numpy)，展平为 1-D list
            feat_list = feat.reshape(-1).tolist()
        else:
            feat_list = []

        targets[str(tid)] = {
            'id': tid,
            'azimuth':   round(float(angle['azimuth_deg']),   3) if angle else None,
            'elevation': round(float(angle['elevation_deg']), 3) if angle else None,
            'features':  feat_list,
        }

    payload = {
        'timestamp': round(time.time(), 3),
        'frame_id':  _frame_id_counter,
        'targets':   targets,
    }
    return json.dumps(payload, ensure_ascii=False).encode()


def _notify_inference_waiters() -> None:
    """推理结果写入 state 后，通过 call_soon_threadsafe 唤醒所有 /ws/inference 协程。"""
    loop = ws._frame_event_loop
    if loop is not None and loop.is_running():
        def _do():
            with ws._inference_waiters_lock:
                for ev in ws._inference_ready_waiters:
                    ev.set()
        loop.call_soon_threadsafe(_do)


# ── 推理函数（在 inference_executor 单线程中执行）─────────────────────
_infer_log_counter = 0   # 控制全流程耗时日志频率（每 30 帧一次）


def inference_and_encode(jpeg_bytes: bytes) -> Optional[bytes]:
    """
    JPEG 解码 → GPU 推理（见 processor.py 中的逐步耗时）→ 更新帧缓冲
    结果不回发给 WebSocket 客户端；浏览器通过 /video/infer MJPEG 流查看。
    """
    global _infer_log_counter
    if ws.processor is None:
        return None

    t_start = time.perf_counter()

    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # CPU JPEG 解码（未被 processor 计时覆盖）
    if frame is None:
        return None

    t_decoded = time.perf_counter()

    with ws.camera_lock:
        result = ws.processor.process_panorama_slices(frame)
    if result[0] is None:
        return None

    t_infer = time.perf_counter()

    _panorama, _yolo_only, annotated, tracked, angle_info = result
    pipeline_ms = (t_infer - t_decoded) * 1000
    decode_ms   = (t_decoded - t_start) * 1000
    total_ms    = (t_infer - t_start) * 1000

    _infer_log_counter += 1
    if _infer_log_counter % 30 == 1:
        print(f"[inference_and_encode] decode={decode_ms:.1f}ms  pipeline={pipeline_ms:.1f}ms"
              f"  total={total_ms:.1f}ms  → 理论上限 {1000/total_ms:.1f} FPS")

    detected_persons = len(tracked) if tracked else 0
    tracking_ids = [str(d.get('track_id', '?')) for d in tracked] if tracked else []
    update_perf(total_ms, detected_persons, tracking_ids)

    # 推理线程内完成耗时操作，不占用 asyncio 事件循环
    infer_json  = _build_inference_json(tracked, angle_info)   # JSON 序列化
    webrtc_vf   = _make_webrtc_frame(annotated)                # BGR→YUV420P

    with ws.frame_lock:
        ws.latest_original_frame    = frame
        ws.latest_annotated_frame   = annotated
        ws.latest_inference_result  = infer_json
        ws.latest_webrtc_frame      = webrtc_vf
        # latest_original_jpeg 已由 camera_ws.drain_recv 在摄像头速率直接更新，此处不重复赋值
    _notify_inference_waiters()     # 立即唤醒所有 /ws/inference 协程

    # GPU 显存碎片清理（每 200 帧一次，原在 _encode_worker 中，现移至此）
    if _infer_log_counter % 200 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 工具函数 ───────────────────────────────────────────────────────────
def _find_free_port(start: int = 8000, max_try: int = 100) -> int:
    for port in range(start, start + max_try):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('0.0.0.0', port)) != 0:
                return port
    raise RuntimeError("找不到可用端口")


def _get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'


# ── 主入口 ─────────────────────────────────────────────────────────────
def main() -> None:
    args = config.parse_args()

    # 1. 初始化处理器（YOLO + OSNet，全景处理器延迟到第一帧）
    ws.processor = FisheyePanoramaYOLOPose(args)
    if not ws.processor.initialize():
        print("初始化失败，程序退出")
        return

    # 2. 注册推理回调 + 启动 GPU 监控线程
    ws.inference_fn = inference_and_encode
    start_gpu_monitor()  # GPU/CPU 指标独立后台线程，不阻塞推理

    # 3. 配置上传方式
    ws.upload_mode = args.upload_mode
    ws.upload_udp_port = args.udp_port
    ws.performance_data['upload_mode'] = args.upload_mode
    if args.upload_mode == 'udp':
        import webui.udp_receiver as udp_receiver
        udp_receiver.configure(args.udp_port)

    port = _find_free_port()
    local_ip = _get_local_ip()

    if args.upload_mode == 'udp':
        upload_desc  = f'UDP (端口 {args.udp_port})'
        stream_cmd   = (f'python camera_client.py ws://{local_ip}:{port}/ws/camera'
                        f' --format udp --udp-port {args.udp_port}')
    else:
        upload_desc  = 'WebSocket'
        stream_cmd   = f'python camera_client.py ws://{local_ip}:{port}/ws/camera'

    print()
    print("=" * 60)
    print("  WebUI 已启动")
    print(f"  本机访问:   http://localhost:{port}")
    print(f"  局域网访问: http://{local_ip}:{port}")
    print(f"  上传方式:   {upload_desc}")
    print(f"  推流命令:   {stream_cmd}")
    print("  Ctrl+C 停止")
    print("=" * 60)
    print()

    # 4. 启动 FastAPI 服务（禁用 WebSocket ping 防止并发写冲突）
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            ws_ping_interval=None,
            ws_ping_timeout=None,
        )
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        ws.processor.cleanup()
        ws.inference_executor.shutdown(wait=False)
        print("程序结束")


if __name__ == "__main__":
    main()
