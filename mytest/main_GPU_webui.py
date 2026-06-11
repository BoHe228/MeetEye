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
from webui.routes import app, _webrtc_pcs
from utils.distance_estimator import HeadPoseDistanceEstimator
from utils.sector import aggregate_sectors


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


# ── 双眼像素距离（原始值，仅用于 JSON 中的 eye_pixel_dist 监控字段）────
def _eye_pixel_dist(keypoints) -> Optional[float]:
    """
    从 YOLO-Pose keypoints 提取双眼像素距离（未做偏转修正）。
    COCO 关键点格式：kpt[1]=左眼, kpt[2]=右眼，每行 [x, y] 或 [x, y, conf]。
    """
    if keypoints is None:
        return None
    kpts = np.array(keypoints)
    if kpts.shape[0] < 3:
        return None
    if kpts.shape[1] >= 3 and (float(kpts[1, 2]) < 0.1 or float(kpts[2, 2]) < 0.1):
        return None
    lx, ly = float(kpts[1, 0]), float(kpts[1, 1])
    rx, ry = float(kpts[2, 0]), float(kpts[2, 1])
    d = ((rx - lx) ** 2 + (ry - ly) ** 2) ** 0.5
    return d if d > 0 else None


# ── 每个跟踪目标独立持有一个距离估算器，用于帧间熔断与姿态修正 ────────
_distance_estimators: dict = {}


# ── 推理结果序列化（在 inference_executor 单线程中调用，天然无竞争）────
_frame_id_counter = 0


def _build_inference_json(tracked: list, angle_info: dict) -> bytes:
    """
    将推理结果序列化为 JSON bytes。
    返回值直接存入 state，可被 /inference/latest 和 /ws/inference 读取。

    angle_info['persons'][i] 与 tracked[i] 严格 1:1 对齐（processor 对无关键点的
    补漏框用合成鼻子点占位）。--sector-output 开启时输出扇区聚合格式，否则按 track_id 输出。

    distance 字段使用 HeadPoseDistanceEstimator 进行偏转角修正，
    每个 track_id 持有独立实例以支持帧间熔断缓存。
    """
    global _frame_id_counter, _distance_estimators
    _frame_id_counter += 1

    # ── 扇区聚合格式（--sector-output）──────────────────────────────────
    if _sector_output:
        sectors, _reps = aggregate_sectors(tracked, angle_info, _num_sectors)
        payload = {
            'timestamp':   round(time.time(), 3),
            'frame_id':    _frame_id_counter,
            'num_sectors': _num_sectors,
            'sectors':     sectors,
        }
        return json.dumps(payload, ensure_ascii=False).encode()

    # ── 兼容：按 track_id 输出 ──────────────────────────────────────────
    persons = (angle_info or {}).get('persons', [])

    targets: dict = {}
    current_tids: set = set()

    for i, det in enumerate(tracked or []):
        tid = int(det.get('track_id', i + 1))
        current_tids.add(tid)
        angle = persons[i] if i < len(persons) else None

        feat = det.get('feature')
        feat_list = feat.reshape(-1).tolist() if feat is not None else []

        # 原始双眼像素距离（未修正，供监控用）
        eye_d = _eye_pixel_dist(det.get('keypoints'))

        # 头部姿态修正距离：每个 track_id 独立维护估算器
        estimator = _distance_estimators.get(tid)
        if estimator is None:
            estimator = HeadPoseDistanceEstimator()
            _distance_estimators[tid] = estimator

        kpts = det.get('keypoints')
        l_eye, r_eye, nose = HeadPoseDistanceEstimator.extract_keypoints(kpts)
        if l_eye is not None:
            distance = estimator.compute_distance(l_eye, r_eye, nose)
        else:
            distance = estimator.last_valid_distance

        targets[str(tid)] = {
            'id':             tid,
            'azimuth':        round(float(angle['azimuth_deg']),   3) if angle else None,
            'elevation':      round(float(angle['elevation_deg']), 3) if angle else None,
            'eye_pixel_dist': round(eye_d,    2) if eye_d    is not None else None,
            'distance':       round(distance, 3) if distance is not None else None,
            'features':       feat_list,
        }

    # 清理不再活跃的轨迹对应的估算器，防止内存持续增长
    stale_tids = [tid for tid in _distance_estimators if tid not in current_tids]
    for tid in stale_tids:
        del _distance_estimators[tid]

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


# ── JSON 结果持久化（--save-json 开启后每帧追加一行到 JSONL 文件）──────
_json_file = None          # io.TextIOWrapper，由 main() 打开，finally 关闭
_json_write_counter = 0    # 用于定期 flush

# ── 扇区聚合输出配置（由 main() 从 args 写入）─────────────────────────
_sector_output = False     # True 时 JSON 改为扇区聚合格式
_num_sectors = 8           # 水平 360° 等分扇区数

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

    detected_persons = len(tracked) if tracked else 0
    tracking_ids = [str(d.get('track_id', '?')) for d in tracked] if tracked else []

    # 推理线程内完成耗时操作，不占用 asyncio 事件循环
    infer_json  = _build_inference_json(tracked, angle_info)   # JSON 序列化
    # WebRTC 帧仅在有活跃连接时生成：无人观看 WebRTC 时省掉每帧整帧 BGR→YUV420P
    # 转换（3840 宽，非小开销）。recv() 在 latest_webrtc_frame 为 None 时自带空白帧兜底。
    webrtc_vf   = _make_webrtc_frame(annotated) if _webrtc_pcs else None

    # ── JSONL 持久化（--save-json，每 30 帧 flush 一次，不影响推理耗时）──
    global _json_file, _json_write_counter
    if _json_file is not None:
        _json_file.write(infer_json.decode() + '\n')
        _json_write_counter += 1
        if _json_write_counter % 30 == 0:
            _json_file.flush()

    with ws.frame_lock:
        ws.latest_original_frame    = frame
        ws.latest_annotated_frame   = annotated
        ws.latest_inference_result  = infer_json
        ws.latest_webrtc_frame      = webrtc_vf
        # latest_original_jpeg 已由 camera_ws.drain_recv 在摄像头速率直接更新，此处不重复赋值
    _notify_inference_waiters()     # 立即唤醒所有 /ws/inference 协程

    # ── 视频录制（record_lock 保护，与 /record/start、/record/stop 互斥）
    with ws.record_lock:
        if ws.is_recording:
            if ws._video_writer_original is None:
                # 首帧时初始化 VideoWriter（此时才知道分辨率）
                oh, ow = frame.shape[:2]
                ah, aw = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                ws._video_writer_original  = cv2.VideoWriter(
                    ws.record_filenames['original'],  fourcc, 25.0, (ow, oh))
                ws._video_writer_annotated = cv2.VideoWriter(
                    ws.record_filenames['annotated'], fourcc, 25.0, (aw, ah))
            ws._video_writer_original.write(frame)
            ws._video_writer_annotated.write(annotated)

    # ── 一帧的全部串行工作（解码 + 推理 + JSON + WebRTC + 录制）至此结束 ──
    # 用整段耗时（loop）而非仅推理段作为「理论上限」基准，使其反映真实循环周期，
    # 否则该数会把 JSON/WebRTC/录制这条尾巴漏掉、一直偏乐观。
    t_end   = time.perf_counter()
    tail_ms = (t_end - t_infer) * 1000   # JSON / WebRTC / 帧缓冲 / 录制
    loop_ms = (t_end - t_start) * 1000
    update_perf(loop_ms, detected_persons, tracking_ids)

    _infer_log_counter += 1
    if _infer_log_counter % 30 == 1:
        fps_str = f"{1000/loop_ms:.1f}" if loop_ms > 0 else "∞"
        print(f"[inference_and_encode] decode={decode_ms:.1f}ms  pipeline={pipeline_ms:.1f}ms"
              f"  tail={tail_ms:.1f}ms(json/webrtc/录制)  loop={loop_ms:.1f}ms"
              f"  → 理论上限 {fps_str} FPS")

    # GPU 显存碎片清理（每 200 帧一次，原在 _encode_worker 中，现移至此）
    if _infer_log_counter % 200 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── 视频文件输入循环（绕开 WebSocket，直接把视频帧喂进 inference_executor）──
_video_running = False   # 供 Ctrl+C 时停止循环


def _video_loop(args) -> None:
    """
    以视频文件替代摄像头推流：
      - 按视频原始 FPS 送帧（推理比视频慢时以推理速度为准，不积压队列）
      - 视频播放完毕后循环重播（保持 WebUI 服务持续可用）
      - 同时更新 state.latest_original_jpeg，使 /video/original MJPEG 端点正常工作
    """
    global _video_running
    import os

    path = args.video_path
    if not os.path.exists(path):
        print(f"[视频] 文件不存在: {path}")
        return

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[视频] 无法打开: {path}")
        return

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1.0 / src_fps
    encode_params  = [cv2.IMWRITE_JPEG_QUALITY, 90]

    print(f"[视频] 已打开: {path}")
    print(f"[视频] 分辨率: {int(cap.get(3))}×{int(cap.get(4))}"
          f"  FPS={src_fps:.1f}  总帧数={total_frm}")

    _video_running = True

    while _video_running:
        t0 = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            # 视频结束 → 退出循环，触发程序关闭
            print("[视频] 播放完毕，正在退出…")
            break

        # BGR → JPEG（与 camera_client.py 推流格式一致）
        ok, buf = cv2.imencode('.jpg', frame, encode_params)
        if not ok:
            continue

        jpeg_bytes = buf.tobytes()

        # 更新原始帧缓冲，使 /video/original MJPEG 端点可用
        with ws.frame_lock:
            ws.latest_original_jpeg = jpeg_bytes

        # 提交推理并等待完成（阻塞式，避免帧积压；推理慢时自动降帧率）
        if ws.inference_fn is not None:
            future = ws.inference_executor.submit(ws.inference_fn, jpeg_bytes)
            try:
                future.result()
            except Exception as e:
                print(f"[视频] 推理异常: {type(e).__name__}: {e}")

        # 按原始 FPS 限速（推理耗时已计入，若推理慢则不额外等待）
        elapsed  = time.perf_counter() - t0
        sleep_ms = frame_interval - elapsed
        if sleep_ms > 0:
            time.sleep(sleep_ms)

    cap.release()
    print("[视频] 播放线程已退出")

    # 视频播完后向主进程发送 SIGINT，触发 uvicorn 优雅关闭和 finally 清理
    import os, signal
    os.kill(os.getpid(), signal.SIGINT)


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

    port = _find_free_port()
    local_ip = _get_local_ip()

    # 3. 视频文件模式：后台线程推帧，无需 camera_client 连接
    video_thread = None
    if getattr(args, 'video_path', None):
        video_thread = threading.Thread(
            target=_video_loop, args=(args,), daemon=True, name='VideoLoop')
        video_thread.start()

    # 扇区聚合输出配置（--sector-output / --num-sectors）
    global _sector_output, _num_sectors
    _sector_output = getattr(args, 'sector_output', False)
    _num_sectors = max(1, int(getattr(args, 'num_sectors', 8)))
    if _sector_output:
        print(f"[JSON] 扇区聚合输出已启用：{_num_sectors} 个扇区")

    # 4. JSONL 输出文件（--save-json）
    global _json_file
    if getattr(args, 'save_json', False):
        import datetime, os
        json_path = getattr(args, 'json_output', None)
        if not json_path:
            os.makedirs(args.output_dir, exist_ok=True)
            ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            src = os.path.splitext(os.path.basename(args.video_path))[0] \
                  if getattr(args, 'video_path', None) else 'camera'
            json_path = os.path.join(args.output_dir, f'{src}_{ts}.jsonl')
        _json_file = open(json_path, 'w', encoding='utf-8', buffering=1)
        print(f"[JSON] 推理结果将保存到: {json_path}")

    # 5. 自动录制（--save-video）：启动即开始录制，复用 inference_and_encode 里的
    #    懒初始化 VideoWriter 逻辑（首帧建写入器）。annotated 路径取 --video-name，
    #    原始帧另存 _original 后缀。退出时在 finally 中释放并做 faststart 修复。
    if getattr(args, 'save_video', False):
        import datetime, os
        os.makedirs(args.output_dir, exist_ok=True)
        name = getattr(args, 'video_name', None)
        if not name:
            ts  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            src = os.path.splitext(os.path.basename(args.video_path))[0] \
                  if getattr(args, 'video_path', None) else 'camera'
            name = f'{src}_{ts}'
        if name.endswith('.mp4'):
            name = name[:-4]
        annotated_path = os.path.join(args.output_dir, f'{name}.mp4')
        original_path  = os.path.join(args.output_dir, f'{name}_original.mp4')
        with ws.record_lock:
            ws.record_filenames = {'original': original_path, 'annotated': annotated_path}
            ws.is_recording = True
        print(f"[录制] --save-video 已启用，自动录制到:\n  {annotated_path}（推理标注）\n  {original_path}（原始帧）")

    print()
    print("=" * 60)
    print("  WebUI 已启动")
    print(f"  本机访问:   http://localhost:{port}")
    print(f"  局域网访问: http://{local_ip}:{port}")
    if getattr(args, 'video_path', None):
        print(f"  输入模式:   视频文件 → {args.video_path}")
    else:
        print(f"  推流命令:   python camera_client.py ws://{local_ip}:{port}/ws/camera")
    print("  Ctrl+C 停止")
    print("=" * 60)
    print()

    # 5. 启动 FastAPI 服务（禁用 WebSocket ping 防止并发写冲突）
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
        global _video_running
        _video_running = False          # 通知视频线程退出

        # 停止自动录制：在 record_lock 下置 is_recording=False 并释放 writer，
        # 确保推理线程不会再写（它在同一把锁下检查 is_recording），再做 faststart 修复。
        with ws.record_lock:
            _was_recording = ws.is_recording
            _rec_files = dict(ws.record_filenames)
            ws.is_recording = False
            ws.record_filenames = {}
            if ws._video_writer_original is not None:
                ws._video_writer_original.release()
                ws._video_writer_original = None
            if ws._video_writer_annotated is not None:
                ws._video_writer_annotated.release()
                ws._video_writer_annotated = None
        if _was_recording:
            print(f"[录制] 已停止并保存: {_rec_files}")
            try:
                from webui.routes import _fix_mp4_faststart
                for _p in _rec_files.values():
                    _fix_mp4_faststart(_p)
            except Exception as _e:
                print(f"[录制] faststart 跳过: {_e}")

        if _json_file is not None:
            _json_file.flush()
            _json_file.close()
            print(f"[JSON] 文件已保存: {_json_file.name}")
        ws.processor.cleanup()
        ws.inference_executor.shutdown(wait=False)
        print("程序结束")


if __name__ == "__main__":
    main()
