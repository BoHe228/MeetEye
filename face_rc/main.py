"""
Fish-eye edge WebUI entry.

This file is the clean replacement for the original
mytest/main_GPU_webui.py deployment path. It owns only:

  camera/video JPEG input -> FaceRCPipeline -> JSON/WebRTC buffers -> WebUI routes

Three input modes are supported:
  1. WebUI default: read a local RK3588 camera device directly.
  2. --video-path reads a local test video.
  3. --camera-device none keeps the old /ws/camera push endpoint.
"""
import json
import logging
import os
import queue
import socket
import signal
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np
import torch

import config
import webui.state as ws
from processor import FaceRCPipeline
from utils.distance_estimator import HeadPoseDistanceEstimator
from utils.sector import aggregate_sectors


class _FilterInvalidHTTP(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Invalid HTTP request" not in record.getMessage()


logging.getLogger("uvicorn.error").addFilter(_FilterInvalidHTTP())

_WEBRTC_MAX_W = 3840
_distance_estimators: dict = {}
_frame_id_counter = 0
_sector_output = False
_num_sectors = 8
_input_running = False
_infer_log_counter = 0
_webrtc_pcs = set()


def _parse_cpu_affinity(value: str) -> Optional[set]:
    text = str(value or "").strip().lower()
    if text in {"", "none", "off", "false", "disable", "disabled"}:
        return None
    cpus = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"invalid CPU range: {part}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(part))
    return cpus or None


def _apply_cpu_affinity(args) -> None:
    affinity = getattr(args, "cpu_affinity", "none")
    try:
        cpus = _parse_cpu_affinity(affinity)
    except Exception as exc:
        print(f"[cpu-affinity] 参数无效，已忽略: {affinity!r} ({exc})")
        return
    if cpus is None:
        print("[cpu-affinity] 已关闭")
        return
    if not hasattr(os, "sched_setaffinity"):
        print("[cpu-affinity] 当前系统不支持 os.sched_setaffinity，已跳过")
        return
    available = os.sched_getaffinity(0)
    target = set(cpus) & set(available)
    if not target:
        print(f"[cpu-affinity] 目标 CPU {sorted(cpus)} 不在可用集合 {sorted(available)} 中，已跳过")
        return
    try:
        os.sched_setaffinity(0, target)
        print(f"[cpu-affinity] 主进程已绑定 CPU: {','.join(str(i) for i in sorted(target))}")
    except Exception as exc:
        print(f"[cpu-affinity] 绑定失败，继续运行: {type(exc).__name__}: {exc}")


def update_perf(*_args, **_kwargs) -> None:
    """No-op until WebUI mode imports the real performance updater."""
    return None


def _make_webrtc_frame(bgr: np.ndarray):
    """Convert an annotated BGR frame to a WebRTC-ready YUV420P frame."""
    import av

    h, w = bgr.shape[:2]
    if w > _WEBRTC_MAX_W:
        bgr = cv2.resize(
            bgr,
            (_WEBRTC_MAX_W, int(h * _WEBRTC_MAX_W / w)),
            interpolation=cv2.INTER_AREA,
        )
    # YUV420P requires even width and height. Some remap/crop combinations can
    # produce odd dimensions, so trim one row/column rather than failing in cvtColor.
    if bgr.shape[1] % 2 != 0 or bgr.shape[0] % 2 != 0:
        even_w = bgr.shape[1] - (bgr.shape[1] % 2)
        even_h = bgr.shape[0] - (bgr.shape[0] % 2)
        bgr = bgr[:even_h, :even_w]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    return av.VideoFrame.from_ndarray(yuv, format="yuv420p")


def _eye_pixel_dist(keypoints) -> Optional[float]:
    """Return raw eye distance for JSON monitoring; it is not used for matching."""
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


def _build_inference_json(tracked: list, angle_info: dict) -> bytes:
    """
    Serialize inference output.

    TrackID is the runtime target id. The face recognition module is
    intentionally absent in this deployment package.
    """
    global _frame_id_counter, _distance_estimators
    _frame_id_counter += 1

    # Sector mode is used by the meeting system: each horizontal sector keeps
    # only the largest representative target.
    if _sector_output:
        sectors, _reps = aggregate_sectors(tracked, angle_info, _num_sectors)
        payload = {
            "timestamp": round(time.time(), 3),
            "frame_id": _frame_id_counter,
            "num_sectors": _num_sectors,
            "sectors": sectors,
        }
        return json.dumps(payload, ensure_ascii=False).encode()

    persons = (angle_info or {}).get("persons", [])
    targets: dict = {}
    current_tids: set = set()

    for i, det in enumerate(tracked or []):
        tid = int(det.get("track_id", i + 1))
        current_tids.add(tid)
        angle = persons[i] if i < len(persons) else None
        eye_d = _eye_pixel_dist(det.get("keypoints"))

        # Keep one distance estimator per TrackID so short invalid keypoint
        # bursts do not make the reported distance flicker to null.
        estimator = _distance_estimators.get(tid)
        if estimator is None:
            estimator = HeadPoseDistanceEstimator()
            _distance_estimators[tid] = estimator
        kpts = det.get("keypoints")
        l_eye, r_eye, nose = HeadPoseDistanceEstimator.extract_keypoints(kpts)
        if l_eye is not None:
            distance = estimator.compute_distance(l_eye, r_eye, nose)
        else:
            distance = estimator.last_valid_distance

        targets[str(tid)] = {
            "id": tid,
            "azimuth": round(float(angle["azimuth_deg"]), 3) if angle else None,
            "elevation": round(float(angle["elevation_deg"]), 3) if angle else None,
            "eye_pixel_dist": round(eye_d, 2) if eye_d is not None else None,
            "distance": round(distance, 3) if distance is not None else None,
        }

    for tid in [tid for tid in _distance_estimators if tid not in current_tids]:
        del _distance_estimators[tid]

    return json.dumps({
        "timestamp": round(time.time(), 3),
        "frame_id": _frame_id_counter,
        "targets": targets,
    }, ensure_ascii=False).encode()


def _notify_inference_waiters() -> None:
    """Wake /ws/inference consumers after a new JSON result is written."""
    loop = ws._frame_event_loop
    if loop is not None and loop.is_running():
        def _do():
            with ws._inference_waiters_lock:
                for ev in ws._inference_ready_waiters:
                    ev.set()
        loop.call_soon_threadsafe(_do)


def inference_and_encode(jpeg_bytes: bytes) -> Optional[bytes]:
    """
    Single-frame backend path.

    It intentionally runs in a single-thread executor. CUDA inference,
    JSON serialization, WebRTC frame preparation, and optional video
    recording complete before the next frame is processed, preventing frame
    backlog on edge devices.
    """
    global _infer_log_counter
    if ws.processor is None:
        return None

    profile_interval = max(
        0,
        int(getattr(getattr(ws.processor, "args", None), "profile_interval", 30)),
    )
    should_profile = profile_interval > 0 and (_infer_log_counter % profile_interval == 0)
    t_start = time.perf_counter()
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    t_decoded = time.perf_counter()

    t_lock_wait_start = time.perf_counter()
    with ws.camera_lock:
        t_lock_acquired = time.perf_counter()
        result = ws.processor.process_panorama_slices(frame)
    if result[0] is None:
        return None
    t_infer = time.perf_counter()

    _panorama, _yolo_only, annotated, tracked, angle_info = result
    infer_json = _build_inference_json(tracked, angle_info)
    t_json = time.perf_counter()
    # WebRTC conversion is skipped when no browser is connected. This saves a
    # full-frame BGR->YUV conversion on headless deployments.
    webrtc_vf = _make_webrtc_frame(annotated) if _webrtc_pcs else None
    t_webrtc = time.perf_counter()

    with ws.frame_lock:
        ws.latest_original_frame = frame
        ws.latest_annotated_frame = annotated
        ws.latest_inference_result = infer_json
        ws.latest_webrtc_frame = webrtc_vf
    t_state = time.perf_counter()
    with ws.jsonl_lock:
        if ws.jsonl_file is not None:
            ws.jsonl_file.write(infer_json)
            ws.jsonl_file.write(b"\n")
    t_jsonl = time.perf_counter()
    _notify_inference_waiters()
    t_notify = time.perf_counter()

    # Recording is lazy-initialized on the first processed frame because only
    # then do we know both original and annotated frame sizes.
    with ws.record_lock:
        if ws.is_recording:
            save_original = bool(ws.record_filenames.get("original"))
            save_annotated = bool(ws.record_filenames.get("annotated"))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            if save_original and ws._video_writer_original is None:
                oh, ow = frame.shape[:2]
                ws._video_writer_original = cv2.VideoWriter(
                    ws.record_filenames["original"], fourcc, 25.0, (ow, oh))
            if save_annotated and ws._video_writer_annotated is None:
                ah, aw = annotated.shape[:2]
                ws._video_writer_annotated = cv2.VideoWriter(
                    ws.record_filenames["annotated"], fourcc, 25.0, (aw, ah))
            if ws._video_writer_original is not None:
                ws._video_writer_original.write(frame)
            if ws._video_writer_annotated is not None:
                ws._video_writer_annotated.write(annotated)
    t_record = time.perf_counter()

    t_end = t_record
    decode_ms = (t_decoded - t_start) * 1000
    lock_wait_ms = (t_lock_acquired - t_lock_wait_start) * 1000
    pipeline_ms = (t_infer - t_lock_acquired) * 1000
    tail_ms = (t_end - t_infer) * 1000
    loop_ms = (t_end - t_start) * 1000
    detected_persons = len(tracked) if tracked else 0
    tracking_ids = [str(det.get("track_id", "?")) for det in tracked] if tracked else []
    update_perf(loop_ms, detected_persons, tracking_ids)

    _infer_log_counter += 1
    if should_profile:
        fps_str = f"{1000 / loop_ms:.1f}" if loop_ms > 0 else "inf"
        tail_parts = [
            ("json", (t_json - t_infer) * 1000),
            ("webrtc_yuv", (t_webrtc - t_json) * 1000),
            ("state_lock", (t_state - t_webrtc) * 1000),
            ("jsonl", (t_jsonl - t_state) * 1000),
            ("notify", (t_notify - t_jsonl) * 1000),
            ("record", (t_record - t_notify) * 1000),
        ]
        slow_tail = sorted(tail_parts, key=lambda item: item[1], reverse=True)[:3]
        tail_detail = " ".join(f"{name}={ms:.1f}ms" for name, ms in tail_parts)
        slow_tail_text = ", ".join(f"{name}:{ms:.1f}ms" for name, ms in slow_tail)
        print(
            f"[frame profile] decode={decode_ms:.1f}ms "
            f"lock_wait={lock_wait_ms:.1f}ms pipeline={pipeline_ms:.1f}ms "
            f"tail={tail_ms:.1f}ms loop={loop_ms:.1f}ms "
            f"webrtc_clients={len(_webrtc_pcs)} -> 理论上限 {fps_str} FPS | "
            f"{tail_detail} | slow_tail={slow_tail_text}"
        )
    if _infer_log_counter % 200 == 0 and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def _submit_jpeg_for_inference(jpeg_bytes: bytes, source_label: str) -> None:
    with ws.frame_lock:
        ws.latest_original_jpeg = jpeg_bytes
    if ws.inference_fn is None:
        return
    future = ws.inference_executor.submit(ws.inference_fn, jpeg_bytes)
    try:
        future.result()
    except Exception as exc:
        print(f"[{source_label}] 推理异常: {type(exc).__name__}: {exc}")


def _video_loop(args) -> None:
    """Feed a local video through the same JPEG path as camera input."""
    global _input_running
    path = args.video_path
    if not os.path.exists(path):
        print(f"[视频] 文件不存在: {path}")
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[视频] 无法打开: {path}")
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1.0 / src_fps
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    print(f"[视频] 已打开: {path}")
    print(f"[视频] 分辨率: {int(cap.get(3))}x{int(cap.get(4))} FPS={src_fps:.1f} 总帧数={total_frm}")

    _input_running = True
    while _input_running:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("[视频] 播放完毕，正在退出...")
            break
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            _submit_jpeg_for_inference(buf.tobytes(), "视频")
        elapsed = time.perf_counter() - t0
        sleep_s = frame_interval - elapsed
        if sleep_s > 0:
            time.sleep(sleep_s)

    cap.release()
    print("[视频] 播放线程已退出")
    os.kill(os.getpid(), signal.SIGINT)


def _camera_device_disabled(value: str) -> bool:
    return str(value or "").strip().lower() in {"", "none", "off", "false", "disabled"}


def _camera_device_to_index(device: str):
    text = str(device or "").strip()
    if text.isdigit():
        return int(text)
    prefix = "/dev/video"
    if text.startswith(prefix) and text[len(prefix):].isdigit():
        return int(text[len(prefix):])
    return text


def _build_camera_ffmpeg_cmd(args) -> list:
    device = str(getattr(args, "camera_device", "/dev/video0"))
    width = int(getattr(args, "camera_width", 1920))
    height = int(getattr(args, "camera_height", 1080))
    fps = float(getattr(args, "camera_fps", 30.0))
    fmt = str(getattr(args, "camera_format", "mjpeg")).lower()

    input_args = ["-f", "v4l2"]
    if fmt == "mjpeg":
        input_args += ["-input_format", "mjpeg"]
    elif fmt == "yuyv":
        input_args += ["-input_format", "yuyv422"]
    input_args += [
        "-video_size", f"{width}x{height}",
        "-framerate", str(int(round(fps))),
        "-i", device,
    ]
    if fmt == "mjpeg":
        return ["ffmpeg", *input_args, "-c:v", "copy", "-f", "mjpeg", "pipe:1"]
    return [
        "ffmpeg", *input_args,
        "-an", "-vf", "format=yuvj420p",
        "-q:v", "4", "-f", "mjpeg", "pipe:1",
    ]


def _drain_proc_stderr(proc: subprocess.Popen, label: str) -> None:
    if proc.stderr is None:
        return
    try:
        for raw in iter(proc.stderr.readline, b""):
            if not raw:
                break
            msg = raw.decode("utf-8", errors="replace").strip()
            if msg:
                print(f"[{label}] {msg}")
    except Exception:
        pass


def _camera_ffmpeg_loop(args) -> bool:
    global _input_running
    cmd = _build_camera_ffmpeg_cmd(args)
    print(f"[摄像头] FFmpeg 采集: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except FileNotFoundError:
        print("[摄像头] 未找到 ffmpeg，回退 OpenCV 采集")
        return False
    except Exception as exc:
        print(f"[摄像头] FFmpeg 启动失败，回退 OpenCV: {type(exc).__name__}: {exc}")
        return False

    stderr_thread = threading.Thread(
        target=_drain_proc_stderr,
        args=(proc, "ffmpeg"),
        daemon=True,
        name="CameraFFmpegStderr",
    )
    stderr_thread.start()

    _input_running = True
    buf = b""
    frame_n = 0
    started = time.perf_counter()
    try:
        while _input_running:
            if proc.stdout is None:
                break
            chunk = proc.stdout.read(524288)
            if not chunk:
                break
            buf += chunk

            latest_jpeg = None
            while True:
                start = buf.find(b"\xff\xd8")
                if start < 0:
                    buf = b""
                    break
                end = buf.find(b"\xff\xd9", start + 2)
                if end < 0:
                    buf = buf[start:]
                    break
                latest_jpeg = buf[start:end + 2]
                buf = buf[end + 2:]

            if latest_jpeg is None:
                continue
            frame_n += 1
            if frame_n == 1:
                print(f"[摄像头] 首帧已获取: {len(latest_jpeg) // 1024} KB")
            elif frame_n % 150 == 0:
                elapsed = max(time.perf_counter() - started, 1e-6)
                print(f"[摄像头] 已处理 {frame_n} 帧，均速 {frame_n / elapsed:.1f} FPS")
            _submit_jpeg_for_inference(latest_jpeg, "摄像头")
    finally:
        had_frame = frame_n > 0
        _input_running = False
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        print("[摄像头] FFmpeg 采集线程已退出")
    if not had_frame:
        print("[摄像头] FFmpeg 未获取到有效帧，回退 OpenCV 采集")
    return had_frame


def _camera_opencv_loop(args) -> None:
    global _input_running
    device = getattr(args, "camera_device", "/dev/video0")
    cap = cv2.VideoCapture(_camera_device_to_index(device))
    if not cap.isOpened():
        print(f"[摄像头] OpenCV 无法打开: {device}")
        return

    width = int(getattr(args, "camera_width", 1920))
    height = int(getattr(args, "camera_height", 1080))
    fps = float(getattr(args, "camera_fps", 30.0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    fmt = str(getattr(args, "camera_format", "mjpeg")).lower()
    if fmt == "mjpeg":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif fmt == "yuyv":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    frame_interval = 1.0 / max(fps, 1.0)
    print(
        f"[摄像头] OpenCV 已打开: {device} "
        f"{int(cap.get(3))}x{int(cap.get(4))} FPS={cap.get(cv2.CAP_PROP_FPS):.1f}"
    )

    _input_running = True
    try:
        while _input_running:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("[摄像头] 读取失败，退出")
                break
            ok, buf = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                _submit_jpeg_for_inference(buf.tobytes(), "摄像头")
            sleep_s = frame_interval - (time.perf_counter() - t0)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        cap.release()
        print("[摄像头] OpenCV 采集线程已退出")


def _camera_loop(args) -> None:
    if not _camera_ffmpeg_loop(args):
        _camera_opencv_loop(args)


def _find_free_port(start: int = 8000, max_try: int = 100) -> int:
    """Pick the first free WebUI port, starting at 8000."""
    for port in range(start, start + max_try):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", port)) != 0:
                return port
    raise RuntimeError("找不到可用端口")


def _get_local_ip() -> str:
    """Best-effort LAN IP used for the startup banner."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _init_runtime(args) -> bool:
    """Initialize shared inference state for WebUI and headless modes."""
    ws.processor = FaceRCPipeline(args)
    if not ws.processor.initialize():
        print("初始化失败，程序退出")
        return False

    ws.inference_fn = inference_and_encode
    ws.webrtc_fps = max(1, min(120, int(getattr(args, "webrtc_fps", 40))))
    ws.save_original_video = bool(getattr(args, "save_original_video", False))

    global _sector_output, _num_sectors
    _sector_output = bool(getattr(args, "sector_output", False))
    _num_sectors = max(1, int(getattr(args, "num_sectors", 8)))
    if _sector_output:
        print(f"[JSON] 扇区聚合输出已启用: {_num_sectors} 个扇区")
    return True


def _default_headless_jsonl_path(args) -> str:
    os.makedirs(args.output_dir, exist_ok=True)
    if args.video_path:
        stem = os.path.splitext(os.path.basename(args.video_path))[0]
    elif not _camera_device_disabled(getattr(args, "camera_device", "/dev/video0")):
        stem = "camera"
    else:
        stem = "headless"
    return os.path.join(
        args.output_dir,
        f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
    )


def _open_headless_capture(args):
    """Open either a finite video file or a local V4L2 camera for headless mode."""
    if getattr(args, "video_path", None):
        if not os.path.exists(args.video_path):
            print(f"[headless] 视频文件不存在: {args.video_path}")
            return None, None
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            print(f"[headless] 无法打开视频: {args.video_path}")
            return None, None
        info = {
            "kind": "video",
            "label": f"视频文件 -> {args.video_path}",
            "fps": cap.get(cv2.CAP_PROP_FPS) or 0.0,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        return cap, info

    device = getattr(args, "camera_device", "/dev/video0")
    if _camera_device_disabled(device):
        print("headless 模式需要指定 --video-path，或指定可用的 --camera-device")
        return None, None

    cap = cv2.VideoCapture(_camera_device_to_index(device))
    if not cap.isOpened():
        print(f"[headless] 无法打开摄像头: {device}")
        return None, None

    width = int(getattr(args, "camera_width", 1920))
    height = int(getattr(args, "camera_height", 1080))
    fps = float(getattr(args, "camera_fps", 30.0))
    fmt = str(getattr(args, "camera_format", "mjpeg")).lower()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if fmt == "mjpeg":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    elif fmt == "yuyv":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    info = {
        "kind": "camera",
        "label": f"本地摄像头 -> {device} ({fmt})",
        "fps": cap.get(cv2.CAP_PROP_FPS) or fps,
        "frames": 0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    return cap, info


def _run_headless(args) -> None:
    """Run video/camera input directly through the pipeline and write JSONL."""
    global _infer_log_counter
    if getattr(args, "save_video", False):
        print("[headless] 已忽略 --save-video：headless 模式只输出 JSONL")

    if not _init_runtime(args):
        return

    output_jsonl = args.output_jsonl or _default_headless_jsonl_path(args)
    output_dir = os.path.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cap, cap_info = _open_headless_capture(args)
    if cap is None:
        if ws.processor is not None:
            ws.processor.cleanup()
        return

    src_fps = float(cap_info.get("fps", 0.0) or 0.0)
    total_frm = int(cap_info.get("frames", 0) or 0)
    max_frames = max(0, int(getattr(args, "max_frames", 0)))
    profile_interval = max(0, int(getattr(args, "profile_interval", 30)))

    print()
    print("=" * 60)
    print("  Fish-eye headless 已启动")
    print(f"  输入源:     {cap_info['label']}")
    print(f"  输出JSONL:  {output_jsonl}")
    if cap_info["kind"] == "video":
        print(f"  视频信息:   {cap_info['width']}x{cap_info['height']} FPS={src_fps:.1f} 总帧数={total_frm}")
    else:
        print(f"  摄像头信息: {cap_info['width']}x{cap_info['height']} FPS={src_fps:.1f} 总帧数=实时")
    if max_frames > 0:
        print(f"  最大帧数:   {max_frames}")
    print("  Ctrl+C 停止")
    print("=" * 60)
    print()

    use_pipeline_parallel = bool(getattr(args, "headless_pipeline_parallel", False))
    pipeline_queue_size = max(1, int(getattr(args, "headless_pipeline_queue_size", 2)))
    use_remap_prefetch = bool(getattr(args, "headless_remap_prefetch", False))
    if use_pipeline_parallel:
        if not bool(getattr(args, "direct_slice_remap", False)):
            print("[headless-pipeline] 需要 --direct-slice-remap，已回退串行")
            use_pipeline_parallel = False
        elif ws.processor is None or not bool(getattr(ws.processor, "direct_slice_remap", False)):
            print("[headless-pipeline] processor 未启用 direct-slice，已回退串行")
            use_pipeline_parallel = False
        else:
            print(
                "[headless-pipeline] 已启用：detection worker 与当前帧 tracker 重叠执行 "
                f"queue={pipeline_queue_size}"
            )
            if use_remap_prefetch:
                print("[headless-pipeline] OpenCL direct-slice remap 预提交已启用")

    processed = 0
    started = time.perf_counter()
    try:
        with open(output_jsonl, "wb") as jf:
            if use_pipeline_parallel:
                frame_queue: "queue.Queue" = queue.Queue(maxsize=pipeline_queue_size)
                result_queue: "queue.Queue" = queue.Queue(maxsize=pipeline_queue_size)
                stop_event = threading.Event()
                sentinel = object()

                def _reader() -> None:
                    frame_idx = 0
                    try:
                        while not stop_event.is_set():
                            if max_frames > 0 and frame_idx >= max_frames:
                                break
                            ret, frame = cap.read()
                            if not ret:
                                break
                            frame_idx += 1
                            frame_queue.put((frame_idx, frame))
                    finally:
                        frame_queue.put(sentinel)

                def _detect_worker() -> None:
                    try:
                        if use_remap_prefetch:
                            pending = None
                            pending_frame_idx = None
                            while True:
                                item = frame_queue.get()
                                if item is sentinel:
                                    if pending is not None:
                                        try:
                                            remapped = ws.processor.finish_direct_slice_frame_remap(pending)
                                            detection_result = ws.processor.detect_direct_slice_remapped(remapped)
                                            result_queue.put((pending_frame_idx, detection_result, None))
                                        except Exception as exc:
                                            result_queue.put((pending_frame_idx, None, exc))
                                    break

                                frame_idx, frame = item
                                try:
                                    next_pending = ws.processor.start_direct_slice_frame_remap(frame)
                                    if pending is not None:
                                        remapped = ws.processor.finish_direct_slice_frame_remap(pending)
                                        detection_result = ws.processor.detect_direct_slice_remapped(remapped)
                                        result_queue.put((pending_frame_idx, detection_result, None))
                                    if next_pending is not None:
                                        pending = next_pending
                                        pending_frame_idx = frame_idx
                                    else:
                                        pending = None
                                        pending_frame_idx = None
                                except Exception as exc:
                                    result_queue.put((frame_idx, None, exc))
                                    pending = None
                                    pending_frame_idx = None
                        else:
                            while True:
                                item = frame_queue.get()
                                if item is sentinel:
                                    break
                                frame_idx, frame = item
                                try:
                                    detection_result = ws.processor.detect_direct_slice_frame(frame)
                                    result_queue.put((frame_idx, detection_result, None))
                                except Exception as exc:
                                    result_queue.put((frame_idx, None, exc))
                    finally:
                        result_queue.put(sentinel)

                reader_thread = threading.Thread(
                    target=_reader,
                    daemon=True,
                    name="HeadlessFrameReader",
                )
                detect_thread = threading.Thread(
                    target=_detect_worker,
                    daemon=True,
                    name="DirectSliceDetect",
                )
                reader_thread.start()
                detect_thread.start()

                try:
                    while True:
                        t0 = time.perf_counter()
                        item = result_queue.get()
                        if item is sentinel:
                            break
                        _frame_idx, detection_result, exc = item
                        if exc is not None:
                            raise exc

                        if detection_result is None:
                            continue

                        _infer_log_counter += 1
                        should_profile = profile_interval > 0 and _infer_log_counter % profile_interval == 0
                        wait_done = time.perf_counter()
                        result = ws.processor.finish_direct_slice_detection(
                            detection_result,
                            print_profile=should_profile,
                            extra_timing=[("检测等待", wait_done)] if should_profile else None,
                        )
                        if result[0] is None:
                            continue
                        _panorama, _yolo_only, _annotated, tracked, angle_info = result
                        infer_json = _build_inference_json(tracked, angle_info)
                        jf.write(infer_json)
                        jf.write(b"\n")
                        processed += 1

                        if should_profile:
                            loop_ms = (time.perf_counter() - t0) * 1000.0
                            fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
                            print(
                                f"[headless profile] frame={processed} "
                                f"loop={loop_ms:.1f}ms theoretical={fps:.1f} FPS "
                                f"targets={len(tracked) if tracked else 0}"
                            )
                finally:
                    stop_event.set()
                    reader_thread.join(timeout=2.0)
                    detect_thread.join(timeout=2.0)
            else:
                while True:
                    if max_frames > 0 and processed >= max_frames:
                        break
                    t0 = time.perf_counter()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = ws.processor.process_panorama_slices(frame)
                    if result[0] is None:
                        continue
                    _panorama, _yolo_only, _annotated, tracked, angle_info = result
                    infer_json = _build_inference_json(tracked, angle_info)
                    jf.write(infer_json)
                    jf.write(b"\n")
                    processed += 1

                    _infer_log_counter += 1
                    if profile_interval > 0 and _infer_log_counter % profile_interval == 0:
                        loop_ms = (time.perf_counter() - t0) * 1000.0
                        fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
                        print(
                            f"[headless profile] frame={processed} "
                            f"loop={loop_ms:.1f}ms theoretical={fps:.1f} FPS "
                            f"targets={len(tracked) if tracked else 0}"
                        )
    except KeyboardInterrupt:
        print("\n[headless] 用户中断")
    finally:
        cap.release()
        if ws.processor is not None:
            ws.processor.cleanup()

    elapsed = time.perf_counter() - started
    avg_fps = processed / elapsed if elapsed > 0 else 0.0
    print(f"[headless] 完成: frames={processed}, elapsed={elapsed:.1f}s, avg_fps={avg_fps:.2f}")
    print(f"[headless] 已保存JSONL: {output_jsonl}")


def _run_webui(args) -> None:
    """Initialize pipeline state and start FastAPI/uvicorn."""
    import uvicorn
    from webui.gpu_info import start_gpu_monitor, update_perf as webui_update_perf
    from webui.routes import _webrtc_pcs as routes_webrtc_pcs, app

    global _webrtc_pcs, update_perf
    _webrtc_pcs = routes_webrtc_pcs
    update_perf = webui_update_perf

    if not _init_runtime(args):
        return
    start_gpu_monitor()

    if getattr(args, "output_jsonl", None):
        output_dir = os.path.dirname(args.output_jsonl)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with ws.jsonl_lock:
            ws.jsonl_path = args.output_jsonl
            ws.jsonl_file = open(args.output_jsonl, "wb")
        print(f"[WebUI] JSONL 输出已启用: {args.output_jsonl}")

    # Optional startup recording follows the original WebUI behavior. Browser
    # button recording still uses the REST endpoints in webui/routes.py.
    if getattr(args, "save_video", False):
        os.makedirs(args.output_dir, exist_ok=True)
        name = getattr(args, "video_name", None)
        if not name:
            src = os.path.splitext(os.path.basename(args.video_path))[0] if args.video_path else "camera"
            name = f"{src}_{time.strftime('%Y%m%d_%H%M%S')}"
        if name.endswith(".mp4"):
            name = name[:-4]
        annotated_path = os.path.join(args.output_dir, f"{name}.mp4")
        save_original = bool(getattr(args, "save_original_video", False))
        original_path = os.path.join(args.output_dir, f"{name}_original.mp4") if save_original else None
        with ws.record_lock:
            ws.record_filenames = {"original": original_path, "annotated": annotated_path}
            ws.is_recording = True
        if original_path:
            print(f"[录制] 自动录制到:\n  {annotated_path}\n  {original_path}")
        else:
            print(f"[录制] 自动录制到:\n  {annotated_path}\n  原始视频: 已关闭")

    port = _find_free_port()
    local_ip = _get_local_ip()
    input_thread = None
    if getattr(args, "video_path", None):
        input_thread = threading.Thread(
            target=_video_loop,
            args=(args,),
            daemon=True,
            name="VideoLoop",
        )
        input_thread.start()
    elif not _camera_device_disabled(getattr(args, "camera_device", "/dev/video0")):
        input_thread = threading.Thread(
            target=_camera_loop,
            args=(args,),
            daemon=True,
            name="CameraLoop",
        )
        input_thread.start()

    print()
    print("=" * 60)
    print("  Fish-eye WebUI 已启动")
    print(f"  本机访问:   http://localhost:{port}")
    print(f"  局域网访问: http://{local_ip}:{port}")
    if getattr(args, "video_path", None):
        print(f"  输入模式:   视频文件 -> {args.video_path}")
    elif not _camera_device_disabled(getattr(args, "camera_device", "/dev/video0")):
        print(
            f"  输入模式:   本地摄像头 -> {args.camera_device} "
            f"{args.camera_width}x{args.camera_height}@{args.camera_fps:g} "
            f"format={args.camera_format}"
        )
    else:
        print(f"  推流端点:   ws://{local_ip}:{port}/ws/camera")
    print("  Ctrl+C 停止")
    print("=" * 60)
    print()

    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="warning",
            log_config=None,
            ws_ping_interval=None,
            ws_ping_timeout=None,
        )
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        global _input_running
        _input_running = False
        if input_thread is not None and input_thread.is_alive():
            input_thread.join(timeout=2.0)
        with ws.record_lock:
            ws.is_recording = False
            if ws._video_writer_original is not None:
                ws._video_writer_original.release()
                ws._video_writer_original = None
            if ws._video_writer_annotated is not None:
                ws._video_writer_annotated.release()
                ws._video_writer_annotated = None
        with ws.jsonl_lock:
            if ws.jsonl_file is not None:
                ws.jsonl_file.close()
                ws.jsonl_file = None
                print(f"[WebUI] 已保存JSONL: {ws.jsonl_path}")
        if ws.processor is not None:
            ws.processor.cleanup()
        ws.inference_executor.shutdown(wait=False)


def main() -> None:
    args = config.parse_args()
    _apply_cpu_affinity(args)
    if getattr(args, "webui", False):
        _run_webui(args)
    else:
        _run_headless(args)


if __name__ == "__main__":
    main()
