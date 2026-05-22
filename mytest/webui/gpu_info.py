"""
GPU 信息查询 + 性能指标更新

GPU/CPU 查询在独立后台线程（_gpu_monitor_thread）每秒运行一次，
与推理线程完全隔离，不阻塞推理流水线。

内存监控：录制期间每 MEM_CHECK_INTERVAL 秒检查一次系统可用内存；
低于 MEM_LOW_MB 时自动停止录制并退出进程（优先保全已录文件）。
"""
import os
import signal
import subprocess
import threading
import time
from typing import Dict, List, Optional

import psutil

from . import state

# ── 可用内存安全阈值（MB）。低于此值时触发录制急停 + 程序退出。────────
MEM_LOW_MB: int = 1000
# 每隔多少秒检查一次内存（录制期间生效）
MEM_CHECK_INTERVAL: int = 10

# ── pynvml 句柄（模块级单例，避免每次 nvmlInit / nvmlShutdown 开销）──────
_nvml_handle = None
_nvml_lock = threading.Lock()


def _init_nvml():
    global _nvml_handle
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except Exception:
        _nvml_handle = None


def get_gpu_info() -> Dict[str, str]:
    """返回 GPU 使用率、显存、温度（优先 pynvml 单例，回退到单次 nvidia-smi 查询）"""
    with _nvml_lock:
        if _nvml_handle is not None:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(_nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                return {
                    "usage": f"{util.gpu}%",
                    "memory": f"{mem.used / 1024**3:.1f}/{mem.total / 1024**3:.1f} GB",
                    "temp": f"{temp}°C",
                }
            except Exception:
                pass
    # 回退：单次 nvidia-smi 查询（所有字段合并一条命令，避免 3 次子进程开销）
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            encoding='utf-8', timeout=3
        ).strip()
        parts = [x.strip() for x in out.split(',')]
        usage, mu, mt, temp = parts[0], parts[1], parts[2], parts[3]
        return {
            "usage": f"{usage}%",
            "memory": f"{float(mu)/1024:.1f}/{float(mt)/1024:.1f} GB",
            "temp": f"{temp}°C",
        }
    except Exception:
        return {"usage": "N/A", "memory": "N/A", "temp": "N/A"}


def _faststart_sync(path: str) -> None:
    """同步执行 ffmpeg faststart，供退出前调用（不开后台线程）。"""
    if not os.path.exists(path):
        return
    tmp = path + '.tmp.mp4'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', path, '-c', 'copy', '-movflags', 'faststart', tmp],
            check=True, capture_output=True, timeout=300,
        )
        os.replace(tmp, path)
        print(f"[内存监控] faststart 完成: {path}")
    except Exception as e:
        print(f"[内存监控] faststart 跳过 ({path}): {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


def _emergency_stop_and_exit(avail_mb: float) -> None:
    """
    可用内存低于阈值时的急停流程：
      1. 停止录制，释放 VideoWriter（保全已写入数据）
      2. 对已保存文件同步执行 faststart（使进度条可用）
      3. 发送 SIGTERM 触发 uvicorn 优雅退出
    此函数只会被调用一次（_mem_shutdown_triggered 保护）。
    """
    print(f"\n{'='*60}")
    print(f"[内存监控] ⚠️  系统可用内存仅剩 {avail_mb:.0f} MB（阈值 {MEM_LOW_MB} MB）")
    print("[内存监控] 正在停止录制并退出程序，请稍候...")
    print(f"{'='*60}\n")

    filenames: dict = {}
    with state.record_lock:
        if state.is_recording:
            state.is_recording = False
            filenames = dict(state.record_filenames)
            state.record_filenames = {}
            if state._video_writer_original is not None:
                state._video_writer_original.release()
                state._video_writer_original = None
            if state._video_writer_annotated is not None:
                state._video_writer_annotated.release()
                state._video_writer_annotated = None

    if filenames:
        print(f"[内存监控] 录制文件已保存: {filenames}")
        for path in filenames.values():
            _faststart_sync(path)
    else:
        print("[内存监控] 当前未在录制，直接退出")

    print("[内存监控] 发送 SIGTERM，uvicorn 开始优雅退出...")
    os.kill(os.getpid(), signal.SIGTERM)


_mem_shutdown_triggered = False   # 防止重复触发
_mem_check_counter      = 0       # 计数器，每 MEM_CHECK_INTERVAL 秒检查一次


def _gpu_monitor_loop() -> None:
    """
    独立后台线程：每秒查询一次 GPU/CPU 指标并写入 performance_data。
    与推理线程完全隔离，nvidia-smi 的延迟不影响推理帧率。
    录制期间额外每 MEM_CHECK_INTERVAL 秒检查系统可用内存。
    """
    global _mem_check_counter, _mem_shutdown_triggered
    _init_nvml()
    while True:
        time.sleep(1.0)
        gpu = get_gpu_info()
        vm  = psutil.virtual_memory()
        cpu_pct   = psutil.cpu_percent(interval=None)
        mem_pct   = vm.percent
        avail_mb  = vm.available / 1024 ** 2

        with state.perf_lock:
            state.performance_data["gpu_usage"]           = gpu["usage"]
            state.performance_data["gpu_memory"]          = gpu["memory"]
            state.performance_data["gpu_temp"]            = gpu["temp"]
            state.performance_data["system_cpu"]          = cpu_pct
            state.performance_data["system_memory"]       = mem_pct
            state.performance_data["system_memory_avail_mb"] = round(avail_mb, 1)

        # ── 录制期间内存检查 ──────────────────────────────────────────
        if not _mem_shutdown_triggered and state.is_recording:
            _mem_check_counter += 1
            if _mem_check_counter >= MEM_CHECK_INTERVAL:
                _mem_check_counter = 0
                if avail_mb < MEM_LOW_MB:
                    _mem_shutdown_triggered = True
                    # 在新线程中执行（避免阻塞 monitor 自身）
                    threading.Thread(
                        target=_emergency_stop_and_exit,
                        args=(avail_mb,),
                        daemon=True,
                        name="mem-emergency-stop",
                    ).start()
        else:
            _mem_check_counter = 0   # 未在录制时重置计数器


def start_gpu_monitor() -> None:
    """启动 GPU 监控后台线程（由 main() 调用一次）"""
    t = threading.Thread(target=_gpu_monitor_loop, daemon=True, name="gpu-monitor")
    t.start()


def update_perf(inference_ms: float = 0.0, detected_persons: int = 0,
                tracking_ids: Optional[List[str]] = None) -> None:
    """
    更新推理侧性能指标（仅计帧率 + 推理耗时 + 检测结果）。
    GPU/CPU 指标由 _gpu_monitor_loop 独立维护，此处不再查询，零阻塞。
    """
    with state.perf_lock:
        now = time.time()
        state.performance_data["_frame_count"] += 1
        dt = now - state.performance_data["_last_fps_time"]
        if dt >= 1.0:
            state.performance_data["fps"] = state.performance_data["_frame_count"] / dt
            state.performance_data["_frame_count"] = 0
            state.performance_data["_last_fps_time"] += dt  # 滑动窗口：对齐到下一整秒边界，避免累积漂移
        state.performance_data["inference_time_ms"] = inference_ms
        state.performance_data["theoretical_fps"] = (1000.0 / inference_ms) if inference_ms > 0 else 0.0
        state.performance_data["detected_persons"] = detected_persons
        if tracking_ids is not None:
            state.performance_data["tracking_ids"] = tracking_ids
