"""
GPU 信息查询 + 性能指标更新

GPU/CPU 查询在独立后台线程（_gpu_monitor_thread）每秒运行一次，
与推理线程完全隔离，不阻塞推理流水线。
"""
import subprocess
import threading
import time
from typing import Dict, List, Optional

import psutil

from . import state

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


def _gpu_monitor_loop() -> None:
    """
    独立后台线程：每秒查询一次 GPU/CPU 指标并写入 performance_data。
    与推理线程完全隔离，nvidia-smi 的延迟不影响推理帧率。
    """
    _init_nvml()
    while True:
        time.sleep(1.0)
        gpu = get_gpu_info()
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_pct = psutil.virtual_memory().percent
        with state.perf_lock:
            state.performance_data["gpu_usage"] = gpu["usage"]
            state.performance_data["gpu_memory"] = gpu["memory"]
            state.performance_data["gpu_temp"] = gpu["temp"]
            state.performance_data["system_cpu"] = cpu_pct
            state.performance_data["system_memory"] = mem_pct


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
