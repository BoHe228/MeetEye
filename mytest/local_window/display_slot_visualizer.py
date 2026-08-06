"""
Display ID 槽图片观察工具

订阅 board_cpp WebUI 的 /ws/slots，按 display_id 固定槽位显示板端发送的
ID 槽缩略图和元信息。

用法:
    python display_slot_visualizer.py
    python display_slot_visualizer.py ws://172.16.30.68:8080
    python display_slot_visualizer.py ws://172.16.30.68:8080/ws/slots
    python display_slot_visualizer.py --test
"""
import argparse
import base64
import io
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 120
matplotlib.rcParams["figure.facecolor"] = "#ffffff"
matplotlib.rcParams["axes.facecolor"] = "#f8fafc"

_COLOR_TEXT = "#172033"
_COLOR_MUTED = "#64748b"
_COLOR_LINE = "#d6dee9"
_COLOR_EMPTY = "#eef2f7"
_COLOR_ACTIVE = "#0f766e"
_COLOR_WARN = "#b45309"
_COLOR_ERROR = "#dc2626"


@dataclass
class SlotState:
    display_id: int
    public_id: int = 0
    raw_track_id: int = 0
    frame_id: int = -1
    face_id: str = "null"
    front_score: float = 0.0
    image: Optional[np.ndarray] = None
    updated_at: float = 0.0


_lock = threading.Lock()
_slots: Dict[int, SlotState] = {}
_display_id_max_count: int = 8
_conn_status: str = "未连接"
_conn_url: str = ""
_recv_count: int = 0
_recv_fps: float = 0.0
_last_msg_time: Optional[float] = None
_conn_open_time: Optional[float] = None
_last_error: str = ""


def normalize_slots_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return "ws://172.16.30.51:8080/ws/slots"
    if url.startswith("http://"):
        url = "ws://" + url[len("http://"):]
    elif url.startswith("https://"):
        url = "wss://" + url[len("https://"):]
    if not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    if not url.endswith("/ws/slots"):
        url = url.rstrip("/") + "/ws/slots"
    return url


def decode_data_url_image(data_url: str) -> Optional[np.ndarray]:
    if not data_url:
        return None
    try:
        payload = data_url.split(",", 1)[1] if "," in data_url else data_url
        raw = base64.b64decode(payload)
        try:
            from PIL import Image
        except ImportError:
            print("缺少依赖：pip install pillow")
            return None
        with Image.open(io.BytesIO(raw)) as img:
            return np.asarray(img.convert("RGB"))
    except Exception:
        return None


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _on_message(ws, message: str) -> None:
    global _display_id_max_count, _recv_count, _recv_fps, _last_msg_time, _last_error
    try:
        data = json.loads(message)
        now = time.monotonic()
        if _last_msg_time is not None:
            dt = now - _last_msg_time
            if dt > 0:
                inst_fps = 1.0 / dt
                _recv_fps = inst_fps if _recv_fps <= 0 else _recv_fps * 0.85 + inst_fps * 0.15
        _last_msg_time = now

        max_count = max(1, _safe_int(data.get("display_id_max_count"), 8))
        incoming = {}
        for item in data.get("slots", []) or []:
            display_id = _safe_int(item.get("display_id"), 0)
            if display_id <= 0:
                continue
            image = decode_data_url_image(item.get("image") or "")
            state = SlotState(
                display_id=display_id,
                public_id=_safe_int(item.get("public_id"), 0),
                raw_track_id=_safe_int(item.get("raw_track_id"), 0),
                frame_id=_safe_int(item.get("frame_id"), -1),
                face_id=str(item.get("face_id") or "null"),
                front_score=_safe_float(item.get("front_score"), 0.0),
                image=image,
                updated_at=now,
            )
            incoming[display_id] = state

        with _lock:
            _display_id_max_count = max_count
            _slots.clear()
            _slots.update(incoming)
            _recv_count += 1
            _last_error = ""
    except Exception as exc:
        with _lock:
            _last_error = str(exc)
        print(f"[ws] 槽图片消息解析错误: {exc}")


def _on_open(ws) -> None:
    global _conn_status, _conn_open_time
    with _lock:
        _conn_status = "已连接"
        _conn_open_time = time.monotonic()
    print("[ws] 已连接到 /ws/slots")


def _on_error(ws, error) -> None:
    global _conn_status, _last_error
    with _lock:
        _conn_status = "连接错误"
        _last_error = str(error)
    print(f"[ws] 错误: {error}")


def _on_close(ws, code, msg) -> None:
    global _conn_status, _conn_open_time
    with _lock:
        _conn_status = "已断开，重连中..."
        _conn_open_time = None
    print(f"[ws] 连接关闭 (code={code})")


def start_ws_thread(url: str) -> threading.Thread:
    try:
        import websocket
    except ImportError:
        print("缺少依赖：pip install websocket-client")
        sys.exit(1)

    def _run() -> None:
        ws = websocket.WebSocketApp(
            url,
            on_message=_on_message,
            on_open=_on_open,
            on_error=_on_error,
            on_close=_on_close,
        )
        ws.run_forever(reconnect=3)

    t = threading.Thread(target=_run, daemon=True, name="slot-ws-recv")
    t.start()
    return t


def make_test_image(slot_id: int) -> np.ndarray:
    h, w = 96, 96
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), dtype=np.uint8)
    colors = [
        (59, 130, 246),
        (16, 185, 129),
        (245, 158, 11),
        (239, 68, 68),
        (139, 92, 246),
        (6, 182, 212),
        (249, 115, 22),
        (132, 204, 22),
    ]
    c = colors[(slot_id - 1) % len(colors)]
    mask = ((xx - w / 2) ** 2 / (w * 0.24) ** 2 + (yy - h / 2) ** 2 / (h * 0.34) ** 2) <= 1
    base[:] = (20, 27, 38)
    base[mask] = c
    return base


def load_test_data() -> None:
    global _display_id_max_count, _conn_status, _conn_url, _recv_count
    now = time.monotonic()
    with _lock:
        _display_id_max_count = 8
        _conn_status = "测试模式"
        _conn_url = "测试数据"
        _recv_count = 1
        _slots.clear()
        for slot_id in range(1, 7):
            _slots[slot_id] = SlotState(
                display_id=slot_id,
                public_id=slot_id + 10,
                raw_track_id=slot_id + 20,
                frame_id=1000 + slot_id,
                face_id=f"face{slot_id}" if slot_id % 2 else "null",
                front_score=700.0 + slot_id * 15,
                image=make_test_image(slot_id),
                updated_at=now,
            )


def snapshot_state():
    with _lock:
        now = time.monotonic()
        last_msg_age = None if _last_msg_time is None else max(0.0, now - _last_msg_time)
        conn_age = None if _conn_open_time is None else max(0.0, now - _conn_open_time)
        return (
            dict(_slots),
            int(_display_id_max_count),
            _conn_status,
            _conn_url,
            int(_recv_count),
            float(_recv_fps),
            _last_error,
            last_msg_age,
            conn_age,
        )


def _status_color(status: str, recv_count: int, last_error: str) -> str:
    if last_error or "错误" in status or "断开" in status:
        return _COLOR_ERROR
    if recv_count <= 0:
        return _COLOR_WARN
    if "已连接" in status or "测试" in status:
        return _COLOR_ACTIVE
    return _COLOR_MUTED


def _fit_text(value: str, max_chars: int) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max(1, max_chars - 1)] + "…"


def _image_extent(image: np.ndarray, box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, x2, y1, y2 = box
    if image is None or image.ndim < 2:
        return box
    h, w = image.shape[:2]
    if w <= 0 or h <= 0:
        return box
    box_w = max(1e-6, x2 - x1)
    box_h = max(1e-6, y2 - y1)
    image_aspect = float(w) / float(h)
    box_aspect = box_w / box_h
    if image_aspect >= box_aspect:
        shown_w = box_w
        shown_h = box_w / image_aspect
        cy = (y1 + y2) * 0.5
        return x1, x2, cy - shown_h * 0.5, cy + shown_h * 0.5
    shown_h = box_h
    shown_w = box_h * image_aspect
    cx = (x1 + x2) * 0.5
    return cx - shown_w * 0.5, cx + shown_w * 0.5, y1, y2


def draw_slot(ax, slot_id: int, state: Optional[SlotState], now: float) -> None:
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if state and state.image is not None:
        active = True
        border = _COLOR_ACTIVE
        header_bg = "#e7f8f4"
        face = _fit_text(state.face_id if state.face_id else "null", 12)
    else:
        active = False
        border = _COLOR_LINE
        header_bg = "#f1f5f9"
        face = "null"

    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, facecolor="#ffffff",
                           edgecolor=border, linewidth=1.5 if active else 1.0, zorder=0))
    ax.add_patch(Rectangle((0.01, 0.84), 0.98, 0.15, facecolor=header_bg,
                           edgecolor="none", zorder=1))
    ax.text(0.06, 0.915, f"ID {slot_id}", ha="left", va="center",
            fontsize=12, fontweight="bold", color=border if active else _COLOR_MUTED,
            transform=ax.transAxes, zorder=3)
    ax.text(0.94, 0.915, face, ha="right", va="center",
            fontsize=9.5, fontweight="bold", color=_COLOR_TEXT,
            transform=ax.transAxes, zorder=3)

    if not active:
        ax.add_patch(Rectangle((0.08, 0.28), 0.84, 0.52, facecolor=_COLOR_EMPTY,
                               edgecolor="#cbd5e1", linewidth=0.8, zorder=1))
        ax.text(
            0.5,
            0.54,
            "空槽",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
            color="#94a3b8",
            fontweight="bold",
            zorder=2,
        )
        ax.text(0.5, 0.15, "等待当前槽图", ha="center", va="center",
                transform=ax.transAxes, fontsize=8.5, color=_COLOR_MUTED)
        return

    image_box = (0.08, 0.92, 0.28, 0.80)
    ax.add_patch(Rectangle((image_box[0], image_box[2]),
                           image_box[1] - image_box[0],
                           image_box[3] - image_box[2],
                           facecolor="#f1f5f9", edgecolor="#cbd5e1",
                           linewidth=0.8, zorder=1))
    ax.imshow(state.image, extent=_image_extent(state.image, image_box),
              aspect="equal", zorder=2)
    age = max(0.0, now - state.updated_at)
    row1 = f"P{state.public_id}   R{state.raw_track_id}   F{state.frame_id}"
    row2 = f"score {state.front_score:.1f}   age {age:.1f}s"
    ax.text(0.06, 0.18, row1, ha="left", va="center",
            transform=ax.transAxes, fontsize=8.3, color=_COLOR_TEXT,
            fontweight="bold", zorder=3)
    ax.text(0.06, 0.085, row2, ha="left", va="center",
            transform=ax.transAxes, fontsize=8.0, color=_COLOR_MUTED, zorder=3)


def draw_status(ax, max_count: int, active: int, status: str, url: str,
                recv_count: int, recv_fps: float, last_error: str,
                last_msg_age: Optional[float], conn_age: Optional[float]) -> None:
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    color = _status_color(status, recv_count, last_error)
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="#f8fafc",
                           edgecolor=_COLOR_LINE, linewidth=1.0))
    ax.add_patch(Rectangle((0, 0), 0.012, 1, facecolor=color, edgecolor="none"))

    if recv_count <= 0:
        if conn_age is None:
            note = "未建立连接"
        else:
            note = f"已连接 {conn_age:.1f}s，尚未收到槽图；确认板端正在推理"
    elif last_msg_age is not None and last_msg_age > 3.0:
        note = f"最近 {last_msg_age:.1f}s 无新槽图；确认板端推理仍在运行并检查 --webui-slot-fps"
    else:
        note = "接收正常"
    if last_error:
        note = f"{note} | error: {last_error}"

    ax.text(0.035, 0.66, "Display ID 槽图片观察", ha="left", va="center",
            fontsize=13, fontweight="bold", color=_COLOR_TEXT)
    ax.text(0.035, 0.27, note, ha="left", va="center",
            fontsize=9.2, color=_COLOR_MUTED)
    stats = f"{status}  |  槽 {active}/{max_count}  |  消息 {recv_count}  |  接收FPS {recv_fps:.1f}"
    ax.text(0.965, 0.66, stats, ha="right", va="center",
            fontsize=9.5, color=color, fontweight="bold")
    ax.text(0.965, 0.27, _fit_text(url, 82), ha="right", va="center",
            fontsize=8.2, color=_COLOR_MUTED)


def draw_grid(status_ax, axes, max_count: int, slots: Dict[int, SlotState],
              status: str, url: str, recv_count: int, recv_fps: float,
              last_error: str, last_msg_age: Optional[float],
              conn_age: Optional[float]) -> None:
    now = time.monotonic()
    active = sum(1 for sid in range(1, max_count + 1) if sid in slots)
    draw_status(
        status_ax,
        max_count,
        active,
        status,
        url,
        recv_count,
        recv_fps,
        last_error,
        last_msg_age,
        conn_age,
    )
    for idx, ax in enumerate(axes, start=1):
        if idx <= max_count:
            ax.set_visible(True)
            draw_slot(ax, idx, slots.get(idx), now)
        else:
            ax.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(description="MeetEye Display ID 槽图片观察工具")
    parser.add_argument(
        "url",
        nargs="?",
        default="ws://172.16.30.51:8080",
        help="WebSocket 地址，例如 ws://172.16.30.68:8080 或 ws://172.16.30.68:8080/ws/slots",
    )
    parser.add_argument("--test", action="store_true", help="使用本地测试槽图片")
    parser.add_argument("--columns", type=int, default=4, help="每行槽数量，默认 4")
    parser.add_argument("--max-slots", type=int, default=8, help="本地最多显示槽数，默认 8")
    parser.add_argument("--refresh", type=float, default=0.2, help="刷新间隔秒数，默认 0.2")
    args = parser.parse_args()

    columns = max(1, min(args.columns, 8))
    max_slots = max(1, min(args.max_slots, 64))
    rows = int(math.ceil(max_slots / columns))

    global _conn_url
    if args.test:
        load_test_data()
    else:
        _conn_url = normalize_slots_url(args.url)
        print(f"=== 连接 {_conn_url} ===")
        start_ws_thread(_conn_url)

    fig = plt.figure(figsize=(columns * 2.25, rows * 1.9 + 0.65))
    grid = fig.add_gridspec(
        rows + 1,
        columns,
        height_ratios=[0.30] + [1.0] * rows,
        hspace=0.08,
        wspace=0.055,
    )
    status_ax = fig.add_subplot(grid[0, :])
    axes = [
        fig.add_subplot(grid[r + 1, c])
        for r in range(rows)
        for c in range(columns)
    ]
    fig.patch.set_facecolor("#ffffff")
    plt.ion()
    fig.subplots_adjust(left=0.018, right=0.99, top=0.965, bottom=0.025)
    print("关闭窗口或按 Ctrl+C 退出")

    last_count = -1
    last_update = 0.0
    try:
        while plt.get_fignums():
            (
                slots,
                server_max,
                status,
                url,
                recv_count,
                recv_fps,
                last_error,
                last_msg_age,
                conn_age,
            ) = snapshot_state()
            max_count = min(max(1, server_max), max_slots)
            now = time.monotonic()
            if recv_count != last_count or now - last_update >= max(args.refresh, 0.05):
                draw_grid(
                    status_ax,
                    axes,
                    max_count,
                    slots,
                    status,
                    url,
                    recv_count,
                    recv_fps,
                    last_error,
                    last_msg_age,
                    conn_age,
                )
                fig.canvas.draw_idle()
                last_count = recv_count
                last_update = now
            plt.pause(max(args.refresh, 0.05))
    except KeyboardInterrupt:
        print("\n程序退出")


if __name__ == "__main__":
    main()
