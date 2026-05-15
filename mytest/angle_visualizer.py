"""
角度实时可视化 — 订阅 /ws/inference，在半球体上实时显示目标方位

用法:
    python angle_visualizer.py                              # 测试模式（预设固定角度）
    python angle_visualizer.py ws://192.168.x.x:8000       # 自动补全端点
    python angle_visualizer.py ws://192.168.x.x:8000/ws/inference
"""
import argparse
import json
import sys
import threading
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401 — 注册 3d projection
from mpl_toolkits.mplot3d import proj3d

matplotlib.rcParams['font.sans-serif'] = [
    'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans'
]
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 共享状态（ws 线程写，主线程读，均在 _lock 下访问）────────────────
_lock = threading.Lock()
_latest_targets: dict = {}   # {tid: {'azimuth': float, 'elevation': float}}
_conn_status: str = "未连接"
_frame_count: int = 0


# ── Arrow3D ───────────────────────────────────────────────────────────
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# ── WebSocket 回调 ────────────────────────────────────────────────────
def _on_message(ws, message):
    global _latest_targets, _frame_count
    try:
        data = json.loads(message)          # json.loads 接受 str 和 bytes
        targets_raw = data.get('targets', {})

        new_targets = {}
        for tid, tdata in targets_raw.items():
            az = tdata.get('azimuth')
            el = tdata.get('elevation')
            if az is None or el is None:    # 角度未计算的目标跳过
                continue
            new_targets[str(tid)] = {'azimuth': float(az), 'elevation': float(el)}

        with _lock:
            _latest_targets = new_targets   # 整帧替换，消除残留目标
            _frame_count += 1
    except Exception as e:
        print(f"[ws] 消息解析错误: {e}")


def _on_open(ws):
    global _conn_status
    with _lock:
        _conn_status = "已连接"
    print("[ws] 已连接到服务器")


def _on_error(ws, error):
    global _conn_status
    with _lock:
        _conn_status = f"连接错误"
    print(f"[ws] 错误: {error}")


def _on_close(ws, code, msg):
    global _conn_status
    with _lock:
        _conn_status = "已断开，重连中…"
    print(f"[ws] 连接关闭 (code={code})")


def start_ws_thread(url: str) -> threading.Thread:
    try:
        import websocket
    except ImportError:
        print("缺少依赖：pip install websocket-client")
        sys.exit(1)

    def _run():
        ws = websocket.WebSocketApp(
            url,
            on_message=_on_message,
            on_open=_on_open,
            on_error=_on_error,
            on_close=_on_close,
        )
        ws.run_forever(reconnect=3)         # 断线后 3 秒自动重连

    t = threading.Thread(target=_run, daemon=True, name="ws-recv")
    t.start()
    return t


# ── 坐标转换 ──────────────────────────────────────────────────────────
def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float,
                            radius: float = 0.85):
    """球坐标 → 笛卡尔。方位角 0° → Y 轴正向（正前方），顺时针增大。"""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    x = radius * np.cos(el) * np.sin(az)
    y = radius * np.cos(el) * np.cos(az)
    z = radius * np.sin(el)
    return x, y, z


# ── 颜色：按 ID 哈希固定，避免目标数变化时颜色跳变 ──────────────────
_CMAP = plt.cm.tab10

def _id_color(tid: str):
    return _CMAP((hash(tid) % 10) / 10)


# ── 静态背景（只在启动时绘制一次）──────────────────────────────────
def build_static_scene(ax) -> None:
    u = np.linspace(0, 2 * np.pi, 36)

    # 上半球纬线
    for phi in np.linspace(0, np.pi / 2, 10):
        ax.plot(np.sin(phi) * np.cos(u),
                np.sin(phi) * np.sin(u),
                np.full_like(u, np.cos(phi)),
                color='gray', alpha=0.12, lw=0.5)
    # 上半球经线
    for theta in u[::4]:
        v = np.linspace(0, np.pi / 2, 20)
        ax.plot(np.sin(v) * np.cos(theta),
                np.sin(v) * np.sin(theta),
                np.cos(v),
                color='gray', alpha=0.12, lw=0.5)

    # 俯视小段（-10° ~ 0°），用仰角公式：r_h=cos(el), z=sin(el)
    for el_deg in np.linspace(-10, 0, 3):
        el = np.deg2rad(el_deg)
        r_h = np.cos(el)
        z_h = np.sin(el)
        ax.plot(r_h * np.cos(u), r_h * np.sin(u),
                np.full_like(u, z_h),
                color='lightgray', alpha=0.12, lw=0.5)

    # 参考圆（仰角公式：r_h=cos(el), z=sin(el)）
    tc = np.linspace(0, 2 * np.pi, 120)
    # 0° 水平面：z=0, r=1
    ax.plot(np.cos(tc), np.sin(tc), np.zeros_like(tc),
            'b-', alpha=0.5, lw=1.5, label='水平面 0°')
    # -10° 平面：z=sin(-10°)≈-0.174, r=cos(-10°)≈0.985
    el_n10 = np.deg2rad(-10)
    r_n10  = np.cos(el_n10)
    z_n10  = np.sin(el_n10)
    ax.plot(r_n10 * np.cos(tc), r_n10 * np.sin(tc),
            np.full_like(tc, z_n10),
            'c--', alpha=0.4, lw=1.0, label='-10° 平面')

    # 方位标注
    for az_deg, label in [(0, 'Front'), (90, 'Right'), (180, 'Back'), (270, 'Left')]:
        az = np.deg2rad(az_deg)
        ax.text(1.28 * np.sin(az), 1.28 * np.cos(az), 0.0,
                label, fontsize=8, ha='center', va='center',
                color='steelblue', alpha=0.8)

    ax.set_xlim([-1.35, 1.35])
    ax.set_ylim([-1.35, 1.35])
    ax.set_zlim([-0.25, 1.2])
    ax.set_xlabel('X (East)',  fontsize=9, labelpad=6)
    ax.set_ylabel('Y (Front)', fontsize=9, labelpad=6)
    ax.set_zlabel('Z (Up)',    fontsize=9, labelpad=6)
    ax.view_init(elev=25, azim=45)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right', fontsize=9)


# ── 动态元素更新（每帧调用）─────────────────────────────────────────
def update_dynamic(ax, points_dict: dict) -> None:
    # 移除上一帧的动态元素
    for coll in list(ax.collections):
        coll.remove()
    for txt in list(ax.texts):
        txt.remove()
    for patch in [p for p in ax.patches if isinstance(p, Arrow3D)]:
        patch.remove()

    for tid, pd in points_dict.items():
        az, el = pd['azimuth'], pd['elevation']
        x, y, z = spherical_to_cartesian(az, el)
        c = _id_color(tid)

        # 目标点
        ax.scatter(x, y, z, color=c, s=130, alpha=0.95,
                   edgecolors='black', linewidth=1.2, depthshade=True)

        # 箭头从 0° 平面圆心（原点）出发
        ax.add_patch(Arrow3D(
            [0, x], [0, y], [0, z],
            mutation_scale=14, arrowstyle='->',
            color=c, alpha=0.75, linewidth=2.2,
        ))

        ax.text(x * 1.18, y * 1.18, z * 1.18,
                f'ID {tid}\nAz {az:.1f}°  El {el:.1f}°',
                fontsize=8, color=c,
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', alpha=0.85, edgecolor='none'))


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    global _latest_targets, _conn_status

    parser = argparse.ArgumentParser(description='角度实时可视化')
    parser.add_argument('url', nargs='?', default=None,
                        help='WebSocket 地址，例如 ws://192.168.1.100:8000')
    parser.add_argument('--test', action='store_true', help='强制使用测试数据')
    args = parser.parse_args()

    test_mode = args.test or (args.url is None)

    if test_mode:
        _conn_status = "测试模式"
        _latest_targets = {
            'A': {'azimuth': 0,   'elevation': 0},
            'B': {'azimuth': 90,  'elevation': 30},
            'C': {'azimuth': 180, 'elevation': 60},
            'D': {'azimuth': 270, 'elevation': -5},
            'E': {'azimuth': 45,  'elevation': 85},
        }
        print("=== 测试模式：显示预设固定角度 ===")
        print("真实模式: python angle_visualizer.py ws://HOST:PORT")
    else:
        url = args.url
        if not url.endswith('/ws/inference'):
            url = url.rstrip('/') + '/ws/inference'
        print(f"=== 连接 {url} ===")
        start_ws_thread(url)

    # matplotlib 必须在主线程运行
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    build_static_scene(ax)
    plt.ion()
    plt.tight_layout()

    last_frame = -1
    print("关闭窗口或按 Ctrl+C 退出")

    try:
        while plt.get_fignums():
            with _lock:
                targets = dict(_latest_targets)
                status  = _conn_status
                fc      = _frame_count

            # 测试模式始终渲染（保持窗口响应），实时模式仅在新帧到达时渲染
            if test_mode or fc != last_frame:
                n = len(targets)
                fig.suptitle(
                    f'实时目标位置  [{status}]  帧 #{fc}  目标 {n} 个',
                    fontsize=12, y=0.99,
                )
                update_dynamic(ax, targets)
                fig.canvas.draw_idle()
                last_frame = fc

            plt.pause(0.033)    # ~30 fps，同时处理窗口事件

    except KeyboardInterrupt:
        print("\n程序退出")


if __name__ == '__main__':
    main()
