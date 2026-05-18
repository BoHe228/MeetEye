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
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['axes.facecolor']   = '#f8fafc'
matplotlib.rcParams['figure.facecolor'] = '#ffffff'
matplotlib.rcParams['grid.color']       = '#e2e8f0'
matplotlib.rcParams['axes.edgecolor']   = '#cbd5e1'
matplotlib.rcParams['xtick.color']      = '#475569'
matplotlib.rcParams['ytick.color']      = '#475569'
matplotlib.rcParams['axes.labelcolor']  = '#334155'
matplotlib.rcParams['axes.titlecolor']  = '#1e293b'

# ── 共享状态（ws 线程写，主线程读，均在 _lock 下访问）────────────────
_lock = threading.Lock()
_latest_targets: dict = {}   # {tid: {'azimuth': float, 'elevation': float, 'distance': float|None}}
_conn_status: str = "未连接"
_frame_count: int = 0

# 动态元素追踪（用于帧间精确清理，避免误删静态标注）
_radar_dynamic_artists: list = []
_sphere_dynamic_artists: list = []   # 3D 文字、连线、弧线（scatter/Arrow3D 由类型清理）

# 雷达最大显示距离（m）
_RADAR_MAX_DIST = 5.0

# ── 专业调色板（低饱和、协调）─────────────────────────────────────────
_PALETTE = [
    '#3b82f6',  # 蓝
    '#10b981',  # 翠绿
    '#f59e0b',  # 琥珀
    '#ef4444',  # 红
    '#8b5cf6',  # 紫
    '#06b6d4',  # 青
    '#f97316',  # 橙
    '#84cc16',  # 黄绿
    '#ec4899',  # 粉
    '#6366f1',  # 靛蓝
]

def _id_color(tid: str) -> str:
    return _PALETTE[hash(tid) % len(_PALETTE)]


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
        data = json.loads(message)
        targets_raw = data.get('targets', {})

        new_targets = {}
        for tid, tdata in targets_raw.items():
            az = tdata.get('azimuth')
            el = tdata.get('elevation')
            if az is None or el is None:
                continue
            dist = tdata.get('distance')
            if dist is None:
                eye_d = tdata.get('eye_pixel_dist')
                if eye_d is not None and float(eye_d) > 0:
                    dist = estimate_distance_from_eyes(float(eye_d))
            new_targets[str(tid)] = {
                'azimuth':   float(az),
                'elevation': float(el),
                'distance':  float(dist) if dist is not None else None,
            }

        with _lock:
            _latest_targets = new_targets
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
        _conn_status = "连接错误"
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
        ws.run_forever(reconnect=3)

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


# ── 双眼像素距离 → 估计物理距离（m，备用）────────────────────────────
_EYE_K1 = 0.024030
_EYE_K2 = 0.044812

def estimate_distance_from_eyes(eye_pixel_dist: float):
    denom = _EYE_K1 * eye_pixel_dist + _EYE_K2
    return 1.0 / denom if denom > 0 else None


# ── 3D 半球体静态背景（只在启动时绘制一次）──────────────────────────
def build_static_scene(ax) -> None:
    u = np.linspace(0, 2 * np.pi, 48)

    for phi in np.linspace(0, np.pi / 2, 12):
        ax.plot(np.sin(phi) * np.cos(u), np.sin(phi) * np.sin(u),
                np.full_like(u, np.cos(phi)),
                color='#64748b', alpha=0.18 if phi == 0 else 0.09, lw=0.6, zorder=1)
    for theta in u[::6]:
        v = np.linspace(0, np.pi / 2, 24)
        ax.plot(np.sin(v) * np.cos(theta), np.sin(v) * np.sin(theta), np.cos(v),
                color='#64748b', alpha=0.09, lw=0.6, zorder=1)

    for el_deg in np.linspace(-10, 0, 4):
        el = np.deg2rad(el_deg)
        r_h = np.cos(el)
        ax.plot(r_h * np.cos(u), r_h * np.sin(u),
                np.full_like(u, np.sin(el)),
                color='#94a3b8', alpha=0.14, lw=0.5, zorder=1)

    tc = np.linspace(0, 2 * np.pi, 180)

    # 0° 水平面
    ax.plot(np.cos(tc), np.sin(tc), np.zeros_like(tc),
            color='#3b82f6', alpha=0.95, lw=2.8, label='水平面 0°', zorder=5)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: F401
    disc_r = np.linspace(0, 1, 40)
    disc_t = np.linspace(0, 2 * np.pi, 80)
    disc_R, disc_T = np.meshgrid(disc_r, disc_t)
    ax.plot_surface(disc_R * np.cos(disc_T), disc_R * np.sin(disc_T),
                    np.zeros_like(disc_R),
                    color='#dbeafe', alpha=0.22, linewidth=0, zorder=1)
    for az_deg in range(0, 360, 30):
        az_r = np.deg2rad(az_deg)
        ax.plot([0, np.sin(az_r)], [0, np.cos(az_r)], [0, 0],
                color='#3b82f6', alpha=0.35 if az_deg % 90 == 0 else 0.16, lw=1.0, zorder=2)

    # -10° 参考平面
    el_n10 = np.deg2rad(-10)
    ax.plot(np.cos(el_n10) * np.cos(tc), np.cos(el_n10) * np.sin(tc),
            np.full_like(tc, np.sin(el_n10)),
            color='#0ea5e9', linestyle='--', alpha=0.55, lw=1.2, label='-10° 平面')

    # 方位静态标注（不会被 update_dynamic 误删）
    for az_deg, label in [(0, '前方'), (90, '右侧'), (180, '后方'), (270, '左侧')]:
        az = np.deg2rad(az_deg)
        ax.text(1.58 * np.sin(az), 1.58 * np.cos(az), 0.02,
                label, fontsize=9, ha='center', va='center',
                color='#1e40af', alpha=0.9, fontweight='medium')

    lim = 1.35
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X (东)',  fontsize=10, labelpad=8, color='#475569')
    ax.set_ylabel('Y (前)', fontsize=10, labelpad=8, color='#475569')
    ax.set_zlabel('Z (上)',  fontsize=10, labelpad=8, color='#475569')
    ax.view_init(elev=28, azim=50)
    ax.grid(True, alpha=0.22, linestyle=':')
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_color('#f1f5f9')
        pane.set_edgecolor('#e2e8f0')
    ax.set_title('3D 半球体 — 目标方位', fontsize=11, pad=14,
                 color='#1e293b', fontweight='medium')
    legend = ax.legend(loc='upper right', fontsize=9,
                       framealpha=0.92, facecolor='#ffffff', edgecolor='#e2e8f0')
    legend.get_frame().set_boxstyle('round,pad=0.3')


# ── 水平角俯视雷达底盘（距离范围 0 ~ 5 m，只调用一次）──────────────
def build_radar_scene(ax) -> None:
    ax.set_aspect('equal')
    margin = _RADAR_MAX_DIST * 1.30
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_facecolor('#f8fafc')

    tc = np.linspace(0, 2 * np.pi, 360)

    # 距离同心圆刻度
    for d in range(1, int(_RADAR_MAX_DIST) + 1):
        at_boundary = (d == int(_RADAR_MAX_DIST))
        ax.plot(d * np.sin(tc), d * np.cos(tc),
                color='#3b82f6' if at_boundary else '#64748b',
                lw=2.4 if at_boundary else 0.75,
                alpha=0.85 if at_boundary else 0.32,
                zorder=2)
        ax.text(0.12, d + 0.17, f'{d} m',
                fontsize=7.5, ha='left', va='bottom', color='#475569', alpha=0.75)

    # 辐射刻度线（每 30°）
    for az_deg in range(0, 360, 30):
        az_r = np.deg2rad(az_deg)
        is_main = az_deg % 90 == 0
        ax.plot([0, _RADAR_MAX_DIST * np.sin(az_r)],
                [0, _RADAR_MAX_DIST * np.cos(az_r)],
                color='#64748b', alpha=0.42 if is_main else 0.20,
                lw=1.1 if is_main else 0.65, zorder=1)
        if not is_main:
            r_lbl = _RADAR_MAX_DIST * 1.14
            ax.text(r_lbl * np.sin(az_r), r_lbl * np.cos(az_r),
                    f'{az_deg}°',
                    fontsize=6.5, ha='center', va='center', color='#64748b', alpha=0.65)

    # 主方向标注
    for az_deg, lbl in [(0, '前方'), (90, '右侧'), (180, '后方'), (270, '左侧')]:
        az_r = np.deg2rad(az_deg)
        r_txt = _RADAR_MAX_DIST * 1.20
        ax.text(r_txt * np.sin(az_r), r_txt * np.cos(az_r), lbl,
                fontsize=10, ha='center', va='center',
                color='#1e40af', fontweight='bold')

    # 中心原点
    ax.plot(0, 0, marker='+', color='#334155', markersize=12,
            markeredgewidth=1.8, alpha=0.75, zorder=5)
    ax.plot(0, 0, marker='o', color='#ffffff', markersize=5,
            markeredgecolor='#334155', markeredgewidth=1.2, zorder=6)

    ax.set_xlabel('东 (m)', fontsize=10, labelpad=8, color='#475569')
    ax.set_ylabel('前 (m)', fontsize=10, labelpad=8, color='#475569')
    ax.set_title(f'2D 雷达俯视 — 距离范围 0 ~ {_RADAR_MAX_DIST:.0f} m',
                 fontsize=11, pad=14, color='#1e293b', fontweight='medium')
    ax.tick_params(labelsize=8, color='#64748b')
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_color('#cbd5e1')
    ax.grid(False)


# ── 雷达图动态目标（每帧调用）────────────────────────────────────────
def update_radar(ax, points_dict: dict) -> None:
    global _radar_dynamic_artists
    for artist in _radar_dynamic_artists:
        try:
            artist.remove()
        except ValueError:
            pass
    _radar_dynamic_artists.clear()

    for tid, pd in points_dict.items():
        az       = pd['azimuth']
        dist     = pd.get('distance')
        has_dist = dist is not None

        d_plot = min(float(dist), _RADAR_MAX_DIST) if has_dist else _RADAR_MAX_DIST
        az_r   = np.deg2rad(az)
        px     = d_plot * np.sin(az_r)
        py     = d_plot * np.cos(az_r)
        c      = _id_color(tid)

        # 中心 → 目标连线（动态层，高于所有静态元素）
        line, = ax.plot([0, px], [0, py],
                        color=c, lw=1.8, alpha=0.60, zorder=10,
                        solid_capstyle='round')
        _radar_dynamic_artists.append(line)

        # 目标点
        sc = ax.scatter(px, py,
                        color=c, s=140 if has_dist else 160, zorder=12,
                        edgecolors='#1e293b', linewidths=1.5,
                        marker='o' if has_dist else '^', alpha=0.95)
        _radar_dynamic_artists.append(sc)

        # 标注：沿径向外推，避免超出边界
        norm = max((px ** 2 + py ** 2) ** 0.5, 0.01)
        ox   = (px / norm) * _RADAR_MAX_DIST * 0.09
        oy   = (py / norm) * _RADAR_MAX_DIST * 0.09
        dist_str = f'{dist:.1f} m' if has_dist else '距离未知'
        txt = ax.text(px + ox, py + oy,
                      f'ID {tid}\n方位 {az:.1f}°\n{dist_str}',
                      fontsize=8, color='#1e293b', zorder=14,
                      bbox=dict(boxstyle='round,pad=0.35',
                                facecolor='#ffffff', alpha=0.82,
                                edgecolor=c, linewidth=1.0))
        _radar_dynamic_artists.append(txt)


# ── 3D 半球体动态元素（每帧调用）────────────────────────────────────
def update_dynamic(ax, points_dict: dict) -> None:
    global _sphere_dynamic_artists

    # 精确清理：只移除追踪的动态文字/连线/弧线，不碰静态方向标注
    for artist in _sphere_dynamic_artists:
        try:
            artist.remove()
        except ValueError:
            pass
    _sphere_dynamic_artists.clear()

    # scatter → ax.collections；Arrow3D → ax.patches（按类型清理，不影响静态元素）
    for coll in list(ax.collections):
        coll.remove()
    for patch in [p for p in ax.patches if isinstance(p, Arrow3D)]:
        patch.remove()

    for tid, pd in points_dict.items():
        az, el = pd['azimuth'], pd['elevation']
        x, y, z = spherical_to_cartesian(az, el)
        c = _id_color(tid)
        az_r = np.deg2rad(az)
        el_r = np.deg2rad(el)

        # 水平投影点（仰角=0 处的对应位置）
        x_h = 0.85 * np.sin(az_r)
        y_h = 0.85 * np.cos(az_r)

        # 垂线：目标 → 水平面投影（动态层）
        drop, = ax.plot([x, x_h], [y, y_h], [z, 0],
                        color=c, alpha=0.28, lw=0.9, linestyle=':', zorder=20)
        _sphere_dynamic_artists.append(drop)

        # 仰角弧（在方位平面内，从水平面到目标）
        t_arc = np.linspace(0, el_r, max(int(abs(el)) // 3 + 6, 8))
        r_arc = 0.40
        arc, = ax.plot(r_arc * np.cos(t_arc) * np.sin(az_r),
                       r_arc * np.cos(t_arc) * np.cos(az_r),
                       r_arc * np.sin(t_arc),
                       color=c, alpha=0.42, lw=1.1, linestyle='--', zorder=22)
        _sphere_dynamic_artists.append(arc)

        # 目标点
        ax.scatter(x, y, z, color=c, s=160, alpha=0.98,
                   edgecolors='#1e293b', linewidths=1.5,
                   depthshade=True, zorder=30)

        # 方向箭头
        ax.add_patch(Arrow3D(
            [0, x], [0, y], [0, z],
            mutation_scale=16, arrowstyle='->',
            color=c, alpha=0.80, linewidth=2.5, zorder=25,
        ))

        # 标注文字
        txt = ax.text(x * 1.22, y * 1.22, z * 1.22,
                      f'ID {tid}\n方位 {az:.1f}°  俯仰 {el:.1f}°',
                      fontsize=8.5, color='#1e293b',
                      bbox=dict(boxstyle='round,pad=0.35',
                                facecolor='#ffffff', alpha=0.82,
                                edgecolor=c, linewidth=1.0),
                      zorder=35)
        _sphere_dynamic_artists.append(txt)


# ── 主函数 ────────────────────────────────────────────────────────────
def main():
    global _latest_targets, _conn_status

    parser = argparse.ArgumentParser(description='MeetEye')
    parser.add_argument('url', nargs='?', default='ws://172.16.30.51:8001',
                        help='WebSocket 地址，例如 ws://192.168.1.100:8000')
    parser.add_argument('--test', action='store_true', help='强制使用测试数据')
    args = parser.parse_args()

    test_mode = args.test or (args.url is None)

    if test_mode:
        _conn_status = "测试模式"
        _latest_targets = {
            'A': {'azimuth':   0, 'elevation':  0,  'distance': 1.0},
            'B': {'azimuth':  90, 'elevation': 30,  'distance': 1.9},
            'C': {'azimuth': 180, 'elevation': 60,  'distance': None},
            'D': {'azimuth': 270, 'elevation': -5,  'distance': 4.7},
            'E': {'azimuth':  45, 'elevation': 85,  'distance': None},
        }
        print("=== 测试模式：显示预设固定角度 ===")
        print("真实模式: python angle_visualizer.py ws://HOST:PORT")
    else:
        url = args.url
        if not url.endswith('/ws/inference'):
            url = url.rstrip('/') + '/ws/inference'
        print(f"=== 连接 {url} ===")
        start_ws_thread(url)

    # 左：3D 半球体，右：俯视雷达
    fig = plt.figure(figsize=(17, 7.5))
    fig.patch.set_facecolor('#ffffff')
    gs  = fig.add_gridspec(1, 2, wspace=0.30)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax2d = fig.add_subplot(gs[0, 1])

    build_static_scene(ax3d)
    build_radar_scene(ax2d)
    plt.ion()
    plt.tight_layout(pad=3.2)

    # 标题对象创建一次，后续只更新文字和颜色（避免每帧重建）
    title_main = fig.suptitle(
        'MeetEye',
        fontsize=15, y=0.985, color='#1e293b', fontweight='medium',
    )
    title_sub = fig.text(
        0.5, 0.935, '',
        ha='center', va='center', fontsize=11, color='#10b981',
    )

    last_frame = -1
    print("关闭窗口或按 Ctrl+C 退出")

    try:
        while plt.get_fignums():
            with _lock:
                targets = dict(_latest_targets)
                status  = _conn_status
                fc      = _frame_count

            if test_mode or fc != last_frame:
                n = len(targets)

                if '已连接' in status or '测试' in status:
                    status_color = '#10b981'
                elif '错误' in status or '断开' in status:
                    status_color = '#ef4444'
                else:
                    status_color = '#f59e0b'

                title_sub.set_text(
                    f'状态: {status}  |  帧 #{fc}  |  目标: {n} 个'
                )
                title_sub.set_color(status_color)

                update_dynamic(ax3d, targets)
                update_radar(ax2d, targets)
                fig.canvas.draw_idle()
                last_frame = fc

            plt.pause(0.033)    # ~30 fps，同时处理窗口事件

    except KeyboardInterrupt:
        print("\n程序退出")


if __name__ == '__main__':
    main()
