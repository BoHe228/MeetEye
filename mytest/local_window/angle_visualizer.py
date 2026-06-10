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
_sector_mode: bool = False   # True：服务端为 --sector-output 扇区格式，标签显示「扇区 N」

# 动态元素追踪（用于帧间精确清理，避免误删静态标注）
_radar_dynamic_artists: list = []
_sphere_dynamic_artists: list = []   # 3D 文字、连线、弧线（scatter/Arrow3D 由类型清理）
_legend_artists: list = []           # 底部图注区域的动态元素

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
    # 尝试将ID解析为数字，如果是数字直接用数字索引
    try:
        idx = int(float(tid))
        return _PALETTE[idx % len(_PALETTE)]
    except (ValueError, TypeError):
        # 如果不是数字，使用改进的哈希方法
        h = 0
        for char in str(tid):
            h = (h * 31 + ord(char)) & 0xFFFFFFFF
        return _PALETTE[h % len(_PALETTE)]


def _disp_label(tid: str, long: bool = False) -> str:
    """扇区模式下显示「扇区 N」（long）/「扇N」（short，点旁）；否则原 ID。"""
    if _sector_mode:
        num = str(tid).lstrip('S')
        return f'扇区 {num}' if long else f'扇{num}'
    return f'ID {tid}' if long else f'{tid}'


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
    global _latest_targets, _frame_count, _sector_mode
    try:
        data = json.loads(message)
        new_targets = {}
        _sector_mode = ('sectors' in data)

        if 'sectors' in data:
            # 扇区聚合格式（--sector-output）：每个有目标的扇区贡献一个点
            # （键用 S<编号>，扇区格式无距离 → distance=None）
            for sid, sdata in data['sectors'].items():
                if not sdata.get('has_target'):
                    continue
                az = sdata.get('azimuth')
                el = sdata.get('elevation')
                if az is None or el is None:
                    continue
                new_targets[f'S{sid}'] = {
                    'azimuth':   float(az),
                    'elevation': float(el),
                    'distance':  None,
                }
        else:
            # 兼容旧格式：按 track_id 输出
            for tid, tdata in data.get('targets', {}).items():
                az = tdata.get('azimuth')
                el = tdata.get('elevation')
                if az is None or el is None:
                    continue
                dist = tdata.get('distance')
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

        # 在点旁边只显示简单的ID编号，不显示完整信息
        norm = max((px ** 2 + py ** 2) ** 0.5, 0.01)
        ox   = (px / norm) * _RADAR_MAX_DIST * 0.06
        oy   = (py / norm) * _RADAR_MAX_DIST * 0.06
        txt = ax.text(px + ox, py + oy,
                      _disp_label(tid),
                      fontsize=9, color='#1e293b', zorder=14,
                      fontweight='bold')
        _radar_dynamic_artists.append(txt)


# ── 更新中间图注（每帧调用）────────────────────────────────────────
def update_legend(fig, points_dict: dict) -> None:
    global _legend_artists
    for artist in _legend_artists:
        try:
            artist.remove()
        except ValueError:
            pass
    _legend_artists.clear()

    # 标题
    title = fig.text(0.5, 0.90, '扇区信息' if _sector_mode else '目标信息',
                     fontsize=12, ha='center',
                     color='#1e293b', fontweight='medium')
    _legend_artists.append(title)

    if not points_dict:
        return

    # 按ID排序，保持图注顺序稳定
    sorted_items = sorted(points_dict.items(), key=lambda x: str(x[0]))

    n = len(sorted_items)
    # 起始位置（基于 figure 坐标），垂直居中，增加行间距
    item_height = 0.10
    total_height = n * item_height
    y_start = 0.5 + total_height / 2 - item_height * 0.2
    x_start = 0.40

    for i, (tid, pd) in enumerate(sorted_items):
        y = y_start - i * item_height

        c = _id_color(tid)
        az = pd['azimuth']
        el = pd['elevation']
        dist = pd.get('distance')

        # 绘制颜色标记点
        point = fig.text(x_start, y, '●', fontsize=14, color=c, va='center')
        _legend_artists.append(point)

        # 标题（扇区模式「扇区 N」，否则「ID xx」）
        id_txt = fig.text(x_start + 0.025, y, _disp_label(tid, long=True), fontsize=10,
                          color='#1e293b', va='center', fontweight='bold')
        _legend_artists.append(id_txt)

        # 方位
        az_txt = fig.text(x_start + 0.025, y - 0.03, f'方位: {az:.1f}°',
                          fontsize=10, color='#475569', va='center')
        _legend_artists.append(az_txt)

        # 俯仰
        el_txt = fig.text(x_start + 0.11, y - 0.03, f'俯仰: {el:.1f}°',
                          fontsize=10, color='#475569', va='center')
        _legend_artists.append(el_txt)

        # 距离
        dist_str = f'{dist:.1f} m' if dist is not None else '—'
        dist_txt = fig.text(x_start + 0.025, y - 0.058, f'距离: {dist_str}',
                            fontsize=10, color='#475569', va='center')
        _legend_artists.append(dist_txt)


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
        ax.scatter(x, y, z, color=c, s=100, alpha=0.98,
                   edgecolors='#1e293b', linewidths=1.5,
                   depthshade=True, zorder=30)

        # 方向箭头
        ax.add_patch(Arrow3D(
            [0, x], [0, y], [0, z],
            mutation_scale=16, arrowstyle='->',
            color=c, alpha=0.80, linewidth=2.5, zorder=25,
        ))

        # 在点旁边只显示简单的ID编号，不显示完整信息
        txt = ax.text(x * 1.18, y * 1.18, z * 1.18,
                      _disp_label(tid),
                      fontsize=10, color='#1e293b',
                      fontweight='bold', zorder=35)
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

    # 左：3D 半球体，中：图注区，右：俯视雷达
    fig = plt.figure(figsize=(10, 4))
    fig.patch.set_facecolor('#ffffff')
    gs  = fig.add_gridspec(1, 3, width_ratios=[1.1, 0.5, 1], wspace=0.3)
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax2d = fig.add_subplot(gs[0, 2])

    build_static_scene(ax3d)
    build_radar_scene(ax2d)
    plt.ion()
    plt.tight_layout(pad=2.5)

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

                _cnt_label = '有目标扇区' if _sector_mode else '目标'
                title_sub.set_text(
                    f'状态: {status}  |  帧 #{fc}  |  {_cnt_label}: {n} 个'
                )
                title_sub.set_color(status_color)

                update_dynamic(ax3d, targets)
                update_radar(ax2d, targets)
                update_legend(fig, targets)
                fig.canvas.draw_idle()
                last_frame = fc

            plt.pause(0.033)    # ~30 fps，同时处理窗口事件

    except KeyboardInterrupt:
        print("\n程序退出")


if __name__ == '__main__':
    main()
