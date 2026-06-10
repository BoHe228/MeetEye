"""
扇区聚合：水平 360° 等分为 num_sectors 个扇区，忽略 track_id，
按目标水平角分入扇区，每扇区取检测框面积最大者作为代表。

供两处共用，保证「谁是扇区代表」判定一致：
  · webui/processor.py —— 用返回的代表下标把代表框画成红色强调
  · main_GPU_webui.py  —— 用返回的扇区字典构建 JSON

依赖 angle_info['persons'][i] 与 tracked[i] 1:1 对齐（processor 对无关键点的
补漏框用合成鼻子点占位，每个目标都有一条角度记录）。
"""
from typing import List, Dict, Optional, Set, Tuple

import cv2
import numpy as np


def aggregate_sectors(
    tracked: List[dict],
    angle_info: Optional[dict],
    num_sectors: int,
) -> Tuple[Dict[str, dict], Set[int]]:
    """
    返回 (sectors, rep_indices)：
      sectors:      {扇区下标str: {'has_target': bool, 'azimuth': float|None, 'elevation': float|None}}
      rep_indices:  被选为扇区代表的 tracked 下标集合（用于高亮其检测框）
    """
    num_sectors = max(1, int(num_sectors))
    best: List[Optional[tuple]] = [None] * num_sectors   # 每扇区: (area, azimuth, elevation, tracked_idx)
    persons = (angle_info or {}).get('persons', [])
    sector_size = 360.0 / num_sectors

    for i, det in enumerate(tracked or []):
        angle = persons[i] if i < len(persons) else None
        if angle is None:
            continue
        az = float(angle['azimuth_deg'])
        el = float(angle['elevation_deg'])
        x1, y1, x2, y2 = det['bbox']
        area = (x2 - x1) * (y2 - y1)
        s = int(az // sector_size) % num_sectors
        if best[s] is None or area > best[s][0]:
            best[s] = (area, az, el, i)

    sectors: Dict[str, dict] = {}
    rep_indices: Set[int] = set()
    for s in range(num_sectors):
        v = best[s]
        sectors[str(s)] = {
            'has_target': v is not None,
            'azimuth':    round(v[1], 3) if v else None,
            'elevation':  round(v[2], 3) if v else None,
        }
        if v is not None:
            rep_indices.add(v[3])
    return sectors, rep_indices


def draw_sector_grid(
    image: np.ndarray,
    num_sectors: int,
    sectors: Optional[Dict[str, dict]] = None,
    inplace: bool = False,
) -> np.ndarray:
    """
    在全景图上画出扇区范围（--show-sectors）。

    映射与 angle_calculator._pixel_to_angle 一致：azimuth = 360·x/W 线性，
    所以扇区边界就是均匀竖线 x = W·s/num_sectors。每个扇区顶部标注
    「编号 + 角度区间」；若传入 sectors 字典，则把「有目标」的扇区底色高亮。

    inplace=False（默认）在副本上绘制；调用处若已持有可改写的缓冲，传 inplace=True
    省掉一次整帧拷贝（实时路径用）。
    """
    num_sectors = max(1, int(num_sectors))
    out = image if inplace else image.copy()
    h, w = out.shape[:2]
    sector_deg = 360.0 / num_sectors

    for s in range(num_sectors):
        x0 = int(round(w * s / num_sectors))
        x1 = int(round(w * (s + 1) / num_sectors))

        # 有目标的扇区：顶部淡色色带强调
        if sectors is not None and sectors.get(str(s), {}).get('has_target'):
            band = out[0:28, x0:x1].copy()
            tint = np.full_like(band, (0, 80, 160))  # BGR 暖橙
            out[0:28, x0:x1] = cv2.addWeighted(band, 0.5, tint, 0.5, 0)

        # 边界竖线（s==0 的左边界即图像左缘，也画出来）
        cv2.line(out, (x0, 0), (x0, h - 1), (0, 255, 255), 1, cv2.LINE_AA)

        # 标注：扇区号 + 角度区间
        label = f"S{s} [{int(s * sector_deg)}-{int((s + 1) * sector_deg)})"
        cv2.putText(out, label, (x0 + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x0 + 4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # 最右边界（= 360°/0° 接缝）
    cv2.line(out, (w - 1, 0), (w - 1, h - 1), (0, 255, 255), 1, cv2.LINE_AA)
    return out
