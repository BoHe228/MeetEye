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
