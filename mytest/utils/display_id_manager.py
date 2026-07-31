from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Set, Tuple


BBox = Tuple[float, float, float, float]


def _bbox(det: dict) -> Optional[BBox]:
    box = det.get("bbox")
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _area(box: Optional[BBox]) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _size_ratio(a: BBox, b: BBox) -> float:
    aw = max(a[2] - a[0], 1.0)
    ah = max(a[3] - a[1], 1.0)
    bw = max(b[2] - b[0], 1.0)
    bh = max(b[3] - b[1], 1.0)
    return min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)


def _center_dist_norm(a: BBox, b: BBox) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    avg_h = max(((a[3] - a[1]) + (b[3] - b[1])) * 0.5, 1.0)
    return math.hypot(acx - bcx, acy - bcy) / avg_h


class DisplayIdManager:
    """Output-only display_id slot manager.

    The manager never rewrites tracker track_id. It only decides which active
    tracks are visible and which 1..N display_id they use.
    """

    def __init__(
        self,
        max_count: int = 8,
        binding_ttl_frames: int = 60,
        reuse_max_frames: int = 900,
        fallback_min_frames: int = 150,
        reuse_center_thresh: float = 2.0,
        reuse_size_ratio: float = 0.4,
    ) -> None:
        self.max_count = max(0, int(max_count))
        self.binding_ttl_frames = max(0, int(binding_ttl_frames))
        self.reuse_max_frames = max(0, int(reuse_max_frames))
        self.fallback_min_frames = max(0, int(fallback_min_frames))
        self.reuse_center_thresh = max(0.0, float(reuse_center_thresh))
        self.reuse_size_ratio = max(0.0, min(float(reuse_size_ratio), 1.0))

        self._display_by_track: Dict[int, int] = {}
        self._free_display_ids: Set[int] = set()
        self._display_counter = 0
        self._display_last_bbox: Dict[int, BBox] = {}
        self._display_last_frame: Dict[int, int] = {}
        self._display_last_track: Dict[int, int] = {}
        self._track_last_frame: Dict[int, int] = {}
        self._track_streak: Dict[int, int] = {}

    def apply(self, detections: List[dict], frame_id: int) -> List[dict]:
        if self.max_count == 0 and not detections:
            return detections

        active_ids = {
            int(det.get("track_id", -1))
            for det in detections
            if int(det.get("track_id", -1)) > 0
        }
        self._release_expired_bindings(active_ids, frame_id)
        self._update_streaks(detections, frame_id)

        active_bbox = {
            int(det.get("track_id", -1)): _bbox(det)
            for det in detections
            if int(det.get("track_id", -1)) > 0
        }

        visible: List[dict] = []
        hidden_ids: Set[int] = set()
        display_index: Dict[int, int] = {}
        process_order = list(range(len(detections)))
        process_order.sort(
            key=lambda idx: self._priority_key(detections[idx]),
            reverse=True,
        )

        for idx in process_order:
            det = detections[idx]
            tid = int(det.get("track_id", -1))
            box = _bbox(det)
            if tid <= 0 or box is None or tid in hidden_ids:
                continue

            display_id = self._display_by_track.get(tid, 0)
            if display_id <= 0:
                display_id = self._choose_reusable_display_id(tid, box, frame_id)
            if display_id <= 0 and (
                self.max_count <= 0 or self._display_counter < self.max_count
            ):
                self._display_counter += 1
                display_id = self._display_counter
            if display_id <= 0:
                replacement = self._choose_replacement(tid, box, active_ids, active_bbox)
                if replacement is None:
                    hidden_ids.add(tid)
                    continue
                display_id, old_tid = replacement
                hidden_ids.add(old_tid)
                old_index = display_index.get(old_tid)
                if old_index is not None:
                    visible[old_index]["display_id"] = None
                self._display_by_track.pop(old_tid, None)

            self._display_by_track[tid] = display_id
            self._free_display_ids.discard(display_id)
            out = dict(det)
            out["display_id"] = display_id
            visible.append(out)
            display_index[tid] = len(visible) - 1
            self._display_last_bbox[display_id] = box
            self._display_last_frame[display_id] = frame_id
            self._display_last_track[display_id] = tid
            self._track_last_frame[tid] = frame_id

        return [det for det in visible if det.get("display_id")]

    def _priority_key(self, det: dict) -> Tuple[int, int, float, int]:
        tid = int(det.get("track_id", -1))
        bound = 1 if tid in self._display_by_track else 0
        streak = self._track_streak.get(tid, 0)
        area = _area(_bbox(det))
        return bound, streak, area, -tid

    def _release_expired_bindings(self, active_ids: Iterable[int], frame_id: int) -> None:
        active = set(active_ids)
        for tid, display_id in list(self._display_by_track.items()):
            if tid in active:
                continue
            last_frame = self._track_last_frame.get(tid, frame_id)
            missing = frame_id - last_frame
            if self.binding_ttl_frames > 0 and missing <= self.binding_ttl_frames:
                continue
            self._display_by_track.pop(tid, None)
            if display_id > 0:
                self._free_display_ids.add(display_id)
                self._display_last_track[display_id] = tid

    def _update_streaks(self, detections: List[dict], frame_id: int) -> None:
        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid <= 0:
                continue
            last = self._track_last_frame.get(tid)
            self._track_streak[tid] = (
                self._track_streak.get(tid, 0) + 1
                if last == frame_id - 1
                else 1
            )

    def _choose_reusable_display_id(self, tid: int, box: BBox, frame_id: int) -> int:
        best_id = 0
        best_score = float("inf")
        for display_id in self._free_display_ids:
            old_box = self._display_last_bbox.get(display_id)
            old_frame = self._display_last_frame.get(display_id)
            if old_box is None or old_frame is None:
                continue
            missing = frame_id - old_frame
            if missing <= 0 or missing > self.reuse_max_frames:
                continue
            if _size_ratio(box, old_box) < self.reuse_size_ratio:
                continue
            dist = _center_dist_norm(box, old_box)
            if dist >= self.reuse_center_thresh:
                continue
            score = dist + missing / max(1, self.reuse_max_frames) * 0.10
            if score < best_score:
                best_score = score
                best_id = display_id
        if best_id > 0:
            return best_id

        best_missing = -1
        best_id = 0
        for display_id in self._free_display_ids:
            old_frame = self._display_last_frame.get(display_id)
            if old_frame is None:
                continue
            missing = frame_id - old_frame
            if missing < self.fallback_min_frames:
                continue
            if missing > best_missing or (missing == best_missing and display_id < best_id):
                best_missing = missing
                best_id = display_id
        return best_id

    def _choose_replacement(
        self,
        new_tid: int,
        new_box: BBox,
        active_ids: Set[int],
        active_bbox: Dict[int, Optional[BBox]],
    ) -> Optional[Tuple[int, int]]:
        new_hits = self._track_streak.get(new_tid, 0)
        new_area = _area(new_box)
        worst: Optional[Tuple[int, float, int, int]] = None
        for old_tid, display_id in self._display_by_track.items():
            if old_tid == new_tid or old_tid not in active_ids:
                continue
            old_box = active_bbox.get(old_tid)
            if old_box is None:
                continue
            old_hits = self._track_streak.get(old_tid, 0)
            old_area = _area(old_box)
            key = (old_hits, old_area, display_id, old_tid)
            if worst is None or key < worst:
                worst = key
        if worst is None:
            return None
        old_hits, old_area, display_id, old_tid = worst
        if new_hits > old_hits or (new_hits == old_hits and new_area > old_area):
            return display_id, old_tid
        return None
