#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def targets(row: dict) -> List[dict]:
    data = row.get("targets")
    if isinstance(data, dict):
        return [v for v in data.values() if isinstance(v, dict)]
    return []


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def circular_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def summarize_numbers(values: Iterable[float]) -> dict:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0}
    vals_sorted = sorted(vals)
    def pct(p):
        idx = min(len(vals_sorted) - 1, max(0, round((len(vals_sorted) - 1) * p / 100.0)))
        return vals_sorted[idx]
    return {
        "n": len(vals_sorted),
        "mean": mean(vals_sorted),
        "median": median(vals_sorted),
        "min": vals_sorted[0],
        "p90": pct(90),
        "p95": pct(95),
        "max": vals_sorted[-1],
    }


def format_summary(label: str, stats: dict, unit: str = "") -> str:
    if not stats or stats.get("n", 0) == 0:
        return f"{label}: n=0"
    suffix = unit
    return (
        f"{label}: n={stats['n']}, mean={stats['mean']:.3f}{suffix}, "
        f"median={stats['median']:.3f}{suffix}, min={stats['min']:.3f}{suffix}, "
        f"p90={stats['p90']:.3f}{suffix}, p95={stats['p95']:.3f}{suffix}, "
        f"max={stats['max']:.3f}{suffix}"
    )


def collect_target_field(rows: List[dict], field: str) -> List[float]:
    vals = []
    for row in rows:
        for target in targets(row):
            value = safe_float(target.get(field))
            if value is not None:
                vals.append(value)
    return vals


def pair_by_frame(fp16_rows: List[dict], int8_rows: List[dict]) -> List[Tuple[dict, dict]]:
    fp16_by_id = {int(row.get("frame_id", i + 1)): row for i, row in enumerate(fp16_rows)}
    int8_by_id = {int(row.get("frame_id", i + 1)): row for i, row in enumerate(int8_rows)}
    ids = sorted(set(fp16_by_id) & set(int8_by_id))
    return [(fp16_by_id[i], int8_by_id[i]) for i in ids]


def greedy_angle_match(fp16_targets: List[dict], int8_targets: List[dict], max_azimuth_delta: float) -> Tuple[list, int, int]:
    fp = []
    it = []
    for idx, target in enumerate(fp16_targets):
        az = safe_float(target.get("azimuth"))
        if az is not None:
            fp.append((idx, target, az))
    for idx, target in enumerate(int8_targets):
        az = safe_float(target.get("azimuth"))
        if az is not None:
            it.append((idx, target, az))

    used_int8 = set()
    matches = []
    for fp_idx, fp_target, fp_az in fp:
        best = None
        for int8_idx, int8_target, int8_az in it:
            if int8_idx in used_int8:
                continue
            da = circular_delta(fp_az, int8_az)
            if best is None or da < best[0]:
                best = (da, int8_idx, int8_target)
        if best is not None and best[0] <= max_azimuth_delta:
            used_int8.add(best[1])
            matches.append((fp_target, best[2], best[0]))
    return matches, max(0, len(fp16_targets) - len(matches)), max(0, len(int8_targets) - len(matches))


def compare(args: argparse.Namespace) -> str:
    left_label = str(getattr(args, "left_label", "FP16"))
    right_label = str(getattr(args, "right_label", "INT8"))
    fp16_path = Path(args.fp16)
    int8_path = Path(args.int8)
    fp16_rows = load_jsonl(fp16_path)
    int8_rows = load_jsonl(int8_path)
    pairs = pair_by_frame(fp16_rows, int8_rows)

    fp16_counts = [len(targets(row)) for row in fp16_rows]
    int8_counts = [len(targets(row)) for row in int8_rows]
    paired_fp16_counts = [len(targets(a)) for a, _ in pairs]
    paired_int8_counts = [len(targets(b)) for _, b in pairs]
    count_diffs = [b - a for a, b in zip(paired_fp16_counts, paired_int8_counts)]

    equal_frames = sum(1 for d in count_diffs if d == 0)
    int8_more_frames = sum(1 for d in count_diffs if d > 0)
    int8_less_frames = sum(1 for d in count_diffs if d < 0)

    diff_counter = Counter(count_diffs)
    top_diffs = sorted(diff_counter.items(), key=lambda kv: (-kv[1], kv[0]))

    worst_less = []
    worst_more = []
    for fp16, int8, diff in zip((a for a, _ in pairs), (b for _, b in pairs), count_diffs):
        item = (
            int(fp16.get("frame_id", 0)),
            len(targets(fp16)),
            len(targets(int8)),
            diff,
        )
        if diff < 0:
            worst_less.append(item)
        elif diff > 0:
            worst_more.append(item)
    worst_less = sorted(worst_less, key=lambda x: (x[3], x[0]))[:20]
    worst_more = sorted(worst_more, key=lambda x: (-x[3], x[0]))[:20]

    match_count = 0
    unmatched_fp16 = 0
    unmatched_int8 = 0
    az_deltas = []
    el_deltas = []
    eye_deltas = []
    dist_deltas = []
    for fp16, int8 in pairs:
        matches, miss_fp16, extra_int8 = greedy_angle_match(
            targets(fp16),
            targets(int8),
            args.match_azimuth_deg,
        )
        unmatched_fp16 += miss_fp16
        unmatched_int8 += extra_int8
        match_count += len(matches)
        for fp_target, int8_target, az_delta in matches:
            az_deltas.append(az_delta)
            fp_el = safe_float(fp_target.get("elevation"))
            int8_el = safe_float(int8_target.get("elevation"))
            if fp_el is not None and int8_el is not None:
                el_deltas.append(abs(int8_el - fp_el))
            fp_eye = safe_float(fp_target.get("eye_pixel_dist"))
            int8_eye = safe_float(int8_target.get("eye_pixel_dist"))
            if fp_eye is not None and int8_eye is not None:
                eye_deltas.append(abs(int8_eye - fp_eye))
            fp_dist = safe_float(fp_target.get("distance"))
            int8_dist = safe_float(int8_target.get("distance"))
            if fp_dist is not None and int8_dist is not None:
                dist_deltas.append(abs(int8_dist - fp_dist))

    total_fp16 = sum(fp16_counts)
    total_int8 = sum(int8_counts)
    paired_total_fp16 = sum(paired_fp16_counts)
    paired_total_int8 = sum(paired_int8_counts)
    ratio = (total_int8 / total_fp16) if total_fp16 else 0.0
    drop_ratio = 1.0 - ratio if total_fp16 else 0.0
    fp16_mean_per_frame = mean(fp16_counts) if fp16_counts else 0.0
    int8_mean_per_frame = mean(int8_counts) if int8_counts else 0.0
    mean_ratio = (int8_mean_per_frame / fp16_mean_per_frame) if fp16_mean_per_frame else 0.0
    mean_drop_ratio = 1.0 - mean_ratio if fp16_mean_per_frame else 0.0

    lines = []
    lines.append(f"YOLOv8n-face {left_label} vs {right_label} JSONL 对比报告")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"{left_label} JSONL: {fp16_path}")
    lines.append(f"{right_label} JSONL: {int8_path}")
    lines.append(f"匹配帧依据: frame_id 交集")
    lines.append("")
    lines.append("总体计数")
    lines.append("-" * 60)
    lines.append(f"{left_label} 文件帧数: {len(fp16_rows)}")
    lines.append(f"{right_label} 文件帧数: {len(int8_rows)}")
    lines.append(f"共同 frame_id 帧数: {len(pairs)}")
    lines.append(f"{left_label} 检测目标总数: {total_fp16}")
    lines.append(f"{right_label} 检测目标总数: {total_int8}")
    lines.append(f"{right_label} - {left_label} 目标总数差: {total_int8 - total_fp16}")
    lines.append(f"{right_label} / {left_label} 总数比例: {ratio:.4f}")
    lines.append(f"相对 {left_label}: {right_label} 保留 {ratio * 100:.2f}%，下降 {drop_ratio * 100:.2f}%")
    lines.append(f"共同帧内 {left_label} 总数: {paired_total_fp16}")
    lines.append(f"共同帧内 {right_label} 总数: {paired_total_int8}")
    lines.append("")
    lines.append("逐帧目标数")
    lines.append("-" * 60)
    lines.append(format_summary(f"{left_label} 每帧目标数", summarize_numbers(fp16_counts)))
    lines.append(format_summary(f"{right_label} 每帧目标数", summarize_numbers(int8_counts)))
    lines.append(format_summary(f"逐帧差值 {right_label}-{left_label}", summarize_numbers(count_diffs)))
    lines.append(f"平均每帧相对 {left_label}: {right_label} 保留 {mean_ratio * 100:.2f}%，下降 {mean_drop_ratio * 100:.2f}%")
    lines.append(f"逐帧目标数相同: {equal_frames} / {len(pairs)} ({equal_frames / len(pairs) * 100:.2f}%)")
    lines.append(f"{right_label} 更多目标的帧: {int8_more_frames} / {len(pairs)} ({int8_more_frames / len(pairs) * 100:.2f}%)")
    lines.append(f"{right_label} 更少目标的帧: {int8_less_frames} / {len(pairs)} ({int8_less_frames / len(pairs) * 100:.2f}%)")
    lines.append(f"逐帧差值分布 diff={right_label}-{left_label}:")
    lines.append("  " + ", ".join(f"{diff}: {cnt}" for diff, cnt in top_diffs))
    lines.append("")
    lines.append("目标字段分布")
    lines.append("-" * 60)
    for field, unit in [
        ("azimuth", "deg"),
        ("elevation", "deg"),
        ("eye_pixel_dist", "px"),
        ("distance", "m"),
    ]:
        lines.append(format_summary(f"{left_label} {field}", summarize_numbers(collect_target_field(fp16_rows, field)), unit))
        lines.append(format_summary(f"{right_label} {field}", summarize_numbers(collect_target_field(int8_rows, field)), unit))
    lines.append("")
    lines.append("按水平角就近匹配后的差异")
    lines.append("-" * 60)
    lines.append(f"匹配规则: 同一 frame_id 内，按 azimuth 贪心匹配，最大允许差 {args.match_azimuth_deg:.1f}deg")
    lines.append(f"匹配目标对数: {match_count}")
    lines.append(f"未匹配 {left_label} 目标数: {unmatched_fp16}")
    lines.append(f"未匹配 {right_label} 目标数: {unmatched_int8}")
    lines.append(format_summary("匹配目标 azimuth 绝对差", summarize_numbers(az_deltas), "deg"))
    lines.append(format_summary("匹配目标 elevation 绝对差", summarize_numbers(el_deltas), "deg"))
    lines.append(format_summary("匹配目标 eye_pixel_dist 绝对差", summarize_numbers(eye_deltas), "px"))
    lines.append(format_summary("匹配目标 distance 绝对差", summarize_numbers(dist_deltas), "m"))
    lines.append("")
    lines.append(f"{right_label} 少检最多的帧")
    lines.append("-" * 60)
    if worst_less:
        for frame_id, fp_count, int8_count, diff in worst_less:
            lines.append(f"frame={frame_id}: {left_label}={fp_count}, {right_label}={int8_count}, diff={diff}")
    else:
        lines.append("无")
    lines.append("")
    lines.append(f"{right_label} 多检最多的帧")
    lines.append("-" * 60)
    if worst_more:
        for frame_id, fp_count, int8_count, diff in worst_more:
            lines.append(f"frame={frame_id}: {left_label}={fp_count}, {right_label}={int8_count}, diff=+{diff}")
    else:
        lines.append("无")
    lines.append("")
    lines.append("结论提示")
    lines.append("-" * 60)
    if total_int8 < total_fp16:
        lines.append(f"{right_label} 总检测数低于 {left_label}，表现为整体更保守或存在更多漏检。")
    elif total_int8 > total_fp16:
        lines.append(f"{right_label} 总检测数高于 {left_label}，表现为整体更激进或可能引入更多误检。")
    else:
        lines.append(f"{right_label} 与 {left_label} 总检测数一致，但仍需看逐帧差异和关键点/角度偏移。")
    lines.append("JSONL 中没有保存置信度和 bbox 坐标，因此本报告无法直接比较 score/bbox IoU。")
    lines.append("若后续要做更严格的量化精度评估，建议 JSON 增加 bbox、confidence、keypoints 原始坐标。")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP16 and INT8 face JSONL outputs.")
    parser.add_argument("--fp16", default="yolo_pose_output/yolov8n-face_fp16.jsonl")
    parser.add_argument("--int8", default="yolo_pose_output/yolov8n-face_int8.jsonl")
    parser.add_argument("--output", default="face_rc/benchmark_results/yolov8n-face_fp16_vs_int8_jsonl_compare.txt")
    parser.add_argument("--left-label", default="FP16")
    parser.add_argument("--right-label", default="INT8")
    parser.add_argument("--match-azimuth-deg", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = compare(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    existed = output.exists() and output.stat().st_size > 0
    with output.open("a", encoding="utf-8") as f:
        if existed:
            f.write("\n\n")
        f.write("#" * 80 + "\n")
        f.write(f"追加时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#" * 80 + "\n\n")
        f.write(report + "\n")
    print(f"appended: {output}")


if __name__ == "__main__":
    main()
