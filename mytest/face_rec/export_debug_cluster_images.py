"""
Export FaceRec debug clusters into per-cluster image folders.

This is useful when a single large HTML page is inconvenient to inspect.

Example:

    python mytest/face_rec/export_debug_cluster_images.py \
        debug_face_dump/session_test_6.4 \
        --threshold 0.55 \
        --min-cluster-size 3
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Sequence

import numpy as np

from cluster_debug_dump import (
    _connected_components,
    _load_features,
    _load_metadata,
    _top_items,
)


def _safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_.#=-]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return (text or "unknown")[:max_len]


def _cluster_dir_name(cid: int, indices: Sequence[int], rows: Sequence[dict]) -> str:
    face_ids = Counter(rows[i].get("face_id") or "unknown" for i in indices)
    face_part = _safe_name(_top_items(face_ids, limit=4).replace(", ", "__"))
    return f"cluster_{cid:03d}_n{len(indices)}_{face_part}"


def _sample_stem(row: dict, rank: int) -> str:
    face_id = row.get("face_id") or "unknown"
    event = row.get("event") or "event"
    return _safe_name(
        f"{rank:04d}_sample{row.get('sample_idx')}_"
        f"frame{row.get('frame_id')}_track{row.get('track_id')}_"
        f"{face_id}_{event}",
        max_len=140,
    )


def _copy_image(dump_dir: Path, rel_path: str | None, out_path: Path) -> bool:
    if not rel_path:
        return False
    src = dump_dir / rel_path
    if not src.exists():
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out_path)
    return True


def _write_cluster_html(
    cluster_dir: Path,
    cid: int,
    indices: Sequence[int],
    rows: Sequence[dict],
) -> None:
    face_ids = Counter(rows[i].get("face_id") or "unknown" for i in indices)
    track_ids = Counter(str(rows[i].get("track_id")) for i in indices)
    events = Counter(rows[i].get("event") or "unknown" for i in indices)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>Cluster {cid:03d}</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:18px;background:#f7f7f7;color:#111}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}",
        ".item{background:#fff;border:1px solid #ddd;padding:8px}",
        ".imgs{display:flex;gap:8px;align-items:flex-start}",
        "img{max-width:120px;max-height:160px;object-fit:contain;border:1px solid #ccc;background:#eee}",
        ".meta{font-size:12px;line-height:1.45;margin-top:6px;word-break:break-all}",
        "</style>",
        f"<h1>Cluster {cid:03d}: n={len(indices)}</h1>",
        f"<p>face_ids: {html.escape(_top_items(face_ids, limit=10))}</p>",
        f"<p>track_ids: {html.escape(_top_items(track_ids, limit=10))}</p>",
        f"<p>events: {html.escape(_top_items(events, limit=10))}</p>",
        "<div class='grid'>",
    ]
    for rank, idx in enumerate(indices, start=1):
        row = rows[idx]
        stem = _sample_stem(row, rank)
        aligned = f"aligned/{stem}.jpg"
        bbox = f"bbox/{stem}.jpg"
        score = row.get("score")
        score_text = "-" if score is None else f"{float(score):.3f}"
        parts.extend([
            "<div class='item'>",
            "<div class='imgs'>",
            f"<div><img src='{html.escape(aligned)}'><br>aligned</div>",
            f"<div><img src='{html.escape(bbox)}'><br>bbox</div>",
            "</div>",
            "<div class='meta'>",
            f"rank={rank}<br>",
            f"sample={html.escape(str(row.get('sample_idx')))}<br>",
            f"frame={html.escape(str(row.get('frame_id')))} track={html.escape(str(row.get('track_id')))}<br>",
            f"face={html.escape(str(row.get('face_id') or 'unknown'))} score={score_text}<br>",
            f"event={html.escape(str(row.get('event') or '-'))}<br>",
            f"bbox={html.escape(str(row.get('bbox')))}",
            "</div></div>",
        ])
    parts.append("</div>")
    (cluster_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _write_summary(
    output_dir: Path,
    clusters: Sequence[Sequence[int]],
    rows: Sequence[dict],
    threshold: float,
    min_cluster_size: int,
) -> None:
    lines = [
        "FaceRec debug cluster export",
        "",
        f"threshold: {threshold:.3f}",
        f"min_cluster_size: {min_cluster_size}",
        f"clusters: {len(clusters)}",
        "",
    ]
    for cid, indices in enumerate(clusters, start=1):
        face_ids = Counter(rows[i].get("face_id") or "unknown" for i in indices)
        track_ids = Counter(str(rows[i].get("track_id")) for i in indices)
        dirname = _cluster_dir_name(cid, indices, rows)
        lines.extend([
            f"cluster {cid:03d}: n={len(indices)}",
            f"  dir: {dirname}",
            f"  face_ids: {_top_items(face_ids, limit=10)}",
            f"  track_ids: {_top_items(track_ids, limit=10)}",
            "",
        ])
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def export_clusters(
    dump_dir: str,
    threshold: float,
    min_cluster_size: int,
    output_dir: str,
) -> None:
    dump_path = Path(dump_dir)
    rows = _load_metadata(str(dump_path))
    features, rows = _load_features(str(dump_path), rows)
    sim = features @ features.T
    clusters = [
        c for c in _connected_components(sim, threshold)
        if len(c) >= min_cluster_size
    ]

    out_path = Path(output_dir) if output_dir else dump_path / (
        f"cluster_exports_t{threshold:.2f}_min{min_cluster_size}".replace(".", "")
    )
    out_path.mkdir(parents=True, exist_ok=True)

    for cid, indices in enumerate(clusters, start=1):
        cluster_dir = out_path / _cluster_dir_name(cid, indices, rows)
        (cluster_dir / "aligned").mkdir(parents=True, exist_ok=True)
        (cluster_dir / "bbox").mkdir(parents=True, exist_ok=True)
        for rank, idx in enumerate(indices, start=1):
            row = rows[idx]
            stem = _sample_stem(row, rank)
            _copy_image(
                dump_path,
                row.get("aligned_face"),
                cluster_dir / "aligned" / f"{stem}.jpg",
            )
            _copy_image(
                dump_path,
                row.get("bbox_crop"),
                cluster_dir / "bbox" / f"{stem}.jpg",
            )
        _write_cluster_html(cluster_dir, cid, indices, rows)

    _write_summary(out_path, clusters, rows, threshold, min_cluster_size)
    print(f"dump_dir: {dump_path.resolve()}")
    print(f"samples: {len(rows)}")
    print(f"clusters exported: {len(clusters)}")
    print(f"output_dir: {out_path.resolve()}")
    print(f"summary: {(out_path / 'summary.txt').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FaceRec debug clusters into per-cluster folders."
    )
    parser.add_argument("dump_dir", help="Directory passed to --face-debug-dump-dir")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_clusters(
        args.dump_dir,
        threshold=args.threshold,
        min_cluster_size=max(1, args.min_cluster_size),
        output_dir=args.output_dir,
    )
