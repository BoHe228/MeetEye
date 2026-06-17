"""
Cluster temporary FaceRec debug dumps.

Input directory is produced by running main_GPU_face_rc_webui.py with:

    --face-debug-dump-dir debug_face_dump/session_xxx

The script clusters saved 512D features by cosine similarity and reports
whether each cluster maps cleanly to one runtime face_id.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np


def _load_metadata(dump_dir: str) -> List[dict]:
    path = os.path.join(dump_dir, "metadata.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid json at {path}:{line_no}: {exc}") from exc
            rows.append(row)
    return rows


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return features / norms


def _load_features(dump_dir: str, rows: List[dict]) -> Tuple[np.ndarray, List[dict]]:
    loaded_rows = []
    features = []
    for row in rows:
        rel = row.get("feature")
        if not rel:
            continue
        path = os.path.join(dump_dir, rel)
        if not os.path.exists(path):
            print(f"[WARN] missing feature, skip: {path}")
            continue
        feat = np.load(path).astype(np.float32).reshape(-1)
        if feat.size != 512:
            print(f"[WARN] feature dim is {feat.size}, skip: {path}")
            continue
        loaded_rows.append(row)
        features.append(feat)

    if not features:
        raise ValueError(f"no usable 512D features in {dump_dir}")
    return _normalize(np.stack(features).astype(np.float32)), loaded_rows


def _connected_components(sim: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim.shape[0]
    seen = np.zeros(n, dtype=bool)
    clusters = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        cluster = []
        while stack:
            idx = stack.pop()
            cluster.append(idx)
            neighbors = np.flatnonzero((sim[idx] >= threshold) & (~seen))
            for nb in neighbors.tolist():
                seen[nb] = True
                stack.append(nb)
        clusters.append(sorted(cluster))
    clusters.sort(key=len, reverse=True)
    return clusters


def _purity(counter: Counter) -> float:
    total = sum(counter.values())
    return 0.0 if total <= 0 else max(counter.values()) / total


def _top_items(counter: Counter, limit: int = 5) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def _write_preview_html(
    dump_dir: str,
    rows: List[dict],
    clusters: List[List[int]],
    output_name: str,
    max_images_per_cluster: int,
) -> str:
    output_path = os.path.join(dump_dir, output_name)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        "<title>Face Debug Clusters</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;margin:20px;background:#f7f7f7;color:#111}",
        ".cluster{background:#fff;border:1px solid #ddd;margin:0 0 16px;padding:12px}",
        ".grid{display:flex;flex-wrap:wrap;gap:8px}",
        ".item{width:132px;font-size:12px;line-height:1.35}",
        "img{width:112px;height:112px;object-fit:cover;border:1px solid #ccc}",
        "</style>",
        "<h1>Face Debug Clusters</h1>",
    ]
    for cid, indices in enumerate(clusters, start=1):
        face_ids = Counter(rows[i].get("face_id") or "unknown" for i in indices)
        track_ids = Counter(str(rows[i].get("track_id")) for i in indices)
        parts.append(
            f"<section class='cluster'><h2>Cluster {cid}: n={len(indices)}, "
            f"face_ids={_top_items(face_ids)}, tracks={_top_items(track_ids)}</h2>"
        )
        parts.append("<div class='grid'>")
        for idx in indices[:max_images_per_cluster]:
            row = rows[idx]
            img = row.get("aligned_face") or row.get("bbox_crop")
            img_html = ""
            if img:
                img_html = f"<img src='{img}' alt='sample {row.get('sample_idx')}'>"
            score = row.get("score")
            score_text = "-" if score is None else f"{float(score):.3f}"
            parts.append(
                "<div class='item'>"
                f"{img_html}<br>"
                f"sample={row.get('sample_idx')}<br>"
                f"frame={row.get('frame_id')} track={row.get('track_id')}<br>"
                f"face={row.get('face_id') or 'unknown'} score={score_text}<br>"
                f"event={row.get('event')}"
                "</div>"
            )
        parts.append("</div></section>")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return output_path


def analyze_dump(
    dump_dir: str,
    threshold: float,
    min_cluster_size: int,
    preview_html: str,
    max_images_per_cluster: int,
) -> None:
    rows = _load_metadata(dump_dir)
    features, rows = _load_features(dump_dir, rows)
    sim = features @ features.T
    clusters = [
        c for c in _connected_components(sim, threshold)
        if len(c) >= min_cluster_size
    ]

    print(f"dump_dir: {os.path.abspath(dump_dir)}")
    print(f"samples: {len(rows)}")
    print(f"threshold: {threshold:.3f}")
    print(f"clusters: {len(clusters)}")
    print()

    face_to_clusters: Dict[str, List[int]] = defaultdict(list)
    for cid, indices in enumerate(clusters, start=1):
        face_ids = Counter(rows[i].get("face_id") or "unknown" for i in indices)
        track_ids = Counter(str(rows[i].get("track_id")) for i in indices)
        events = Counter(rows[i].get("event") or "unknown" for i in indices)
        cluster_sim = sim[np.ix_(indices, indices)]
        if len(indices) > 1:
            tri = cluster_sim[np.triu_indices(len(indices), k=1)]
            min_sim = float(np.min(tri))
            mean_sim = float(np.mean(tri))
        else:
            min_sim = 1.0
            mean_sim = 1.0

        for face_id in face_ids:
            face_to_clusters[face_id].append(cid)

        print(f"cluster {cid:03d}  n={len(indices)}  purity={_purity(face_ids):.3f}")
        print(f"  face_ids: {_top_items(face_ids)}")
        print(f"  track_ids: {_top_items(track_ids)}")
        print(f"  events: {_top_items(events)}")
        print(f"  pairwise_cosine: mean={mean_sim:.3f}, min={min_sim:.3f}")

    split_faces = {
        face_id: ids
        for face_id, ids in face_to_clusters.items()
        if face_id != "unknown" and len(ids) > 1
    }
    mixed_clusters = []
    for cid, indices in enumerate(clusters, start=1):
        face_ids = {rows[i].get("face_id") or "unknown" for i in indices}
        face_ids.discard("unknown")
        if len(face_ids) > 1:
            mixed_clusters.append(cid)

    print()
    print("summary:")
    print(f"  mixed_clusters(face cluster contains multiple face_id): {mixed_clusters or '-'}")
    if split_faces:
        text = ", ".join(f"{face_id}->{ids}" for face_id, ids in sorted(split_faces.items()))
        print(f"  split_face_ids(one face_id appears in multiple clusters): {text}")
    else:
        print("  split_face_ids(one face_id appears in multiple clusters): -")

    if preview_html:
        path = _write_preview_html(
            dump_dir, rows, clusters, preview_html, max_images_per_cluster
        )
        print(f"  preview_html: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster FaceRec debug dump features and compare clusters to face_id."
    )
    parser.add_argument("dump_dir", help="Directory passed to --face-debug-dump-dir")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Cosine similarity threshold for connected-component clustering.")
    parser.add_argument("--min-cluster-size", type=int, default=1,
                        help="Hide clusters smaller than this size.")
    parser.add_argument("--preview-html", type=str, default="clusters.html",
                        help="Write an HTML crop preview in dump_dir; empty string disables it.")
    parser.add_argument("--max-images-per-cluster", type=int, default=40,
                        help="Maximum images shown for each cluster in the HTML preview.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_dump(
        args.dump_dir,
        threshold=args.threshold,
        min_cluster_size=max(1, args.min_cluster_size),
        preview_html=args.preview_html,
        max_images_per_cluster=max(1, args.max_images_per_cluster),
    )
