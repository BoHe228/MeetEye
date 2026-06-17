"""
Plot FaceRec debug dump clustering results.

Example:

    python mytest/face_rec/plot_debug_clusters.py \
        debug_face_dump/session_test \
        --threshold 0.55 \
        --min-cluster-size 3
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Dict, List, Sequence

import numpy as np

from cluster_debug_dump import (
    _connected_components,
    _load_features,
    _load_metadata,
)


def _prepare_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _project(features: np.ndarray, method: str, perplexity: float, random_state: int) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=random_state).fit_transform(features)

    from sklearn.manifold import TSNE

    n = features.shape[0]
    safe_perplexity = min(float(perplexity), max(2.0, (n - 1) / 3.0))
    return TSNE(
        n_components=2,
        metric="cosine",
        init="random",
        learning_rate="auto",
        perplexity=safe_perplexity,
        random_state=random_state,
    ).fit_transform(features)


def _cluster_labels(
    features: np.ndarray,
    threshold: float,
    min_cluster_size: int,
) -> tuple[List[str], List[List[int]]]:
    sim = features @ features.T
    clusters = _connected_components(sim, threshold)
    kept = [indices for indices in clusters if len(indices) >= min_cluster_size]

    labels = ["small/noise"] * features.shape[0]
    for cid, indices in enumerate(kept, start=1):
        label = f"C{cid:02d}"
        for idx in indices:
            labels[idx] = label
    return labels, kept


def _face_labels(rows: Sequence[dict]) -> List[str]:
    return [str(row.get("face_id") or "unknown") for row in rows]


def _label_color_map(labels: Sequence[str], plt) -> Dict[str, object]:
    ordered = [label for label, _count in Counter(labels).most_common()]
    cmap = plt.get_cmap("tab20")
    colors = {}
    color_idx = 0
    for label in ordered:
        if label in {"small/noise", "unknown"}:
            colors[label] = "#b0b0b0"
        else:
            colors[label] = cmap(color_idx % 20)
            color_idx += 1
    return colors


def _plot_scatter(
    points: np.ndarray,
    labels: Sequence[str],
    title: str,
    output_path: str,
    xlabel: str,
    ylabel: str,
    annotate_clusters: bool = False,
) -> None:
    plt = _prepare_matplotlib()
    colors = _label_color_map(labels, plt)
    counts = Counter(labels)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)

    for label, _count in counts.most_common():
        mask = np.array([item == label for item in labels], dtype=bool)
        alpha = 0.35 if label in {"small/noise", "unknown"} else 0.82
        size = 24 if label in {"small/noise", "unknown"} else 34
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=size,
            c=[colors[label]],
            label=f"{label} ({counts[label]})",
            alpha=alpha,
            linewidths=0.25,
            edgecolors="white",
        )
        if annotate_clusters and label != "small/noise":
            cx = float(np.mean(points[mask, 0]))
            cy = float(np.mean(points[mask, 1]))
            ax.text(cx, cy, label, fontsize=10, weight="bold", ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.22)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_heatmap(
    cluster_labels: Sequence[str],
    face_labels: Sequence[str],
    output_path: str,
) -> None:
    plt = _prepare_matplotlib()
    cluster_order = [
        label for label, _count in Counter(cluster_labels).most_common()
        if label != "small/noise"
    ]
    if "small/noise" in cluster_labels:
        cluster_order.append("small/noise")
    face_order = [label for label, _count in Counter(face_labels).most_common()]

    matrix = np.zeros((len(cluster_order), len(face_order)), dtype=np.int32)
    cluster_to_idx = {label: idx for idx, label in enumerate(cluster_order)}
    face_to_idx = {label: idx for idx, label in enumerate(face_order)}
    for cluster, face in zip(cluster_labels, face_labels):
        matrix[cluster_to_idx[cluster], face_to_idx[face]] += 1

    fig_w = max(9.0, 1.0 + len(face_order) * 0.85)
    fig_h = max(6.0, 1.2 + len(cluster_order) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_title("Cluster x FaceID count")
    ax.set_xlabel("runtime face_id")
    ax.set_ylabel("embedding cluster")
    ax.set_xticks(np.arange(len(face_order)), labels=face_order, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(cluster_order)), labels=cluster_order)

    max_value = int(matrix.max()) if matrix.size else 0
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = int(matrix[y, x])
            if value <= 0:
                continue
            color = "white" if max_value and value > max_value * 0.45 else "black"
            ax.text(x, y, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_dump(
    dump_dir: str,
    threshold: float,
    min_cluster_size: int,
    method: str,
    perplexity: float,
    random_state: int,
    output_prefix: str,
) -> None:
    rows = _load_metadata(dump_dir)
    features, rows = _load_features(dump_dir, rows)
    points = _project(features, method, perplexity, random_state)
    cluster_labels, clusters = _cluster_labels(features, threshold, min_cluster_size)
    faces = _face_labels(rows)

    suffix = f"{method}_t{threshold:.2f}_min{min_cluster_size}".replace(".", "")
    prefix = output_prefix or f"cluster_plot_{suffix}"

    cluster_png = os.path.join(dump_dir, f"{prefix}_by_cluster.png")
    face_png = os.path.join(dump_dir, f"{prefix}_by_faceid.png")
    heatmap_png = os.path.join(dump_dir, f"{prefix}_heatmap.png")

    axis_name = method.upper()
    _plot_scatter(
        points,
        cluster_labels,
        title=f"Face embeddings by cluster ({method}, threshold={threshold:.2f}, min={min_cluster_size})",
        output_path=cluster_png,
        xlabel=f"{axis_name}-1",
        ylabel=f"{axis_name}-2",
        annotate_clusters=True,
    )
    _plot_scatter(
        points,
        faces,
        title=f"Face embeddings by runtime face_id ({method})",
        output_path=face_png,
        xlabel=f"{axis_name}-1",
        ylabel=f"{axis_name}-2",
    )
    _plot_heatmap(cluster_labels, faces, heatmap_png)

    mixed = []
    for cid, indices in enumerate(clusters, start=1):
        face_set = {faces[idx] for idx in indices}
        face_set.discard("unknown")
        if len(face_set) > 1:
            mixed.append(f"C{cid:02d}")

    print(f"dump_dir: {os.path.abspath(dump_dir)}")
    print(f"samples: {len(rows)}")
    print(f"clusters >= min size: {len(clusters)}")
    print(f"mixed clusters: {mixed or '-'}")
    print(f"cluster scatter: {cluster_png}")
    print(f"face_id scatter: {face_png}")
    print(f"heatmap: {heatmap_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot FaceRec debug dump clusters.")
    parser.add_argument("dump_dir", help="Directory passed to --face-debug-dump-dir")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--output-prefix", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_dump(
        args.dump_dir,
        threshold=args.threshold,
        min_cluster_size=max(1, args.min_cluster_size),
        method=args.method,
        perplexity=args.perplexity,
        random_state=args.random_state,
        output_prefix=args.output_prefix,
    )
