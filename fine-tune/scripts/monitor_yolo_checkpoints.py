#!/usr/bin/env python3
"""Monitor a YOLO training run and keep extra checkpoints by validation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "runs/pose/yolo26n-pose-omnilab"


@dataclass(frozen=True)
class MetricRule:
    name: str
    column: str
    mode: str = "max"


METRIC_RULES = [
    MetricRule("best_pose_map5095", "metrics/mAP50-95(P)"),
    MetricRule("best_pose_map50", "metrics/mAP50(P)"),
    MetricRule("best_pose_recall", "metrics/recall(P)"),
    MetricRule("best_pose_precision", "metrics/precision(P)"),
    MetricRule("best_box_map5095", "metrics/mAP50-95(B)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="YOLO run directory.")
    parser.add_argument("--interval", type=float, default=30.0, help="Polling interval in seconds.")
    parser.add_argument("--top-k", type=int, default=3, help="Keep top K checkpoints for each metric.")
    parser.add_argument("--every", type=int, default=5, help="Also keep every N epochs. Set 0 to disable.")
    parser.add_argument("--keep-last", type=int, default=3, help="Keep the latest N epoch snapshots.")
    parser.add_argument("--once", action="store_true", help="Run one update and exit.")
    return parser.parse_args()


def clean_key(key: str) -> str:
    return key.replace("/", "_").replace("(", "").replace(")", "").replace("-", "")


def read_results(results_path: Path) -> list[dict[str, float]]:
    if not results_path.exists():
        return []

    rows: list[dict[str, float]] = []
    with results_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            row: dict[str, float] = {}
            for key, value in raw.items():
                if key is None:
                    continue
                key = key.strip()
                value = (value or "").strip()
                if not key or not value:
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    continue
            if "epoch" in row:
                rows.append(row)
    return rows


def checkpoint_name(epoch: int, row: dict[str, float]) -> str:
    pose_map = row.get("metrics/mAP50-95(P)", 0.0)
    pose_recall = row.get("metrics/recall(P)", 0.0)
    return f"epoch_{epoch:03d}_posemap{pose_map:.4f}_poserecall{pose_recall:.4f}.pt"


def wait_for_stable_file(path: Path, delay: float = 1.0, attempts: int = 5) -> bool:
    previous = -1
    for _ in range(attempts):
        if not path.exists():
            return False
        current = path.stat().st_size
        if current > 0 and current == previous:
            return True
        previous = current
        time.sleep(delay)
    return path.exists() and path.stat().st_size > 0


def copy_checkpoint(src: Path, dst: Path, epoch: int, row: dict[str, float]) -> bool:
    if dst.exists():
        return False
    if not wait_for_stable_file(src):
        return False
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)
    meta = {
        "epoch": epoch,
        "source": str(src),
        "checkpoint": str(dst),
        "metrics": {key: row.get(key) for key in row if key.startswith("metrics/")},
        "losses": {key: row.get(key) for key in row if "loss" in key},
    }
    dst.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return True


def desired_epochs(rows: list[dict[str, float]], args: argparse.Namespace) -> dict[int, set[str]]:
    wanted: dict[int, set[str]] = {}

    def add(epoch: int, reason: str) -> None:
        wanted.setdefault(epoch, set()).add(reason)

    for rule in METRIC_RULES:
        candidates = [r for r in rows if rule.column in r]
        reverse = rule.mode == "max"
        for row in sorted(candidates, key=lambda r: r[rule.column], reverse=reverse)[: args.top_k]:
            add(int(row["epoch"]), rule.name)

    if args.every > 0:
        for row in rows:
            epoch = int(row["epoch"])
            if epoch > 0 and epoch % args.every == 0:
                add(epoch, f"every_{args.every}")

    if args.keep_last > 0:
        for row in sorted(rows, key=lambda r: r["epoch"], reverse=True)[: args.keep_last]:
            add(int(row["epoch"]), "latest")

    return wanted


def index_existing(out_dir: Path) -> dict[int, Path]:
    existing: dict[int, Path] = {}
    for path in sorted(out_dir.glob("epoch_*.pt")):
        parts = path.stem.split("_")
        if len(parts) < 2:
            continue
        try:
            epoch = int(parts[1])
        except ValueError:
            continue
        existing[epoch] = path
    return existing


def write_summary(out_dir: Path, rows: list[dict[str, float]], wanted: dict[int, set[str]], existing: dict[int, Path]) -> None:
    lines = ["epoch,reasons,checkpoint,pose_map50,pose_map50_95,pose_recall,box_map50_95"]
    rows_by_epoch = {int(r["epoch"]): r for r in rows}
    for epoch in sorted(wanted):
        row = rows_by_epoch[epoch]
        ckpt = existing.get(epoch, "")
        lines.append(
            ",".join(
                [
                    str(epoch),
                    "|".join(sorted(wanted[epoch])),
                    str(ckpt),
                    f"{row.get('metrics/mAP50(P)', 0.0):.6f}",
                    f"{row.get('metrics/mAP50-95(P)', 0.0):.6f}",
                    f"{row.get('metrics/recall(P)', 0.0):.6f}",
                    f"{row.get('metrics/mAP50-95(B)', 0.0):.6f}",
                ]
            )
        )
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_once(run_dir: Path, args: argparse.Namespace) -> int:
    results_path = run_dir / "results.csv"
    weights_dir = run_dir / "weights"
    last_path = weights_dir / "last.pt"
    out_dir = weights_dir / "monitored"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_results(results_path)
    if not rows:
        print(f"No completed epochs found yet: {results_path}")
        return 0

    wanted = desired_epochs(rows, args)
    existing = index_existing(out_dir)
    rows_by_epoch = {int(r["epoch"]): r for r in rows}
    latest_epoch = int(max(r["epoch"] for r in rows))
    copied = 0

    for epoch in sorted(wanted):
        if epoch in existing:
            continue
        row = rows_by_epoch[epoch]
        dst = out_dir / checkpoint_name(epoch, row)

        periodic_src = weights_dir / f"epoch_{epoch}.pt"
        src = periodic_src if periodic_src.exists() else last_path

        if epoch != latest_epoch and src == last_path:
            print(f"Skip epoch {epoch}: no epoch-specific checkpoint exists.")
            continue

        if copy_checkpoint(src, dst, epoch, row):
            existing[epoch] = dst
            copied += 1
            reasons = "|".join(sorted(wanted[epoch]))
            print(f"Saved epoch {epoch}: {dst.name} ({reasons})")

    write_summary(out_dir, rows, wanted, existing)
    print(f"epochs={len(rows)} latest={latest_epoch} copied={copied} summary={out_dir / 'summary.csv'}")
    return copied


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    print(f"Monitoring: {run_dir}")
    while True:
        update_once(run_dir, args)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
