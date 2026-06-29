#!/usr/bin/env python3
"""Run the C API rknn_run benchmark for a manifest of RKNN models.

Run this on the RK3588 board after syncing face_rc and the converted RKNNs.
It generates CSV and Markdown tables with the same timing scope for every
model: warmup excluded, only rknn_run() timed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "face_rc" / "yolo_model" / "RK3588" / "benchmark_640" / "manifest.json"
DEFAULT_OUT_DIR = ROOT / "face_rc" / "benchmark_results"


MEAN_RE = re.compile(r"mean:\s+([0-9.]+)\s+ms\s+\(([0-9.]+)\s+FPS")
MEDIAN_RE = re.compile(r"median:\s+([0-9.]+)\s+ms")
P90_RE = re.compile(r"p90 / p95:\s+([0-9.]+)\s+/\s+([0-9.]+)\s+ms")
MINMAX_RE = re.compile(r"min / max:\s+([0-9.]+)\s+/\s+([0-9.]+)\s+ms")
WALL_RE = re.compile(r"wall FPS:\s+([0-9.]+)")


def parse_core_masks(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_models(manifest_path: Path) -> List[dict]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = []
    for item in data.get("models", []):
        if item.get("status") != "ok" or not item.get("rknn_path"):
            continue
        models.append(item)
    if not models:
        raise RuntimeError(f"no runnable RKNN model records in {manifest_path}")
    return models


def parse_output(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {"raw_output": text}
    for key, regex in [
        ("mean", MEAN_RE),
        ("median", MEDIAN_RE),
        ("p90", P90_RE),
        ("minmax", MINMAX_RE),
        ("wall", WALL_RE),
    ]:
        match = regex.search(text)
        if not match:
            continue
        if key == "mean":
            result["mean_ms"] = match.group(1)
            result["fps_by_mean"] = match.group(2)
        elif key == "median":
            result["median_ms"] = match.group(1)
        elif key == "p90":
            result["p90_ms"] = match.group(1)
            result["p95_ms"] = match.group(2)
        elif key == "minmax":
            result["min_ms"] = match.group(1)
            result["max_ms"] = match.group(2)
        elif key == "wall":
            result["wall_fps"] = match.group(1)
    return result


def ensure_benchmark_binary(binary: Path) -> None:
    if binary.is_file():
        return
    build_script = ROOT / "face_rc" / "tools" / "build_rknn_run_benchmark.sh"
    if not build_script.is_file():
        raise FileNotFoundError(f"benchmark binary missing and build script not found: {build_script}")
    subprocess.run(["bash", str(build_script)], cwd=str(ROOT), check=True)
    if not binary.is_file():
        raise FileNotFoundError(f"benchmark binary still missing after build: {binary}")


def run_one(binary: Path, model_path: Path, loops: int, warmup: int, core_mask: str) -> Dict[str, str]:
    env = os.environ.copy()
    lib_dir = binary.parent / "lib"
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [str(binary), str(model_path), str(loops), str(warmup), str(core_mask)]
    print("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(proc.stdout)
    parsed = parse_output(proc.stdout)
    parsed["returncode"] = str(proc.returncode)
    if proc.returncode != 0:
        if proc.returncode < 0:
            parsed["error"] = f"benchmark killed by signal {-proc.returncode}"
        else:
            parsed["error"] = f"benchmark failed: returncode={proc.returncode}"
        if not proc.stdout.strip():
            parsed["error"] += "; no benchmark output"
        print(f"[benchmark error] {model_path}: {parsed['error']}")
    elif not proc.stdout.strip():
        parsed["error"] = "benchmark returned 0 but produced no output"
        print(f"[benchmark error] {model_path}: {parsed['error']}")
    return parsed


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fieldnames = [
        "name",
        "family",
        "task",
        "imgsz",
        "batch",
        "dtype",
        "quant_mode",
        "core_mask",
        "mean_ms",
        "fps_by_mean",
        "median_ms",
        "p90_ms",
        "p95_ms",
        "min_ms",
        "max_ms",
        "wall_fps",
        "returncode",
        "rknn_path",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    lines = [
        "# RK3588 RKNN Benchmark",
        "",
        "Timing scope: C API, warmup excluded, only `rknn_run()` timed.",
        "",
        "| model | family | task | input | quant | core | mean ms | FPS | p95 ms | status |",
        "|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        input_desc = f"{row.get('batch')}x{row.get('imgsz')}x{row.get('imgsz')}"
        lines.append(
            "| {name} | {family} | {task} | {input_desc} | {quant}/{dtype} | {core} | "
            "{mean} | {fps} | {p95} | {status} |".format(
                name=row.get("name", ""),
                family=row.get("family", ""),
                task=row.get("task", ""),
                input_desc=input_desc,
                quant=row.get("quant_mode", ""),
                dtype=row.get("dtype", ""),
                core=row.get("core_mask", ""),
                mean=row.get("mean_ms", ""),
                fps=row.get("fps_by_mean", ""),
                p95=row.get("p95_ms", ""),
                status=row.get("error") or "ok",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark RKNN models from a manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--loops", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument(
        "--core-masks",
        default="7",
        help="comma-separated C API core masks, e.g. 1 or 1,7. RK3588: 1=core0, 2=core1, 4=core2, 7=all",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument(
        "--binary",
        default=str(ROOT / "face_rc" / "tools" / "bin" / "rknn_run_benchmark"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    binary = Path(args.binary).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_benchmark_binary(binary)

    models = load_models(manifest_path)
    rows: List[dict] = []
    for item in models:
        model_path = ROOT / item["rknn_path"]
        for core_mask in parse_core_masks(args.core_masks):
            parsed = run_one(binary, model_path, args.loops, args.warmup, core_mask)
            rows.append(
                {
                    **item,
                    **parsed,
                    "core_mask": core_mask,
                }
            )

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"rknn_run_benchmark_{stamp}.csv"
    md_path = out_dir / f"rknn_run_benchmark_{stamp}.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print(f"csv: {csv_path}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
