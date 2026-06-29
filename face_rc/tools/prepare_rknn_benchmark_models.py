#!/usr/bin/env python3
"""Prepare RK3588 RKNN models for same-size benchmark comparison.

The output is a directory under face_rc containing:
  - one subdirectory per model with ONNX/RKNN/metadata,
  - manifest.json for the board-side benchmark runner.

Default benchmark policy:
  - batch=1
  - imgsz=640
  - RK3588 target
  - INT8 quantization
  - official YOLOv8-pose uses Rockchip's hybrid quantization recipe
  - YOLO26 exports disable end2end postprocess before RKNN conversion
  - YOLO26 and generic Ultralytics exports use plain INT8 quantization first
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = ROOT / "face_rc" / "yolo_model" / "RK3588" / "benchmark_640"
DEFAULT_DATASET = ROOT / "rknn_model_zoo" / "datasets" / "COCO" / "coco_subset_20.txt"
DEFAULT_CONVERT_PY = Path("/tmp/rknn_convert_env/bin/python")


@dataclass
class ModelSpec:
    name: str
    task: str
    family: str
    source: str
    quant_mode: str = "plain"
    source_type: str = "pt"
    end2end: Optional[bool] = None
    enabled: bool = True


SPECS: List[ModelSpec] = [
    ModelSpec(
        name="yolov8n-face",
        task="face-pose",
        family="YOLOv8",
        source="yolo_model/yolov8n-face.pt",
        quant_mode="plain",
    ),
    ModelSpec(
        name="yolov8n-pose-official",
        task="person-pose",
        family="YOLOv8",
        source="face_rc/yolo_model/RK3588/yolov8-pose/yolov8n-pose.onnx",
        source_type="onnx",
        quant_mode="yolov8_pose_hybrid",
    ),
    ModelSpec(
        name="yolo26n",
        task="detect",
        family="YOLO26",
        source="yolo_model/yolo26n.pt",
        quant_mode="plain",
        end2end=False,
    ),
    ModelSpec(
        name="yolo26n-pose",
        task="person-pose",
        family="YOLO26",
        source="yolo_model/yolo26n-pose.pt",
        quant_mode="plain",
        end2end=False,
    ),
    ModelSpec(
        name="yolo26s-pose",
        task="person-pose",
        family="YOLO26",
        source="yolo_model/yolo26s-pose.pt",
        quant_mode="plain",
        end2end=False,
    ),
    ModelSpec(
        name="yolo26n-seg",
        task="segment",
        family="YOLO26",
        source="yolo_model/yolo26n-seg.pt",
        quant_mode="plain",
        end2end=False,
    ),
    ModelSpec(
        name="yolo26x-pose",
        task="person-pose",
        family="YOLO26",
        source="yolo_model/yolo26x-pose.pt",
        quant_mode="plain",
        end2end=False,
    ),
]


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    printable = " ".join(cmd)
    print(f"\n$ {printable}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def rel_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def model_dir_name(spec: ModelSpec, imgsz: int, dtype: str) -> str:
    q = "hybrid" if spec.quant_mode != "plain" else "plain"
    if spec.end2end is False:
        q += "_noe2e"
    return f"{spec.name}-{imgsz}-{dtype}-{q}"


def export_pt_to_onnx(spec: ModelSpec, out_dir: Path, imgsz: int, batch: int, opset: int) -> Path:
    source = ROOT / spec.source
    if not source.is_file():
        raise FileNotFoundError(f"source model not found: {source}")

    final_onnx = out_dir / f"{spec.name}-{imgsz}-b{batch}.onnx"
    if final_onnx.is_file():
        print(f"[skip] ONNX exists: {final_onnx}")
        return final_onnx

    # Ultralytics writes next to the source by default. Use a temporary export
    # directory under the target model dir to avoid touching source folders.
    tmp_source = out_dir / source.name
    if not tmp_source.exists():
        shutil.copy2(source, tmp_source)

    export_args = [
        "format='onnx'",
        f"imgsz={imgsz}",
        f"batch={batch}",
        f"opset={opset}",
        "dynamic=False",
        "simplify=True",
        "half=False",
        "int8=False",
        "nms=False",
    ]
    if spec.end2end is not None:
        export_args.append(f"end2end={spec.end2end}")
    code = (
        "from ultralytics import YOLO\n"
        f"m = YOLO(r'{tmp_source}')\n"
        f"m.export({','.join(export_args)})\n"
    )
    run([sys.executable, "-c", code], cwd=ROOT)

    generated = tmp_source.with_suffix(".onnx")
    if not generated.is_file():
        candidates = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f"Ultralytics export did not create ONNX for {spec.name}")
        generated = candidates[0]
    generated.rename(final_onnx)
    print(f"[onnx] {final_onnx}")
    return final_onnx


def copy_onnx(spec: ModelSpec, out_dir: Path) -> Path:
    source = ROOT / spec.source
    if not source.is_file():
        raise FileNotFoundError(f"source ONNX not found: {source}")
    final_onnx = out_dir / source.name
    if not final_onnx.is_file() or source.stat().st_mtime > final_onnx.stat().st_mtime:
        shutil.copy2(source, final_onnx)
    print(f"[onnx] {final_onnx}")
    return final_onnx


def convert_onnx_to_rknn(
    spec: ModelSpec,
    onnx_path: Path,
    out_dir: Path,
    imgsz: int,
    batch: int,
    dtype: str,
    dataset: Path,
    convert_python: Path,
    force: bool,
) -> Path:
    rknn_path = out_dir / f"{spec.name}-{imgsz}-b{batch}-rk3588-{dtype}.rknn"
    if rknn_path.is_file() and not force:
        print(f"[skip] RKNN exists: {rknn_path}")
        return rknn_path
    if not convert_python.is_file():
        raise FileNotFoundError(f"RKNN conversion python not found: {convert_python}")

    helper = ROOT / "face_rc" / "tools" / "rknn_convert_model.py"
    run(
        [
            str(convert_python),
            str(helper),
            "--onnx",
            str(onnx_path),
            "--output",
            str(rknn_path),
            "--dataset",
            str(dataset),
            "--target-platform",
            "rk3588",
            "--dtype",
            dtype,
            "--quant-mode",
            spec.quant_mode,
        ],
        cwd=out_dir,
    )
    return rknn_path


def write_metadata(
    out_dir: Path,
    spec: ModelSpec,
    onnx_path: Path,
    rknn_path: Optional[Path],
    imgsz: int,
    batch: int,
    dtype: str,
    status: str,
    error: Optional[str] = None,
) -> dict:
    record = {
        **asdict(spec),
        "imgsz": imgsz,
        "batch": batch,
        "dtype": dtype,
        "end2end": spec.end2end,
        "onnx_path": rel_or_abs(onnx_path, ROOT) if onnx_path else "",
        "rknn_path": rel_or_abs(rknn_path, ROOT) if rknn_path else "",
        "status": status,
        "error": error or "",
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record


def selected_specs(names: Iterable[str]) -> List[ModelSpec]:
    requested = [n for n in names if n]
    if not requested or requested == ["all"]:
        return SPECS
    by_name = {s.name: s for s in SPECS}
    missing = [n for n in requested if n not in by_name]
    if missing:
        raise ValueError(f"unknown model(s): {', '.join(missing)}")
    return [by_name[n] for n in requested]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RKNN benchmark models.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--convert-python", default=str(DEFAULT_CONVERT_PY))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--dtype", default="i8", choices=["i8", "fp"])
    parser.add_argument(
        "--models",
        default="all",
        help="comma-separated names or all. Names: " + ", ".join(s.name for s in SPECS),
    )
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument("--force-convert", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="write failed records and continue converting the remaining models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    dataset = Path(args.dataset).resolve()
    # Keep the venv launcher path as-is. Calling Path.resolve() follows the
    # `bin/python -> base-python` symlink and can accidentally bypass the venv.
    convert_python = Path(args.convert_python).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in selected_specs(args.models.split(",")):
        out_dir = output_root / model_dir_name(spec, args.imgsz, args.dtype)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n===== {spec.name} ({spec.family}, {spec.task}, {spec.quant_mode}) =====")
        try:
            if args.skip_export:
                candidates = sorted(out_dir.glob("*.onnx"))
                if not candidates:
                    raise FileNotFoundError(f"--skip-export set but no ONNX in {out_dir}")
                onnx_path = candidates[0]
            elif spec.source_type == "onnx":
                onnx_path = copy_onnx(spec, out_dir)
            else:
                onnx_path = export_pt_to_onnx(spec, out_dir, args.imgsz, args.batch, args.opset)

            rknn_path = None
            if not args.skip_convert:
                rknn_path = convert_onnx_to_rknn(
                    spec,
                    onnx_path,
                    out_dir,
                    args.imgsz,
                    args.batch,
                    args.dtype,
                    dataset,
                    convert_python,
                    args.force_convert,
                )
            records.append(
                write_metadata(
                    out_dir,
                    spec,
                    onnx_path,
                    rknn_path,
                    args.imgsz,
                    args.batch,
                    args.dtype,
                    "ok",
                )
            )
        except Exception as exc:
            print(f"[error] {spec.name}: {exc}", file=sys.stderr)
            records.append(
                write_metadata(
                    out_dir,
                    spec,
                    locals().get("onnx_path"),
                    locals().get("rknn_path"),
                    args.imgsz,
                    args.batch,
                    args.dtype,
                    "failed",
                    str(exc),
                )
            )
            if not args.continue_on_error:
                raise

    manifest = {
        "imgsz": args.imgsz,
        "batch": args.batch,
        "dtype": args.dtype,
        "target_platform": "rk3588",
        "dataset": rel_or_abs(dataset, ROOT),
        "models": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
