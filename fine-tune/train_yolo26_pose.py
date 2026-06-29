#!/usr/bin/env python3
"""Fine-tune yolo26n-pose.pt on a prepared YOLO-pose dataset."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

# Prefer the project-local Ultralytics copy bundled in fine-tune/ over any
# globally installed package.
sys.path.insert(0, str(ROOT))


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/yolo26n-pose.pt", help="Initial pose checkpoint.")
    parser.add_argument("--data", default="datasets/omnilab_zhankai/omnilab_zhankai.yaml", help="Dataset YAML.")
    parser.add_argument("--project", default="runs/pose", help="Training output directory.")
    parser.add_argument("--name", default="yolo26n-pose-omnilab", help="Run name.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0, help="Examples: 0, 0,1, cpu. Default lets Ultralytics choose.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--optimizer", default="auto", help="Ultralytics optimizer, e.g. auto, AdamW, SGD.")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers, e.g. 10.")
    parser.add_argument("--resume", action="store_true", help="Resume the run from the latest checkpoint.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--cache", action="store_true", help="Cache images in RAM/disk as Ultralytics decides.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)
    project_path = resolve_path(args.project)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    train_kwargs = {
        "data": str(data_path),
        "task": "pose",
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "project": str(project_path),
        "name": args.name,
        "patience": args.patience,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "warmup_epochs": args.warmup_epochs,
        "mosaic": args.mosaic,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "close_mosaic": args.close_mosaic,
        "seed": args.seed,
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "cache": args.cache,
        "amp": args.amp,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze

    print("Fine-tune configuration:")
    print(f"  model:   {model_path}")
    print(f"  data:    {data_path}")
    print(f"  project: {project_path / args.name}")
    print(f"  imgsz:   {args.imgsz}")
    print(f"  batch:   {args.batch}")
    print(f"  epochs:  {args.epochs}")

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
