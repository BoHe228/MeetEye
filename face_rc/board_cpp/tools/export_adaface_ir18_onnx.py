#!/usr/bin/env python3
"""Export MeetEye AdaFace IR-18 checkpoint to ONNX.

The exported model keeps only the 512D L2-normalized embedding output. Input is
float32 NCHW, shape [1, 3, 112, 112], and should already be normalized with
(pixel / 255 - 0.5) / 0.5, matching mytest/face_rec/face_rec_manager.py.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AdaFace IR-18 to ONNX.")
    parser.add_argument(
        "--checkpoint",
        default="face_rec_model/adaface_ir18_webface4m.ckpt",
        help="AdaFace IR-18 .ckpt path",
    )
    parser.add_argument(
        "--output",
        default="face_rc/board_cpp/models/adaface_ir18_112.onnx",
        help="output ONNX path",
    )
    parser.add_argument("--opset", type=int, default=12)
    return parser.parse_args()


class EmbeddingOnly(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature, _norm = self.model(x)
        return feature


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> None:
    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                break
    clean = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        clean[new_key] = value
    missing, unexpected = model.load_state_dict(clean, strict=False)
    if missing:
        print(f"[AdaFaceExport] missing keys: {len(missing)}")
    if unexpected:
        print(f"[AdaFaceExport] unexpected keys: {len(unexpected)}")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    adaface_dir = repo_root / "mytest" / "face_rec" / "AdaFace"
    if str(adaface_dir) not in sys.path:
        sys.path.insert(0, str(adaface_dir))

    from net import build_model  # noqa: E402

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    model = build_model("ir_18")
    load_checkpoint(model, checkpoint)
    model.eval()
    wrapped = EmbeddingOnly(model).eval()

    dummy = torch.randn(1, 3, 112, 112, dtype=torch.float32)
    export_kwargs = {
        "input_names": ["input"],
        "output_names": ["embedding"],
        "opset_version": args.opset,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False
    torch.onnx.export(wrapped, dummy, str(output), **export_kwargs)
    print(f"[AdaFaceExport] ONNX saved: {output}")
    print("[AdaFaceExport] input: float32 NCHW [1,3,112,112], normalized to [-1,1]")
    print("[AdaFaceExport] output: embedding float32 [1,512], L2-normalized")


if __name__ == "__main__":
    main()
