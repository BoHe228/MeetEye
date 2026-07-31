#!/usr/bin/env python3
"""Convert exported AdaFace ONNX to RKNN for RK3588.

Default dtype is fp to preserve face embedding quality. INT8 is supported when a
calibration dataset file is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AdaFace ONNX to RKNN.")
    parser.add_argument(
        "--onnx",
        default="face_rc/board_cpp/models/adaface_ir18_112.onnx",
        help="input ONNX path",
    )
    parser.add_argument(
        "--output",
        default="face_rc/board_cpp/models/adaface_ir18_112.rknn",
        help="output RKNN path",
    )
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument("--dtype", choices=["fp", "i8"], default="fp")
    parser.add_argument(
        "--dataset",
        default="",
        help="calibration dataset list for INT8; not used for fp",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    onnx_path = Path(args.onnx)
    if not onnx_path.is_absolute():
        onnx_path = repo_root / onnx_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    dataset = Path(args.dataset) if args.dataset else None
    if dataset is not None and not dataset.is_absolute():
        dataset = repo_root / dataset
    if args.dtype == "i8" and (dataset is None or not dataset.is_file()):
        raise FileNotFoundError("--dataset is required for INT8 AdaFace conversion")

    rknn = RKNN(verbose=args.verbose)
    try:
        print("--> Config AdaFace RKNN")
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            target_platform=args.target_platform,
        )
        print("done")

        print("--> Load ONNX")
        ret = rknn.load_onnx(
            model=str(onnx_path),
            inputs=["input"],
            input_size_list=[[1, 3, 112, 112]],
        )
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        print("done")

        print("--> Build RKNN")
        if args.dtype == "fp":
            ret = rknn.build(do_quantization=False)
        else:
            ret = rknn.build(do_quantization=True, dataset=str(dataset))
        if ret != 0:
            raise RuntimeError(f"build failed: {ret}")
        print("done")

        print("--> Export RKNN")
        ret = rknn.export_rknn(str(output_path))
        if ret != 0:
            raise RuntimeError(f"export_rknn failed: {ret}")
        print(f"output_path: {output_path}")
        print("done")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
