#!/usr/bin/env python3
"""Convert one ONNX model to RKNN for RK3588.

This script is intended to run inside an environment that has
`rknn-toolkit2` installed. It is kept small so the orchestration script can
call it with a temporary conversion venv.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from rknn.api import RKNN


YOLOV8_POSE_HYBRID_NODES = [
    [
        "/model.22/cv4.0/cv4.0.0/act/Mul_output_0",
        "/model.22/Concat_6_output_0",
    ],
    [
        "/model.22/cv4.1/cv4.1.0/act/Mul_output_0",
        "/model.22/Concat_6_output_0",
    ],
    [
        "/model.22/cv4.2/cv4.2.0/act/Mul_output_0",
        "/model.22/Concat_6_output_0",
    ],
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert one ONNX model to RKNN.")
    parser.add_argument("--onnx", required=True, help="input ONNX model")
    parser.add_argument("--output", required=True, help="output RKNN path")
    parser.add_argument("--dataset", required=True, help="calibration image list")
    parser.add_argument("--target-platform", default="rk3588")
    parser.add_argument(
        "--dtype",
        default="i8",
        choices=["i8", "fp"],
        help="i8 enables INT8 quantization; fp disables quantization",
    )
    parser.add_argument(
        "--quant-mode",
        default="plain",
        choices=["plain", "yolov8_pose_hybrid"],
        help="plain uses rknn.build(); yolov8_pose_hybrid matches the official model-zoo pose example",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    onnx_path = Path(args.onnx).resolve()
    output_path = Path(args.output).resolve()
    dataset_path = Path(args.dataset).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if args.dtype == "i8" and not dataset_path.is_file():
        raise FileNotFoundError(f"dataset list not found: {dataset_path}")

    rknn = RKNN(verbose=args.verbose)
    try:
        print("--> Config model")
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=args.target_platform,
        )
        print("done")

        print("--> Loading model")
        ret = rknn.load_onnx(model=str(onnx_path))
        if ret != 0:
            raise RuntimeError(f"load_onnx failed: {ret}")
        print("done")

        print("--> Building model")
        do_quant = args.dtype == "i8"
        if not do_quant:
            ret = rknn.build(do_quantization=False)
            if ret != 0:
                raise RuntimeError(f"build fp failed: {ret}")
        elif args.quant_mode == "plain":
            ret = rknn.build(do_quantization=True, dataset=str(dataset_path))
            if ret != 0:
                raise RuntimeError(f"build int8 failed: {ret}")
        elif args.quant_mode == "yolov8_pose_hybrid":
            old_cwd = os.getcwd()
            os.chdir(str(output_path.parent))
            try:
                model_name = onnx_path.stem
                ret = rknn.hybrid_quantization_step1(
                    dataset=str(dataset_path),
                    proposal=False,
                    custom_hybrid=YOLOV8_POSE_HYBRID_NODES,
                )
                if ret != 0:
                    raise RuntimeError(f"hybrid_quantization_step1 failed: {ret}")
                ret = rknn.hybrid_quantization_step2(
                    model_input=f"{model_name}.model",
                    data_input=f"{model_name}.data",
                    model_quantization_cfg=f"{model_name}.quantization.cfg",
                )
                if ret != 0:
                    raise RuntimeError(f"hybrid_quantization_step2 failed: {ret}")
            finally:
                os.chdir(old_cwd)
        else:
            raise ValueError(f"unsupported quant mode: {args.quant_mode}")
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
