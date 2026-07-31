#!/usr/bin/env python3
"""Generate board_cpp maps for a rectified wide-angle full-frame input.

The output format matches meeteye_cpp_smoke.cpp:
  map_x.bin/map_y.bin       direct OpenCL remap into one 640x640 RKNN input
  base_map_x.bin/base_map_y.bin  processed wide image -> original camera frame
  meta.txt                  shape, letterbox and wide-angle intrinsic metadata
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import yaml


def python_round(value: float) -> int:
    return int(round(value))


def load_calibration_yaml(path: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    camera_matrix = np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    dist = np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1, 1)
    image_size = data.get("image_size", {})
    width = int(image_size.get("width", 0))
    height = int(image_size.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image_size in calibration yaml: {path}")
    return camera_matrix, dist, width, height


def letterbox(process_w: int, process_h: int, imgsz: int) -> Dict[str, float]:
    gain = min(imgsz / float(process_h), imgsz / float(process_w))
    new_w = python_round(process_w * gain)
    new_h = python_round(process_h * gain)
    pad_w = imgsz - new_w
    pad_h = imgsz - new_h
    left = python_round(pad_w / 2.0 - 0.1)
    top = python_round(pad_h / 2.0 - 0.1)
    return {
        "gain": float(gain),
        "left": int(left),
        "top": int(top),
        "new_width": int(new_w),
        "new_height": int(new_h),
    }


def identity_base_map(width: int, height: int) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    map_x = np.broadcast_to(xs[None, :], (height, width)).astype(np.float32, copy=True)
    map_y = np.broadcast_to(ys[:, None], (height, width)).astype(np.float32, copy=True)
    return map_x, map_y


def build_base_maps(
    camera_matrix: np.ndarray,
    dist: np.ndarray,
    source_w: int,
    source_h: int,
    alpha: float,
    crop: bool,
    no_undistort: bool,
    output_w: int,
    output_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    if no_undistort:
        base_x, base_y = identity_base_map(source_w, source_h)
        new_camera_matrix = camera_matrix.copy()
        roi = {"x": 0, "y": 0, "w": source_w, "h": source_h}
    else:
        new_camera_matrix, roi_tuple = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist,
            (source_w, source_h),
            float(alpha),
            (source_w, source_h),
        )
        base_x, base_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            dist,
            None,
            new_camera_matrix,
            (source_w, source_h),
            cv2.CV_32FC1,
        )
        x, y, w, h = [int(v) for v in roi_tuple]
        if not crop or w <= 0 or h <= 0:
            x, y, w, h = 0, 0, source_w, source_h
        base_x = base_x[y:y + h, x:x + w]
        base_y = base_y[y:y + h, x:x + w]
        new_camera_matrix = new_camera_matrix.copy()
        new_camera_matrix[0, 2] -= float(x)
        new_camera_matrix[1, 2] -= float(y)
        roi = {"x": x, "y": y, "w": w, "h": h}

    process_h, process_w = base_x.shape[:2]
    if output_w <= 0:
      output_w = process_w
    if output_h <= 0:
      output_h = process_h
    if output_w <= 0 or output_h <= 0:
        raise ValueError("output width/height must be positive")

    if output_w != process_w or output_h != process_h:
        scale_x = output_w / float(process_w)
        scale_y = output_h / float(process_h)
        base_x = cv2.resize(base_x, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        base_y = cv2.resize(base_y, (output_w, output_h), interpolation=cv2.INTER_LINEAR)
        new_camera_matrix = new_camera_matrix.copy()
        new_camera_matrix[0, 0] *= scale_x
        new_camera_matrix[0, 2] *= scale_x
        new_camera_matrix[1, 1] *= scale_y
        new_camera_matrix[1, 2] *= scale_y
        roi["resized_w"] = int(output_w)
        roi["resized_h"] = int(output_h)

    return (
        np.ascontiguousarray(base_x, dtype=np.float32),
        np.ascontiguousarray(base_y, dtype=np.float32),
        new_camera_matrix,
        roi,
    )


def build_direct_map(
    base_x: np.ndarray,
    base_y: np.ndarray,
    imgsz: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    process_h, process_w = base_x.shape[:2]
    lb = letterbox(process_w, process_h, imgsz)
    new_w = int(lb["new_width"])
    new_h = int(lb["new_height"])
    gain = float(lb["gain"])

    xs = (np.arange(new_w, dtype=np.float32) / gain).astype(np.float32)
    ys = (np.arange(new_h, dtype=np.float32) / gain).astype(np.float32)
    grid_x = np.ascontiguousarray(np.broadcast_to(xs[None, :], (new_h, new_w)), dtype=np.float32)
    grid_y = np.ascontiguousarray(np.broadcast_to(ys[:, None], (new_h, new_w)), dtype=np.float32)

    direct_x = cv2.remap(
        base_x,
        grid_x,
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    direct_y = cv2.remap(
        base_y,
        grid_x,
        grid_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return (
        direct_x[np.newaxis, :, :].astype(np.float32, copy=False),
        direct_y[np.newaxis, :, :].astype(np.float32, copy=False),
        lb,
    )


def write_meta(
    out_dir: Path,
    calib_yaml: Path,
    source_w: int,
    source_h: int,
    base_x: np.ndarray,
    new_camera_matrix: np.ndarray,
    lb: Dict[str, float],
    roi: Dict[str, int],
    alpha: float,
    crop: bool,
    no_undistort: bool,
    imgsz: int,
) -> None:
    process_h, process_w = base_x.shape[:2]
    roi_h = int(lb["new_height"])
    roi_w = int(lb["new_width"])
    lines = [
        "metadata_format=direct_slice_cpp_v1",
        "projection_mode=wide_angle",
        f"wide_calib_yaml={calib_yaml}",
        f"wide_undistort_alpha={float(alpha):.12g}",
        f"wide_undistort_crop={1 if crop else 0}",
        f"wide_no_undistort={1 if no_undistort else 0}",
        f"wide_roi_x={roi.get('x', 0)}",
        f"wide_roi_y={roi.get('y', 0)}",
        f"wide_roi_w={roi.get('w', process_w)}",
        f"wide_roi_h={roi.get('h', process_h)}",
        "num_slices=1",
        f"roi_h={roi_h}",
        f"roi_w={roi_w}",
        f"imgsz={int(imgsz)}",
        "slice_overlap=0",
        f"process_width={process_w}",
        f"process_height={process_h}",
        f"base_output_width={process_w}",
        f"base_output_height={process_h}",
        f"img_width={source_w}",
        f"img_height={source_h}",
        f"base_map_width={process_w}",
        f"base_map_height={process_h}",
        f"wide_fx={float(new_camera_matrix[0, 0]):.12g}",
        f"wide_fy={float(new_camera_matrix[1, 1]):.12g}",
        f"wide_cx={float(new_camera_matrix[0, 2]):.12g}",
        f"wide_cy={float(new_camera_matrix[1, 2]):.12g}",
        "slice0.slice_idx=0",
        "slice0.start_x=0",
        "slice0.actual_start_x=0",
        f"slice0.end_x={process_w}",
        f"slice0.slice_width={process_w}",
        f"slice0.slice_height={process_h}",
        f"slice0.original_width={process_w}",
        f"slice0.original_height={process_h}",
        "slice0.wrap_around=0",
        f"slice0.gain={float(lb['gain']):.12g}",
        f"slice0.left={int(lb['left'])}",
        f"slice0.top={int(lb['top'])}",
        f"slice0.new_width={roi_w}",
        f"slice0.new_height={roi_h}",
    ]
    (out_dir / "meta.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate board_cpp single-input wide-angle rectification maps."
    )
    parser.add_argument("--calib-yaml", required=True, help="OpenCV calibration yaml")
    parser.add_argument("--cpp-output-dir", required=True, help="output map directory")
    parser.add_argument("--imgsz", type=int, default=640, help="RKNN input size")
    parser.add_argument("--source-width", type=int, default=0, help="override source width")
    parser.add_argument("--source-height", type=int, default=0, help="override source height")
    parser.add_argument("--wide-undistort-alpha", type=float, default=0.5)
    parser.add_argument("--wide-undistort-crop", dest="wide_undistort_crop", action="store_true", default=True)
    parser.add_argument("--no-wide-undistort-crop", dest="wide_undistort_crop", action="store_false")
    parser.add_argument("--wide-no-undistort", action="store_true", help="identity map; only letterbox")
    parser.add_argument("--output-width", type=int, default=0, help="processed rectified width; 0 keeps ROI/full width")
    parser.add_argument("--output-height", type=int, default=0, help="processed rectified height; 0 keeps ROI/full height")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    calib_yaml = Path(args.calib_yaml)
    out_dir = Path(args.cpp_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix, dist, yaml_w, yaml_h = load_calibration_yaml(calib_yaml)
    source_w = int(args.source_width) if args.source_width > 0 else yaml_w
    source_h = int(args.source_height) if args.source_height > 0 else yaml_h
    if args.imgsz <= 0:
        raise ValueError("--imgsz must be positive")
    if source_w <= 0 or source_h <= 0:
        raise ValueError("source width/height must be positive")
    if not math.isfinite(args.wide_undistort_alpha):
        raise ValueError("--wide-undistort-alpha must be finite")

    base_x, base_y, new_camera_matrix, roi = build_base_maps(
        camera_matrix,
        dist,
        source_w,
        source_h,
        args.wide_undistort_alpha,
        args.wide_undistort_crop,
        args.wide_no_undistort,
        args.output_width,
        args.output_height,
    )
    direct_x, direct_y, lb = build_direct_map(base_x, base_y, args.imgsz)

    direct_x.tofile(out_dir / "map_x.bin")
    direct_y.tofile(out_dir / "map_y.bin")
    base_x.tofile(out_dir / "base_map_x.bin")
    base_y.tofile(out_dir / "base_map_y.bin")
    write_meta(
        out_dir,
        calib_yaml,
        source_w,
        source_h,
        base_x,
        new_camera_matrix,
        lb,
        roi,
        args.wide_undistort_alpha,
        args.wide_undistort_crop,
        args.wide_no_undistort,
        args.imgsz,
    )

    print(f"wrote: {out_dir}")
    print(f"source: {source_w}x{source_h}")
    print(f"processed: {base_x.shape[1]}x{base_x.shape[0]}")
    print(f"letterbox: {int(lb['new_width'])}x{int(lb['new_height'])}, top={int(lb['top'])}, left={int(lb['left'])}")
    print(
        "wide intrinsics: "
        f"fx={new_camera_matrix[0, 0]:.3f}, fy={new_camera_matrix[1, 1]:.3f}, "
        f"cx={new_camera_matrix[0, 2]:.3f}, cy={new_camera_matrix[1, 2]:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
