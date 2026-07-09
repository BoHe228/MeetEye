#!/usr/bin/env python3
"""Export the Python direct-slice .npz map into plain files for C++ runtime.

This script is a development/preparation tool. Run it on a machine with numpy
available, such as the Firefly board or a normal Ubuntu host. The Buildroot
runtime only needs the generated map_x.bin, map_y.bin, and meta.txt files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


SLICE_INFO_FIELDS = (
    "slice_idx",
    "start_x",
    "actual_start_x",
    "end_x",
    "slice_width",
    "slice_height",
    "original_width",
    "original_height",
    "wrap_around",
)

LETTERBOX_INFO_FIELDS = (
    "gain",
    "left",
    "top",
    "new_width",
    "new_height",
)


def scalar(data: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in data:
        return default
    value = data[key]
    try:
        return value.item()
    except Exception:
        return value


def decode_metadata(data: np.lib.npyio.NpzFile) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if "slice_infos_array" in data and "letterbox_infos_array" in data:
        slice_infos: List[Dict[str, Any]] = []
        for row in np.asarray(data["slice_infos_array"]):
            item: Dict[str, Any] = {}
            for key, value in zip(SLICE_INFO_FIELDS, row):
                item[key] = bool(int(value)) if key == "wrap_around" else int(value)
            slice_infos.append(item)

        letterbox_infos: List[Dict[str, Any]] = []
        for row in np.asarray(data["letterbox_infos_array"]):
            item = {}
            for key, value in zip(LETTERBOX_INFO_FIELDS, row):
                item[key] = float(value) if key == "gain" else int(round(float(value)))
            letterbox_infos.append(item)
        return slice_infos, letterbox_infos

    if "slice_infos_json" in data and "letterbox_infos_json" in data:
        return (
            json.loads(str(data["slice_infos_json"].item())),
            json.loads(str(data["letterbox_infos_json"].item())),
        )

    return (
        [dict(item) for item in data["slice_infos"]],
        [dict(item) for item in data["letterbox_infos"]],
    )


def write_meta(path: Path, lines: Iterable[Tuple[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, value in lines:
            if isinstance(value, bool):
                value = 1 if value else 0
            f.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export direct-slice map for board_cpp.")
    parser.add_argument(
        "--input",
        default="../board/maps/6.22_2560_yolo_slices_640.npz",
        help="input direct-slice .npz map file",
    )
    parser.add_argument(
        "--output",
        default="map_export",
        help="output directory for C++ runtime map files",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(str(input_path), allow_pickle=True)
    map_x = np.ascontiguousarray(np.asarray(data["slice_map_x"], dtype=np.float32))
    map_y = np.ascontiguousarray(np.asarray(data["slice_map_y"], dtype=np.float32))
    if map_x.shape != map_y.shape or map_x.ndim != 3:
        raise ValueError(f"invalid map shape: map_x={map_x.shape}, map_y={map_y.shape}")

    slice_infos, letterbox_infos = decode_metadata(data)
    if len(slice_infos) != map_x.shape[0] or len(letterbox_infos) != map_x.shape[0]:
        raise ValueError(
            "metadata count mismatch: "
            f"maps={map_x.shape[0]}, slices={len(slice_infos)}, letterbox={len(letterbox_infos)}"
        )

    map_x.tofile(output_dir / "map_x.bin")
    map_y.tofile(output_dir / "map_y.bin")

    num_slices, roi_h, roi_w = map_x.shape
    imgsz = int(scalar(data, "imgsz", roi_h))
    process_width = int(scalar(data, "process_width", slice_infos[0]["original_width"]))
    process_height = int(scalar(data, "process_height", slice_infos[0]["original_height"]))
    base_output_width = int(scalar(data, "base_output_width", process_width) or process_width)
    base_output_height = int(scalar(data, "base_output_height", process_height) or process_height)

    base_map_x = None
    base_map_y = None
    if "base_map_x" in data and "base_map_y" in data:
        base_map_x = np.ascontiguousarray(np.asarray(data["base_map_x"], dtype=np.float32))
        base_map_y = np.ascontiguousarray(np.asarray(data["base_map_y"], dtype=np.float32))
    else:
        base_map_file = str(scalar(data, "base_map_file", "") or "")
        if base_map_file:
            base_path = Path(base_map_file)
            if not base_path.is_absolute():
                base_path = input_path.parent / base_path
            if base_path.exists():
                base = np.load(str(base_path), allow_pickle=True)
                if "map_x" in base and "map_y" in base:
                    base_map_x = np.ascontiguousarray(np.asarray(base["map_x"], dtype=np.float32))
                    base_map_y = np.ascontiguousarray(np.asarray(base["map_y"], dtype=np.float32))

    if base_map_x is not None and base_map_y is not None:
        if base_map_x.shape != base_map_y.shape or base_map_x.ndim != 2:
            raise ValueError(f"invalid base map shape: x={base_map_x.shape}, y={base_map_y.shape}")
        map_h, map_w = base_map_x.shape
        crop_h = base_output_height // 3
        crop_h = max(0, min(crop_h, map_h))
        crop_end = max(crop_h, min(base_output_height, map_h))
        base_map_x = np.ascontiguousarray(base_map_x[crop_h:crop_end], dtype=np.float32)
        base_map_y = np.ascontiguousarray(base_map_y[crop_h:crop_end], dtype=np.float32)
        if base_map_x.shape != (process_height, process_width):
            raise ValueError(
                "base map shape after crop does not match process shape: "
                f"base={base_map_x.shape}, process={(process_height, process_width)}. "
                "Regenerate the map with matching process dimensions or add resize support."
            )
        base_map_x.tofile(output_dir / "base_map_x.bin")
        base_map_y.tofile(output_dir / "base_map_y.bin")

    lines: List[Tuple[str, Any]] = [
        ("source_npz", input_path),
        ("num_slices", int(num_slices)),
        ("roi_h", int(roi_h)),
        ("roi_w", int(roi_w)),
        ("imgsz", imgsz),
        ("process_width", process_width),
        ("process_height", process_height),
        ("base_output_width", base_output_width),
        ("base_output_height", base_output_height),
        ("img_width", int(scalar(data, "img_width", 0) or 0)),
        ("img_height", int(scalar(data, "img_height", 0) or 0)),
        ("radius", int(scalar(data, "radius", 0) or 0)),
    ]
    if base_map_x is not None and base_map_y is not None:
        lines.extend(
            [
                ("base_map_width", int(base_map_x.shape[1])),
                ("base_map_height", int(base_map_x.shape[0])),
            ]
        )

    if "center" in data:
        center = np.asarray(data["center"]).reshape(-1)
        if center.size >= 2:
            lines.append(("center_x", int(center[0])))
            lines.append(("center_y", int(center[1])))

    for idx, (slice_info, letterbox_info) in enumerate(zip(slice_infos, letterbox_infos)):
        prefix = f"slice{idx}"
        for key in SLICE_INFO_FIELDS:
            lines.append((f"{prefix}.{key}", slice_info[key]))
        for key in LETTERBOX_INFO_FIELDS:
            lines.append((f"{prefix}.{key}", letterbox_info[key]))

    write_meta(output_dir / "meta.txt", lines)
    print(f"exported: {output_dir}")
    print(f"map: slices={num_slices} roi={roi_w}x{roi_h} imgsz={imgsz}")
    print(f"process: {process_width}x{process_height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
