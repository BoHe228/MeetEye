#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${ROOT_DIR}/bin/meeteye_cpp_smoke"
MODEL="${ROOT_DIR}/models/yolov8n-face-640-b1-int8-hybrid-split-kptconf-rk3588.rknn"
MAP_DIR="${ROOT_DIR}/maps/6.22_2560_yolo_slices_640_cpp"

if [[ ! -x "${BIN}" ]]; then
  echo "missing executable: ${BIN}" >&2
  echo "build it on an aarch64 RK3588 board first: bash build.sh" >&2
  exit 1
fi

IMAGE=""
IMAGE_LIST=""
CAMERA_DEVICE=""
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --image-list)
      IMAGE_LIST="${2:-}"
      shift 2
      ;;
    --camera-device)
      CAMERA_DEVICE="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --map-dir)
      MAP_DIR="${2:-}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "${IMAGE}" && -z "${IMAGE_LIST}" && -z "${CAMERA_DEVICE}" ]]; then
  echo "usage: $0 --image frame.jpg | --image-list frames.txt | --camera-device /dev/video0 [extra meeteye_cpp_smoke args]" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${ROOT_DIR}/lib:${ROOT_DIR}/lib/lib:${LD_LIBRARY_PATH:-}"

CMD=("${BIN}" --map-dir "${MAP_DIR}" --model "${MODEL}")
if [[ -n "${IMAGE}" ]]; then
  CMD+=(--image "${IMAGE}")
fi
if [[ -n "${IMAGE_LIST}" ]]; then
  CMD+=(--image-list "${IMAGE_LIST}")
fi
if [[ -n "${CAMERA_DEVICE}" ]]; then
  CMD+=(--camera-device "${CAMERA_DEVICE}")
fi
CMD+=("${EXTRA_ARGS[@]}")

exec "${CMD[@]}"
