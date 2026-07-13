#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/tools/generate_board_cpp_maps_from_camera.cpp"
OUT_DIR="${ROOT_DIR}/tools/bin"
OUT="${OUT_DIR}/generate_board_cpp_maps_from_camera"

mkdir -p "${OUT_DIR}"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:-}"

"${CXX}" -O3 -DNDEBUG -std=c++14 ${CXXFLAGS} \
  "${SRC}" \
  -ldl \
  -o "${OUT}"

echo "built: ${OUT}"
echo "turbojpeg: loaded with dlopen at runtime for MJPEG camera input"
