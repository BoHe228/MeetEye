#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/tools/hybrid_sort_native.cpp"
OUT_DIR="${ROOT_DIR}/lib"
OUT="${OUT_DIR}/libhybrid_sort_native.so"

mkdir -p "${OUT_DIR}"
g++ -O3 -DNDEBUG -std=c++14 -fPIC -shared \
  "${SRC}" \
  -o "${OUT}"

echo "built: ${OUT}"
