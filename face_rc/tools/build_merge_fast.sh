#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT_DIR}/face_rc/tools/merge_fast.cpp"
OUT_DIR="${ROOT_DIR}/face_rc/tools/bin"
OUT="${OUT_DIR}/libmerge_fast.so"

mkdir -p "${OUT_DIR}"
g++ -O3 -DNDEBUG -std=c++14 -fPIC -shared \
  "${SRC}" \
  -o "${OUT}"

echo "built: ${OUT}"
