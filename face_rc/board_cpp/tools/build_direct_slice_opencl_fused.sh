#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/tools/direct_slice_opencl_fused.cpp"
OUT_DIR="${ROOT_DIR}/lib"
OUT="${OUT_DIR}/libdirect_slice_opencl_fused.so"

mkdir -p "${OUT_DIR}"

OPENCL_LIB="${OPENCL_LIB:-}"
if [[ -z "${OPENCL_LIB}" ]]; then
  for cand in \
    "${ROOT_DIR}/lib/libOpenCL.so" \
    "${ROOT_DIR}/lib/libOpenCL.so.1" \
    /usr/lib/aarch64-linux-gnu/libOpenCL.so \
    /usr/lib/aarch64-linux-gnu/libOpenCL.so.1 \
    /usr/lib/libOpenCL.so \
    /usr/lib/libOpenCL.so.1 \
    /lib/aarch64-linux-gnu/libOpenCL.so \
    /lib/aarch64-linux-gnu/libOpenCL.so.1; do
    if [[ -f "${cand}" ]]; then
      OPENCL_LIB="${cand}"
      break
    fi
  done
fi
if [[ -z "${OPENCL_LIB}" ]]; then
  OPENCL_LIB="-lOpenCL"
fi

g++ -O3 -DNDEBUG -std=c++14 -fPIC -shared \
  "${SRC}" \
  -o "${OUT}" \
  "${OPENCL_LIB}" \
  -Wl,-rpath,'$ORIGIN'

echo "built: ${OUT}"
echo "opencl lib: ${OPENCL_LIB}"
