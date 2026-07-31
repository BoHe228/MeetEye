#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${ROOT_DIR}/bin"
OUT="${OUT_DIR}/meeteye_cpp_smoke"
SRC="${ROOT_DIR}/meeteye_cpp_smoke.cpp"

mkdir -p "${OUT_DIR}"

CXX="${CXX:-g++}"
CXXFLAGS="${CXXFLAGS:-}"
TURBOJPEG_LIB="${TURBOJPEG_LIB:-}"

if [[ -z "${TURBOJPEG_LIB}" ]]; then
  for candidate in \
    /usr/lib/aarch64-linux-gnu/libturbojpeg.so.0 \
    /usr/lib/aarch64-linux-gnu/libturbojpeg.so \
    /usr/lib/libturbojpeg.so.0 \
    /usr/lib/libturbojpeg.so \
    /lib/aarch64-linux-gnu/libturbojpeg.so.0 \
    /lib/aarch64-linux-gnu/libturbojpeg.so; do
    if [[ -f "${candidate}" ]]; then
      TURBOJPEG_LIB="${candidate}"
      break
    fi
  done
fi

if [[ -z "${TURBOJPEG_LIB}" ]]; then
  TURBOJPEG_LIB="-lturbojpeg"
fi

"${CXX}" -O3 -DNDEBUG -std=c++14 ${CXXFLAGS} \
  "${SRC}" \
  -L"${ROOT_DIR}/lib" \
  -L"${ROOT_DIR}/lib/lib" \
  -lrknn_capi_parallel \
  -ladaface_rknn \
  -ldirect_slice_opencl_fused \
  -lhybrid_sort_native \
  "${TURBOJPEG_LIB}" \
  -pthread \
  -ldl \
  -Wl,--allow-shlib-undefined \
  -Wl,-rpath,'$ORIGIN/../lib' \
  -Wl,-rpath,'$ORIGIN/../lib/lib' \
  -o "${OUT}"

echo "built: ${OUT}"
echo "turbojpeg: ${TURBOJPEG_LIB}"
