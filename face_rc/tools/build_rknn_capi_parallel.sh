#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${ROOT_DIR}/face_rc/tools/rknn_capi_parallel.cpp"
OUT_DIR="${ROOT_DIR}/face_rc/tools/bin"
OUT="${OUT_DIR}/librknn_capi_parallel.so"
LOCAL_RKNN_RT_LIB="${OUT_DIR}/lib/librknnrt.so"
RKNN_API_DIR="${ROOT_DIR}/RKSDK/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api"
RKNN_INCLUDE_DIR="${RKNN_API_DIR}/include"

if [[ ! -f "${RKNN_INCLUDE_DIR}/rknn_api.h" ]]; then
  RKNN_API_HEADER="$(find "${ROOT_DIR}/RKSDK" "${ROOT_DIR}/rknn_model_zoo" /usr/include /usr/local/include "${VIRTUAL_ENV:-/tmp/not-a-venv}" \
    -name rknn_api.h 2>/dev/null | head -n 1 || true)"
  if [[ -n "${RKNN_API_HEADER}" ]]; then
    RKNN_INCLUDE_DIR="$(dirname "${RKNN_API_HEADER}")"
  fi
fi

if [[ ! -f "${RKNN_INCLUDE_DIR}/rknn_api.h" ]]; then
  echo "No rknn_api.h found." >&2
  exit 1
fi

ARCH="$(uname -m)"
if [[ "${ARCH}" == "aarch64" || "${ARCH}" == "arm64" ]]; then
  LIB_ARCH="aarch64"
else
  LIB_ARCH="x86_64"
fi

RKNN_RT_LIB=""
if [[ -f "${LOCAL_RKNN_RT_LIB}" ]]; then
  RKNN_RT_LIB="${LOCAL_RKNN_RT_LIB}"
elif [[ -f "${RKNN_API_DIR}/${LIB_ARCH}/librknnrt.so" ]]; then
  RKNN_RT_LIB="${RKNN_API_DIR}/${LIB_ARCH}/librknnrt.so"
else
  for candidate in \
    /usr/lib/librknnrt.so \
    /usr/lib64/librknnrt.so \
    /usr/lib/aarch64-linux-gnu/librknnrt.so \
    /lib/aarch64-linux-gnu/librknnrt.so; do
    if [[ -f "${candidate}" ]]; then
      RKNN_RT_LIB="${candidate}"
      break
    fi
  done
fi

if [[ -z "${RKNN_RT_LIB}" ]]; then
  RKNN_RT_LIB="$(ldconfig -p 2>/dev/null | awk '/librknnrt\.so/{print $NF; exit}' || true)"
fi

if [[ -z "${RKNN_RT_LIB}" || ! -f "${RKNN_RT_LIB}" ]]; then
  echo "No librknnrt.so found." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
g++ -O3 -DNDEBUG -std=c++14 -fPIC -shared -pthread \
  -I"${RKNN_INCLUDE_DIR}" \
  "${SRC}" \
  "${RKNN_RT_LIB}" \
  -Wl,-rpath,'$ORIGIN/lib' \
  -o "${OUT}"

echo "built: ${OUT}"
echo "include dir: ${RKNN_INCLUDE_DIR}"
echo "runtime lib source: ${RKNN_RT_LIB}"
echo "runtime lib copied: skipped"
if [[ -f "${LOCAL_RKNN_RT_LIB}" ]]; then
  echo "runtime used at run time: ${LOCAL_RKNN_RT_LIB}"
else
  echo "runtime used at run time: system library search path"
fi
