#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT_DIR}/tools/rknn_capi_parallel.cpp"
OUT_DIR="${ROOT_DIR}/lib"
OUT="${OUT_DIR}/librknn_capi_parallel.so"
LOCAL_RKNN_RT_LIB="${OUT_DIR}/lib/librknnrt.so"
RKNN_INCLUDE_DIR="${ROOT_DIR}/tools/rknn/include"
RKNN_RT_SOURCE="${ROOT_DIR}/tools/rknn/lib/librknnrt.so"

if [[ ! -f "${RKNN_INCLUDE_DIR}/rknn_api.h" ]]; then
  RKNN_API_HEADER="$(find /usr/include /usr/local/include \
    -name rknn_api.h 2>/dev/null | head -n 1 || true)"
  if [[ -n "${RKNN_API_HEADER}" ]]; then
    RKNN_INCLUDE_DIR="$(dirname "${RKNN_API_HEADER}")"
  fi
fi

if [[ ! -f "${RKNN_INCLUDE_DIR}/rknn_api.h" ]]; then
  echo "No rknn_api.h found." >&2
  exit 1
fi

RKNN_RT_LIB=""
if [[ -f "${LOCAL_RKNN_RT_LIB}" ]]; then
  RKNN_RT_LIB="${LOCAL_RKNN_RT_LIB}"
elif [[ -f "${RKNN_RT_SOURCE}" ]]; then
  RKNN_RT_LIB="${RKNN_RT_SOURCE}"
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

mkdir -p "${OUT_DIR}/lib"
if [[ "${RKNN_RT_LIB}" != "${LOCAL_RKNN_RT_LIB}" ]]; then
  cp -f "${RKNN_RT_LIB}" "${LOCAL_RKNN_RT_LIB}"
fi

g++ -O3 -DNDEBUG -std=c++14 -fPIC -shared -pthread \
  -I"${RKNN_INCLUDE_DIR}" \
  "${SRC}" \
  "${LOCAL_RKNN_RT_LIB}" \
  -ldl \
  -Wl,-rpath,'$ORIGIN/lib' \
  -o "${OUT}"

echo "built: ${OUT}"
echo "include dir: ${RKNN_INCLUDE_DIR}"
echo "runtime lib source: ${RKNN_RT_LIB}"
echo "runtime used at run time: ${LOCAL_RKNN_RT_LIB}"
