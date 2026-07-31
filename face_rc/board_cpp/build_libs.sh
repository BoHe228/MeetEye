#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${ROOT_DIR}/tools/build_direct_slice_opencl_fused.sh"
bash "${ROOT_DIR}/tools/build_rknn_capi_parallel.sh"
bash "${ROOT_DIR}/tools/build_adaface_rknn.sh"
bash "${ROOT_DIR}/tools/build_hybrid_sort_native.sh"
bash "${ROOT_DIR}/tools/build_merge_fast.sh"
bash "${ROOT_DIR}/tools/build_tracker_assoc_fast.sh"

echo "built board_cpp native libs under: ${ROOT_DIR}/lib"
