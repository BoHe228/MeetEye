#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/build_merge_fast.sh"
bash "${SCRIPT_DIR}/build_tracker_assoc_fast.sh"
