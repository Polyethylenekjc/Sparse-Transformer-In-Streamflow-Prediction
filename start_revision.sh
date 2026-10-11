#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Prefer the already activated environment. No automatic package downloads.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
if [ "$#" -eq 0 ]; then set -- prepare; fi
exec "${PYTHON:-python}" run_revision.py "$@"
