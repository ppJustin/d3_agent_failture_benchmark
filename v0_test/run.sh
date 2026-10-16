#!/usr/bin/env bash
set -euo pipefail
V0_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /cpfs01/zijian.pan/conda-envs/d3_v1/bin/python -u "$V0_DIR/run_full.py" "$@"
