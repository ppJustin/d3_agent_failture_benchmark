#!/usr/bin/env bash
set -euo pipefail
IMAGE_TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /cpfs01/zijian.pan/conda-envs/d3_v1/bin/python -u "$IMAGE_TEST_DIR/run.py" "$@"
