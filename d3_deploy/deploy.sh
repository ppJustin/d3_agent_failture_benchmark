#!/usr/bin/env bash
set -euo pipefail
D3_DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /cpfs01/zijian.pan/conda-envs/d3_v1/bin/python "$D3_DEPLOY_DIR/deploy.py" "$@"
