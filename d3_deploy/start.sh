#!/usr/bin/env bash
set -euo pipefail
D3_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_VISIBLE_DEVICES="${D3_GPU:-0}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
exec /cpfs01/zijian.pan/conda-envs/d3_v1/bin/python -u "$D3_DIR/model/d3_server.py" --model "$D3_DIR/model" --device cuda:0 --batch-size 1 --host 127.0.0.1 --port 8899 "$@"
