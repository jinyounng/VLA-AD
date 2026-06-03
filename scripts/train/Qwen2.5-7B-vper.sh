#!/usr/bin/env bash

CONFIG="${1:-projects/configs/spacedrive/spacedrive_vper_qwen.py}"
GPUS="${2:-2}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29501}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"

python -m torch.distributed.launch \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --use_env \
    --nproc_per_node="${GPUS}" \
    --master_port="${PORT}" \
    "${SCRIPT_DIR}/train.py" \
    "${CONFIG}" \
    --seed 888 \
    --launcher pytorch \
    ${@:3}
