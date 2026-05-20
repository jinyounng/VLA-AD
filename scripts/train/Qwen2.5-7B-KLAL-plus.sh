#!/usr/bin/env bash

CONFIG="${1:-projects/KLAL/spacedrive_plus_qwen_klal.py}"
GPUS="${2:-1}"
WORK_DIR="${WORK_DIR:-workspace/spacedrive_plus_qwen_klal}"
BATCH_SIZE="${BATCH_SIZE:-1}"
CUMULATIVE_ITERS="${CUMULATIVE_ITERS:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29695}"
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
    --work-dir "${WORK_DIR}" \
    --cfg-options \
    data.samples_per_gpu="${BATCH_SIZE}" \
    optimizer_config.cumulative_iters="${CUMULATIVE_ITERS}" \
    ${@:3}
