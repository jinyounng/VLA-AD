#!/usr/bin/env bash
# Qwen2.5-VL-3B-Instruct — same entry as Qwen2.5-7B-plus.sh; config: spacedrive_qwen_3B_plus.py
#
# Before training (see docs/data_model.md):
#   huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ckpts/Qwen2.5-VL-3B-Instruct
#   set ckpt_path in tools/token_additor.py, then: python tools/token_additor.py
#   (output dir should match: ckpts/Qwen2.5-VL-3B-Instruct-with-new-special-tokens/)
#   update projects/mmdet3d_plugin/datasets/utils/constants.py token ids as needed
#
# Usage:
#   bash scripts/train/Qwen2.5-3B.sh
#   bash scripts/train/Qwen2.5-3B.sh projects/configs/spacedrive/spacedrive_qwen_3B_plus.py 1
    
CONFIG="${1:-projects/configs/spacedrive/spacedrive_qwen_3B_plus.py}"
GPUS="${2:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-29510}"
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
