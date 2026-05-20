#!/usr/bin/env bash
# SmolVLM-256M LoRA fine-tuning on NuScenes VQA data
# Target HW: 1x A6000 48GB (or any GPU with >=24GB)
# Usage:
#   bash scripts/train/smolvlm-256M_LoRA.sh
#   bash scripts/train/smolvlm-256M_LoRA.sh --num_train_epochs 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Defaults (override via env vars) ──
MODEL_PATH="${MODEL_PATH:-ckpts/SmolVLM-256M-Instruct}"
DATA_ROOT="${DATA_ROOT:-/data/jykim/projects/OpenDriveVLA/data/nuscenes}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace/smolvlm-256M-lora}"
GPUS="${GPUS:-1}"

if [ "${GPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=${GPUS}"
else
    LAUNCHER="python"
fi

${LAUNCHER} "${SCRIPT_DIR}/train_smolvlm.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --data_root "${DATA_ROOT}" \
    --anno_path "nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl" \
    --vqa_dir "vqa/train/" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 2048 \
    --use_lora True \
    --lora_rank 8 \
    --lora_alpha 8 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 6 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 3 \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none \
    --seed 888 \
    "$@"
