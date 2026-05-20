#!/usr/bin/env bash
# SmolLM-135M LoRA fine-tuning on NuScenes driving data
# Target HW: 1x A6000 Ada 48GB
# Usage:
#   bash scripts/train/smollm2-135M_LoRA.sh
#   bash scripts/train/smollm2-135M_LoRA.sh --num_train_epochs 10

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Defaults (override via env vars or CLI passthrough) ──
MODEL_PATH="${MODEL_PATH:-ckpts/SmolLM-135M}"
DATA_PATH="${DATA_PATH:-/data/jykim/projects/OpenDriveVLA/data/nuscenes/vqa/train/}"
OUTPUT_DIR="${OUTPUT_DIR:-workspace/smollm2-135M-lora}"
GPUS="${GPUS:-1}"

if [ "${GPUS}" -gt 1 ]; then
    LAUNCHER="torchrun --nproc_per_node=${GPUS}"
else
    LAUNCHER="python"
fi

# effective batch = 128 * 2 * 1 GPU = 256
${LAUNCHER} "${SCRIPT_DIR}/train_smollm.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length 2048 \
    --use_lora True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 6 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 5 \
    --bf16 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to none \
    --seed 888 \
    "$@"
