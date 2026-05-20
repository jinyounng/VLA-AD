#!/usr/bin/env bash
# SpaceDrive + SmolVLM LoRA fine-tuning launcher.
# Edit variables below (or override via env) before running.

set -euo pipefail

CONFIG="${CONFIG:-projects/configs/spacedrive/spacedrive_smolvlm.py}"
CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-1,2}"   # e.g., "2" or "0,1,2"
# Keep GPUS in sync with CUDA_DEVICE_ID length to avoid NCCL duplicate-GPU errors.
NUM_VISIBLE_DEVICES=$(awk -F',' '{print NF}' <<< "${CUDA_DEVICE_ID}")
GPUS="${GPUS:-${NUM_VISIBLE_DEVICES}}"
if [ "${GPUS}" -ne "${NUM_VISIBLE_DEVICES}" ]; then
  echo "[WARN] GPUS(${GPUS}) != number of CUDA_DEVICE_ID entries(${NUM_VISIBLE_DEVICES}). Using ${NUM_VISIBLE_DEVICES}."
  GPUS="${NUM_VISIBLE_DEVICES}"
fi

# LoRA defaults (override via env if needed)
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-5}"
ACCUM_ITERS="${ACCUM_ITERS:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-6}"
WORK_DIR="${WORK_DIR:-workspace/spacedrive_smolvlm_lora_plus}"
WARMUP_ITERS="${WARMUP_ITERS:-500}"
WARMUP_RATIO="${WARMUP_RATIO:-0.3}"
MIN_LR_RATIO="${MIN_LR_RATIO:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
SEED="${SEED:-888}"
DATASET_SIZE="${DATASET_SIZE:-28130}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

ITERS_PER_EPOCH=$(( DATASET_SIZE / (GPUS * BATCH_SIZE) ))
if [ "${ITERS_PER_EPOCH}" -lt 1 ]; then ITERS_PER_EPOCH=1; fi
MAX_ITERS=$(( NUM_EPOCHS * ITERS_PER_EPOCH ))
CKPT_INTERVAL=$(( ITERS_PER_EPOCH / 2 ))
if [ "${CKPT_INTERVAL}" -lt 1 ]; then CKPT_INTERVAL=1; fi

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ID}"

bash "$(dirname "$0")/dist_train.sh" "$CONFIG" "$GPUS" \
  --seed "${SEED}" \
  --cfg-options \
  model.use_lora=True \
  ego_status=PE \
  lr="${LR}" \
  data.samples_per_gpu="${BATCH_SIZE}" \
  optimizer_config.cumulative_iters="${ACCUM_ITERS}" \
  optimizer.weight_decay="${WEIGHT_DECAY}" \
  lr_config.warmup_iters="${WARMUP_ITERS}" \
  lr_config.warmup_ratio="${WARMUP_RATIO}" \
  lr_config.min_lr_ratio="${MIN_LR_RATIO}" \
  find_unused_parameters=True \
  num_gpus="${GPUS}" \
  batch_size="${BATCH_SIZE}" \
  num_iters_per_epoch="${ITERS_PER_EPOCH}" \
  data.shuffler_sampler.num_iters_to_seq="${ITERS_PER_EPOCH}" \
  runner.max_iters="${MAX_ITERS}" \
  checkpoint_config.interval="${CKPT_INTERVAL}" \
  evaluation.interval="${MAX_ITERS}" \
  log_config.interval="${LOG_INTERVAL}" \
  num_epochs="${NUM_EPOCHS}" \
  work_dir="${WORK_DIR}" \
  "$@"
