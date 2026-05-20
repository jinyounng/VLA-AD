#!/usr/bin/env bash
# SmolLM-135M inference / evaluation
# Usage:
#   bash scripts/test/test_smollm.sh
#   bash scripts/test/test_smollm.sh --eval_loss
#   MODEL_PATH=workspace/smollm2-135M-lora/final_model BASE_MODEL_PATH=ckpts/SmolLM-135M bash scripts/test/test_smollm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Defaults ──
MODEL_PATH="${MODEL_PATH:-workspace/smollm2-135M-full/final_model}"
DATA_PATH="${DATA_PATH:-/data/jykim/projects/OpenDriveVLA/data/nuscenes/vqa/val/}"
OUTPUT_PATH="${OUTPUT_PATH:-workspace/smollm_test_results.json}"

CMD="python ${SCRIPT_DIR}/test_smollm.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --batch_size 8 \
    --torch_dtype bfloat16"

# LoRA인 경우 base model 경로 추가
if [ -n "${BASE_MODEL_PATH:-}" ]; then
    CMD="${CMD} --base_model_path ${BASE_MODEL_PATH}"
fi

${CMD} "$@"
