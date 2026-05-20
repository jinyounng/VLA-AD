#!/usr/bin/env bash
# SmolVLM-256M inference / evaluation on NuScenes VQA
# Usage:
#   bash scripts/test/test_smolvlm.sh
#   MODEL_PATH=workspace/smolvlm-256M-lora/final_model BASE_MODEL_PATH=ckpts/SmolVLM-256M-Instruct bash scripts/test/test_smolvlm.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ── Defaults ──
MODEL_PATH="${MODEL_PATH:-workspace/smolvlm-256M-full/final_model}"
DATA_ROOT="${DATA_ROOT:-/data/jykim/projects/OpenDriveVLA/data/nuscenes}"
OUTPUT_PATH="${OUTPUT_PATH:-workspace/smolvlm_test_results.json}"

CMD="python ${SCRIPT_DIR}/test_smolvlm.py \
    --model_path ${MODEL_PATH} \
    --data_root ${DATA_ROOT} \
    --anno_path nuscenes2d_ego_temporal_infos_val.pkl \
    --vqa_dir vqa/val/ \
    --output_path ${OUTPUT_PATH} \
    --max_new_tokens 256 \
    --temperature 0.0 \
    --torch_dtype bfloat16"

# LoRA인 경우 base model 경로 추가
if [ -n "${BASE_MODEL_PATH:-}" ]; then
    CMD="${CMD} --base_model_path ${BASE_MODEL_PATH}"
fi

${CMD} "$@"
