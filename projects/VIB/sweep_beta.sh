#!/usr/bin/env bash
#
# Beta sweep for VIB experiments.
# Launches one training run per beta value sequentially.
#
# Usage:
#   bash projects/VIB/sweep_beta.sh [GPUS] [EXTRA_ARGS...]
#
# Examples:
#   bash projects/VIB/sweep_beta.sh 2
#   bash projects/VIB/sweep_beta.sh 4 --resume-from workspace/spacedrive_vib_qwen/latest.pth

set -euo pipefail

GPUS="${1:-2}"
shift || true
EXTRA_ARGS="${*}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
CONFIG="projects/configs/spacedrive/spacedrive_vib_qwen.py"

BETAS=(0.00001 0.0001 0.001 0.01 0.1)

for BETA in "${BETAS[@]}"; do
    WORK_DIR="workspace/spacedrive_vib_beta_${BETA}/"
    echo "====================================================="
    echo " VIB beta sweep: beta=${BETA}"
    echo " work_dir: ${WORK_DIR}"
    echo "====================================================="

    bash "${REPO_ROOT}/scripts/train/train_vib.sh" \
        "${CONFIG}" \
        "${GPUS}" \
        --cfg-options \
            "model.vib_beta=${BETA}" \
            "work_dir=${WORK_DIR}" \
        ${EXTRA_ARGS}
done

echo "===== Beta sweep complete ====="
