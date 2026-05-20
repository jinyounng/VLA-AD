#!/usr/bin/env bash
# SpaceDrive + SmolVLM planning inference (wrapper around dist_test_vlm_planning.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG="${CONFIG:-projects/configs/spacedrive/spacedrive_smolvlm.py}"
GPUS="${GPUS:-2}"

exec bash "${SCRIPT_DIR}/dist_test_vlm_planning.sh" "${CONFIG}" "${GPUS}" "$@"
