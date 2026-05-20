#!/usr/bin/env bash
# Distributed VLM planning inference with only past ego xyz coordinates ablated.
#
# This launcher uses scripts/test/test_ablate_past_ego_xyz.py, which temporarily
# zeros memory_egopose[:, :, :3, 3] during generation and restores memory before
# the next temporal update.  It does not edit the model or config files.
#
# Usage:
#   CHECKPOINT=/path/to/iter_x.pth CUDA_DEVICE_ID=4 \
#     bash scripts/test/dist_test_ablate_past_ego_xyz.sh <config.py> <num_gpus> \
#     --cfg-options model.save_path=workspace/.../_results_past_xyz_zero/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG="${1:?usage: $0 <config.py> <num_gpus> [-- extra args to test.py ...]}"
GPUS="${2:?usage: $0 <config.py> <num_gpus> ...}"
shift 2

CUDA_DEVICE_ID="${CUDA_DEVICE_ID:-}"
if [ -n "${CUDA_DEVICE_ID}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE_ID}"
  _n=$(awk -F',' '{print NF}' <<< "${CUDA_DEVICE_ID}")
  if [ "${GPUS}" -ne "${_n}" ]; then
    echo "[WARN] GPUS(${GPUS}) != len(CUDA_DEVICE_ID)=${_n}; using ${_n}"
    GPUS="${_n}"
  fi
fi

CHECKPOINT="${CHECKPOINT:-}"
if [ -z "${CHECKPOINT}" ]; then
  _cfg="${CONFIG}"
  case "${_cfg}" in
    /*) ;;
    *) _cfg="${PROJECT_ROOT}/${_cfg}" ;;
  esac
  CHECKPOINT="$(python3 -c "import importlib.util; p='${_cfg}'; spec=importlib.util.spec_from_file_location('cfg', p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(m.work_dir + 'latest.pth')")"
fi
echo "CHECKPOINT=${CHECKPOINT}"
echo "ABLATION=past_ego_xyz_zero"

PORT="${PORT:-29518}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
SEED="${SEED:-888}"

python3 -m torch.distributed.launch \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --use_env \
  --nproc_per_node="${GPUS}" \
  --master_port="${PORT}" \
  "${SCRIPT_DIR}/test_ablate_past_ego_xyz.py" \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --seed "${SEED}" \
  --launcher pytorch \
  --format-only \
  "$@"
