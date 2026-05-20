#!/usr/bin/env bash
# Distributed VLM planning inference with one semantic ego-status group ablated.
#
# Usage:
#   CHECKPOINT=/path/to/iter.pth CUDA_DEVICE_ID=0,1,2 \
#     bash scripts/test/dist_test_ablate_ego_status_group.sh <config.py> <num_gpus> <group> \
#     --cfg-options model.save_path=workspace/.../<group>_zero_results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG="${1:?usage: $0 <config.py> <num_gpus> <group> [-- extra args to test.py ...]}"
GPUS="${2:?usage: $0 <config.py> <num_gpus> <group> ...}"
ABLATE_GROUP="${3:?usage: $0 <config.py> <num_gpus> <group> ...}"
shift 3

export SPACEDRIVE_ABLATE_EGO_GROUP="${ABLATE_GROUP}"

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
echo "ABLATION=ego_status_group_${ABLATE_GROUP}_zero"

PORT="${PORT:-29524}"
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
  "${SCRIPT_DIR}/test_ablate_ego_status_group.py" \
  "${CONFIG}" \
  "${CHECKPOINT}" \
  --seed "${SEED}" \
  --launcher pytorch \
  --format-only \
  "$@"
