#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONFIG="${1:?usage: $0 <config.py> <num_gpus> [extra test args...]}"
GPUS="${2:?usage: $0 <config.py> <num_gpus> [extra test args...]}"
shift 2

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
SEED=${SEED:-888}

CHECKPOINT=${CHECKPOINT:-"/data/jykim/projects/SpaceDrive/workspace/spacedrive_plus_qwen/iter_21096.pth"}
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

VISUAL_ATTN_ARGS=()
VISUAL_ATTN_BIAS="${VISUAL_ATTN_BIAS:-1}"
VISUAL_ATTN_ALPHA="${VISUAL_ATTN_ALPHA:-3}"
VISUAL_TOKEN_START="${VISUAL_TOKEN_START:--1}"
VISUAL_TOKEN_END="${VISUAL_TOKEN_END:--1}"
VISUAL_ATTN_LAYER_START="${VISUAL_ATTN_LAYER_START:--1}"
VISUAL_ATTN_LAYER_END="${VISUAL_ATTN_LAYER_END:--1}"
VISUAL_ATTN_FORCE_EAGER="${VISUAL_ATTN_FORCE_EAGER:-1}"

if [ "$VISUAL_ATTN_BIAS" = "1" ]; then
    VISUAL_ATTN_ARGS+=(--visual-attn-bias)
fi
VISUAL_ATTN_ARGS+=(--visual-attn-alpha "$VISUAL_ATTN_ALPHA")
VISUAL_ATTN_ARGS+=(--visual-token-start "$VISUAL_TOKEN_START")
VISUAL_ATTN_ARGS+=(--visual-token-end "$VISUAL_TOKEN_END")
VISUAL_ATTN_ARGS+=(--visual-attn-layer-start "$VISUAL_ATTN_LAYER_START")
VISUAL_ATTN_ARGS+=(--visual-attn-layer-end "$VISUAL_ATTN_LAYER_END")

if [ "$VISUAL_ATTN_FORCE_EAGER" = "0" ]; then
    VISUAL_ATTN_ARGS+=(--no-force-eager-attn)
fi

echo CHECKPOINT: "$CHECKPOINT"
echo VISUAL_ATTN_BIAS: "$VISUAL_ATTN_BIAS"
echo VISUAL_ATTN_ALPHA: "$VISUAL_ATTN_ALPHA"
echo VISUAL_TOKEN_RANGE: "[$VISUAL_TOKEN_START, $VISUAL_TOKEN_END)"
echo VISUAL_ATTN_LAYER_RANGE: "[$VISUAL_ATTN_LAYER_START, $VISUAL_ATTN_LAYER_END]"
echo VISUAL_ATTN_FORCE_EAGER: "$VISUAL_ATTN_FORCE_EAGER"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" \
python3 -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "${SCRIPT_DIR}/test_visual_attn_bias.py" \
    "$CONFIG" \
    "$CHECKPOINT" \
    --seed "$SEED" \
    --launcher pytorch \
    "${VISUAL_ATTN_ARGS[@]}" \
    "$@"
