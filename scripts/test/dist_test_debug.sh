#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
CKPT_ITER=${CKPT_ITER:-iter_21096.pth}
CKPT_DIR=${CKPT_DIR:-/data/jykim/projects/SpaceDrive/ckpts/spacedrive_plus-qwen2.5vl-7b-unidepth}
DEBUG_FULL_TEST=${DEBUG_FULL_TEST:-0}

if [ -z "$CONFIG" ] || [ -z "$GPUS" ]; then
    echo "Usage: bash scripts/test/dist_test_debug.sh <CONFIG> <GPUS> [extra test.py args...]"
    echo "Optional env:"
    echo "  CKPT_DIR=/path/to/ckpt_dir   (default: ${CKPT_DIR})"
    echo "  CKPT_ITER=iter_xxxxx.pth     (default: ${CKPT_ITER})"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config file not found: $CONFIG"
    exit 1
fi

CHECKPOINT="${CKPT_DIR%/}/${CKPT_ITER}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo CHECKPOINT: $CHECKPOINT

DEBUG_ARGS="--debug-only-one-sample"
if [ "$DEBUG_FULL_TEST" = "1" ]; then
    DEBUG_ARGS=""
fi

PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test_debug.py \
    $CONFIG \
    $CHECKPOINT \
    --seed 888 \
    --launcher pytorch \
    --dump-one-input \
    $DEBUG_ARGS \
    ${@:3}
