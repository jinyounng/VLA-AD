#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}



# Load config and find the checkpoint path
CKPT_ITER='latest.pth'
CHECKPOINT=$(python -c "import importlib.util; spec = importlib.util.spec_from_file_location('config', '$CONFIG'); config = importlib.util.module_from_spec(spec); spec.loader.exec_module(config); print(config.work_dir + '$CKPT_ITER')")
if [ -z "$CHECKPOINT" ]; then
    echo "Checkpoint not found in config file. Please provide a checkpoint path."
    exit 1
fi

echo CHECKPOINT: $CHECKPOINT


PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --use_env \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/test.py \
    $CONFIG \
    $CHECKPOINT \
    --seed 888 \
    --launcher pytorch \
    ${@:3}
