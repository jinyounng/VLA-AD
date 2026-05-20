#!/usr/bin/env bash
# SpaceDrive training with SmolVLM-256M backbone
# 2 GPUs (physical id 1,2); per-GPU batch in config is set so global effective batch matches 1-GPU run

CONFIG=projects/configs/spacedrive/spacedrive_smolvlm.py
GPUS=${GPUS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2}

bash $(dirname "$0")/dist_train.sh $CONFIG $GPUS ${@:1}
