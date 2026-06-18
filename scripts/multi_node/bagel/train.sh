#!/bin/bash
# Multi-node training script for Edit-R2。
# Usage: bash train.sh <NODE_RANK>
#   NODE_RANK: 0-indexed rank of the current machine (0, 1, 2, ...)
#
# Run this script on every node with its corresponding rank:
#   Node 0: bash train.sh 0
#   Node 1: bash train.sh 1
#   ...

## Cluster configuration
GPUS_PER_NODE=6 # Number of GPUs to use per node
NUM_MACHINES=4 # Total number of nodes
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_PORT=19501
MASTER_ADDR=${MASTER_ADDR:-"<SET_YOUR_MASTER_NODE_IP>"}   # IP of rank-0 node
RANK=$1 # Passed as first argument

## NCCL settings 
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3
export NCCL_TIMEOUT=5400000

## Wandb
export WANDB_API_KEY=${WANDB_API_KEY:-""}
export WANDB_MODE=online
export WANDB_DIR=${WANDB_DIR:-"$(pwd)"}

## Python path
# Assumes the script is run from the project root
export PYTHONPATH="$(pwd):${PYTHONPATH}"
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch_error.json

## GPU selection
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 # First two GPUs are reserved for reward model

# Reward server
export EDIVAL_SERVER_URL="http://127.0.0.1:12342"

## Launch
accelerate launch \
    --config_file scripts/accelerate_configs/fsdp_hybrid_2node.yaml \
    --num_machines ${NUM_MACHINES} --num_processes ${NUM_PROCESSES} \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_editr2.py \
    --config config/grpo.py:editr2 \
    2>&1 | tee train_$(date +%Y%m%d_%H%M%S).log
