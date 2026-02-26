#!/bin/bash

if [ "$#" -lt 1 ]; then
  echo "usage: $0 MODEL_PATH"
  exit
fi

module load cuda/12.3.2 anaconda3.2023.09-0

set -e

set -x

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)
conda activate $GIT_ROOT_PATH/envs/vllm-0.10.1

set -x

hostname

MODEL=$1
LOCAL_DIR=$(pwd)
#LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MNS=32
MAX_MODEL_LEN=10000
N_GPUS=1

QUANTZ=""

# Start engine server
vllm serve $MODEL --host localhost --port 8000 --gpu-memory-utilization 0.9 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MNS --tensor-parallel-size $N_GPUS $QUANTZ & SERVER_PID=$!

# Wait for server start
python3 $GIT_ROOT_PATH/util/wait_vllm.py

NUM_PROMPTS_VAL=$(( 4*$MNS )) 
REQUEST_RATE_VAL="90000@1@100000"
SCHEDULER="none"

bash $LOCAL_DIR/intermediary-scripts/run-single-bench.sh $MODEL 0 $MNS $NUM_PROMPTS_VAL $REQUEST_RATE_VAL $SCHEDULER

