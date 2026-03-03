#!/bin/bash

module load cuda/12.3.2 anaconda3.2023.09-0

set -e

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)
MODELS_BASE_PATH=/snfs1/llm-models
BENCHMARK_PATH=$GIT_ROOT_PATH/../vllm/benchmarks
DATASET_PATH=$GIT_ROOT_PATH/datasets

conda activate $GIT_ROOT_PATH/envs/vllm-0.9.2

set -x

export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/studies/3-goodput-many-scheds/sjf-oracle-logs/:$PYTHONPATH"

# Params
MAX_CONCUR=32
NUM_PROMPTS=320
MODEL="llama-3.2-3B-Instruct"

# Start vllm
export VLLM_USE_V1=0; vllm serve $MODELS_BASE_PATH/$MODEL --host localhost --port 8000 --gpu-memory-utilization 0.9 --max-model-len 10000 --max-num-seqs $MAX_CONCUR --tensor-parallel-size 1 --scheduler-cls oracle.Scheduler & SERVER_PID=$!

# Wait for server start
python3 $GIT_ROOT_PATH/util/wait_vllm.py

# Run benchmark with oracle
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODELS_BASE_PATH/$MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name sharegpt \
        --dataset-path $GIT_ROOT_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --max-concurrency $MAX_CONCUR


# Kill server
kill -s TERM $SERVER_PID
wait $SERVER_PID


