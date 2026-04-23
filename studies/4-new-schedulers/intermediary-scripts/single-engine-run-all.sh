set -e
set +x

module load anaconda3.2023.09-0

conda activate ../../envs/vllm-0.16.0/

set -x

#export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/studies/4-new-schedulers:$PYTHONPATH"
export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/schedulers-v2:$PYTHONPATH"


GIT_ROOT_PATH=$(git rev-parse --show-toplevel)

OUTPUTS_PATH=./vllm_outputs
#rm -rf $OUTPUTS_PATH
mkdir -p $OUTPUTS_PATH

MODEL=$1
SCHEDULER=$2
OFFLOADING=$3
KV_MEM=$4

MODEL_NAME=$(basename $MODEL)

if [[ "$SCHEDULER" == "none" ]]; then
  SCHEDULER_PARAM=""
else
  SCHEDULER_PARAM="--scheduler-cls $SCHEDULER"
fi

if [[ "$OFFLOADING" == "none" ]]; then
  OFFLOADING_PARAM=""
else
  OFFLOADING_PARAM="--kv-offloading-size $OFFLOADING"
fi

KV_MEM_PARAM="--gpu-memory-utilization $KV_MEM"

# Base LLM params
MAX_MODEL_LEN_PARAM="--max-model-len 2000"
MNS_PARAM="--max-num-seqs 30"
TP_PARAM="--tensor-parallel-size 1"

BASE_FILENAME=${MODEL_NAME}-${SCHEDULER}-offld${OFFLOADING}-kvmem${KV_MEM}
SERVER_FILENAME=vllm-${BASE_FILENAME}.log

vllm serve $MODEL --host localhost --port 8000 $KV_MEM_PARAM $MAX_MODEL_LEN_PARAM $MNS_PARAM $TP_PARAM $OFFLOADING_PARAM $SCHEDULER_PARAM --disable-hybrid-kv-cache-manager >$OUTPUTS_PATH/$SERVER_FILENAME 2>&1 & SERVER_PID=$!

# Wait for server start
python3 $GIT_ROOT_PATH/util/wait_vllm.py

# === Benchmarks =============================================

# This benchmark only runs on this env...
set +x
conda deactivate
conda activate $GIT_ROOT_PATH/envs/vllm-0.9.2
set -x

DO_THROUGHPUT=1
DO_SHARE=1

NUM_PROMPTS=30
REQUEST_RATE=999
BURSTNESS=1
MAX_CONCUR=200

BENCHMARK_PATH=$GIT_ROOT_PATH/../vllm/benchmarks
DATASET_PATH=$GIT_ROOT_PATH/datasets

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -O "$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

JSON_FLAGS="--save-result --result-dir $OUTPUTS_PATH --save-detailed"

if [ "$DO_THROUGHPUT" = "1" ]; then
THOUGHPUT_FILENAME="res-$BASE_FILENAME-throughput.json"
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name random \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
        --random-input-len 1 --random-output-len 1900 --ignore-eos \
	$JSON_FLAGS --result-filename $THOUGHPUT_FILENAME
fi

if [ "$DO_SHARE" = "1" ]; then
SHARE_FILENAME="res-$BASE_FILENAME-share.json"
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
	--num-prompts $(( 10 * $NUM_PROMPTS )) \
        --dataset-name sharegpt \
        --dataset-path $GIT_ROOT_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
	$JSON_FLAGS --result-filename $SHARE_FILENAME
fi

# === end Benchmarks ========================================

# Kill server
#set +e
kill -s TERM $SERVER_PID
wait $SERVER_PID
set -e



