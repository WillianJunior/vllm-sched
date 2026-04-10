module load cuda/12.3.2 anaconda3.2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"

set -x
set -e

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)
conda activate $GIT_ROOT_PATH/envs/vllm-0.9.2

BENCHMARK_PATH=$GIT_ROOT_PATH/../vllm/benchmarks
DATASET_PATH=$GIT_ROOT_PATH/datasets

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -O "$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

MODEL=/snfs1/llm-models/llama-3.2-3B-Instruct/
NUM_PROMPTS=30
REQUEST_RATE=999
BURSTNESS=1
MAX_CONCUR=200

DO_RANDOM=1
DO_SHARE=1

if [ "$DO_RANDOM" = "1" ]; then
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name random \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
	--random-input-len 1 --random-output-len 1900 --ignore-eos
fi

if [ "$DO_SHARE" = "1" ]; then
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name sharegpt \
        --dataset-path $GIT_ROOT_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR
fi
