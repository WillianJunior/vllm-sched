if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <model_path> <is_share_gpt [0|1]> <num_prompts> <request_rate> <burstness> <max_concur> <output_paths>"
    exit 1
fi


set -e
set +x


# === Benchmarks =============================================

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)

# This benchmark only runs on this env...
module unload anaconda3.2023.09-0
module load anaconda3.2023.09-0
conda deactivate
conda activate $GIT_ROOT_PATH/envs/vllm-0.9.2
set -x


MODEL=$1
IS_SHARE=$2
if [ "$IS_SHARE" = "1" ]; then
    DO_THROUGHPUT=0
    DO_SHARE=1
else
    DO_THROUGHPUT=1
    DO_SHARE=0
fi

NUM_PROMPTS=$3
REQUEST_RATE=$4
BURSTNESS=$5
MAX_CONCUR=$6
OUTPUTS_PATH=$7

BENCHMARK_PATH=$GIT_ROOT_PATH/../vllm/benchmarks
DATASET_PATH=$GIT_ROOT_PATH/datasets

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -O "$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

JSON_FLAGS="--save-result --result-dir $OUTPUTS_PATH --save-detailed"

# Clear kv-cache for prefix reuse
curl -X POST "http://localhost:8000/reset_prefix_cache"

if [ "$DO_THROUGHPUT" = "1" ]; then
BENCH_FILENAME="res-$BASE_FILENAME-throughput.json"
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name random \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
        --random-input-len 1 --random-output-len 1200 --ignore-eos \
	$JSON_FLAGS --result-filename $BENCH_FILENAME
fi

if [ "$DO_SHARE" = "1" ]; then
BENCH_FILENAME="res-$BASE_FILENAME-share.json"
python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
	--num-prompts $NUM_PROMPTS \
        --dataset-name sharegpt \
        --dataset-path $GIT_ROOT_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
	$JSON_FLAGS --result-filename $BENCH_FILENAME
fi


