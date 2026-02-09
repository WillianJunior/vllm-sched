module load cuda/12.3.2 python/3.12.1 uv/0.8.9
source /sonic_home/willianjunior/vllm-segment/envs/vllm-prebuilt/bin/activate

if [ "$#" -lt 5 ]; then
  echo "$0 usage: GOODPUT MAX_CONCUR NUM_PROMPTS REQUEST_RATE"
  exit
fi

DATASET_PATH=/sonic_home/willianjunior/vllm-segment/git/vllm-sched/datasets

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# benchmark script in git
cd /sonic_home/willianjunior/vllm-segment/git/vllm/benchmarks


GOODPUT=$1
MAX_CONCUR=$2 # default: None
NUM_PROMPTS=$3
REQUEST_RATE=$4

python3 benchmark_serving.py --base-url http://localhost:8000 --backend vllm --model /snfs1/llm-models/llama-3.2-3B-Instruct --max-concurrency $MAX_CONCUR --num-prompts $NUM_PROMPTS --goodput e2el:$GOODPUT --save-result --result-dir ./results --dataset-name sharegpt --dataset-path ../../../datasets/ShareGPT_V3_unfiltered_cleaned_split.json --percentile-metrics ttft,tpot,itl,e2el --request-rate $REQUEST_RATE

