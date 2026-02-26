#module load cuda/12.3.2 python/3.12.1 uv/0.8.9

set -e

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)

#source /sonic_home/willianjunior/vllm-segment/envs/vllm-prebuilt/bin/activate

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $GIT_ROOT_PATH/envs/vllm-0.10.1

if [ "$#" -ne 6 ]; then
  echo "usage: $0 MODEL GOODPUT MAX_CONCUR NUM_PROMPTS REQUEST_RATE SCHEDULER"
  exit
fi

set -x

BENCHMARK_PATH=$GIT_ROOT_PATH/../vllm/benchmarks
DATASET_PATH=$GIT_ROOT_PATH/datasets

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -O "$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# benchmark script in git
#cd /sonic_home/willianjunior/vllm-segment/git/vllm/benchmarks

MODEL=$1
GOODPUT=$2 # not used... use RBG
MAX_CONCUR=$3 # default: None
NUM_PROMPTS=$4
RBG=$5  # req_rate@burst@goodput_difficulty
SCHEDULER=$6

REQUEST_RATE="${RBG%%@*}"
BURSTNESS="${RBG#*@}"
BURSTNESS="${BURSTNESS%%@*}"
GOODPUT="${RBG##*@}"

# base throughput in tokens/sec/req
#BASE_THROUGHPUT=75 # 2380 tokens/s, MNS=32 1/900 input/output tokens 320 reqs rtx3090
#TPOT_GOODPUT=$(printf "%.2f\n" $(echo "scale=2; 1000*$GOODPUT_DIFFICULTY/$BASE_THROUGHPUT" | bc))
E2EL_GOODPUT=$GOODPUT

RESULT_FILENAME=res-$(basename $MODEL)-tpot_goodput${TPOT_GOODPUT}-req_rate${REQUEST_RATE}-burst${BURTSTNESS}-${SCHEDULER}.json


#python3 $BENCHMARK_PATH/benchmark_serving.py \
#	--base-url http://localhost:8000 --backend openai \
#	--endpoint /v1/completions	\
#	--model $MODEL \
#	--ignore-eos --dataset-name random \
#	--num-prompts $NUM_PROMPTS \
#	--random-input-len 1 --random-output-len 900 \
#	--percentile-metrics ttft,tpot,itl,e2el \
#	--request-rate $REQUEST_RATE --burstiness $BURSTNESS \
#	--goodput e2el:$E2EL_GOODPUT \
#	--max-concurrency $MAX_CONCUR
	#--goodput tpot:$TPOT_GOODPUT


python3 $BENCHMARK_PATH/benchmark_serving.py \
	--base-url http://localhost:8000 \
	--backend vllm --model $MODEL \
	--num-prompts $NUM_PROMPTS \
	--goodput e2el:$E2EL_GOODPUT --result-filename $RESULT_FILENAME \
	--save-result --result-dir ./results --dataset-name sharegpt \
	--dataset-path $GIT_ROOT_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
	--percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
	--request-rate $REQUEST_RATE --burstiness $BURSTNESS \
	--max-concurrency $MAX_CONCUR



