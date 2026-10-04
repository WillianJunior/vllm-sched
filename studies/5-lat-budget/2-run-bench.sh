module load cuda/12.3.2 anaconda3.2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"

set -x
set -e

SCHEDULER_NAME=$1
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILENAME="${SCHEDULER_NAME}_${TIMESTAMP}.json"

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)
#conda activate $GIT_ROOT_PATH/envs/vllm-0.9.2
conda activate ../../envs/vllm-0.16.0-itl/

WILLIAN_BASE="/sonic_home/willianjunior/vllm-segment/git"
BENCHMARK_PATH="$WILLIAN_BASE/vllm/benchmarks"
DATASET_PATH="$GIT_ROOT_PATH/datasets"

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -O "$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

MODEL=/snfs1/llm-models/llama-3.2-3B-Instruct/
NUM_PROMPTS=1000
REQUEST_RATE=999
BURSTNESS=0.3
MAX_CONCUR=400

DO_RANDOM=0
DO_SHARE=1

if [ "$DO_RANDOM" = "1" ]; then
vllm bench serve \
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
vllm bench serve \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name sharegpt \
        --dataset-path $GIT_ROOT_PATH/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
	--goodput "itl:60" \
	--save-result \
	--result-dir results \
	--result-filename $RESULT_FILENAME \
	--save-detailed

echo "Limpando métricas não necessárias do arquivo results/$RESULT_FILENAME..."
python3 -c "
import json
import os

file_path = os.path.join('results', '$RESULT_FILENAME')

with open(file_path, 'r') as f:
    data = json.load(f)

chaves_para_remover = ['input_lens', 'output_lens', 'itls', 'ttfts', 'tpots', 'generated_texts', 'errors', 'start_times']

for chave in chaves_para_remover:
        data.pop(chave, None)

with open(file_path, 'w') as f:
    json.dump(data, f, indent=4)
"
echo "Benchmark processado e salvo com sucesso em results/$RESULT_FILENAME"
fi
