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
DO_SHARE=0
DO_MULTITURN=1

REQUEST_RATE=${REQUEST_RATE:-4.0}
BENCHMARK_DURATION=${BENCHMARK_DURATION:-1800}

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

if [ "$DO_MULTITURN" = "1" ]; then
    SHAREGPT_JSON="$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json"
    SHAREGPT_JSONL="$DATASET_PATH/ShareGPT_V3_multiturn_clean.jsonl"

    if [ ! -f "$SHAREGPT_JSONL" ] || [ "${FORCE_REGEN_DATASET:-0}" = "1" ]; then
        echo "Gerando dataset multi-turn JSONL filtrado a partir de $SHAREGPT_JSON..."
        python3 -c '
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)

for i, x in enumerate(data):
    turns = [{"text": m["value"]} for m in x.get("conversations", []) if m.get("from") in ("human", "user") and m.get("value", "").strip()]
    if len(turns) >= 2:
        # Limita caracteres totais da conversa para prevenir context overflow (> 8000 tokens)
        total_chars = sum(len(m.get("value", "")) for m in x.get("conversations", []))
        if total_chars <= 18000:
            print(json.dumps({"session_id": x.get("id", str(i)), "turns": turns}, ensure_ascii=False))
' "$SHAREGPT_JSON" > "$SHAREGPT_JSONL"
        echo "Dataset multi-turn JSONL gerado com sucesso em $SHAREGPT_JSONL"
    fi

    which aiperf >/dev/null 2>&1 || pip install aiperf

    ARTIFACT_DIR="results/${SCHEDULER_NAME}_aiperf_${TIMESTAMP}"
    mkdir -p "$ARTIFACT_DIR"

    echo "Iniciando benchmark multi-turn com AIPerf..."
    aiperf profile \
        --model "$MODEL" \
        --endpoint-type chat \
        --url http://localhost:8000 \
        --streaming \
        --input-file "$SHAREGPT_JSONL" \
        --custom-dataset-type multi_turn \
        --goodput "inter_token_latency:60" \
        --request-rate 4.0 \
        --arrival-pattern constant \
        --benchmark-duration 1800 \
        --random-seed 42 \
        --artifact-dir "$ARTIFACT_DIR"

    echo "Benchmark AIPerf Multi-Turn finalizado. Artefatos salvos em $ARTIFACT_DIR"

    echo "Convertendo métricas do AIPerf para o formato padrão do projeto em results/$RESULT_FILENAME..."
    python3 "$GIT_ROOT_PATH/studies/5-lat-budget/export_aiperf_to_legacy.py" \
        --artifact-dir "$ARTIFACT_DIR" \
        --output-file "results/$RESULT_FILENAME" \
        --model "$MODEL" \
        --concurrency "$CONCURRENCY"
fi
