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

mkdir -p $OUTPUTS_PATH
TMP_OUTPUTS_PATH="/tmp/$USER/vllm_outputs"
mkdir -p $TMP_OUTPUTS_PATH

WILLIAN_BASE="/sonic_home/willianjunior/vllm-segment/git"
BENCHMARK_PATH="$WILLIAN_BASE/vllm/benchmarks"
DATASET_PATH="$WILLIAN_BASE/vllm-sched/datasets"

# Download dataset if not available
[ -f $DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json ] || wget -O "$DATASET_PATH/ShareGPT_V3_unfiltered_cleaned_split.json" https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

QWEN_MODEL_PATH="/snfs2/guilherme.farany/models/Qwen-3.5-35B-A3B"

# Verifica se o diretório existe e se não está vazio
if [ ! -d "$QWEN_MODEL_PATH" ] || [ -z "$(ls -A $QWEN_MODEL_PATH)" ]; then
    echo "Modelo Qwen não encontrado no cache local. Iniciando o download..."
    mkdir -p "$QWEN_MODEL_PATH"
    
    # Faz o download usando o huggingface-cli (já incluído no env do vllm)
    huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir "$QWEN_MODEL_PATH"
fi

JSON_FLAGS="--save-result --result-dir $TMP_OUTPUTS_PATH --save-detailed --goodput itl:50"

# Clear kv-cache for prefix reuse
curl -X POST "http://localhost:8000/reset_prefix_cache"

if [ "$DO_THROUGHPUT" = "1" ]; then
    BENCH_FILENAME="res-$BASE_FILENAME-throughput.json"
    OUTPUT_LEN=$(( (TOTAL_KV_TOKENS / NUM_PROMPTS) - 10 ))
    python3 $BENCHMARK_PATH/benchmark_serving.py \
        --base-url http://localhost:8000 \
        --backend vllm --model $MODEL \
        --num-prompts $NUM_PROMPTS \
        --dataset-name random \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,75,90,99 \
        --request-rate $REQUEST_RATE --burstiness $BURSTNESS \
        --max-concurrency $MAX_CONCUR \
        --random-input-len 10 --random-output-len $OUTPUT_LEN --ignore-eos \
	$JSON_FLAGS --result-filename $BENCH_FILENAME
fi

if [ "$DO_SHARE" = "1" ]; then
    BENCH_FILENAME="res-${BASE_FILENAME}-reqs${NUM_PROMPTS}-burst${BURSTNESS}.json"
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

# Limpeza, e cálculo do goodput

TEST_CASE="share"
[ "$DO_THROUGHPUT" = "1" ] && TEST_CASE="throughput"

NOME_ESCALONADOR=$SCHEDULER
if [ "$SCHEDULER" = "rr-v3.Scheduler" ]; then
    NOME_ESCALONADOR="SuperInfer"
elif [ "$SCHEDULER" = "teste_rr.Scheduler" ]; then
    NOME_ESCALONADOR="Round-Robin"
elif [ "$SCHEDULER" = "none" ]; then
    NOME_ESCALONADOR="Baseline-FCFS"
fi

echo "Limpando JSON e injetando metadados..."

# Processamento apenas para metadados e limpeza
python3 -c '
import json, sys

filename = sys.argv[1]

# Abre o arquivo JSON gerado pelo benchmark
with open(filename, "r") as f:
    data = json.load(f)

# Injeta metadados
data["scheduler"] = sys.argv[2]
data["offload"] = sys.argv[3]
data["kvmem"] = sys.argv[4]
data["test_case"] = sys.argv[5]

# Limpa o arquivo deletando as listas gigantes
for key in ["input_lens", "output_lens", "ttfts", "itls", "tpots", "generated_texts", "errors", "start_times"]:
    data.pop(key, None)

# Salva de volta no mesmo arquivo no /tmp
with open(filename, "w") as f:
    json.dump(data, f, indent=4)

' "$TMP_OUTPUTS_PATH/$BENCH_FILENAME" "$NOME_ESCALONADOR" "$OFFLOADING" "$KV_MEM" "$TEST_CASE"

mv "$TMP_OUTPUTS_PATH/$BENCH_FILENAME" "$OUTPUTS_PATH/$BENCH_FILENAME"

echo "JSON limpo (com Goodput calculado para limite de ${ITL_LIMIT}s) transferido para $OUTPUTS_PATH."
