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

JSON_FLAGS="--save-result --result-dir $TMP_OUTPUTS_PATH --save-detailed"

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
fi

# Definição limite de ITL (Inter-Token Latency) para cálculo do Goodput
# Tenta pegar a variável VLT_TBT_SLO que foi exportada no start.sh. 
ITL_LIMIT=${VLT_TBT_SLO:-0.05} 

echo "Calculando Goodput com a meta auditada de ITL_LIMIT=${ITL_LIMIT}s..."

# Processamento e Cálculo de Goodput
python3 -c '
import json, sys

filename = sys.argv[1]
limit = float(sys.argv[2])

# Abre o arquivo JSON gerado pelo benchmark
with open(filename, "r") as f:
    data = json.load(f)

# Tenta pegar a lista de itls ou tpots (suporta diferentes versões do vLLM)
metrics = data.get("itls", data.get("tpots", []))
valid_count = 0
total = len(metrics)

# Percorre os tempos de todos os prompts
for m in metrics:
    if isinstance(m, list): # Se for lista de tokens, faz a média
        avg = sum(m) / len(m) if len(m) > 0 else 0
    elif m is None: # Se não gerou tokens (OOM ou erro)
        avg = 0
    else: # Se já for um número direto
        avg = float(m)
    
    # Verifica se cumpriu o SLO estipulado
    if avg <= limit:
        valid_count += 1

# Calcula a vazão de requisições bem-sucedidas (Goodput real)
duration = data.get("duration", 1.0)
if duration <= 0: duration = 1.0

if total > 0:
    data["request_goodput"] = valid_count / duration
    data["goodput_percentage"] = (valid_count / total) * 100
else:
    data["request_goodput"] = 0.0
    data["goodput_percentage"] = 0.0

# Injeta metadados
data["itl_limit_applied"] = limit
data["scheduler"] = sys.argv[3]
data["offload"] = sys.argv[4]
data["kvmem"] = sys.argv[5]
data["test_case"] = sys.argv[6]

# Limpa o arquivo deletando as listas gigantes
for key in ["input_lens", "output_lens", "ttfts", "itls", "tpots", "generated_texts", "errors"]:
    data.pop(key, None)

# Salva de volta no mesmo arquivo no /tmp
with open(filename, "w") as f:
    json.dump(data, f, indent=4)

' "$TMP_OUTPUTS_PATH/$BENCH_FILENAME" "$ITL_LIMIT" "$NOME_ESCALONADOR" "$OFFLOADING" "$KV_MEM" "$TEST_CASE"

cp "$TMP_OUTPUTS_PATH/$BENCH_FILENAME" "$OUTPUTS_PATH/$BENCH_FILENAME"

echo "JSON limpo (com Goodput calculado para limite de ${ITL_LIMIT}s) transferido para $OUTPUTS_PATH."
