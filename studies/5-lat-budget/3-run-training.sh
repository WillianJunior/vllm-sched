#!/bin/bash

module load cuda/12.3.2 anaconda3.2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate ../../envs/vllm-0.16.0-itl/

# Localiza o arquivo de log (no diretorio atual, no pai ou na raiz do repo)
LOG_FILE="${LOG_FILE:-Profiling_100k_multi_turn_new.log}"
if [ ! -f "$LOG_FILE" ]; then
    if [ -f "../../Profiling_100k_multi_turn_new.log" ]; then
        LOG_FILE="../../Profiling_100k_multi_turn_new.log"
    elif [ -f "../Profiling_100k_multi_turn_new.log" ]; then
        LOG_FILE="../Profiling_100k_multi_turn_new.log"
    elif [ -f "$HOME/Profiling_100k_multi_turn_new.log" ]; then
        LOG_FILE="$HOME/Profiling_100k_multi_turn_new.log"
    fi
fi

if [ ! -f "$LOG_FILE" ]; then
    echo "[ERRO] Arquivo de profiling '$LOG_FILE' não foi encontrado!"
    echo "Verifique se o profiling foi executado ou forneça o caminho: LOG_FILE=/caminho/log.log bash 3-run-training.sh"
    exit 1
fi

echo "Usando arquivo de profiling: $LOG_FILE"
echo "Lendo log e construindo base de dados em training_data.txt..."

awk '
BEGIN {
    # Imprime o cabeçalho
    print "step lat_escalonador lat_gpu batch_size token_budget decode_reqs prefill_reqs kv_blocks_used kv_cache_usage num_preemptions" > "training_data.txt"
}
/\[lat_prof_gpu\]/ {
    # 1. A GPU trabalha PRIMEIRO e cospe a latência física
    if(match($0, /lat=([0-9.]+)/, lg)) {
        current_lat_gpu = lg[1]
    }
}
/\[step[0-9]+\] lat:/ {
    # 2. O escalonador finaliza a chamada e anota a latência "suja" DEPOIS
    match($0, /\[step([0-9]+)\]/, s); step_id = s[1]
    match($0, /lat: ([0-9.]+)/, l); lat_esc[step_id] = l[1]
}
/used_tokens_budget:/ {
    # 3. O escalonador cospe as features POR ÚLTIMO
    match($0, /\[step([0-9]+)\]/, s); step_id = s[1]
    if(match($0, /used_tokens_budget: ([-0-9]+)/, a)) tb = a[1]
    if(match($0, /kv_blocks_used: ([-0-9]+)/, b)) kv = b[1]
    if(match($0, /decode_reqs: ([-0-9]+)/, c)) dc = c[1]
    if(match($0, /batch_size: ([-0-9]+)/, d)) bs = d[1]
    if(match($0, /prefill_reqs: ([-0-9]+)/, e)) pf = e[1]
    if(match($0, /kv_cache_usage: ([-0-9.]+)/, f)) ku = f[1]
    if(match($0, /num_preemptions: ([-0-9.]+)/, g)) np = g[1]
    
    # Como essa é a última peça do quebra-cabeça, juntamos tudo e salvamos!
    if (bs != "-1" && current_lat_gpu != "") {
        print step_id, (lat_esc[step_id] * 1000), current_lat_gpu, bs, tb, dc, pf, kv, ku, np >> "training_data.txt"
        
        # Limpa o current_lat_gpu para o próximo ciclo
        current_lat_gpu = ""
    }
}' $LOG_FILE

echo "Base construída! Iniciando o treinamento do modelo..."

export VLLM_PREEMPTION_PENALTY="${VLLM_PREEMPTION_PENALTY:-50.0}"
export VLLM_PREEMPTION_QUOTA="${VLLM_PREEMPTION_QUOTA:-3000}"

#python3 clean_training_data.py
python3 estimate_lat.py