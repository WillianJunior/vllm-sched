#!/bin/bash

module load cuda/12.3.2 anaconda3.2023.09-0
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate ../../envs/vllm-0.16.0-itl/

LOG_FILE="metricas.log" 

echo "Lendo log e construindo base de dados..."

awk '
BEGIN {
    # Imprime o cabeçalho
    print "step lat_escalonador lat_gpu batch_size token_budget decode_reqs prefill_reqs kv_blocks_used" > "training_data.txt"
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
    
    # Como essa é a última peça do quebra-cabeça, juntamos tudo e salvamos!
    if (bs != "-1" && current_lat_gpu != "") {
        print step_id, (lat_esc[step_id] * 1000), current_lat_gpu, bs, tb, dc, pf, kv >> "training_data.txt"
        
        # Limpa o current_lat_gpu para o próximo ciclo
        current_lat_gpu = ""
    }
}' $LOG_FILE

echo "Base construída! Iniciando o treinamento do modelo..."

python3 clean_training_data.py
python3 estimate_lat.py