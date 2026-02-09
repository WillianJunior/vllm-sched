#!/bin/bash
#SBATCH --mail-user=guns945@gmail.com
#SBATCH --mail-type=ALL

source /sonic_home/willianjunior/vllm-segment/envs/vllm-v0/bin/activate
cd /sonic_home/willianjunior/vllm-segment/git/vllm-sched/schedulers

N=1000

export VLLM_USE_V1=0; python3 -u -m vllm.entrypoints.openai.api_server \
	--model /snfs1/llm-models/llama-3.2-3B-Instruct --seed 0 \
	--host localhost --max-model-len 10000 --max-num-seqs 20 \
	--enable-chunked-prefill \
	--port 8000 --swap-space 8 --disable-log-requests \
	> vllm-server-oracle.log 2>&1 &

# waits for vllm to start
python3 /sonic_home/willianjunior/vllm-segment/git/vllm-sched/sched-test/wait_vllm.py

# run benchmark
cd /sonic_home/willianjunior/vllm-segment/git/vllm/benchmarks
MINE=8001; VLLM=8000; 
python3 -u benchmark_serving.py --base-url http://localhost:$VLLM \
	--backend vllm --model /snfs1/llm-models/llama-3.2-3B-Instruct \
	--max-concurrency ${N} --num-prompts ${N} \
	--goodput ttft:3000 tpot:100 --dataset-name sharegpt \
	--dataset-path ../../../datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
	--percentile-metrics ttft,tpot,itl,e2el

pkill -f "vllm.entrypoints.openai.api_server"
