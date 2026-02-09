#!/bin/bash
#SBATCH --mail-user=guns945@gmail.com
#SBATCH --mail-type=ALL

source /sonic_home/willianjunior/vllm-segment/envs/vllm-v0/bin/activate
cd /sonic_home/willianjunior/vllm-segment/git/vllm-sched/sched-test

N=10000

export VLLM_USE_V1=0; python3 -u -m vllm.entrypoints.openai.api_server \
	--model /snfs1/llm-models/llama-3.2-3B-Instruct --seed 0 \
	--host localhost --max-model-len 4000 --max-num-seqs 100 \
	--scheduler-cls oracle.Scheduler --enable-chunked-prefill \
	--port 8000 --swap-space 8 --disable-log-requests \
	> vllm-server-oracle.log 2>&1 &

# waits for vllm to start
python3 /sonic_home/willianjunior/vllm-segment/git/vllm-sched/sched-test/wait_vllm.py

# run benchmark
cd /sonic_home/willianjunior/vllm-segment/git/vllm/benchmarks
MINE=8001; VLLM=8000; 
python3 benchmark_serving.py --base-url http://localhost:$VLLM \
	--backend vllm --model /snfs1/llm-models/llama-3.2-3B-Instruct \
	--max-concurrency 200 --num-prompts ${N} \
	--goodput ttft:3000 tpot:100 --dataset-name sharegpt \
	--dataset-path ../../../datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
	--percentile-metrics ttft,tpot,itl,e2el

# waits for all outputs from vllm-server to be flushed
sleep 5

pkill -f "vllm.entrypoints.openai.api_server"
cp /sonic_home/willianjunior/vllm-segment/git/vllm/benchmarks/prompts.txt /sonic_home/willianjunior/vllm-segment/git/vllm-sched/sched-test/

cd /sonic_home/willianjunior/vllm-segment/git/vllm-sched/sched-test
echo "#REQ_TOKEN PROMPT DECODE" > oracle-n${N}.txt;
grep cmpl vllm-server-oracle.log | awk '{print $4 " " $6 " " $8}' > oracle-decode-lens.txt.tmp
sort -k1 -n oracle-decode-lens.txt.tmp >> oracle-n${N}.txt
sed '2d' oracle-n${N}.txt > oracle.tmp # remove the second line: a replicated test for benchmark
mv oracle.tmp oracle-n${N}.txt

rm oracle-decode-lens.txt.tmp
rm vllm-server-oracle.log

# remove windows ^M
tr -d '\r' < prompts.txt > prompts.txt.tmp
mv prompts.txt.tmp prompts.txt

QRF_MODEL=../llm-len-regression/models/random-forest-model-335.pkl.qrf

python3 -u qrf-prep.py --llm-model /snfs1/llm-models/llama-3.2-3B-Instruct --prompts prompts.txt --random-forest-model ../llm-len-regression/models/random-forest-model-335.pkl.qrf > estimations.log

sort -k1 -n estimations.log >> oracle-est-n${N}.txt
rm estimations.log

mv oracle-est-n${N}.txt datasets/
mv oracle-n${N}.txt datasets/
mv prompts.txt datasets/
