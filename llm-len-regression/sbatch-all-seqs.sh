#!/bin/bash

source /sonic_home/willianjunior/vllm-segment/prep.sh
cd /sonic_home/willianjunior/vllm-segment/llm-len-regression

export VLLM_USE_V1=0; python3 -m vllm.entrypoints.openai.api_server --model /snfs1/llm-models/llama-3.2-3B-Instruct --seed 0 --host localhost --max-model-len 100000 --max-num-seqs 100 --enable-chunked-prefill --port 8000 --disable-log-requests &

python3 -u run-all-seqs.py > all-seqs-len.log



