# Done manually

MNS=32

vllm serve /snfs1/llm-models/llama-3.2-3B-Instruct/ --gpu-memory-utilization 0.9 --max-model-len 1000 --max-num-seqs $MNS

# finding throughput and tokens/sec/req
python3 benchmark_serving.py --base-url http://localhost:8000 --backend vllm --endpoint /v1/completions --model /snfs1/llm-models/llama-3.2-3B-Instruct/ --ignore-eos --dataset-name random --num-prompts $(($MNS * 10)) --random-input-len 1 --random-output-len 900

# testing goodput
python3 benchmark_serving.py --base-url http://localhost:8000 --backend vllm --endpoint /v1/completions --model /snfs1/llm-models/llama-3.2-3B-Instruct/ --ignore-eos --dataset-name random --num-prompts $(($MNS * 2)) --random-input-len 1 --random-output-len 900 --percentile-metrics ttft,tpot,itl,e2el --goodput e2el:13000 --request-rate 4 --burstiness 0.2

