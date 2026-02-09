module load cuda/12.3.2 python/3.12.1 uv/0.8.9
source /sonic_home/willianjunior/vllm-segment/envs/vllm-prebuilt/bin/activate

wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

cd /sonic_home/willianjunior/vllm-segment/git/vllm/benchmarks
python3 benchmark_serving.py --base-url http://localhost:8000 --backend vllm --model /snfs1/llm-models/llama-3.2-3B-Instruct --max-concurrency 300 --num-prompts 1000 --goodput ttft:3000 tpot:100 --save-result --result-dir ./results --dataset-name sharegpt --dataset-path ../../../datasets/ShareGPT_V3_unfiltered_cleaned_split.json --percentile-metrics ttft,tpot,itl,e2el

