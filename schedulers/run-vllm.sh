if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source /sonic_home/willianjunior/vllm-segment/prep.sh
fi
export VLLM_USE_V1=0; python3 -u -m vllm.entrypoints.openai.api_server --model /snfs1/llm-models/llama-3.2-3B-Instruct --seed 0 --host localhost --max-model-len 4000 --max-num-seqs 1 --scheduler-cls eevdf2.EEVDF --enable-chunked-prefill --gpu-memory-utilization 0.47 --port 8000 --swap-space 8 --disable-log-requests

