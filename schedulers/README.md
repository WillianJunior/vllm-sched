#How to run the scheduler:
cd to this path (path of scheduler)

```command
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server --model /snfs1/llm-models/llama-3.2-3B-Instruct --seed 0 --host localhost --max-model-len 8000 --max-num-seqs 100 --enable-sleep-mode --scheduler-cls fcfs.MyFCFSSched
```


