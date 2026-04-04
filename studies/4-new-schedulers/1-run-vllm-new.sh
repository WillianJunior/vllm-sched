module load anaconda3.2023.09-0

conda activate ../../envs/vllm-0.16.0/

export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/studies/4-new-schedulers:$PYTHONPATH"

export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/schedulers-v2:$PYTHONPATH"

MNS=1

vllm serve /snfs1/llm-models/llama-3.2-3B-Instruct/ --host localhost --port 8000 --gpu-memory-utilization 0.28 --max-model-len 2000 --max-num-seqs $MNS --tensor-parallel-size 1 --scheduler-cls $1
#new_scheduler.Scheduler


