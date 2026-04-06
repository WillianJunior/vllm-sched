module load anaconda3.2023.09-0

conda activate ../../envs/vllm-0.16.0/

export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/studies/4-new-schedulers:$PYTHONPATH"

export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/schedulers-v2:$PYTHONPATH"

MNS=20

# For 6 reqs:
# mem for 4090: 0.28
# mem for 3090: 0.29

# For 40
# mem for 3090: 

vllm serve /snfs1/llm-models/llama-3.2-3B-Instruct/ --host localhost --port 8000 --gpu-memory-utilization 0.60 --max-model-len 2000 --max-num-seqs $MNS --tensor-parallel-size 1 --kv-offloading-size 5 --disable-hybrid-kv-cache-manager --scheduler-cls $1
#new_scheduler.Scheduler


