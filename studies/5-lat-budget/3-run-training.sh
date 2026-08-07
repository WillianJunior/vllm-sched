module load anaconda3.2023.09-0

conda activate ../../envs/vllm-0.16.0/

# Assuming we have a log file with:
# (EngineCore_DP0 pid=590575) [sched][lat_prof][step2298] lat: 0.0005115789826959372
# (EngineCore_DP0 pid=590575) [sched][lat_prof][step2298] used_tokens_budget: 1 kv_blocks_used: 110 decode_reqs: 1 batch_size: 1 prefill_reqs: 0

LOG_FILE=profiling-mns400-tb8000.log

awk 'BEGIN{print "step lat batch_size token_budget decode_reqs prefill_reqs kv_blocks_used"} /lat:/{lat=$NF} /used_tokens_budget:/{match($0,/used_tokens_budget: ([0-9]+)/,a); match($0,/kv_blocks_used: ([0-9]+)/,b); match($0,/decode_reqs: ([0-9]+)/,c); match($0,/batch_size: ([0-9]+)/,d); match($0,/prefill_reqs: ([0-9]+)/,e); if(++i>1) print i-1, lat, d[1], a[1], c[1], e[1], b[1]}' $LOG_FILE > training_data.txt

python3 estimate_lat.py
