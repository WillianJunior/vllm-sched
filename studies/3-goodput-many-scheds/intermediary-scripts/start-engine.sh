
#/bin/bash

SCHEDULER=$1
PARAMS=( GOODPUT MAX_CONCUR NUM_PROMPTS REQUEST_RATE REP )
GOODPUT=( 100 ) # need to find out
MAX_CONCUR=( "None" ) # can leave to request_rate
NUM_PROMPTS=( 20 )
REQUEST_RATE=( 1 ) # need ot find out
REP=( $(seq 1) ) # todo: can later replicate

module load cuda/12.3.2 python/3.12.1 uv/0.8.9
source /sonic_home/willianjunior/vllm-segment/envs/vllm-prebuilt/bin/activate

# Start engine server
vllm serve --model $MODEL --gpu-memory-utilization 0.9 --max-model-len 10000 --max-num-seqs $MNS --tensor-parallel-size $N_GPUS $QUANTZ & SERVER_PID=$!

# Wait for server start
python3 /sonic_home/willianjunior/vllm-segment/git/vllm-sched/util/wait_vllm.py


# Compute sizes for parameters sets
TESTS_SIZES=()
TOTAL_TESTS=1
for p in "${PARAMS[@]}"; do
  eval "n=\${#$p[@]}"
  TESTS_SIZES+=("$n")
  TOTAL_TESTS=$((TOTAL_TESTS * n))
done

# Run each parameters set
for ((i=0; i<TOTAL_TESTS; i++)); do
	idx=$i
	for ((k=${#PARAMS[@]}-1; k>=0; k--)); do
		arr=${PARAMS[k]}
		n=${TESTS_SIZES[k]}
		eval "val=(\"\${$arr[@]}\")"
		# dynamically assign to variable name
		eval "${PARAMS[k]}_VAL=\"\${val[idx % n]}\""
		idx=$((idx / n))
	done
	echo Testing goodput=$GOODPUT_VAL max_concurrency=$MAX_CONCUR_VAL num_prompts=$NUM_PROMPTS_VAL request_rate=$REQUEST_RATE_VAL rep=$REP_VAL
	bash /sonic_home/willianjunior/vllm-segment/git/vllm-sched/studies/3-goodput-many-scheds/run-bench.sh $GOODPUT_VAL $MAX_CONCUR_VAL $NUM_PROMPTS_VAL $REQUEST_RATE_VAL

done

# Kill server
kill -s TERM $SERVER_PID
wait $SERVER_PID

