#/bin/bash

if [ "$#" -lt 1 ]; then
  echo "usage: $0 MODEL_PATH [SCHEDULER]"
  exit
fi

#module load cuda/12.3.2 python/3.12.1 uv/0.8.9
module load cuda/12.3.2 anaconda3.2023.09-0

set -e

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)

#source /sonic_home/willianjunior/vllm-segment/envs/vllm-prebuilt/bin/activate

#source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $GIT_ROOT_PATH/envs/vllm-0.10.1

set -x

MODEL=$1
SCHEDULER=$2

LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MNS=32
MAX_MODEL_LEN=10000
N_GPUS=1

PARAMS=( GOODPUT MAX_CONCUR NUM_PROMPTS REQUEST_RATE REP )
GOODPUT=( 100 ) # need to find out
MAX_CONCUR=( "None" ) # can leave to request_rate
NUM_PROMPTS=( $((4*$MNS)) )

# req_rate@burst@goodput
# values from run on gorgona5: p50, p75, p90, p99
REQUEST_RATE=( "3@0.01@2610" "3@0.01@5481" "3@0.01@7714" "3@0.01@12171" )
REP=( $(seq 1) ) # todo: can later replicate

QUANTZ=""
if [ -n "$SCHEDULER" ]; then
    SCHEDULER="--scheduler $SCHEDULER"
fi

# Start engine server
vllm serve $MODEL --host localhost --port 8000 --gpu-memory-utilization 0.9 --max-model-len $MAX_MODEL_LEN --max-num-seqs $MNS --tensor-parallel-size $N_GPUS $QUANTZ $SCHEDULER & SERVER_PID=$!

# Wait for server start
python3 $GIT_ROOT_PATH/util/wait_vllm.py

if [ -n "$SCHEDULER" ]; then
    SCHEDULER=$(echo "$SCHEDULER" | awk '{print $NF}')
else
    SCHEDULER="baseline"
fi

# Compute sizes for parameters sets
set +x
TESTS_SIZES=()
TOTAL_TESTS=1
for p in "${PARAMS[@]}"; do
  eval "n=\${#$p[@]}"
  TESTS_SIZES+=("$n")
  TOTAL_TESTS=$((TOTAL_TESTS * n))
done
set -x

# Run each parameters set
for ((i=0; i<TOTAL_TESTS; i++)); do
	# Get parameters
	set +x
	idx=$i
	for ((k=${#PARAMS[@]}-1; k>=0; k--)); do
		arr=${PARAMS[k]}
		n=${TESTS_SIZES[k]}
		eval "val=(\"\${$arr[@]}\")"
		# dynamically assign to variable name
		eval "${PARAMS[k]}_VAL=\"\${val[idx % n]}\""
		idx=$((idx / n))
	done
	set -x

	echo Testing $MODEL $SCHEDULER goodput=$GOODPUT_VAL max_concurrency=$MAX_CONCUR_VAL num_prompts=$NUM_PROMPTS_VAL request_rate=$REQUEST_RATE_VAL rep=$REP_VAL
	bash $LOCAL_DIR/run-single-bench.sh $MODEL $GOODPUT_VAL $(( $MAX_CONCUR_VAL * 1000 )) $NUM_PROMPTS_VAL $REQUEST_RATE_VAL $SCHEDULER

done

# Kill server
kill -s TERM $SERVER_PID
wait $SERVER_PID

