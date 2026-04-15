#!/bin/bash

set -x
set -e

# Make schedulers available for import
GIT_ROOT_PATH=$(git rev-parse --show-toplevel)
export PYTHONPATH="$GIT_ROOT_PATH/schedulers/:$PYTHONPATH"

MODELS_BASE_PATH=/snfs1/llm-models

MODELS=( "llama-3.2-3B-Instruct" )
SCHEDULERS=( none rr.Scheduler )  # RR
#SCHEDULERS=( rr.Scheduler )  # RR
#SCHEDULERS=( none )  # RR
#OFFLOADING=( none 5 20 )
OFFLOADING=( 20 )
#KV_MEM=( 0.3 0.95 ) # RTX 3090 MNS 30
KV_MEM=( 0.3 )
#KV_MEM=( 0.29 0.95 ) # RTX 3090
#KV_MEM=( 0.95 ) # RTX 3090

PARAMS=( MODELS SCHEDULERS OFFLOADING KV_MEM )

# Compute sizes for parameters sets
set +x
TESTS_SIZES=()
TOTAL_TESTS=1
for p in "${PARAMS[@]}"; do
  echo $p
  eval "n=\${#$p[@]}"
  TESTS_SIZES+=("$n")
  TOTAL_TESTS=$((TOTAL_TESTS * n))
  echo $n
  echo $TOTAL_TESTS
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

        bash intermediary-scripts/single-engine-run-all.sh $MODELS_BASE_PATH/$MODELS_VAL $SCHEDULERS_VAL $OFFLOADING_VAL $KV_MEM_VAL
done

