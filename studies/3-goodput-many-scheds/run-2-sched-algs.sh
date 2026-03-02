#!/bin/bash

set -x

# Make schedulers available for import
export PYTHONPATH="/home/vip/willianjunior/git/vllm-sched/schedulers/:$PYTHONPATH"

#MODELS_BASE_PATH=/scratch/willian/models
MODELS_BASE_PATH=/snfs1/llm-models

#MODELS=( "llama/Llama-3.1-8B-Instruct" )
MODELS=( "llama-3.2-3B-Instruct" )
SCHEDULERS=(  )

PARAMS=( MODELS SCHEDULERS )

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

	bash intermediary-scripts/single-engine-run-all.sh $MODELS_BASE_PATH/$MODEL_VAL $SCHEDULER_VAL
done

