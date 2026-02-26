#!/bin/bash

set -x

#MODELS_BASE_PATH=/scratch/willian/models
MODELS_BASE_PATH=/snfs1/llm-models
#MODELS=( "llama/Llama-3.1-8B-Instruct" )
MODELS=( "llama-3.2-3B-Instruct" )

for MODEL in "${MODELS[@]}"; do
    bash intermediary-scripts/single-engine-run-all.sh $MODELS_BASE_PATH/$MODEL
done

