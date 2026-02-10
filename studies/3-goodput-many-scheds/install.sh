#!/bin/bash

# all succede or fail
set -e

export BASE_PATH=$(git rev-parse --show-toplevel)/envs
export VLLM_ENV_NAME=vllm-0.10.1
export PYTHON_VERSION=3.11

# create and activate conda env
conda create -y -p "$BASE_PATH/$VLLM_ENV_NAME" python="$PYTHON_VERSION"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$BASE_PATH/$VLLM_ENV_NAME"

# install uv and vllm
pip3 install --upgrade uv
#uv pip install "transformers==4.33.0" "tokenizers==0.13.3"
#uv pip install vllm==0.10.2

uv pip install "vllm==0.10.1" "transformers==4.57.6"
uv pip install pandas datasets
