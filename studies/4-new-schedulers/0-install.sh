#!/bin/bash

module load anaconda3.2023.09-0

# all succede or fail
set -e -x

export BASE_PATH=$(git rev-parse --show-toplevel)/envs
#export VLLM_ENV_NAME=vllm-0.10.1
#export VLLM_ENV_NAME=vllm-0.9.2
export VLLM_ENV_NAME=vllm-0.16.0
export PYTHON_VERSION=3.11

# create and activate conda env
conda create -y -p "$BASE_PATH/$VLLM_ENV_NAME" python="$PYTHON_VERSION"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$BASE_PATH/$VLLM_ENV_NAME"

# install uv and vllm
pip3 install --upgrade uv
#uv pip install "transformers==4.33.0" "tokenizers==0.13.3"
#uv pip install vllm==0.10.2

# for blackwell gpus (sm_120)
# still not working...
#uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

#uv pip install "vllm==0.10.1" "transformers==4.57.6"
#uv pip install "vllm==0.9.2" "transformers==4.53.0"
uv pip install "vllm==0.16.0"

uv pip install pandas datasets scikit-learn joblib
