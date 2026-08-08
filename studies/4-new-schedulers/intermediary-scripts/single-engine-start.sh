if [ "$#" -ne 7 ]; then
    echo "Usage: $0 <model_path> <scheduler_cls> <scheduler_path> <offloading> <kv_mem> <max_num_seqs> <max_model_len>"
    exit 1
fi

set -e
set +x

module unload anaconda3.2023.09-0
module load anaconda3.2023.09-0
conda deactivate
conda activate ../../../envs/vllm-0.16.0/

set -x

MODEL=$1
SCHEDULER=$2
SCHEDULE_PATH=$3
OFFLOADING=$4
KV_MEM=$5
MNS=$6
MAX_MODEL_LEN=$7

GIT_ROOT_PATH=$(git rev-parse --show-toplevel)

#export PYTHONPATH="/sonic_home/willianjunior/vllm-segment/git/vllm-sched/studies/4-new-schedulers:$PYTHONPATH"
export PYTHONPATH="$GIT_ROOT_PATH/$SCHEDULE_PATH:$PYTHONPATH"


TMP_OUTPUTS_PATH=/tmp/$USER/vllm_outputs
mkdir -p $TMP_OUTPUTS_PATH

MODEL_NAME=$(basename $MODEL)

if [[ "$SCHEDULER" == "none" ]]; then
  SCHEDULER_PARAM="--scheduler-cls vllm.v1.core.sched.scheduler.Scheduler"
else
  SCHEDULER_PARAM="--scheduler-cls $SCHEDULER"
fi

if [[ "$OFFLOADING" == "none" ]]; then
  OFFLOADING_PARAM=""
else
  OFFLOADING_PARAM="--kv-offloading-size $OFFLOADING"
fi

KV_MEM_PARAM="--gpu-memory-utilization $KV_MEM"

# Base LLM params
MAX_MODEL_LEN_PARAM="--max-model-len $MAX_MODEL_LEN"
MNS_PARAM="--max-num-seqs $MNS"
TP_PARAM="--tensor-parallel-size 1"

CHUNKED_PREFILL_PARAM="--enable-chunked-prefill --max-num-batched-tokens 512"

# Adicão do MNS (Max Number of Seqs) e o timestamp atual para garantir que o nome seja único
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BASE_FILENAME=${MODEL_NAME}-${SCHEDULER}-offld${OFFLOADING}-kvmem${KV_MEM}-mns${MNS}-${TIMESTAMP}
SERVER_FILENAME=vllm-${BASE_FILENAME}.log

export VLLM_SERVER_DEV_MODE=1 # allow reset prefix caching

export VLT_TBT_SLO=0.050  # Padrão

echo ">>> Iniciando SuperInfer com TBT_SLO configurado para: $VLT_TBT_SLO segundos <<<"

vllm serve $MODEL --host localhost --port 8000 $KV_MEM_PARAM $MAX_MODEL_LEN_PARAM $MNS_PARAM $TP_PARAM $OFFLOADING_PARAM $SCHEDULER_PARAM --disable-hybrid-kv-cache-manager >$TMP_OUTPUTS_PATH/$SERVER_FILENAME 2>&1 & SERVER_PID=$!

# Wait for server start
python3 $GIT_ROOT_PATH/util/wait_vllm.py

