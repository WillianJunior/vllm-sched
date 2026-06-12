if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <server_pid> <server_log_path> <new_server_log_path>"
    exit 1
fi

set -x

SERVER_PID=$1
SERVER_LOG_PATH=$2
NEW_SERVER_LOG_PATH=$3

if jq -ne '.num_prompts == .completed' $OUTPUTS_PATH/$BENCH_FILENAME >/dev/null; then
    echo "Successfull requests less than total requests submitted..."
    BENCH_RETURN=1
else
    BENCH_RETURN=0
fi

ps -p "$pid" >/dev/null 2>&1
IS_VLLM_ALIVE=$?

# === end Benchmarks ========================================

# Kill server
#set +e
kill -s TERM $SERVER_PID
wait $SERVER_PID
set -e

cp $SERVER_LOG_PATH $NEW_SERVER_LOG_PATH
#cp $TMP_OUTPUTS_PATH/$SERVER_FILENAME $OUTPUTS_PATH/$SERVER_FILENAME

if [ "$IS_VLLM_ALIVE" -ne 0 ] || [ "$BENCH_RETURN" -ne 0 ]; then
    echo "Execution failed. Log on $NEW_SERVER_LOG_PATH"
    exit 1
fi

