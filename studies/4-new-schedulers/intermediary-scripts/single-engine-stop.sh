if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <server_pid> <server_log_path> <new_server_log_path> <json_result_file>"
    exit 1
fi

set -x

SERVER_PID=$1
SERVER_LOG_PATH=$2
NEW_SERVER_LOG_PATH=$3
JSON_RESULT_FILE=$4

#if jq -e '.num_prompts == .completed' $SERVER_LOG_PATH >/dev/null; then
#    BENCH_RETURN=0
#else
#    echo "Successfull requests less than total requests submitted:"
#    BENCH_RETURN=1
#fi

jq -e '.num_prompts == .completed' "$JSON_RESULT_FILE" >/dev/null
BENCH_RETURN=$?

ps -p "$SERVER_PID" >/dev/null 2>&1
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

