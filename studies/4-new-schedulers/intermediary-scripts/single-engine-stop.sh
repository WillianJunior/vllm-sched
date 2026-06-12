if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <server_pid> <server_log_path> <new_server_log_path>"
    exit 1
fi

set -x

SERVER_PID=$1
SERVER_LOG_PATH=$2
NEW_SERVER_LOG_PATH=$3

# === end Benchmarks ========================================

# Kill server
#set +e
kill -s TERM $SERVER_PID
wait $SERVER_PID
set -e

cp $SERVER_LOG_PATH $NEW_SERVER_LOG_PATH
#cp $TMP_OUTPUTS_PATH/$SERVER_FILENAME $OUTPUTS_PATH/$SERVER_FILENAME


