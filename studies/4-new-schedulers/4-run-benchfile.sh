#!/bin/bash

BENCHFILE=$1

set -x
set -e

while IFS= read -r line; do
    # skip empty lines
    [[ -z "$line" ]] && continue

    # skip comments
    [[ "$line" =~ ^[[:space:]]*# ]] && continue

    echo "Testing: $line"

    read -r -a arr <<< "$line"
    server_config=("${arr[@]:0:7}")
    bench_config=("${arr[@]:7}")

    echo "=== vLLM start ==========================================================="
    source intermediary-scripts/single-engine-start.sh "${server_config[@]}"

    echo "=== benchmarking  ========================================================"
    source intermediary-scripts/run-single-bench.sh "${bench_config[@]}"

    echo "=== vLLM stop ============================================================"
    bash intermediary-scripts/single-engine-stop.sh $SERVER_PID $TMP_OUTPUTS_PATH/$SERVER_FILENAME $OUTPUTS_PATH/$SERVER_FILENAME

done < $BENCHFILE

