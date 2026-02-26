#!/bin/bash
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <log_file.out>"
  exit 1
fi

F=$1;echo "req_rate burstness slo req_throughput req_goodput total_token_throughput e2el:p50 e2el:p75 e2el:p90 e2el:p95"; (grep -v echo $F | grep Testing | awk '{print $7}' | tr '@' ' ' | tr '=' ' ' | awk '{print $2 " " $3 " " $4}' | transpose; grep "Request throughput" $F | awk '{print $4}' | transpose; grep "Request goodput" $F | awk '{print $4}' | transpose; grep "Total Token throughput" $F | awk '{print $5}' | transpose; grep "P50 E2EL" $F | awk '{print $4}' | transpose; grep "P75 E2EL" $F | awk '{print $4}' | transpose; grep "P90 E2EL" $F | awk '{print $4}' | transpose; grep "P99 E2EL" $F | awk '{print $4}' | transpose;) | transpose | tr "." ","

