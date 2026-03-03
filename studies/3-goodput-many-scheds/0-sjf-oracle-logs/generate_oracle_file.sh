(echo "# REQ_TOKEN_ID PROMPT_TOKENS DECODE_TOKENS"; grep "Finished cmpl" 0-sjf-oracle-logs/results-0-oracle.out | awk '{print $4 " " $6 " " $8}') > oracle.log
