# First run the oracle.Scheduler cls | tee oracle.log

# TOKEN_ID prompt output
(grep cmpl oracle.log | grep logger | awk '{for (i=1; i<NF; i++) if ($i == "prompt_token_ids:") print $7 " " $(i+4)}' | sort -k 1 | awk '{print $2}' | transpose; grep cmpl oracle.log | grep Finished | sort -k 2 | awk '{print $4 " " $6}' | transpose) | transpose
