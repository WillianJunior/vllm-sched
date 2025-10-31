IN_FILE=$1
OUT_FILE=$2
grep Prompt_len $IN_FILE | sed 's/\]//g' | sort -k2 -n | awk '{print $7}' > $OUT_FILE
