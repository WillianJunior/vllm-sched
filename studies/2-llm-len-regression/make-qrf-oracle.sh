echo "#ID prompt_tokens decode_tokens estimated_tokens" > qrf-oracle.txt; ((cat preds950.log | grep -v "#" | transpose; cat all-seqs-len40kmodel-10kgen.log | grep -v "======" | sed "s/\]/ /g" | sort -k2 -n | head -950 | awk '{print $7}' | transpose) | transpose) | sed "s/;/ /g" | awk '{print $1 " " $3 " " $6 " " $5}' >> qrf-oracle.txt

# make filtered version
cat qrf-oracle.txt | awk '$3 > 1' | awk '$3 < 10000' > qrf-oracle-filtered.txt
