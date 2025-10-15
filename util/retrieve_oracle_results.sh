#!/bin/bash

if [[ $# -ne 1 ]]; then
	echo "usage: ./script.sh <FILE.log>"
	echo "output: REQ_ID_TOKEN num_prompt_tokens num_output_tokens"
	exit
fi

#alias transpose='tpse'
transpose(){
        awk '
        {
            for (i=1; i<=NF; i++)  {
                a[NR,i] = $i
            }
        }
        NF>p { p = NF }
        END {
            for(j=1; j<=p; j++) {
                str=a[1,j]
                for(i=2; i<=NR; i++){
                    str=str" "a[i,j];
                }
                print str
            }
        }' $1
}


#tpse(tpse(grep cmpl $1 | grep logger | awk '{for (i=1; i<NF; i++) if ($i == "prompt_token_ids:") print $7 " " $(i+4)}' | sort -k 1 | awk '{print $2}'); tpse(grep cmpl oracle.log | grep Finished | sort -k 2 | awk '{print $4 " " $6}'))
(grep cmpl $1 | grep logger | awk '{for (i=1; i<NF; i++) if ($i == "prompt_token_ids:") print $7 " " $(i+4)}' | sort -k 1 | awk '{print $2}' | transpose; grep cmpl oracle.log | grep Finished | sort -k 2 | awk '{print $4 " " $6}' | transpose) | transpose | sed 's/,//g'

