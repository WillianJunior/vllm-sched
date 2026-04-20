import os
import sys
import json

# All json fields from benchmark serve
all_columns = ['date', 'backend', 'model_id', 'tokenizer_id', 'num_prompts', 'request_rate', 'burstiness', 'max_concurrency', 'duration', 'completed', 'total_input_tokens', 'total_output_tokens', 'request_throughput', 'request_goodput', 'output_throughput', 'total_token_throughput', 'input_lens', 'output_lens', 'ttfts', 'itls', 'generated_texts', 'errors', 'mean_ttft_ms', 'median_ttft_ms', 'std_ttft_ms', 'p50_ttft_ms', 'p75_ttft_ms', 'p90_ttft_ms', 'p99_ttft_ms', 'mean_tpot_ms', 'median_tpot_ms', 'std_tpot_ms', 'p50_tpot_ms', 'p75_tpot_ms', 'p90_tpot_ms', 'p99_tpot_ms', 'mean_itl_ms', 'median_itl_ms', 'std_itl_ms', 'p50_itl_ms', 'p75_itl_ms', 'p90_itl_ms', 'p99_itl_ms', 'mean_e2el_ms', 'median_e2el_ms', 'std_e2el_ms', 'p50_e2el_ms', 'p75_e2el_ms', 'p90_e2el_ms', 'p99_e2el_ms']

to_ignore_cols = ['generated_texts', 'errors', 'ttfts', 'itls', 'input_lens', 'output_lens']

PATH = sys.argv[1]  # first command-line argument

COLUMNS_OF_INTEREST = ['output_throughput', 'total_token_throughput', ]  # change this


def parse_filename(filename):
    # remove extension
    name = os.path.splitext(filename)[0]
    parts = name.split("-")
    # last 4 fields are fixed structure
    scheduler, offload, kvmem, test_case = parts[-4:]

    #if scheduler == 'none':
    #    scheduler = 'FCFS'

    offload = offload.strip('offld')
    kvmem = kvmem.strip('kvmem')

    return_dict = {}
    return_dict['scheduler'] = scheduler
    return_dict['offload'] = offload
    return_dict['kvmem'] = kvmem
    return_dict['test_case'] = test_case

    return return_dict


results = []

for fname in os.listdir(PATH):
    if not fname.endswith(".json"):
        continue

    filename_fields = parse_filename(fname)
    file_path = os.path.join(PATH, fname)

    with open(file_path) as f:
        data = json.load(f)

    data.update(filename_fields)
    data['model_id'] = os.path.basename(data['model_id'])

    for col in to_ignore_cols:
        del data[col]

    results.append(data)

print(results)
