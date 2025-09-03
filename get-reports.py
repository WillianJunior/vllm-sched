import requests
import concurrent.futures
from time import time, sleep
from prometheus_client.parser import text_string_to_metric_families
from collections import defaultdict
import sys

url_base = "http://127.0.0.1:8000"

# Retry logic to wait until the server is up
def wait_for_server(timeout=60, retry_interval=2):
    start_time = time()
    while time() - start_time < timeout:
        try:
            response = requests.get(url_base + "/health")
            if response.status_code < 500:
                print("Server is up!")
                return True
        except requests.exceptions.ConnectionError:
            pass  # Server not up yet
        print(f"Waiting for server at {url_base}...")
        sleep(retry_interval)
    print("Timeout waiting for server.")
    return False

def get_vllm_metrics_json(should_get_hist=False):
    response = requests.get(url_base + '/metrics')
    response.raise_for_status()  # Raise error if request failed
    raw_metrics = response.text

    metrics = {}

    for family in text_string_to_metric_families(raw_metrics):
        #print(family)
        f_name = family.name

        data = defaultdict(lambda: [])
        for sample in family.samples:
            is_not_hist = 'bucket' not in sample.name
            if is_not_hist or should_get_hist:
                data[sample.name].append((sample.labels, sample.value))
            
        #    metric = {
        #        "name": sample.name,
        #        "labels": sample.labels,
        #        "value": sample.value,
        #        "type": family.type,
        #        "help": family.documentation
        #    }
            #metrics_json.append(metric)
        final_data = {}
        for s_name, samples in data.items():
            if len(samples) == 1:
                label, value = samples[0]
                final_data[s_name] = value
            else:
                final_data[s_name] = samples
        metrics[f_name] = final_data
    return metrics

def print_metrics():
    metrics = get_vllm_metrics_json()

    def get_avg(name):
       return metrics[name][name + '_sum'] / metrics[name][name + '_count']

    avg_latency = get_avg('vllm:e2e_request_latency_seconds')

    prefix_name = 'vllm:prefix_cache_'
    prefix_hit_rate = metrics[prefix_name + 'hits'][prefix_name + 'hits_total'] / metrics[prefix_name + 'queries'][prefix_name + 'queries_total']

    req_run_name = 'vllm:num_requests_running'
    num_req_run = metrics[req_run_name][req_run_name]

    req_wait_name = 'vllm:num_requests_waiting'
    num_req_wait = metrics[req_wait_name][req_wait_name]

    gpu_cache_name = 'vllm:gpu_cache_usage_perc'
    gpu_cache_use = metrics[gpu_cache_name][gpu_cache_name]

    kv_cache_name = 'vllm:kv_cache_usage_perc'
    kv_cache_use = metrics[kv_cache_name][kv_cache_name]

    n_preempt_name = 'vllm:num_preemptions'
    num_preemptions = metrics[n_preempt_name][n_preempt_name + '_total']

    avg_ttft = get_avg('vllm:time_to_first_token_seconds')
    avg_tpot = get_avg('vllm:time_per_output_token_seconds')

    avg_req_queue_time = get_avg('vllm:request_queue_time_seconds')
    avg_req_inf_time = get_avg('vllm:request_inference_time_seconds')
    avg_req_pref_time = get_avg('vllm:request_prefill_time_seconds')
    avg_req_dec_time = get_avg('vllm:request_decode_time_seconds')

    print(f"{avg_latency}|{prefix_hit_rate}|{num_req_run}|{num_req_wait}|{gpu_cache_use}|{kv_cache_use}|{num_preemptions}|{avg_ttft}|{avg_tpot}|{avg_req_queue_time}|{avg_req_inf_time}|{avg_req_pref_time}|{avg_req_dec_time}")

def main():
    assert len(sys.argv) == 2, "Must set the logging interval"

    wait_for_server()
    print(f"avg_latency|prefix_hit_rate|num_req_run|num_req_wait|gpu_cache_use|kv_cache_use|num_preemptions|avg_ttft|avg_tpot|avg_req_queue_time|avg_req_inf_time|avg_req_pref_time|avg_req_dec_time")
    while True:
        print_metrics()
        sleep(int(sys.argv[1]))


if __name__ == '__main__':
    main()


