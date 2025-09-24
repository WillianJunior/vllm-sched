from time import time, sleep
import argparse
from datasets import load_dataset
from math import ceil
from tqdm import tqdm
import requests
import concurrent.futures


url_base = "http://127.0.0.1:8000"

def prep_parser():
    parser = argparse.ArgumentParser(
        description="Submitter for sequences of LLM datasets.")

    parser.add_argument(
        '--bs', 
        type=int, 
        help='Batch size of a submission (default=256).', 
        default=256,
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        help='Number of tokens for large payload (default=2048)',
        default=2048,
    )

    parser.add_argument(
        '-b',
        action='store_true',
        dest='is_baseline',
        help='Baseline run, without the long context seq.',
    )



    return parser

# Retry logic to wait until the server is up
def wait_for_server(timeout=600, retry_interval=2):
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


def submit_request(num_seqs, max_tokens=None):
    url_request = url_base + "/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"  # vLLM requires this even if it's not used
    }
    
    seqs =  num_seqs * ["talk"]
    payload_large = {
        "model": "/snfs1/llm-models/llama-3.2-3B-Instruct",
        "prompt": ["talk"],
        "max_tokens": max_tokens,
        "stop": None,
        "temperature": 0.0
    }

    seqs = num_seqs//16 * ["talk"]
    payload_small = {
        "model": "/snfs1/llm-models/llama-3.2-3B-Instruct",
        "prompt": seqs,
        "max_tokens": 16,
        "stop": None,
        "temperature": 0.0
    }

    #print(requests.post(url_request, headers=headers, json=payload_large))

    def send_req(payload):
        return requests.post(url_request, headers=headers, json=payload)

    executor = concurrent.futures.ThreadPoolExecutor()
    if max_tokens is not None:
        future = executor.submit(send_req, payload_large)
    else:
        future = executor.submit(send_req, payload_small)

    return executor, future

def wait_all_requests(futures):
    results = []
    for _executor, future in tqdm(futures, desc="Waiting for returning reqs"):
        result = future.result()
        results.append(result)
    return results

def write_stats(results):
    for r in results:
        assert r.status_code == 200
        #print(r.json()) nothing yet...
        #return

def main():
    parser = prep_parser()
    args = parser.parse_args()

    bs = args.bs
    max_tokens = args.max_tokens
    
    if args.is_baseline:
        num_small_seqs = int(bs * max_tokens)
    else:
        num_small_seqs = int((bs-1) * max_tokens)

    assert wait_for_server(), "Couldn't connect to vLLM server."

    reps = 3
    for r in range(reps):
        futures = []
        t0 = time()
        if not args.is_baseline:
            futures.append(submit_request(1, max_tokens))

        #for i in tqdm(range(max_tokens//4), total=max_tokens//4):
        futures.append(submit_request(num_small_seqs))
        results = wait_all_requests(futures)
        t1 = time()

        print(f'time: {t1-t0:.4f}')

    write_stats(results)


if __name__ == '__main__':
    main()
