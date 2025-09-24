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
        '--dataset', 
        type=str, 
        help='Dataset to execute. Example: tatsu-lab/alpaca.', 
        required=True,
    )

    parser.add_argument(
        '--interval', 
        type=int, 
        help='Interval in seconds between batch submissions (default=10).', 
        default=10,
    )

    parser.add_argument(
        '--bs', 
        type=int, 
        help='Batch size of a submission (default=250).', 
        default=250,
    )

    parser.add_argument(
        '--max-batches',
        type=int,
        help='Number of batches to execute (default=-1, i.e., all)',
        default=-1,
    )


    return parser

def get_requests(dataset_name):
    requests = []
    if dataset_name == 'tatsu-lab/alpaca':
        ds = load_dataset(dataset_name)
        for d in tqdm(ds['train'], desc="Preping requests from dataset"):
            instruction = d['instruction']
            input_txt = d['input']
            r = """Below is an instruction that describes a task,\
                paired with an input that provides further context.\
                Write a response that appropriately completes the request.\
                \n\n### Instruction:\n"""
            r += instruction + "\n\n"
            if len(input_txt) > 0:
                r += "### Input:\n"
                r += input_txt
            r += "\n\n### Response:\n"
            requests.append(r)
    elif dataset_name == 'Aeala/ShareGPT_Vicuna_unfiltered':
        ds = load_dataset(dataset_name)
        for d in tqdm(ds['train'], desc="Preping requests from dataset"):
            chat = []
            conv = d['conversations']
            for i in range(len(conv) // 2):
                chat.append(conv[i]['value'])
            requests.append(chat)
    elif dataset_name == 'databricks/databricks-dolly-15k':
        ds = load_dataset(dataset_name)
        for d in tqdm(ds['train'], desc="Preping requests from dataset"):
            r = """Below is an instruction that describes a task,\
                paired with an input that provides further context.\
                Write a response that appropriately completes the request.\
                \n\n### Instruction:\n"""
            r += d['instruction'] + "\n\n"
            if len(d['context']) > 0:
                r += "### Context:\n"
                r +=  d['context']
            r += "\n\n### Response:\n"
            requests.append(r)
    elif dataset_name == 'cnn_dailymail':
        ds = load_dataset("cnn_dailymail", "3.0.0")
        for d in tqdm(ds['train'], desc="Preping requests from dataset"):
            r = """Below is an article, which should be summarized.\
                Please write a short text with important highlights of\
                the article below.\
                \n\n### Article:\n"""
            r += d['article'] + "\n"
            requests.append(r)

    assert len(requests) > 0, f"dataset name not recognized"

    return requests

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


def submit_request(prompts_batch):
    url_request = url_base + "/v1/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY"  # vLLM requires this even if it's not used
    }

    payload = {
        "model": "/snfs1/llm-models/llama-3.2-3B-Instruct",
        "prompt": prompts_batch,
        "max_tokens": 2048,
        "temperature": 0.0
    }

    lens = [isinstance(p, list) for p in prompts_batch]
    assert not any(lens), "Chatting not yet supported..."

    def send_req():
        return requests.post(url_request, headers=headers, json=payload)

    executor = concurrent.futures.ThreadPoolExecutor()
    future = executor.submit(send_req)
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
    
    requests = get_requests(args.dataset)
    #requests = requests[:50]
    if args.max_batches > 0:
        num_batches = args.max_batches
    else:
        num_batches = ceil(len(requests) / args.bs)
    futures = []

    assert wait_for_server(), "Couldn't connect to vLLM server."

    # Submit each batch of requests
    for b in tqdm(range(num_batches), total=num_batches, desc="Sending batches"):
        init = b * args.bs
        end = init + args.bs + 1
        batch = requests[init:end]
        futures.append(submit_request(batch))
        if b < num_batches - 1:
            # don't sleep the last interval
            sleep(args.interval)

    results = wait_all_requests(futures)
    write_stats(results)


if __name__ == '__main__':
    main()
