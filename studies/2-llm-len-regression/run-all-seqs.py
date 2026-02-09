import asyncio
import aiohttp
from time import time, sleep
import requests


# Local vLLM server endpoint
VLLM_URL = "http://localhost:8000/v1/completions"   # or /v1/chat/completions
MODEL_NAME = "/snfs1/llm-models/llama-3.2-3B-Instruct"
REQ_FILE = "prompts.txt"
MAX_TOKENS = 80000
CONCURRENCY = 200   # number of concurrent requests (tune based on hardware)
SEED=0

url_base = "http://localhost:8000"

# Retry logic to wait until the server is up
def wait_for_server(timeout=600, retry_interval=1):
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


async def send_request(req_id, prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.0,
        "seed": SEED,
        "max_tokens": MAX_TOKENS
    }

    #print(prompt)

    try:
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(force_close=True, limit=0)) as session:
            async with session.post(VLLM_URL, json=payload, timeout=90000000) as resp:
                data = await resp.json()
                output_text = data["choices"][0]["text"]
                n_output_tokens = data.get("usage", {}).get("completion_tokens")
                usage = data.get("usage", {}).get("prompt_tokens")
                #print(data)
                print(f"[Request {req_id}] Prompt_len: {usage} Output length: {n_output_tokens} tokens")
                if n_output_tokens < 5:
                    print(f"============== bad: {prompt} XXXXXXXXXXXXXXX {data}")
    except Exception as e:
        print(f"[Request {req_id}] Failed: {e}")

async def main():
    with open(REQ_FILE, "r", encoding="utf-8") as f:
        requests_list = [line.strip() for line in f if line.strip()]

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def sem_task(req_id, prompt):
        async with semaphore:
            await send_request(req_id, prompt)

    tasks = [asyncio.create_task(sem_task(i + 1, p)) for i, p in enumerate(requests_list)]

    # Process tasks as they finish
    for finished_task in asyncio.as_completed(tasks):
        await finished_task

    #connector = aiohttp.TCPConnector(limit=CONCURRENCY)
    #async with aiohttp.ClientSession(connector=connector) as session:
    #    tasks = [
    #        send_request(session, i, prompt)
    #        for i, prompt in enumerate(requests_list[:])
    #    ]
    #    await asyncio.gather(*tasks)

if __name__ == "__main__":
    wait_for_server()
    asyncio.run(main())

