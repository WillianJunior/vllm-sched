import time
import requests

def wait_for_vllm(url="http://localhost:8000/health", interval=1):
    print(f"Waiting for vLLM server at {url} ...")
    while True:
        try:
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print("vLLM server is up!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(interval)

if __name__ == "__main__":
    wait_for_vllm()
