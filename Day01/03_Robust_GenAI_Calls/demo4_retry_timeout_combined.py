# demo4_retry_timeout_combined.py

import time
import concurrent.futures
from dotenv import load_dotenv
from openai import OpenAI

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
client = OpenAI()
MODEL_NAME = os.getenv("MODEL_NAME")


def llm_call(prompt: str) -> str:
    """
    Core LLM call reused across retries and timeout handling
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def call_llm_safe(prompt: str, retries: int = 3, timeout: int = 5) -> str:
    """
    Robust LLM wrapper with retry + timeout
    """
    for attempt in range(retries):
        print(f"\nAttempt {attempt + 1} of {retries} with timeout {timeout} seconds...")
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(llm_call, prompt)
                return future.result(timeout=timeout)

        except concurrent.futures.TimeoutError:
            print(f"Attempt {attempt + 1}: Timeout occurred")
            timeout += (timeout * (attempt+1))

        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")

        # Backoff before retry
        time.sleep(2 ** attempt)

    return "Error: Failed after retries and timeouts."


if __name__ == "__main__":
    prompt = "How does AI improve inventory management at Walmart?"
    output = call_llm_safe(prompt)
    print("LLM Response:\n", output)
