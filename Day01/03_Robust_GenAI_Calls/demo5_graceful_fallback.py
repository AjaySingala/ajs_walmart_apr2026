# demo5_graceful_fallback.py

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
    Core LLM call function
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def call_llm_robust(prompt: str, retries: int = 3, timeout: int = 5) -> str:
    """
    Fully robust LLM wrapper:
    - Retry logic
    - Timeout handling
    - Graceful fallback
    """
    for attempt in range(retries):
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(llm_call, prompt)
                return future.result(timeout=timeout)

        except concurrent.futures.TimeoutError:
            print(f"Attempt {attempt + 1}: Timeout")

        except Exception as e:
            print(f"Attempt {attempt + 1}: Error - {e}")

        time.sleep(2 ** attempt)

    # Graceful business-friendly fallback response
    return (
        "We're currently experiencing high demand. "
        "Please try again in a few moments."
    )


if __name__ == "__main__":
    prompt = "Explain supply chain resilience in retail."
    output = call_llm_robust(prompt)
    print("LLM Response:\n", output)
