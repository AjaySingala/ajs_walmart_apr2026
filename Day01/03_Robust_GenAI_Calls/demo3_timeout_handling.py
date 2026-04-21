# demo3_timeout_handling.py

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
    Core LLM call function (used inside timeout wrapper)
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


def call_llm_with_timeout(prompt: str, timeout: int = 5) -> str:
    """
    LLM call with timeout protection
    Cancels execution if it exceeds timeout
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(llm_call, prompt)

        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return "Error: Request timed out."


if __name__ == "__main__":
    prompt = "Explain demand forecasting in retail."
    output = call_llm_with_timeout(prompt)
    print("LLM Response:\n", output)
