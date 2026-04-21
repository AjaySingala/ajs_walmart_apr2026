# demo2_retry_logic.py

import time
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

def call_llm_with_retry(prompt: str, retries: int = 3, delay: int = 2) -> str:
    """
    LLM call with retry logic
    Retries on failure with exponential backoff
    """
    for attempt in range(retries):
        print(f"\nAttempt {attempt + 1} of {retries}...")
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")

            # Exponential backoff before retrying
            time.sleep(delay * (2 ** attempt))

    # If all retries fail
    return "Error: Unable to process request after multiple attempts."


if __name__ == "__main__":
    prompt = "Summarize Walmart's use of AI in retail."
    output = call_llm_with_retry(prompt)
    print("LLM Response:\n", output)
