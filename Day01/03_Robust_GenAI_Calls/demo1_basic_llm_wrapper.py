# demo1_basic_llm_wrapper.py

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
# Initialize OpenAI client
client = OpenAI()

MODEL_NAME = os.getenv("MODEL_NAME")


def call_llm(prompt: str) -> str:
    """
    Basic LLM call wrapper
    Takes a prompt and returns the model response
    """
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    # Extract and return text output
    return response.choices[0].message.content


if __name__ == "__main__":
    prompt = "Explain supply chain optimization in 2 lines."
    output = call_llm(prompt)
    print("LLM Response:\n", output)
