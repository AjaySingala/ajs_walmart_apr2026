# demo1_basic_call.py

from openai import OpenAI
from dotenv import load_dotenv

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.

# Initialize OpenAI client using API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Simple function to call LLM with a single user prompt
def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# Run example
if __name__ == "__main__":
    answer = ask_llm("Explain GenAI in 2 lines")
    print("LLM Response:\n", answer)
