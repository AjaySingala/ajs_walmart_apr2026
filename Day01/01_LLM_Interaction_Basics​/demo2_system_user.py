# demo2_system_user.py

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
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function with system + user separation
def ask_llm(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},  # Sets behavior/persona
            {"role": "user", "content": user_prompt}       # Actual user query
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    system = "You are a helpful AI trainer explaining concepts simply."
    user = "Explain Large Language Models"

    print(ask_llm(system, user))
