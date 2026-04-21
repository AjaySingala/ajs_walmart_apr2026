# demo3_context_injection.py

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

# Function with system, user, and dynamic context
def ask_llm(system_prompt: str, user_prompt: str, context: str) -> str:
    
    # Combine context with user prompt dynamically
    full_prompt = f"""
    Context:
    {context}

    Question:
    {user_prompt}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    system = "You are a retail analytics expert."
    context = "Walmart sales increased by 20% in Q2 due to seasonal demand."
    user = "Why did sales increase?"

    print(ask_llm(system, user, context))
