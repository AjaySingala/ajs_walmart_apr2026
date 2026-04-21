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

def generate_text(prompt: str) -> str:
    response = client.responses.create(
        model="gpt-4o-mini",
        input=prompt,
        max_output_tokens=200
    )
    return response.output_text


# Demo
prompt = "Write a short explanation of how AI helps retail supply chains."
output = generate_text(prompt)

print("Generated Text:\n", output)
