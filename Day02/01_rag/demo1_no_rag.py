# Simple LLM call WITHOUT any context → prone to hallucination

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
from openai import OpenAI

# Initialize OpenAI client using API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Ask a question NOT present in any provided context
question = "What is the meal allowance for employees at Walmart?"

# Direct LLM call (no grounding)
response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question}
    ]
)

print("\n--- LLM Response (No RAG) ---")
print(f"Question: {question} \n")
print(response.choices[0].message.content)
