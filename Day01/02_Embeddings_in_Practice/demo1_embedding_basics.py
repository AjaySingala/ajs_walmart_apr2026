# Demo 1: Generate embedding for a single sentence

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

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample input text
text = "Walmart improves supply chain efficiency using AI."

# Generate embedding
response = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=text
)

# Extract embedding vector
embedding_vector = response.data[0].embedding

print("Text:", text)
print("Embedding vector length:", len(embedding_vector))
print("First 10 values:", embedding_vector[:10])
