# Demo 4: Convert user query into embedding

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

# User query
query = "How does AI help in supply chain?"

# Generate query embedding
response = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=query
)

query_embedding = response.data[0].embedding

print("Query:", query)
print("Embedding length:", len(query_embedding))
print("First 10 values:", query_embedding[:10])
