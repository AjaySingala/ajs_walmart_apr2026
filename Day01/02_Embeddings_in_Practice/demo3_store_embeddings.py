# Demo 3: Store embeddings in a simple in-memory structure

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

# Corpus
documents = [
    "Walmart uses AI to optimize inventory management.",
    "Customer demand forecasting improves supply chain efficiency.",
    "AI chatbots enhance customer support experience.",
    "Retail analytics helps in pricing optimization."
]

# Generate embeddings
response = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=documents
)

# Simple vector store (list of dicts)
vector_store = []

for i, emb in enumerate(response.data):
    vector_store.append({
        "id": i,
        "text": documents[i],
        "embedding": emb.embedding
    })

# Show stored structure
print("Vector Store Created:\n")

for item in vector_store:
    print(f"ID: {item['id']}")
    print(f"Text: {item['text']}")
    print(f"Embedding sample: {item['embedding'][:5]}")
    print("-" * 50)
