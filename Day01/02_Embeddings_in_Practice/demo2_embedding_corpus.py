# Demo 2: Generate embeddings for a small document corpus

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

# Small document corpus (can relate to Walmart use cases)
documents = [
    "Walmart uses AI to optimize inventory management.",
    "Customer demand forecasting improves supply chain efficiency.",
    "AI chatbots enhance customer support experience.",
    "Retail analytics helps in pricing optimization."
]

# Generate embeddings for all documents
response = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=documents  # Passing list of texts
)

# Store embeddings with corresponding text
corpus_embeddings = []

for i, emb in enumerate(response.data):
    corpus_embeddings.append({
        "text": documents[i],
        "embedding": emb.embedding
    })

# Print summary
print("Generated embeddings for corpus:\n")

for item in corpus_embeddings:
    print(f"Text: {item['text']}")
    print(f"Vector length: {len(item['embedding'])}")
    print("-" * 50)
