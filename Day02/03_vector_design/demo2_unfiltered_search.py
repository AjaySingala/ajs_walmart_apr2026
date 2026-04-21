from dotenv import load_dotenv
from openai import OpenAI
import numpy as np

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
client = OpenAI()

# Reuse same vector store structure
vector_store = []

def get_embedding(text):
    """Generate embedding"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_document(doc_id, text, metadata):
    """Store document with embedding"""
    embedding = get_embedding(text)
    vector_store.append({
        "id": doc_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    })

def search(query, top_k=2):
    """Perform semantic search without filtering"""
    query_embedding = get_embedding(query)

    results = []
    for doc in vector_store:
        score = cosine_similarity(query_embedding, doc["embedding"])
        results.append((score, doc))

    # Sort by similarity score
    results.sort(reverse=True, key=lambda x: x[0])

    return results[:top_k]

# ---- Add same documents ----
add_document("doc1", "Walmart uses AI for inventory optimization.", {"category": "supply_chain", "region": "US"})
add_document("doc2", "Customer demand forecasting improves logistics efficiency.", {"category": "supply_chain", "region": "India"})
add_document("doc3", "Personalized recommendations increase e-commerce sales.", {"category": "marketing", "region": "US"})

# ---- Query ----
query = "How does Walmart improve inventory?"

results = search(query)

print("\nTop Results (Unfiltered):")
for score, doc in results:
    print(f"{doc['id']} | Score: {score:.4f} | {doc['text']}")
