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

vector_store = []

def get_embedding(text):
    """Generate embedding"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    """Compute similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_document(doc_id, text, metadata):
    """Store document"""
    embedding = get_embedding(text)
    vector_store.append({
        "id": doc_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    })

def filter_docs(filter_criteria):
    """Filter documents based on metadata"""
    filtered = []

    for doc in vector_store:
        match = True
        for key, value in filter_criteria.items():
            if doc["metadata"].get(key) != value:
                match = False
                break
        if match:
            filtered.append(doc)

    return filtered

def search(query, top_k=2, filters=None):
    """Search with optional metadata filters"""
    query_embedding = get_embedding(query)

    docs = vector_store
    if filters:
        docs = filter_docs(filters)

    results = []
    for doc in docs:
        score = cosine_similarity(query_embedding, doc["embedding"])
        results.append((score, doc))

    results.sort(reverse=True, key=lambda x: x[0])

    return results[:top_k]

# ---- Add documents ----
add_document("doc1", "Walmart uses AI for inventory optimization.", {"category": "supply_chain", "region": "US"})
add_document("doc2", "Customer demand forecasting improves logistics efficiency.", {"category": "supply_chain", "region": "India"})
add_document("doc3", "Personalized recommendations increase e-commerce sales.", {"category": "marketing", "region": "US"})

# ---- Query without filter ----
print("\nUnfiltered Search:")
results = search("inventory optimization")
for score, doc in results:
    print(f"{doc['id']} | {score:.4f} | {doc['metadata']}")

# ---- Query with filter ----
print("\nFiltered Search (region=India):")
results = search("inventory optimization", filters={"region": "India"})
for score, doc in results:
    print(f"{doc['id']} | {score:.4f} | {doc['metadata']}")
