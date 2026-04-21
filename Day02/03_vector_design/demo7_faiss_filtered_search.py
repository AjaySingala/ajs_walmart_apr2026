from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
client = OpenAI()

index = None
documents = []

def get_embedding(text):
    """Generate embedding"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def add_document(doc_id, text, metadata):
    """Add document"""
    global index

    embedding = get_embedding(text)

    if index is None:
        index = faiss.IndexFlatL2(len(embedding))

    index.add(np.array([embedding]))

    documents.append({
        "id": doc_id,
        "text": text,
        "metadata": metadata
    })

def filter_metadata(doc, filters):
    """Check if document matches filter"""
    return all(doc["metadata"].get(k) == v for k, v in filters.items())

def search(query, top_k=5, filters=None):
    """Search with optional metadata filtering"""
    query_embedding = get_embedding(query)

    distances, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    results = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]

        # Apply filter AFTER retrieval
        if filters and not filter_metadata(doc, filters):
            continue

        results.append((distances[0][i], doc))

    return results

# ---- Add documents ----
add_document("doc1", "Walmart uses AI for inventory optimization.", {"category": "supply_chain", "region": "US"})
add_document("doc2", "Demand forecasting improves logistics efficiency.", {"category": "supply_chain", "region": "India"})
add_document("doc3", "Personalized recommendations boost sales.", {"category": "marketing", "region": "US"})

# ---- Unfiltered ----
print("\nUnfiltered Search:")
for dist, doc in search("inventory optimization"):
    print(doc["id"], doc["metadata"])

# ---- Filtered ----
print("\nFiltered Search (region=India):")
for dist, doc in search("inventory optimization", filters={"region": "India"}):
    print(doc["id"], doc["metadata"])
