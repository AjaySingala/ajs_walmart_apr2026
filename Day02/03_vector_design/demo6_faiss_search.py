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
    """Add document to FAISS"""
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

def search(query, top_k=2):
    """Search FAISS index"""
    query_embedding = get_embedding(query)

    distances, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    results = []
    for i, idx in enumerate(indices[0]):
        results.append((distances[0][i], documents[idx]))

    return results

# ---- Add documents ----
add_document("doc1", "Walmart uses AI for inventory optimization.", {"category": "supply_chain", "region": "US"})
add_document("doc2", "Demand forecasting improves logistics efficiency.", {"category": "supply_chain", "region": "India"})
add_document("doc3", "Personalized recommendations boost sales.", {"category": "marketing", "region": "US"})

# ---- Query ----
results = search("inventory optimization")

print("\nTop Results (FAISS Unfiltered):")
for dist, doc in results:
    print(f"{doc['id']} | Distance: {dist:.4f} | {doc['text']}")
