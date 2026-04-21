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

# Initialize FAISS index (we'll set dimension after first embedding)
index = None

# Metadata store (parallel to FAISS index)
documents = []

def get_embedding(text):
    """Generate embedding using OpenAI"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return np.array(response.data[0].embedding, dtype="float32")

def add_document(doc_id, text, metadata):
    """Add document to FAISS index + metadata store"""
    global index

    embedding = get_embedding(text)

    # Initialize FAISS index with correct dimension
    if index is None:
        dimension = len(embedding)
        index = faiss.IndexFlatL2(dimension)

    index.add(np.array([embedding]))  # Add vector to FAISS

    documents.append({
        "id": doc_id,
        "text": text,
        "metadata": metadata
    })

# ---- Add documents ----
add_document("doc1", "Walmart uses AI for inventory optimization.", {"category": "supply_chain", "region": "US"})
add_document("doc2", "Demand forecasting improves logistics efficiency.", {"category": "supply_chain", "region": "India"})
add_document("doc3", "Personalized recommendations boost sales.", {"category": "marketing", "region": "US"})

print(f"Total vectors in FAISS index: {index.ntotal}")
