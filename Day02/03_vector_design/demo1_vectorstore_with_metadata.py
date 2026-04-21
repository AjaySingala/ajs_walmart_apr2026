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

# Initialize OpenAI client
client = OpenAI()

# In-memory vector store (simple list for demo)
vector_store = []

def get_embedding(text):
    """Generate embedding for given text"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding

def add_document(doc_id, text, metadata):
    """Add document with embedding and metadata to vector store"""
    embedding = get_embedding(text)

    vector_store.append({
        "id": doc_id,
        "text": text,
        "embedding": embedding,
        "metadata": metadata
    })

# ---- Sample Documents ----
add_document(
    "doc1",
    "Walmart uses AI for inventory optimization.",
    {"category": "supply_chain", "region": "US"}
)

add_document(
    "doc2",
    "Customer demand forecasting improves logistics efficiency.",
    {"category": "supply_chain", "region": "India"}
)

add_document(
    "doc3",
    "Personalized recommendations increase e-commerce sales.",
    {"category": "marketing", "region": "US"}
)

# Print stored documents
for doc in vector_store:
    print(f"ID: {doc['id']}, Metadata: {doc['metadata']}")
