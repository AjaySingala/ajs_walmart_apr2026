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

def filter_docs(filters):
    """Filter based on metadata"""
    return [
        doc for doc in vector_store
        if all(doc["metadata"].get(k) == v for k, v in filters.items())
    ]

def search(query, top_k=2, filters=None):
    """Retrieve relevant docs"""
    query_embedding = get_embedding(query)

    docs = filter_docs(filters) if filters else vector_store

    scored = []
    for doc in docs:
        score = cosine_similarity(query_embedding, doc["embedding"])
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return scored[:top_k]

def generate_answer(query, retrieved_docs):
    """Use LLM to generate final response"""
    context = "\n".join([doc["text"] for _, doc in retrieved_docs])

    prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---- Add documents ----
add_document("doc1", "Walmart uses AI for inventory optimization.", {"category": "supply_chain", "region": "US"})
add_document("doc2", "Demand forecasting improves supply chain efficiency in India.", {"category": "supply_chain", "region": "India"})
add_document("doc3", "Personalization drives online sales growth.", {"category": "marketing", "region": "US"})

# ---- Query ----
query = "How is AI used in supply chain?"

# Without filter
docs = search(query)
print("\nAnswer (Unfiltered):")
print(generate_answer(query, docs))

# With filter
docs_filtered = search(query, filters={"region": "India"})
print("\nAnswer (Filtered - India):")
print(generate_answer(query, docs_filtered))
