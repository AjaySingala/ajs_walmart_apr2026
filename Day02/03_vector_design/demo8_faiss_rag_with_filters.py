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
    """Apply metadata filter"""
    return all(doc["metadata"].get(k) == v for k, v in filters.items())

def search(query, top_k=5, filters=None):
    """Retrieve relevant docs"""
    query_embedding = get_embedding(query)

    distances, indices = index.search(
        np.array([query_embedding]),
        top_k
    )

    results = []
    for i, idx in enumerate(indices[0]):
        doc = documents[idx]

        if filters and not filter_metadata(doc, filters):
            continue

        results.append(doc)

    return results[:2]

def generate_answer(query, docs):
    """Generate answer using LLM"""
    context = "\n".join([doc["text"] for doc in docs])

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
add_document("doc3", "Personalization drives e-commerce growth.", {"category": "marketing", "region": "US"})

query = "How is AI used in supply chain?"

# Without filter
docs = search(query)
print("\nAnswer (Unfiltered):")
print(generate_answer(query, docs))

# With filter
docs_filtered = search(query, filters={"region": "India"})
print("\nAnswer (Filtered - India):")
print(generate_answer(query, docs_filtered))
