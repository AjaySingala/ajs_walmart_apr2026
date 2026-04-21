# Demo 5: Semantic Search using Cosine Similarity

from openai import OpenAI
from dotenv import load_dotenv
import math

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Step 1: Document Corpus
# -----------------------------
documents = [
    "Walmart uses AI to optimize inventory management.",
    "Customer demand forecasting improves supply chain efficiency.",
    "AI chatbots enhance customer support experience.",
    "Retail analytics helps in pricing optimization."
]

# -----------------------------
# Step 2: Generate Embeddings
# -----------------------------
response = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=documents
)

# Create vector store
vector_store = []

for i, emb in enumerate(response.data):
    vector_store.append({
        "id": i,
        "text": documents[i],
        "embedding": emb.embedding
    })

# -----------------------------
# Step 3: Cosine Similarity Function
# -----------------------------
def cosine_similarity(vec1, vec2):
    # Dot product of two vectors
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Magnitude of vectors
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    # Cosine similarity formula
    return dot_product / (magnitude1 * magnitude2)

# -----------------------------
# Step 4: Query Input
# -----------------------------
query = "How does AI help in supply chain?"
# query = "inventory optimization using AI"
# query = "customer service automation with chatbots"
# query = "pricing strategy using analytics"

query_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
)

query_embedding = query_response.data[0].embedding

# -----------------------------
# Step 5: Compute Similarity
# -----------------------------
results = []

for item in vector_store:
    similarity = cosine_similarity(query_embedding, item["embedding"])
    
    results.append({
        "text": item["text"],
        "similarity": similarity
    })

# -----------------------------
# Step 6: Sort Results
# -----------------------------
results = sorted(results, key=lambda x: x["similarity"], reverse=True)

# -----------------------------
# Step 7: Display Results
# -----------------------------
print("\n🔍 Query:", query)
print("\nTop Matches:\n")

for result in results:
    print(f"Score: {result['similarity']:.4f}")
    print(f"Text: {result['text']}")
    print("-" * 50)

# # Show Top-K Results Only
# top_k = 2
# print("\n" + "=" * 50)
# print("\n🔍 Query:", query)
# print(f"\nTop-{top_k} Matches:\n")

# for result in results[:top_k]:
#     print(f"Score: {result['similarity']:.4f}")
#     print(f"Text: {result['text']}")
#     print("-" * 50)
