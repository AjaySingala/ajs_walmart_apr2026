# Add Embeddings + Search.
# Step 2: Convert document into embeddings and perform similarity search

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
from openai import OpenAI

# Initialize OpenAI client using API key
print(f"Initialize client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load document.
print(f"Load document...")
with open("company_policy.txt", "r", encoding="utf-8") as f:
    document = f.read()

# Split into simple chunks (basic chunking for demo)
print(f"Chunking...")
chunks = document.split("\n")

# Create embeddings for each chunk.
print(f"Embedding...")
embeddings = []
for chunk in chunks:
    if chunk.strip():  # skip empty lines
        emb = client.embeddings.create(
            model=os.getenv("TEXT_EMBEDDING_MODEL"),
            input=chunk
        )
        embeddings.append((chunk, emb.data[0].embedding))

# Function to calculate similarity (dot product)
def similarity(a, b):
    print(f"Calculate similarity...")
    return sum(x*y for x, y in zip(a, b))

# User query.
# query = "What is the meal allowance?"
query = "When should travel claims be submitted?"

# Embed query.
print(f"Embed query...")
query_embedding = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=query
).data[0].embedding

# Find most relevant chunk.
print(f"Find most relevant chunk...")
best_chunk = max(
    embeddings,
    key=lambda x: similarity(query_embedding, x[1])
)

print(f"\n Query: {query}")
print("\n--- Most Relevant Chunk ---")
print(best_chunk[0])
