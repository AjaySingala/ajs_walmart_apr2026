# Full RAG (Grounded Answer).
# Step 3: Full RAG → retrieve + augment + generate grounded answer

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path).
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
from openai import OpenAI

# Initialize OpenAI client using API key
print(f"Initialize client...")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load document.
print(f"\n Load document...")
with open("company_policy.txt", "r", encoding="utf-8") as f:
    document = f.read()

# Chunking.
print(f"\n Chunking...")
chunks = document.split("\n")

# Create embeddings.
print(f"\n Create embeddings...")
embeddings = []
for chunk in chunks:
    if chunk.strip():
        emb = client.embeddings.create(
            model=os.getenv("TEXT_EMBEDDING_MODEL"),
            input=chunk
        )
        embeddings.append((chunk, emb.data[0].embedding))

# Similarity function.
def similarity(a, b):
    print(f"\n similarity()...")
    return sum(x*y for x, y in zip(a, b))

# Query
query = "What is the meal allowance for employees?"
# query = "What is the internet reimbursement policy?"

# Embed query.
print(f"\n Embed query...")
query_embedding = client.embeddings.create(
    model=os.getenv("TEXT_EMBEDDING_MODEL"),
    input=query
).data[0].embedding

# Retrieve top relevant chunk.
print(f"\n Retrieve top relevant chunk...")
best_chunk = max(
    embeddings,
    key=lambda x: similarity(query_embedding, x[1])
)[0]

# Augment prompt with retrieved context.
augmented_prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{best_chunk}

Question:
{query}
"""

# Call LLM with grounded context.
print(f"\n Call LLM with grounded context...")
response = client.chat.completions.create(
    model=os.getenv("MODEL_NAME"),
    messages=[
        {"role": "system", "content": "You are a precise assistant."},
        {"role": "user", "content": augmented_prompt}
    ]
)

print("\n--- RAG Response (Grounded) ---")
print(response.choices[0].message.content)

# # A very strict prompt:
# augmented_prompt = f"""
# You are answering a policy question.

# STRICT RULES:
# - Only use the provided context
# - Do NOT guess
# - If unsure, say "I don't know"

# Context:
# {best_chunk}

# Question:
# {query}
# """
