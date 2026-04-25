from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, folder_path)

import config

# Start.
import demo1_chromadb_index

llm_client = OpenAI()

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("TEXT_EMBEDDING_MODEL")
)

# # Initialize Chroma client (persists to disk)
# client = chromadb.Client()
# Use PersistentClient (this is the key change)
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="walmart_knowledge",
    embedding_function=embedding_function
)

def retrieve_docs(query, filters=None):
    """Retrieve documents along with distances"""
    results = collection.query(
        query_texts=[query],
        n_results=3,
        where=filters,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    return docs, distances

def is_relevant(docs, distances, threshold=0.8):
    """
    Strict relevance check
    Only allow answer if similarity is strong enough
    """
    if not docs:
        return False

    return min(distances) < threshold

def generate_strict_answer(query, docs):
    """Generate answer ONLY from context"""
    print(f"\n query: {query}")
    context = "\n".join(docs)

    prompt = f"""
    You are a strict RAG system.

    Rules:
    - Answer ONLY using the provided context
    - Do NOT use prior knowledge
    - If the answer is not explicitly present, say:
      "I don't have enough information in the provided context."

    Context:
    {context}

    Question:
    {query}
    """

    response = llm_client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# ---- Query ----
query = "How is AI used in supply chain?"
# query = "What is quantum computing?"  # intentionally outside domain
# query = "What is the GDP of France?"

docs, distances = retrieve_docs(query)

print("\nRetrieved Docs:", docs)
print("Distances:", distances)

# ---- Strict RAG Decision ----
if is_relevant(docs, distances):
    print("\nStrict RAG Answer:")
    print(generate_strict_answer(query, docs))
else:
    print("\nStrict RAG Response:")
    print("I don't have enough information in the provided context.")
