from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, folder_path)

import config

# Start.
# Create embedding function using OpenAI
embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=os.getenv("TEXT_EMBEDDING_MODEL")
)

# # Initialize Chroma client (persists to disk)
# client = chromadb.Client()
# Use PersistentClient (this is the key change)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
collection = client.get_or_create_collection(
    name="walmart_knowledge",
    embedding_function=embedding_function
)

# ---- Add documents with metadata ----
collection.add(
    documents=[
        "Walmart uses AI for inventory optimization.",
        "Demand forecasting improves logistics efficiency in India.",
        "Personalized recommendations increase e-commerce sales."
    ],
    metadatas=[
        {"category": "supply_chain", "region": "US"},
        {"category": "supply_chain", "region": "India"},
        {"category": "marketing", "region": "US"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

print("Documents indexed successfully with metadata!")
