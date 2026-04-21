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

# ---- Query ----
results = collection.query(
    query_texts=["How does Walmart optimize inventory?"],
    n_results=2
)

print("\nUnfiltered Results:")
for i in range(len(results["documents"][0])):
    print(f"{results['ids'][0][i]} | {results['documents'][0][i]}")
