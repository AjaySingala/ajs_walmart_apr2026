from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
# Create LLM using model from env
def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0  # deterministic output for demos
    )

# Create embedding model from env
def get_embeddings():
    return OpenAIEmbeddings(
        model=os.getenv("TEXT_EMBEDDING_MODEL")
    )
