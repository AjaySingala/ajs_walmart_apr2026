from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.

# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model=os.getenv("TEXT_EMBEDDING_MODEL")
)

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0
)

# Create or load persistent Chroma DB
vectorstore = Chroma(
    collection_name="policy_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Sample documents (you can replace with PDFs later)
docs = [
    Document(page_content="Employees are entitled to 20 days of paid leave annually.", metadata={"source": "LeavePolicy"}),
    Document(page_content="Travel reimbursement is allowed only for approved business trips.", metadata={"source": "TravelPolicy"}),
    Document(page_content="Hotel bookings should not exceed $150 per night.", metadata={"source": "TravelPolicy"}),
    Document(page_content="Casual leave cannot be carried forward to next year.", metadata={"source": "LeavePolicy"}),
]

def load_data():
    # Load only once (avoid duplicates)
    if len(vectorstore.get()["documents"]) == 0:
        vectorstore.add_documents(docs)
        print("Documents indexed.")
