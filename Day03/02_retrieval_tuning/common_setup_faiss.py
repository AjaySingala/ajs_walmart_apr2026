from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Set env vars from config.py.
import sys
import os

# Add the folder path
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config


# Initialize embedding model
embeddings = OpenAIEmbeddings(
    model=os.getenv("TEXT_EMBEDDING_MODEL")
)

# Initialize LLM
llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0
)

# Sample documents
docs = [
    Document(page_content="Employees are entitled to 20 days of paid leave annually.", metadata={"source": "LeavePolicy"}),
    Document(page_content="Travel reimbursement is allowed only for approved business trips.", metadata={"source": "TravelPolicy"}),
    Document(page_content="Hotel bookings should not exceed $150 per night.", metadata={"source": "TravelPolicy"}),
    Document(page_content="Casual leave cannot be carried forward to next year.", metadata={"source": "LeavePolicy"}),
]

# Initialize FAISS vectorstore (empty initially)
vectorstore = None


def load_data():
    global vectorstore

    if vectorstore is None:
        # Create FAISS index from documents
        vectorstore = FAISS.from_documents(docs, embeddings)
        print("Documents indexed into FAISS.")
