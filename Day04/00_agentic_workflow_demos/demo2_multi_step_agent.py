# Multi-Step Chatbot.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from common_setup import documents as docs

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def analyze_query(query):
    print("\n analyze_query()...")
    return llm.invoke(f"Classify into [leave, travel, other]: {query}").content

def retrieve_context(query):
    print("\n retrieve_context()...")
    docs = retriever.invoke(query)
    return "\n".join([d.page_content for d in docs])

def generate_answer(query, context):
    print("\n generate_answer()...")
    if not context:
        return "I don't know."

    return llm.invoke(f"""
Answer ONLY from context.

Context:
{context}

Question:
{query}
""").content

def workflow(query):
    print("\n workflow()...")
    print("\n[Step 1] Analyze")
    print(analyze_query(query))

    print("\n[Step 2] Retrieve")
    context = retrieve_context(query)
    print(context)

    print("\n[Step 3] Answer")
    return generate_answer(query, context)

if __name__ == "__main__":
    print("=== Multi-Step Chatbot ===\n")

    while True:
        q = input("You: ")
        if q == "exit":
            break

        print("Bot:", workflow(q))

# What is leave policy?
# Travel reimbursement limit?
# How many leave days?
# GDP of France?
