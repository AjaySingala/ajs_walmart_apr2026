# Convert RAG → Simple Agent.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate

from langchain.agents import create_agent

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
# -------------------------
# STEP 1: Documents
# -------------------------
from common_setup import documents as docs

# -------------------------
# STEP 2: Vector Store
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -------------------------
# STEP 3: Tool
# -------------------------
@tool
def rag_search(query: str) -> str:
    """Search company policies"""
    print(f"\n Tool: rag_search()...")
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n".join([d.page_content for d in docs])

# -------------------------
# STEP 4: LLM
# -------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# STEP 5: CREATE AGENT
# -------------------------
agent = create_agent(
    model=llm,
    tools=[rag_search],
    system_prompt="""
You are a company policy assistant.

Rules:
- Use tools when needed
- Answer ONLY from company policy
- If not found, say "I don't know"
"""
)

# -------------------------
# CHATBOT LOOP
# -------------------------
if __name__ == "__main__":
    print("=== RAG Agent Chatbot (LangChain 1.2) ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        response = agent.invoke({"messages": [{"role": "user", "content": query}]})

        print("Bot:", response["messages"][-1].content)
        
# What is leave policy?
# Travel reimbursement limit?
# How many leave days?
# GDP of France?
