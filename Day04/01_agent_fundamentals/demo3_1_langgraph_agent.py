# DEMO 3: LangGraph Agent.
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langgraph.graph import StateGraph

from typing import TypedDict

from common_setup import documents

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    temperature=0
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("TEXT_EMBEDDING_MODEL")
)

# Create vector store
vectorstore = Chroma.from_texts(
    documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# State Definition
# -----------------------------
class AgentState(TypedDict):
    query: str
    context: str
    answer: str


# -----------------------------
# Node 1: Retrieve
# -----------------------------
def retrieve(state: AgentState):
    print("\n Node: retrieve()...")
    query = state["query"]

    docs = retriever.invoke(query)

    # print("\n--- Retrieved Docs ---")
    # for d in docs:
    #     print("-", d.page_content)

    context = "\n".join([d.page_content for d in docs])

    return {"context": context}


# -----------------------------
# Node 2: Generate Answer
# -----------------------------
def generate(state: AgentState):
    print("\n Node: generate()...")
    prompt = f"""
Answer the question using ONLY the context below.

Context:
{state['context']}

Question:
{state['query']}
"""

    response = llm.invoke(prompt)

    return {"answer": response.content}


# -----------------------------
# Build Graph
# -----------------------------
print("\n Build Graph...")

graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")

app = graph.compile()


# -----------------------------
# Chatbot Loop
# -----------------------------
if __name__ == "__main__":
    print("\n=== LANGGRAPH AGENT CHATBOT ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        result = app.invoke({"query": user_input})

        print(f"Bot: {result['answer']}\n")

# Queries to try:
# What is the leave policy?
# How many leave days do employees get?
# What is the travel reimbursement policy?
# What is the leave carry forward policy?
# What is the GDP of France?

# Show limitation (multi-intent) (Still does not work):
# What is the leave policy and what expenses can I claim?
