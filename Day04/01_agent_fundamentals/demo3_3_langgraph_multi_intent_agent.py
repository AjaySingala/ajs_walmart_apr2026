# No hard-coding to call tool.
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from langgraph.graph import StateGraph
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# TOOL (RAG)
# -----------------------------
@tool
def policy_tool(query: str) -> str:
    """Answer company policy questions using internal documents"""
    print("\n Tool: policy_tool()...")

    docs = retriever.invoke(query)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using ONLY the context below.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content.strip()


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    answers: List[str]
    final_answer: str


# -----------------------------
# Node 1: Decompose Query
# -----------------------------
def decompose(state: AgentState):
    print("\n Node: decompose()...")
    prompt = f"""
Break the question into smaller independent questions.

Question:
{state['query']}

Return as a list.
"""

    response = llm.invoke(prompt).content

    # Simple parsing (safe fallback)
    sub_queries = [q.strip("- ").strip() for q in response.split("\n") if q.strip()]

    return {"sub_queries": sub_queries, "answers": []}


# -----------------------------
# Node 2: Answer Each Sub-query
# -----------------------------
def answer_subqueries(state: AgentState):
    print("\n Node: answer_subqueries()...")
    answers = []

    for q in state["sub_queries"]:
        ans = policy_tool.invoke(q)
        answers.append(f"{q} → {ans}")

    return {"answers": answers}


# -----------------------------
# Node 3: Combine Answers
# -----------------------------
def combine(state: AgentState):
    print("\n Node: combine()...")
    combined_text = "\n".join(state["answers"])

    prompt = f"""
Combine the following answers into a clean final response.

{combined_text}
"""

    final = llm.invoke(prompt).content

    return {"final_answer": final}


# -----------------------------
# Build Graph
# -----------------------------
print("\n Build Graph...")
graph = StateGraph(AgentState)

graph.add_node("decompose", decompose)
graph.add_node("answer_subqueries", answer_subqueries)
graph.add_node("combine", combine)

graph.set_entry_point("decompose")
graph.add_edge("decompose", "answer_subqueries")
graph.add_edge("answer_subqueries", "combine")

app = graph.compile()


# -----------------------------
# Chatbot
# -----------------------------
if __name__ == "__main__":
    print("\n=== MULTI-INTENT AGENT (LANGGRAPH) ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        result = app.invoke({"query": user_input})

        print(f"Bot: {result['final_answer']}\n")

# Queries to try:
# What is the leave policy?
# How many leave days do employees get?
# What is the travel reimbursement policy?
# What is the leave carry forward policy?
# What is the GDP of France?
# Say hello

# Show limitation (multi-intent) (Still does not work):
# What is the leave policy and what expenses can I claim?
# What is the leave policy and what expenses can I reimburse?
