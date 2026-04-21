# Hard-code logic to call tool.
from dotenv import load_dotenv
from typing import TypedDict, Literal

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

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# TOOL (RAG)
# -----------------------------
@tool
def policy_tool(query: str) -> str:
    """Use this tool to answer company policy questions"""
    print("\n Tool: policy_tool()...")

    docs = retriever.invoke(query)

    # print("\n--- Tool Retrieved Docs ---")
    # for d in docs:
    #     print("-", d.page_content)

    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer using ONLY this context.

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content


# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    query: str
    decision: Literal["tool", "direct"]
    answer: str


# -----------------------------
# Node 1: Decide (LLM decides tool usage)
# -----------------------------
def decide(state: AgentState):
    print("\n Node: decide()...")
    prompt = f"""
Decide how to answer the question.

If it is about company policy → say TOOL
Otherwise → say DIRECT

Question: {state['query']}
"""

    response = llm.invoke(prompt).content.lower()

    decision = "tool" if "tool" in response else "direct"

    return {"decision": decision}


# -----------------------------
# Node 2: Tool Node
# -----------------------------
def use_tool(state: AgentState):
    print("\n Node: use_tool()...")
    result = policy_tool.invoke(state["query"])
    return {"answer": result}


# -----------------------------
# Node 3: Direct Answer
# -----------------------------
def direct_answer(state: AgentState):
    print("\n Node: direct_answer()...")
    response = llm.invoke(state["query"])
    return {"answer": response.content}


# -----------------------------
# Routing
# -----------------------------
def route(state: AgentState):
    print("\n route()...")
    return "use_tool" if state["decision"] == "tool" else "direct_answer"


# -----------------------------
# Build Graph
# -----------------------------
print("\n Build Graph...")
graph = StateGraph(AgentState)

graph.add_node("decide", decide)
graph.add_node("use_tool", use_tool)
graph.add_node("direct_answer", direct_answer)

graph.set_entry_point("decide")

graph.add_conditional_edges("decide", route)

app = graph.compile()


# -----------------------------
# Chatbot
# -----------------------------
if __name__ == "__main__":
    print("\n=== LANGGRAPH AGENT (WITH TOOL) ===")
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
# Say hello

# Show limitation (multi-intent) (Still does not work):
# What is the leave policy and what expenses can I claim?
