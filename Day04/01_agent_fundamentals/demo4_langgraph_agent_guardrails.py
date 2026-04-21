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


# TOOL
@tool
def policy_tool(query: str) -> str:
    """Answer policy questions only"""
    print("\n Tool: policy_tool()...")

    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Answer ONLY using context.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content


# STATE
class AgentState(TypedDict):
    query: str
    decision: Literal["tool", "reject"]
    answer: str


# CLASSIFIER (STRICT)
def classify(state: AgentState):
    print("\n classify()...")
    prompt = f"""
Is this strictly a company policy question?

Answer YES or NO.

Question: {state['query']}
"""

    response = llm.invoke(prompt).content.lower()

    decision = "tool" if "yes" in response else "reject"

    return {"decision": decision}


# TOOL NODE
def use_tool(state: AgentState):
    print("\n Node: use_tool()...")
    return {"answer": policy_tool.invoke(state["query"])}


# REJECT NODE
def reject(state: AgentState):
    print("\n Node: reject()...")
    return {"answer": "I don't know"}


# ROUTING
def route(state: AgentState):
    print("\n route()...")
    return "use_tool" if state["decision"] == "tool" else "reject"


# GRAPH
print("\n build Graph...")
graph = StateGraph(AgentState)

graph.add_node("classify", classify)
graph.add_node("use_tool", use_tool)
graph.add_node("reject", reject)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route)

app = graph.compile()


# CHATBOT
if __name__ == "__main__":
    print("\n=== LANGGRAPH GUARDED AGENT (WITH TOOL) ===")
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
