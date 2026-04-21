# Demo: Tool Failure & Fallback Handling using LangGraph

import re
from typing import TypedDict, Annotated, Sequence

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

# LangGraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
# -------------------------
# STEP 1: Dataset
# -------------------------
from common_setup import documents as docs

# -------------------------
# STEP 2: Vector Store
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# -------------------------
# STEP 3: TOOLS
# -------------------------
@tool
def rag_search(query: str) -> str:
    """Search company policy with relevance filtering"""
    
    results = vectorstore.similarity_search_with_score(query, k=2)

    # for doc, score in results:
    #     print(f"content: {doc.page_content} (Score: {score})")

    if not results:
        return "NO_CONTEXT"

    # 🔥 KEY LOGIC: threshold
    threshold = 0.5  # adjust for demo
    # threshold = 1.2  # adjust for demo
    # NOTE:
    # FAISS returns distance, not similarity.
    # Lower score = More similar.
    # Higher score = Less similar.
    # These are DISTANCES, not similarities.
    # So, either increase "distance threshold" to 1.2 or more,
    # OR Convert distance → similarity and then filter:
    #   threshold = 1 / (1 + score) # Similarity.

    filtered = []
    for doc, score in results:
        if score < threshold:   # lower score = more relevant (FAISS)
            filtered.append(doc.page_content)

    if not filtered:
        return "NO_CONTEXT"

    return "\n".join(filtered)

@tool
def fallback_llm(query: str) -> str:
    """General knowledge fallback"""
    print(f"\n Tool: fallback_llm()...")
    return llm.invoke(query).content

@tool
def calculate_reimbursement(text: str) -> str:
    """Calculate reimbursement based on amount and days"""
    print(f"\n Tool: calculate_reimbursement()...")
    try:
        # Extract numbers
        amounts = re.findall(r"\d+", text)
        if len(amounts) < 2:
            return "Could not parse input"

        amount_per_day = int(amounts[0])
        days = int(amounts[1])

        total = amount_per_day * days

        return f"Total reimbursement = ${total}"

    except Exception as e:
        print("Calculation error")
        return "Calculation error"

tools = [rag_search, fallback_llm, calculate_reimbursement]

# -------------------------
# STEP 4: STATE
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int

MAX_RETRIES = 2

# -------------------------
# STEP 5: LLM NODE (Agent Brain)
# -------------------------
def agent_node(state: AgentState):
    print(f"\n Node: agent_node()...")
    messages = state["messages"]

    system_prompt = """
You are an intelligent assistant.

Rules:
1. Use rag_search first
2. If rag_search returns NO_CONTEXT:
   - Retry with improved query
   - After retries → use fallback_llm
3. Use calculate_reimbursement for math questions
4. Do NOT guess if unsure
"""

    response = llm.bind_tools(tools).invoke(
        [{"role": "system", "content": system_prompt}] + messages
    )

    return {"messages": [response]}

# -------------------------
# STEP 6: ROUTER
# -------------------------
def should_continue(state: AgentState):
    print(f"\n Router: should_continue()...")
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    return END

# -------------------------
# STEP 7: RETRY NODE (NEW)
# -------------------------
def retry_node(state: AgentState):
    retry_count = state.get("retry_count", 0)

    print(f"🔁 Retry attempt: {retry_count + 1}")

    if retry_count >= MAX_RETRIES:
        print("❌ Max retries reached → forcing fallback")
        return {
            "retry_count": retry_count,
            "messages": []   # no new query
        }

    # Get last user query
    user_messages = [m for m in state["messages"] if m.type == "human"]
    last_query = user_messages[-1].content

    # 🔥 Simple query reformulation
    new_query = last_query + " (more specific, company policy details)"

    return {
        "messages": [HumanMessage(content=new_query)],
        "retry_count": retry_count + 1
    }

# -------------------------
# STEP 8: TOOL NODE
# -------------------------
tool_node = ToolNode(tools)

# -------------------------
# BUILD GRAPH
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("retry", retry_node)

graph.set_entry_point("agent")

# Agent decides → tool or finish
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)

# After tool → check if retry needed
def post_tool_router(state: AgentState):
    print("post_tool_router()...")

    last_msg = state["messages"][-1]
    retry_count = state.get("retry_count", 0)

    # print("🔍 Tool output:", last_msg)

    # If RAG failed
    if "NO_CONTEXT" in str(last_msg):

        if retry_count < MAX_RETRIES:
            print("⚠️ No context → retrying")
            return "retry"
        else:
            print("🚀 Switching to fallback after retries")
            return "agent"   # let agent choose fallback tool

    return "agent"

graph.add_conditional_edges(
    "tools",
    post_tool_router,
    {
        "retry": "retry",
        "agent": "agent",
    },
)

graph.add_edge("retry", "agent")

app = graph.compile()

# -------------------------
# CHATBOT LOOP
# -------------------------
if __name__ == "__main__":
    print("=== LangGraph Tool Agent Chatbot ===")
    print("Type 'exit' to quit\n")

    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break

        result = app.invoke({
            "messages": [HumanMessage(content=query)],
            "retry_count": 0
        })

        print("Bot:", result["messages"][-1].content)

# What is leave policy?
# Travel reimbursement limit?
# How many leave days?
# GDP of France?
# Meal expense cap?
# If I spend $120 per day for 3 days, how much reimbursement?
# If I spend twenty dollars per day for three days, how much reimbursement?
# I spent 50 per day for 4 days


