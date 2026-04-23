"""
Demo 1: Evaluation, Observability & Cost Awareness
Builds directly on: demo3_langgraph_agent_chatbot.py

Adds:
- Output evaluation
- Scoring
- Observability (tracing)
- Debugging
- Cost estimation
"""

import json
import re
import time
from typing import TypedDict, Annotated, Sequence, List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Set env vars from config.py.
import sys
import os

import config

# Start.
# -------------------------
# ENV + MODEL SETUP
# -------------------------
MODEL_NAME = os.getenv("MODEL_NAME")
EMBED_MODEL = os.getenv("TEXT_EMBEDDING_MODEL")

llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# -------------------------
# STEP 1: Dataset
# -------------------------
from common_setup import documents as docs

# -------------------------
# STEP 2: Vector Store
# -------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -------------------------
# STEP 3: TOOLS
# -------------------------
@tool
def rag_search(query: str) -> str:
    """Search company policy with filtering"""
    print(f"\n rag_search()...")
    results = vectorstore.similarity_search_with_score(query, k=2)

    # for doc, score in results:
    #     print(f"content: {doc.page_content} (Score: {score})")

    if not results:
        return "NO_CONTEXT"

    # 🔥 KEY LOGIC: threshold
    # threshold = 0.5  # adjust for demo
    threshold = 1.2  # adjust for demo
    # NOTE:
    # FAISS returns distance, not similarity.
    # Lower score = More similar.
    # Higher score = Less similar.
    # These are DISTANCES, not similarities.
    # So, either increase "distance threshold" to 1.2 or more,
    # OR Convert distance → similarity and then filter:
    #   threshold = 1 / (1 + score) # Similarity.
    filtered = [doc.page_content for doc, score in results if score < threshold]

    if not filtered:
        return "NO_CONTEXT"

    return "\n".join(filtered)


@tool
def fallback_llm(query: str) -> str:
    """Fallback general LLM"""
    # print(f"\n fallback_llm()...")
    return llm.invoke(query).content


@tool
def calculate_reimbursement(text: str) -> str:
    """Calculate reimbursement"""
    # print(f"\n calculate_reimbursement()...")
    try:
        nums = re.findall(r"\d+", text)
        if len(nums) < 2:
            return "Calculation error"

        total = int(nums[0]) * int(nums[1])
        return f"Total reimbursement = ${total}"

    except:
        return "Calculation error"


tools = [rag_search, fallback_llm, calculate_reimbursement]

# -------------------------
# 🔍 NEW: OBSERVABILITY TRACE
# -------------------------
def add_trace(state, step, info):
    """Store execution trace for debugging"""
    # print(f"\n add_trace(state, {step}, info)...")

    state["traces"].append({
        "step": step,
        "info": str(info),
        "time": time.time()
    })


# -------------------------
# STEP 4: STATE (EXTENDED)
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int
    traces: List[Dict[str, Any]]   # NEW


MAX_RETRIES = 2

# -------------------------
# STEP 5: AGENT NODE
# -------------------------
def agent_node(state: AgentState):
    # print(f"\n NODE: agent_node ()...")
    add_trace(state, "agent_start", state["messages"][-1].content)

    system_prompt = """
You are an intelligent assistant.

Rules:
1. Use rag_search first
2. If NO_CONTEXT:
   - Retry
   - Then fallback_llm
3. Use calculate_reimbursement for math
4. Do NOT guess
"""

    response = llm.bind_tools(tools).invoke(
        [{"role": "system", "content": system_prompt}] + state["messages"]
    )

    add_trace(state, "agent_response", response.content)

    return {"messages": [response]}


# -------------------------
# STEP 6: ROUTER
# -------------------------
def should_continue(state: AgentState):
    # print(f"\n Router: should_continue()...")
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    return END


# -------------------------
# STEP 7: RETRY NODE
# -------------------------
def retry_node(state: AgentState):
    # print(f"\n NODE: retry_node()...")
    retry_count = state.get("retry_count", 0)

    add_trace(state, "retry", f"Attempt {retry_count + 1}")

    print(f"🔁 Retry attempt: {retry_count + 1}")
    if retry_count >= MAX_RETRIES:
        return {"retry_count": retry_count, "messages": []}

    last_query = [m for m in state["messages"] if m.type == "human"][-1].content
    new_query = last_query + " (more specific)"

    return {
        "messages": [HumanMessage(content=new_query)],
        "retry_count": retry_count + 1
    }


# -------------------------
# STEP 8: TOOL NODE
# -------------------------
tool_node = ToolNode(tools)


# -------------------------
# 🔍 NEW: EVALUATION LOGIC
# -------------------------
def evaluate_response(text: str) -> Dict:
    """Simple rule-based evaluation"""
    # print(f"\n evaluate_response({text})...")
    score = 0
    issues = []

    if len(text) > 10:
        score += 1
    else:
        issues.append("Too short")

    if "$" in text or "policy" in text.lower():
        score += 1
    else:
        issues.append("Low relevance")

    if "error" in text.lower():
        issues.append("Error present")

    print(f"\n score: {score}. issues: {issues}...")

    return {"score": score, "issues": issues}


# -------------------------
# 🔍 NEW: EVALUATION NODE
# -------------------------
def evaluation_node(state: AgentState):
    # print(f"\n NODE: evaluation_node()...")
    last_msg = state["messages"][-1].content

    result = evaluate_response(last_msg)
    add_trace(state, "evaluation", result)

    return state


# -------------------------
# 🔍 NEW: DEBUG NODE
# -------------------------
def debug_node(state: AgentState):
    print("\n--- TRACE LOG ---")
    for t in state["traces"]:
        print(f"{t['step']} → {t['info']}")
    print("-----------------\n")

    return state


# -------------------------
# 🔍 NEW: COST ESTIMATION
# -------------------------
import tiktoken

# Pricing (USD per 1M tokens)
INPUT_COST_PER_1M = 0.15
OUTPUT_COST_PER_1M = 0.60

def count_tokens(text: str, model: str):
    """Count tokens using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    token_count = len(encoding.encode(text))
    print(f"token_count: {token_count}")
    return token_count


def estimate_cost(input_text: str, output_text: str):
    """Estimate realistic OpenAI cost in USD"""
    # print(f"\n estimation_cost()...")

    print(f"input_text: {input_text}")
    input_tokens = count_tokens(input_text, os.getenv("MODEL_NAME"))

    print(f"output_text: {output_text}")
    output_tokens = count_tokens(output_text, os.getenv("MODEL_NAME"))

    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M

    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,

        # Human readable format
        "input_cost_usd": f"${input_cost:.8f}",
        "output_cost_usd": f"${output_cost:.8f}",
        "total_cost_usd": f"${total_cost:.8f}",

        # Business metric
        "cost_per_1k_queries": f"${(total_cost * 1000):.4f}"
    }


# Tool routing
def post_tool_router(state: AgentState):
    # print(f"\n post_tool_router...")
    last_msg = state["messages"][-1]
    retry_count = state.get("retry_count", 0)

    if "NO_CONTEXT" in str(last_msg):
        if retry_count < MAX_RETRIES:
            return "retry"
        else:
            return "agent"

    return "agent"

# -------------------------
# BUILD GRAPH (EXTENDED)
# -------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("retry", retry_node)
graph.add_node("evaluate", evaluation_node)
graph.add_node("debug", debug_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: "evaluate"},
)

graph.add_conditional_edges(
    "tools",
    post_tool_router,
    {"retry": "retry", "agent": "agent"},
)

graph.add_edge("retry", "agent")

# After evaluation → debug
graph.add_edge("evaluate", "debug")
graph.add_edge("debug", END)

app = graph.compile()


# -------------------------
# DEMOS
# -------------------------
if __name__ == "__main__":
    print("\n=== Demo 1: Poor Output ===")

    result = app.invoke({
        "messages": [HumanMessage(content="???")],
        "retry_count": 0,
        "traces": []
    })

    print("Final:", result["messages"][-1].content)

    print("\n=== Demo 2: Good Query ===")

    # query = "If I spend $120 per day for 3 days, how much reimbursement?"
    query = "What is the leave policy?"
    result = app.invoke({
        "messages": [HumanMessage(content=query)],
        "retry_count": 0,
        "traces": []
    })

    print("Final:", result["messages"][-1].content)

    print("\n=== Demo 3: Multi-output Scoring ===")

    samples = [
        "Total reimbursement = $360",
        "maybe 100",
        "error happened"
    ]

    for s in samples:
        print("\nText:", s)
        print(evaluate_response(s))

    print("\n=== Demo 4: Cost Awareness ===")

    response = result["messages"][-1].content
    cost_info = estimate_cost(query, response)
    print("\n--- COST BREAKDOWN ---")
    print(json.dumps(cost_info, indent=2))

    print("\n=== Demo 5: Failure Debug ===")

    result = app.invoke({
        "messages": [HumanMessage(content="random nonsense")],
        "retry_count": 0,
        "traces": []
    })

    print("Retries:", result["retry_count"])
    