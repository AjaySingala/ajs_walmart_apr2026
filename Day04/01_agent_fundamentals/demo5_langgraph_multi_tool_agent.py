# Multi-Tool Agent with Guardrails.
from dotenv import load_dotenv
from typing import TypedDict, List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import END

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
# TOOL 1: Policy Tool (RAG)
# -----------------------------
@tool
def policy_tool(query: str) -> str:
    """Use for company policy questions (leave, travel, expenses)"""
    print("\n Tool: policy_tool()...")

    docs = retriever.invoke(query)

    # print("\n--- POLICY TOOL ---")
    # for d in docs:
    #     print("-", d.page_content)

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
# TOOL 2: Calculator Tool
# -----------------------------
@tool
def calculator_tool(expression: str) -> str:
    """Use for math calculations. Input must be a valid math expression."""
    print("\n Tool: calculator_tool()...")

    print("\n--- CALCULATOR TOOL ---")
    print("Expression:", expression)

    try:
        result = eval(expression)
        return str(result)
    except Exception:
        return "Error in calculation"


# -----------------------------
# Register tools dynamically
# -----------------------------
tools = [policy_tool, calculator_tool]
tool_map = {t.name: t for t in tools}

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# State
# -----------------------------
class AgentState(TypedDict):
    messages: List
    in_scope: bool

# -----------------------------
# Node 1: Scope Check (Input Guardrail)
# -----------------------------
def scope_check_node(state: AgentState):
    user_query = state["messages"][-1].content

    prompt = f"""
Determine if the question can be answered using:
1. Company policies (leave, travel, expenses)
2. OR calculations based on those policies

Answer YES or NO.

Examples:
- "What is leave policy?" → YES
- "What is meal reimbursement?" → YES
- "If I spend $120 per day for 3 days, how much will be reimbursed?" → YES
- "What is GDP?" → NO

Question: {user_query}
"""

    result = llm.invoke(prompt).content.lower()

    return {"in_scope": "yes" in result}

# -----------------------------
# Node 2: Agent (LLM decides tools)
# -----------------------------
def agent_node(state: AgentState):
    print("\n Node: agent_node()...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": state["messages"] + [response]}


# -----------------------------
# Node 3: Execute tools (dynamic)
# -----------------------------
def tool_node(state: AgentState):
    print("\n Node: tool_node()...")
    last_message = state["messages"][-1]

    tool_messages = []

    for call in last_message.tool_calls:
        tool_name = call["name"]
        tool_args = call["args"]

        tool = tool_map[tool_name]

        result = tool.invoke(tool_args)

        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=call["id"]
            )
        )

    return {"messages": state["messages"] + tool_messages}

# -----------------------------
# Node 4: Reject (out-of-scope)
# -----------------------------
def reject_node(state: AgentState):
    print("\n Node: reject_node()...")
    return {"messages": state["messages"] + [HumanMessage(content="I don't know")]}


# -----------------------------
# Routing 1: Scope decision
# -----------------------------
def route_scope(state: AgentState):
    print("\n Node: route_scope()...")
    return "agent_node" if state["in_scope"] else "reject_node"


# -----------------------------
# Routing 2: Tool usage
# -----------------------------
def route_tools(state: AgentState):
    print("\n Node: route_tools()...")
    last_message = state["messages"][-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    return END

# -----------------------------
# Build Graph
# -----------------------------
print("\n Build Graph...")
graph = StateGraph(AgentState)

graph.add_node("scope_check_node", scope_check_node)
graph.add_node("agent_node", agent_node)
graph.add_node("tool_node", tool_node)
graph.add_node("reject_node", reject_node)

graph.set_entry_point("scope_check_node")

# Scope routing
graph.add_conditional_edges(
    "scope_check_node",
    route_scope,
    {
        "agent_node": "agent_node",
        "reject_node": "reject_node"
    }
)

# Tool routing
graph.add_conditional_edges(
    "agent_node",
    route_tools,
    {
        "tool_node": "tool_node",
        END: END
    }
)

graph.add_edge("tool_node", "agent_node")
graph.add_edge("reject_node", END)

app = graph.compile()

# -----------------------------
# Chatbot
# -----------------------------
if __name__ == "__main__":
    print("\n=== MULTI-TOOL AGENT ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        result = app.invoke({
            "messages": [HumanMessage(content=user_input)]
        })

        final_msg = result["messages"][-1]

        print(f"Bot: {final_msg.content}\n")

# Query:
# If I spend $120 on meals for 3 days, how much will be reimbursed?
