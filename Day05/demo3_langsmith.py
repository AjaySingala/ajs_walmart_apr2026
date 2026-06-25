# DEMO 3 — LLM Agents with LangSmith Monitoring
# smith.langchain.com

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import asyncio
import os


# # Enable LangSmith tracing
# os.environ["LANGCHAIN_TRACING"] = "true"
# os.environ["LANGCHAIN_PROJECT"] = "ajs-agent-demo-01"
# # Set your API key externally for security
# LANGCHAIN_API_KEY=""

print(os.getenv("LANGCHAIN_TRACING"))
print(os.getenv("LANGCHAIN_API_KEY"))
print(os.getenv("LANGCHAIN_PROJECT"))

# print(os.getenv("LANGSMITH_TRACING"))
# print(os.getenv("LANGSMITH_API_KEY"))
# print(os.getenv("LANGSMITH_PROJECT"))
# print(os.getenv("LANGSMITH_ENDPOINT"))

class AgentState(TypedDict):
    transactions: List[dict]
    findings: List[dict]


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI audit agent."),
    ("human", "Analyze transaction: {txn}. Return structured findings.")
])


async def monitored_node(state: AgentState):
    print("\n📊 Running with LangSmith Tracing...\n")

    chain = prompt | llm

    async def process(txn):
        print(f"\n worker_node() -> process(): Txn Id: {txn['id']}")

        # Each LLM call is automatically traced in LangSmith
        response = await chain.ainvoke({"txn": txn})

        return {
            "id": txn["id"],
            "analysis": response.content
        }

    results = await asyncio.gather(*(process(t) for t in state["transactions"]))

    return {"findings": results}


# Build graph
builder = StateGraph(AgentState)
builder.add_node("monitored", monitored_node)
builder.set_entry_point("monitored")
builder.add_edge("monitored", END)

graph = builder.compile()


if __name__ == "__main__":
    input_state = {
        "transactions": [
            {"id": 201, "amount": 20000, "country": "IN"},
            {"id": 202, "amount": 50000, "country": "US"},
            {"id": 203, "amount": 120000, "country": "Unknown"},
        ],
        "findings": []
    }

    result = asyncio.run(graph.ainvoke(input_state))
    print("\nFinal Output:", result)
