# Demo 2: Tool-based Chatbot
# Same chatbot, but internally using a tool abstraction
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common_setup import get_llm, get_embeddings, documents

vectorstore = Chroma.from_texts(
    documents,
    embedding=get_embeddings(),
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
llm = get_llm()

prompt = ChatPromptTemplate.from_template(
    """Answer ONLY from the context.

Context:
{context}

Question:
{question}
"""
)

@tool
def policy_qa_tool(query: str) -> str:
    """
    Use this tool to answer ANY questions related to:
    - leave policy
    - travel policy
    - expense policy
    - HR rules

    Always use this tool for policy-related queries.
    """

    print(f"\n policy_qa_tool()...")
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": query})


if __name__ == "__main__":
    print("\n=== TOOL CHATBOT ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = policy_qa_tool.invoke(user_input)
        print(f"Bot: {response}\n")

# Queries to try:
# What is the leave policy?
# How many leave days do employees get?
# What is the travel reimbursement policy?
# What is the leave carry forward policy?
# What is the GDP of France?

# Show limitation (multi-intent) (Still does not work):
# What is the leave policy and what expenses can I claim?
