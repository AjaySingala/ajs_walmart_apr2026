# Demo 1: RAG Chatbot (Chain).
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from common_setup import get_llm, get_embeddings, documents

# Create vector store
vectorstore = Chroma.from_texts(
    documents,
    embedding=get_embeddings(),
    persist_directory="./chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_template(
    """Answer ONLY from the context.

Context:
{context}

Question:
{question}
"""
)

llm = get_llm()
chain = prompt | llm | StrOutputParser()


def run_rag(query):
    """Retrieve relevant docs and generate answer"""
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    return chain.invoke({"context": context, "question": query})


if __name__ == "__main__":
    print("\n=== RAG CHATBOT ===")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = run_rag(user_input)
        print(f"Bot: {response}\n")

# Queries to try:
# What is the leave policy?
# What expenses are reimbursable?
# How many leave days do employees get?
# What is the travel reimbursement policy?
# What is the leave carry forward policy?
# What is the GDP of France?

# Show limitation (multi-intent) (Will not work):
# What is the leave policy and what expenses can I claim?
