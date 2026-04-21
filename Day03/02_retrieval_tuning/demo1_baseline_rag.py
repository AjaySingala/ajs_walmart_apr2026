# Demo 1 — Baseline RAG (Top-k = 4, no tuning)
from common_setup import vectorstore, llm, load_data

load_data()

query = "What is the leave policy?"

# Retrieve top-k documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
docs = retriever.invoke(query)

# Combine context
context = "\n".join([doc.page_content for doc in docs])

# Simple prompt
prompt = f"""
Answer the question based on the context below:

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print(f"\nQuery: {query}")
print("\n--- Retrieved Docs ---")
for d in docs:
    print("-", d.page_content)

print("\n--- Answer ---")
print(response.content)
