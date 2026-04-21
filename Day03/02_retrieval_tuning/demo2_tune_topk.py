# Demo 2 — Tune Top-k.
from common_setup import vectorstore, llm

query = "What is the leave policy?"

# Try smaller k
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
docs = retriever.invoke(query)

context = "\n".join([doc.page_content for doc in docs])

prompt = f"""
Answer based only on the context:

{context}

Question: {query}
"""

response = llm.invoke(prompt)

print(f"\nQuery: {query}")
print("\n--- Top-k = 2 Docs ---")
for d in docs:
    print("-", d.page_content)

print("\n--- Answer ---")
print(response.content)
