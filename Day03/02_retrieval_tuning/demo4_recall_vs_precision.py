# Demo 4 — Recall vs Precision (Threshold Filtering).
# Threshold ↓ → higher precision
# Threshold ↑ → higher recall

from common_setup import vectorstore, llm

query = "What is the leave policy?"

results = vectorstore.similarity_search_with_score(query, k=5)

original_context = ""
for doc, score in results:
    original_context += doc.page_content + f" (Score: {score})" + "\n"

# Apply threshold filtering
threshold = 0.5  # tune this
# threshold = 0.95  # tune this
filtered_docs = [doc for doc, score in results if score < threshold]

context = "\n".join([doc.page_content for doc in filtered_docs])

prompt = f"""
Answer ONLY from the context.

{context}

Question: {query}
"""

response = llm.invoke(prompt)

print(f"\nQuery: {query}")
print("\n--- Original Context ---")
print(original_context)

print("\n--- Filtered Docs ---")
for d in filtered_docs:
    print("-", d.page_content)

print("\n--- Answer ---")
print(response.content)
