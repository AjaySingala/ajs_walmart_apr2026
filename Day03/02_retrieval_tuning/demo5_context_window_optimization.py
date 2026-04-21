# Demo 5 — Context Window Optimization.
from common_setup import vectorstore, llm

query = "Explain leave rules"
# query = "Explain travel rules"

results = vectorstore.similarity_search_with_score(query, k=5)
original_docs = ""
for doc, score in results:
    original_docs += doc.page_content + f" (Score: {score})" + "\n"

# Sort by relevance (best first)
sorted_docs = sorted(results, key=lambda x: x[1])
original_context = ""
for doc, score in sorted_docs:
    original_context += doc.page_content + f" (Score: {score}) - (Length: {len(doc.page_content)})" "\n"

# Limit total context size
max_chars = 50  # Tune this.
context = ""

for doc, score in sorted_docs:
    if len(context) + len(doc.page_content) < max_chars:
        context += doc.page_content + "\n"

prompt = f"""
Answer ONLY from the context.

{context}

Question: {query}
"""

response = llm.invoke(prompt)

print(f"\nQuery: {query}")
# print("\n--- Original Docs ---")
# print(original_docs)

print("\n--- Original Context ---")
print(original_context)

print("\n--- Optimized Context ---")
print(context)
print(f"Length: {len(context)}")

print("\n--- Answer ---")
print(response.content)
