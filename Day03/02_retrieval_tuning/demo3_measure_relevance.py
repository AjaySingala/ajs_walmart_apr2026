# Demo 3 — Measure Relevance (Scores).
# Lower score = more relevant (Chroma distance)
# Helps justify tuning decisions

from common_setup import vectorstore

query = "What is the leave policy?"

# Get similarity scores
results = vectorstore.similarity_search_with_score(query, k=4)

print(f"\nQuery: {query}")
print("\n--- Retrieved with Scores ---")
for doc, score in results:
    print(f"Score: {round(score, 4)} | {doc.page_content}")
