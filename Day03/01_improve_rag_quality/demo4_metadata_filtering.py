from rag_core import index_documents, retrieve, generate_answer

index_documents()

query = "How is AI used?"

# Apply metadata filter
docs = retrieve(query, top_k=3, where={"category": "AI"})

context = "\n".join(docs)

answer = generate_answer(context, query)

print("Query:", query)
print("\nFiltered Docs:", docs)
print("\nAnswer:", answer)
