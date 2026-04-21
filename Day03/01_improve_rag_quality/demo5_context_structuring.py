from rag_core import index_documents, retrieve, generate_answer

index_documents()

query = "How does AI help Walmart?"

docs = retrieve(query, top_k=3)

def structure_context(docs):
    """Add structure to improve LLM understanding"""
    structured = ""
    for i, doc in enumerate(docs, 1):
        structured += f"[Source {i}]: {doc}\n"
    return structured


context = structure_context(docs)

answer = generate_answer(context, query)

print("Query:", query)
print("\nStructured Context:\n", context)
print("\nAnswer:", answer)
