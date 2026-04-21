# Demo 1: Baseline Strict RAG
from rag_core import index_documents, retrieve, generate_answer

# Index documents (run once)
index_documents()

# query = "How does Walmart use AI?"
query = "How does Walmart improve supply chain using AI?"

# Simple top-k retrieval
docs = retrieve(query, top_k=2)

context = "\n".join(docs)

answer = generate_answer(context, query)

print("Query:", query)
print("\nRetrieved Docs:", docs)
print("\nAnswer:", answer)
