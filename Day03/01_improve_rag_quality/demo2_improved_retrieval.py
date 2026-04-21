from rag_core import index_documents, retrieve, embed_text, generate_answer
import numpy as np

# Index documents
index_documents()

query = "How does Walmart use AI in supply chain?"
# query = "Explain how Walmart combines AI and supply chain optimization"

# Step 1: Retrieve more documents (increase recall)
docs = retrieve(query, top_k=5)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def rerank_with_embeddings(docs, query):
    """Re-rank documents using embedding similarity"""
    query_emb = embed_text(query)

    scored = []
    for doc in docs:
        doc_emb = embed_text(doc)
        score = cosine_similarity(query_emb, doc_emb)
        scored.append((score, doc))

    # Sort by similarity
    scored.sort(reverse=True)

    for score, doc in scored:
        print(f"Score: {score:.4f} | Doc: {doc[:60]}...")
        
    # Return top 2 most relevant
    return [doc for _, doc in scored[:2]]


# Step 2: Re-rank using embeddings
docs = rerank_with_embeddings(docs, query)

context = "\n".join(docs)

answer = generate_answer(context, query)

print("Query:", query)
print("\nRe-ranked Docs:", docs)
print("\nAnswer:", answer)
