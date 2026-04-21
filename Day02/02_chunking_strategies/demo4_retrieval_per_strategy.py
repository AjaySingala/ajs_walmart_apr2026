from common_setup import get_embedding, cosine_similarity
from demo1_fixed_chunking import fixed_chunk
from demo2_recursive_chunking import recursive_chunk
from demo3_overlap_chunking import overlapping_chunk


def build_vector_store(chunks):
    """Convert chunks into (text, embedding) pairs"""
    return [(chunk, get_embedding(chunk)) for chunk in chunks]


def retrieve(query, vector_store, top_k=2):
    """Retrieve top-k most relevant chunks"""
    query_embedding = get_embedding(query)

    scored = []
    for text, emb in vector_store:
        score = cosine_similarity(query_embedding, emb)
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    query = "How does Walmart improve supply chain efficiency?"
    print(f"\nQuery: {query}")

    # Try different strategies
    strategies = {
        "Fixed": fixed_chunk(document, 80),
        "Recursive": recursive_chunk(document, 80),
        "Overlap": overlapping_chunk(document, 80, 20)
    }

    for name, chunks in strategies.items():
        print(f"\n--- {name} Strategy ---")

        store = build_vector_store(chunks)
        results = retrieve(query, store)

        for res, score in results:
            print(f"\nScore: {score:.4f}")
            print(res)
