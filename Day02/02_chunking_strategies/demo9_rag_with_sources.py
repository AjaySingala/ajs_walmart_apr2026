from demo4_retrieval_per_strategy import build_vector_store, retrieve


def retrieve_with_scores(query, vector_store, top_k=2):
    """Return chunks with scores"""
    from common_setup import get_embedding, cosine_similarity

    query_embedding = get_embedding(query)

    scored = []
    for text, emb in vector_store:
        score = cosine_similarity(query_embedding, emb)
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def format_sources(results):
    """Format sources for display"""
    return "\n\n".join(
        [f"[Score: {score:.4f}] {text}" for text, score in results]
    )


if __name__ == "__main__":
    from demo2_recursive_chunking import recursive_chunk
    from demo8_contextual_rag import generate_answer_with_structure

    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    query = "How does Walmart improve supply chain efficiency?"

    chunks = recursive_chunk(document, 80)
    store = build_vector_store(chunks)

    results = retrieve_with_scores(query, store)
    context = [text for text, _ in results]

    answer = generate_answer_with_structure(query, context)

    print("\nAnswer:\n", answer)
    print("\nSources:\n", format_sources(results))
    