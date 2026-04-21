from demo1_fixed_chunking import fixed_chunk
from demo2_recursive_chunking import recursive_chunk
from demo3_overlap_chunking import overlapping_chunk
from demo6_basic_rag import build_vector_store, retrieve, generate_answer


def run_rag(document, query, strategy="recursive"):
    """Run RAG pipeline with selected chunking strategy"""

    if strategy == "fixed":
        chunks = fixed_chunk(document, 80)
    elif strategy == "overlap":
        chunks = overlapping_chunk(document, 80, 20)
    else:
        chunks = recursive_chunk(document, 80)

    store = build_vector_store(chunks)
    context = retrieve(query, store)
    answer = generate_answer(query, context)

    return answer


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    query = "How does Walmart improve supply chain efficiency?"

    for strategy in ["fixed", "recursive", "overlap"]:
        print(f"\n--- Strategy: {strategy} ---")
        print(run_rag(document, query, strategy))
        