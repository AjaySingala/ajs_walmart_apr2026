# DEMO 5 — Side-by-Side Comparison.
from demo4_retrieval_per_strategy import build_vector_store, retrieve
from demo1_fixed_chunking import fixed_chunk
from demo2_recursive_chunking import recursive_chunk
from demo3_overlap_chunking import overlapping_chunk


def compare_strategies(document, query):
    """Compare retrieval outputs across chunking strategies"""
    
    strategies = {
        "Fixed": fixed_chunk(document, 80),
        "Recursive": recursive_chunk(document, 80),
        "Overlap": overlapping_chunk(document, 80, 20)
    }

    results = {}

    for name, chunks in strategies.items():
        store = build_vector_store(chunks)
        top_result = retrieve(query, store, top_k=1)[0]
        results[name] = top_result

    return results


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    queries = [
        "inventory optimization",
        "customer satisfaction",
        "supply chain efficiency"
    ]

    for query in queries:
        print(f"\n==============================")
        print(f"Query: {query}")
        print(f"==============================")

        results = compare_strategies(document, query)

        for strategy, (text, score) in results.items():
            print(f"\n[{strategy}] Score: {score:.4f}")
            print(text)
            