from demo2_recursive_chunking import recursive_chunk
from demo4_retrieval_per_strategy import build_vector_store, retrieve
from demo8_contextual_rag import generate_answer_with_structure


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    query = "How does Walmart improve supply chain efficiency?"
    # query = "What is the GDP of Canada?"

    # Step 1: Chunk document (reuse previous demo)
    chunks = recursive_chunk(document, 80)

    # Step 2: Build vector store (embeddings)
    store = build_vector_store(chunks)

    # Step 3: Retrieve relevant chunks
    context_chunks = retrieve(query, store)

    # Step 4: NEW → Structured answer generation
    answer = generate_answer_with_structure(query, context_chunks)

    print("\n=== STRUCTURED RAG OUTPUT ===\n")
    print(answer)
    