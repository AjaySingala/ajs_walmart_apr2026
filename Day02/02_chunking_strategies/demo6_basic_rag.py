from common_setup import client, MODEL_NAME, get_embedding, cosine_similarity
from demo2_recursive_chunking import recursive_chunk


def build_vector_store(chunks):
    """Store chunks with embeddings"""
    return [(chunk, get_embedding(chunk)) for chunk in chunks]


def retrieve(query, vector_store, top_k=2):
    """Retrieve most relevant chunks"""
    query_embedding = get_embedding(query)

    scored = []
    for text, emb in vector_store:
        score = cosine_similarity(query_embedding, emb)
        scored.append((text, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [text for text, _ in scored[:top_k]]


def generate_answer(query, context_chunks):
    """Generate answer using retrieved context"""
    
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Answer the question using ONLY the context below.
    
    Context:
    {context}
    
    Question:
    {query}
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # low temperature for factual answers
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    query = "How does Walmart improve supply chain efficiency?"

    # Step 1: Chunk
    chunks = recursive_chunk(document, 80)

    # Step 2: Embed
    store = build_vector_store(chunks)

    # Step 3: Retrieve
    context = retrieve(query, store)

    # Step 4: Generate
    answer = generate_answer(query, context)

    print("\nFinal Answer:\n", answer)
    