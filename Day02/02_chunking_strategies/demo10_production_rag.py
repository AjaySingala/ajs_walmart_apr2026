from common_setup import get_embedding, cosine_similarity, client, MODEL_NAME
from demo1_fixed_chunking import fixed_chunk
from demo2_recursive_chunking import recursive_chunk
from demo3_overlap_chunking import overlapping_chunk


class RAGPipeline:
    """Reusable RAG pipeline with pluggable strategies"""

    def __init__(self, strategy="recursive", chunk_size=80, overlap=20):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store = []

    def chunk_document(self, document):
        """Apply selected chunking strategy"""
        if self.strategy == "fixed":
            return fixed_chunk(document, self.chunk_size)
        elif self.strategy == "overlap":
            return overlapping_chunk(document, self.chunk_size, self.overlap)
        else:
            return recursive_chunk(document, self.chunk_size)

    def index(self, document):
        """Create vector store from document"""
        chunks = self.chunk_document(document)
        self.vector_store = [(c, get_embedding(c)) for c in chunks]

    def retrieve(self, query, top_k=2):
        """Retrieve top-k chunks"""
        query_embedding = get_embedding(query)

        scored = []
        for text, emb in self.vector_store:
            score = cosine_similarity(query_embedding, emb)
            scored.append((text, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def generate(self, query, retrieved_chunks):
        """Generate answer from retrieved chunks"""
        context = "\n\n".join([text for text, _ in retrieved_chunks])

        prompt = f"""
        Answer using ONLY the context.

        Context:
        {context}

        Question:
        {query}
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )

        return response.choices[0].message.content

    def run(self, document, query):
        """End-to-end RAG execution"""
        self.index(document)
        retrieved = self.retrieve(query)
        answer = self.generate(query, retrieved)

        return {
            "answer": answer,
            "sources": retrieved
        }


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    query = "How does Walmart improve supply chain efficiency?"

    rag = RAGPipeline(strategy="overlap")
    result = rag.run(document, query)

    print("\nAnswer:\n", result["answer"])
    print("\nSources:\n")
    for text, score in result["sources"]:
        print(f"[{score:.4f}] {text}")
        