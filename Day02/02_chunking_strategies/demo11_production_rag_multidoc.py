import os
from common_setup import get_embedding, cosine_similarity, client, MODEL_NAME
from demo1_fixed_chunking import fixed_chunk
from demo2_recursive_chunking import recursive_chunk
from demo3_overlap_chunking import overlapping_chunk


class RAGPipeline:
    """Production-ready RAG with multi-document + source attribution"""

    def __init__(self, strategy="recursive", chunk_size=80, overlap=20):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.vector_store = []  # Will store (text, embedding, metadata)

    def chunk_document(self, document):
        """Apply selected chunking strategy"""
        if self.strategy == "fixed":
            return fixed_chunk(document, self.chunk_size)
        elif self.strategy == "overlap":
            return overlapping_chunk(document, self.chunk_size, self.overlap)
        else:
            return recursive_chunk(document, self.chunk_size)

    def load_documents_from_folder(self, folder_path):
        """Load all .txt files from a folder"""
        documents = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                    documents.append((filename, text))  # keep filename as metadata

        return documents
    
    def index_documents(self, folder_path):
        """Index all documents in the folder"""
        docs = self.load_documents_from_folder(folder_path)

        for filename, text in docs:
            chunks = self.chunk_document(text)

            for chunk in chunks:
                embedding = get_embedding(chunk)

                # Store text + embedding + metadata
                self.vector_store.append({
                    "text": chunk,
                    "embedding": embedding,
                    "source": filename
                })

    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant chunks with metadata"""
        query_embedding = get_embedding(query)

        scored = []
        for item in self.vector_store:
            score = cosine_similarity(query_embedding, item["embedding"])
            scored.append({
                "text": item["text"],
                "score": score,
                "source": item["source"]
            })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def generate(self, query, retrieved_chunks):
        """Generate answer using retrieved chunks"""
        
        context = "\n\n".join([item["text"] for item in retrieved_chunks])

        prompt = f"""
        You are an enterprise AI assistant.

        Instructions:
        - Answer ONLY from the context
        - Be concise
        - If not found, say "Not available in context"

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

    def run(self, folder_path, query):
        """End-to-end RAG for multiple documents"""
        self.index_documents(folder_path)
        retrieved = self.retrieve(query)
        answer = self.generate(query, retrieved)

        return {
            "answer": answer,
            "sources": retrieved
        }


if __name__ == "__main__":
    # Folder containing text files
    folder_path = "./documents"

    query = "How does Walmart improve customer satisfaction?"

    rag = RAGPipeline(strategy="overlap")
    result = rag.run(folder_path, query)

    print("\n=== FINAL ANSWER ===\n")
    print(result["answer"])

    print("\n=== SOURCES ===\n")
    for item in result["sources"]:
        print(f"[{item['score']:.4f}] ({item['source']})")
        print(item["text"])
        print("-" * 50)
        