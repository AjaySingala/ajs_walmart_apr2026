import os
import faiss
import numpy as np
from pypdf import PdfReader
from collections import Counter

from common_setup import client, MODEL_NAME, get_embedding
from demo2_recursive_chunking import recursive_chunk


class FAISSRAGPipeline:
    """Production-grade RAG with FAISS + multi-doc + reranking + streaming"""

    def __init__(self, chunk_size=200):
        self.chunk_size = chunk_size
        self.index = None
        self.documents = []  # stores metadata + text

    # -----------------------------
    # 📄 LOAD DOCUMENTS (TXT + PDF)
    # -----------------------------
    def load_documents(self, folder_path):
        """Load TXT and PDF files"""
        docs = []

        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)

            if filename.endswith(".txt"):
                with open(filepath, "r", encoding="utf-8") as f:
                    docs.append({
                        "text": f.read(),
                        "source": filename,
                        "type": "text"
                    })

            elif filename.endswith(".pdf"):
                reader = PdfReader(filepath)
                text = "\n".join([page.extract_text() or "" for page in reader.pages])

                docs.append({
                    "text": text,
                    "source": filename,
                    "type": "pdf"
                })

        return docs

    # -----------------------------
    # ✂️ CHUNK + EMBED
    # -----------------------------
    def chunk_and_embed(self, docs):
        """Chunk documents and generate embeddings"""
        all_embeddings = []

        for doc in docs:
            chunks = recursive_chunk(doc["text"], self.chunk_size)

            for chunk in chunks:
                embedding = get_embedding(chunk)

                self.documents.append({
                    "text": chunk,
                    "source": doc["source"],
                    "type": doc["type"]
                })

                all_embeddings.append(embedding)

        return np.array(all_embeddings).astype("float32")

    # -----------------------------
    # 🧠 BUILD FAISS INDEX
    # -----------------------------
    def build_index(self, folder_path):
        """Create FAISS index"""
        docs = self.load_documents(folder_path)
        embeddings = self.chunk_and_embed(docs)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)  # simple + fast
        self.index.add(embeddings)

    # -----------------------------
    # 🔍 RETRIEVE
    # -----------------------------
    def retrieve(self, query, top_k=5, filter_type=None):
        """Retrieve top-k with optional metadata filtering"""
        query_embedding = np.array([get_embedding(query)]).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k * 2)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc = self.documents[idx]

            # Apply metadata filter
            if filter_type and doc["type"] != filter_type:
                continue

            results.append({
                "text": doc["text"],
                "source": doc["source"],
                "type": doc["type"],
                "score": float(1 / (1 + dist))  # convert distance → similarity
            })

            if len(results) >= top_k:
                break

        return results

    # -----------------------------
    # 🔥 RERANK (LLM-based)
    # -----------------------------
    def rerank(self, query, retrieved_chunks, top_k=3):
        """Rerank using LLM (semantic refinement)"""

        prompt = f"""
        Given the query and chunks, rank the most relevant ones.

        Query: {query}

        Chunks:
        {retrieved_chunks}

        Return ONLY the indices of top {top_k} most relevant chunks.
        """

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            indices = eval(response.choices[0].message.content)
            return [retrieved_chunks[i] for i in indices]
        except:
            return retrieved_chunks[:top_k]

    # -----------------------------
    # 💬 GENERATE (STREAMING)
    # -----------------------------
    def generate_stream(self, query, chunks):
        """Streaming answer generation"""

        context = "\n\n".join([c["text"] for c in chunks])

        prompt = f"""
        Answer using ONLY the context.

        Context:
        {context}

        Question:
        {query}
        """

        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            stream=True
        )

        print("\n=== STREAMING ANSWER ===\n")

        full_response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_response += token

        print("\n")
        return full_response

    # -----------------------------
    # 🚀 RUN PIPELINE
    # -----------------------------
    def run(self, folder_path, query, filter_type=None):
        """End-to-end execution"""

        self.build_index(folder_path)

        # Step 1: Retrieve
        retrieved = self.retrieve(query, filter_type=filter_type)

        # Step 2: Rerank
        reranked = self.rerank(query, retrieved)

        # Step 3: Generate (streaming)
        answer = self.generate_stream(query, reranked)

        return {
            "answer": answer,
            "sources": reranked
        }


# -----------------------------
# ▶️ MAIN
# -----------------------------
if __name__ == "__main__":
    folder_path = "./documents"

    query = "How does Walmart improve supply chain efficiency?"

    rag = FAISSRAGPipeline()

    result = rag.run(folder_path, query, filter_type="pdf")  # try "pdf" or "text"

    print("\n=== SOURCES ===\n")

    for item in result["sources"]:
        print(f"[{item['score']:.4f}] ({item['source']} - {item['type']})")
        print(item["text"])
        print("-" * 50)

    # Bonus: most contributing source
    sources = [item["source"] for item in result["sources"]]
    print("\nTop Source:", Counter(sources).most_common(1))
