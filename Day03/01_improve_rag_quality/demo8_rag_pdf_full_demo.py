import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from PyPDF2 import PdfReader

# ================================
# SETUP
# ================================
# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
client = OpenAI()

chroma_client = chromadb.PersistentClient(path="./chroma_db_pdf")
collection = chroma_client.get_or_create_collection(name="pdf_rag_demo")

# ================================
# EMBEDDING
# ================================
def embed_text(text):
    print(f"\n embed_text()...")
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding


# ================================
# PDF LOADER
# ================================
def load_pdf(file_path):
    """Extract text from PDF"""
    print(f"\n load_pdf({file_path})...")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# ================================
# CHUNKING
# ================================
def chunk_text(text, chunk_size=200, overlap=50):
    """Simple chunking"""
    print(f"\n chunk_text()...")
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


# ================================
# INDEX MULTIPLE PDFs
# ================================
def index_pdfs():
    print(f"\n index_pdfs()...")
    files = [
        ("documents/leave_policy.pdf", "LeavePolicy"),
        ("documents/travel_policy.pdf", "TravelPolicy")
    ]

    doc_id = 1

    for file_path, category in files:
        text = load_pdf(file_path)
        chunks = chunk_text(text)

        for chunk in chunks:
            collection.upsert(
                ids=[str(doc_id)],
                documents=[chunk],
                embeddings=[embed_text(chunk)],
                metadatas=[{"source": category}]
            )
            doc_id += 1


# ================================
# RETRIEVAL
# ================================
def retrieve(query, top_k=3, where=None):
    print(f"\n retrieve(query={query}, top_k={top_k}, where={where})...")
    query_emb = embed_text(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where
    )

    return results["documents"][0] if results["documents"] else []

def retrieve_with_scores(query, top_k=3, where=None):
    """Retrieve docs with similarity scores"""
    query_emb = embed_text(query)

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where,
        include=["documents", "distances"]
    )

    docs = results["documents"][0]
    distances = results["distances"][0]

    # Convert distance → similarity
    scored_docs = []
    for doc, dist in zip(docs, distances):
        similarity = 1 - dist
        scored_docs.append((similarity, doc))

    return scored_docs

# ================================
# COSINE SIMILARITY
# ================================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================================
# QUERY (SAME FOR ALL DEMOS)
# ================================
QUERY = "What are the rules for employee travel reimbursement and approvals?"
print(f"\n QUERY = {QUERY}")

# ================================
# DEMO 1: BASELINE
# ================================
def demo1():
    print(f"\n demo1()...")
    docs = retrieve(QUERY, top_k=2)
    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from context.

Context:
{context}

Question: {QUERY}
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 1 ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 2: RERANK
# ================================
def demo2():
    print(f"\n demo2()...")
    docs = retrieve(QUERY, top_k=5)

    query_emb = embed_text(QUERY)

    scored = []
    for doc in docs:
        score = cosine_similarity(query_emb, embed_text(doc))
        scored.append((score, doc))

    scored.sort(reverse=True)
    docs = [doc for _, doc in scored[:2]]

    context = "\n".join(docs)

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 2 (RERANK) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 3: QUERY REWRITE
# ================================
def demo3():
    print(f"\n demo3()...")
    rewrite = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"Rewrite: {QUERY}"}],
        temperature=0
    ).choices[0].message.content

    docs = retrieve(rewrite, top_k=3)

    context = "\n".join(docs)

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"{context}\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 3 (QUERY REWRITE) ---")
    print("Rewrite:", rewrite)
    print(res.choices[0].message.content)


# ================================
# DEMO 4: METADATA FILTER
# ================================
def demo4(source="TravelPolicy"):
    print(f"\n demo4(source={source})...")
    docs = retrieve(QUERY, top_k=3, where={"source": source})

    context = "\n".join(docs)

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"{context}\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 4 (FILTER) ---")
    print(res.choices[0].message.content)

def demo4_improved(source="TravelPolicy"):
    scored_docs = retrieve_with_scores(
        QUERY,
        top_k=3,
        where={"source": source}
    )

    # KEY FIX: Filter by relevance threshold
    filtered_docs = [doc for score, doc in scored_docs if score > 0.75]

    if not filtered_docs:
        context = ""
    else:
        context = "\n".join(filtered_docs)

    prompt = f"""
Answer ONLY from the context.
If the context is empty or not relevant, say "I don't know".

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 4 (FILTER FIXED) ---")
    print("Filtered Docs:", filtered_docs)
    print("Answer:", response.choices[0].message.content)

# ================================
# DEMO 5: STRUCTURED CONTEXT
# ================================
def demo5():
    print(f"\n demo5()...")
    docs = retrieve(QUERY, top_k=3)

    context = ""
    for i, d in enumerate(docs, 1):
        context += f"[Source {i}]: {d}\n"

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": f"{context}\nQ:{QUERY}"}],
        temperature=0
    )

    print("\n--- DEMO 5 (STRUCTURED) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 6: PROMPT IMPROVEMENT
# ================================
def demo6():
    print(f"\n demo6()...")
    docs = retrieve(QUERY, top_k=3)
    context = "\n".join(docs)

    prompt = f"""
- Answer only from context
- Be structured
- If unsure say "I don't know"

Context:
{context}

Question: {QUERY}
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 6 (PROMPT) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 7: HALLUCINATION
# ================================
def demo7():
    print(f"\n demo7()...")
    docs = retrieve(QUERY, top_k=1)
    context = "\n".join(docs)

    prompt = f"""
Answer using your knowledge also.

Context:
{context}

Question: {QUERY}
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    print("\n--- DEMO 7 (HALLUCINATION) ---")
    print(res.choices[0].message.content)


# ================================
# DEMO 8: NO DATA
# ================================
def demo8():
    print(f"\n demo8()...")
    query = "What is Walmart's stock trading strategy?"
    print(f"\n Query = {query}")

    docs = retrieve(query, top_k=2)
    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from context.
If not found say "I don't know"

Context:
{context}

Q: {query}
"""

    res = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 8 (NO DATA) ---")
    print(res.choices[0].message.content)


# ================================
# RUN
# ================================
if __name__ == "__main__":
    index_pdfs()

    demo1()
    demo2()
    demo3()
    demo4()
    demo4("LeavePolicy")
    demo4_improved("LeavePolicy")
    demo5()
    demo6()
    demo7()
    demo8()
