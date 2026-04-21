import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

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

# Persistent ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_demo")

# -------------------------------
# EMBEDDING FUNCTION
# -------------------------------
def embed_text(text):
    """Generate embedding for text"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding


# -------------------------------
# INDEX RICH DOCUMENTS
# -------------------------------
def index_documents():
    """Index richer documents with overlap, noise, and partial info"""

    docs = [
        {
            "id": "1",
            "text": """Walmart uses machine learning models to optimize inventory management.
            These models analyze historical sales data and store-level demand patterns to ensure
            shelves are stocked efficiently.""",
            "category": "AI"
        },
        {
            "id": "2",
            "text": """Walmart applies demand forecasting techniques in its supply chain.
            Forecasting models consider seasonality, promotions, and regional trends to improve
            replenishment and reduce logistics costs.""",
            "category": "SupplyChain"
        },
        {
            "id": "3",
            "text": """Walmart integrates AI with supply chain operations by combining demand forecasting
            and inventory optimization. This allows faster restocking and better warehouse distribution.""",
            "category": "SupplyChain"
        },
        {
            "id": "4",
            "text": """Walmart uses generative AI in customer service for chatbots handling queries like returns,
            refunds, and product information. This improves response time and customer satisfaction.""",
            "category": "AI"
        },
        {
            "id": "5",
            "text": """Retail businesses like Walmart are affected by seasonal demand such as holidays and weather.
            These trends influence supply chain planning but are not directly controlled by AI systems.""",
            "category": "Retail"
        },
        {
            "id": "6",
            "text": """Some legacy systems in retail supply chains still rely on manual processes.
            Not all Walmart operations are fully automated with AI.""",
            "category": "Noise"
        },
        {
            "id": "99",
            "text": """This is some garbage text that has been added.
            Just has the words AI and inventory in it. Let's see if it shows up in any responses.""",
            "category": "SupplyChain"
        }
    ]

    for doc in docs:
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[embed_text(doc["text"])],
            metadatas=[{"category": doc["category"]}]
        )


# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve(query, top_k=3, where=None):
    """Retrieve documents from vector DB"""
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where
    )

    return results["documents"][0] if results["documents"] else []


# -------------------------------
# COSINE SIMILARITY
# -------------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ================================
# DEMO QUERY (SAME FOR ALL)
# ================================

QUERY = "How does Walmart use AI in supply chain?"

# ================================
# DEMO 1: BASELINE RAG
# ================================
def demo1():
    docs = retrieve(QUERY, top_k=2)
    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from the context.
If not found, say "I don't know".

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 1: BASELINE ---")
    print("\nDocs:", docs)
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 2: BETTER RETRIEVAL + RERANK
# ================================
def demo2():
    docs = retrieve(QUERY, top_k=5)

    query_emb = embed_text(QUERY)

    scored = []
    for doc in docs:
        score = cosine_similarity(query_emb, embed_text(doc))
        scored.append((score, doc))

    scored.sort(reverse=True)
    docs = [doc for _, doc in scored[:2]]

    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 2: RERANKED ---")
    print("\nDocs:", docs)
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 3: QUERY REFORMULATION
# ================================
def demo3():
    rewrite_prompt = f"""
Rewrite the query to improve retrieval by adding keywords.

Query: {QUERY}
"""

    rewritten = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": rewrite_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    docs = retrieve(rewritten, top_k=3)

    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 3: QUERY REWRITE ---")
    print("\nRewritten Query:", rewritten)
    print("\nDocs:", docs)
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 4: METADATA FILTERING
# ================================
def demo4(category: str = "SupplyChain"):
    docs = retrieve(QUERY, top_k=3, where={"category": category})

    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print(f"\n--- DEMO 4: METADATA FILTER ({category}) ---")
    print("\nDocs:", docs)
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 5: CONTEXT STRUCTURING
# ================================
def demo5():
    docs = retrieve(QUERY, top_k=3)

    context = ""
    for i, doc in enumerate(docs, 1):
        context += f"[Source {i}]: {doc}\n"

    prompt = f"""
Use the sources below to answer.

{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 5: STRUCTURED CONTEXT ---")
    print("\nStructured Context:\n", context)
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 6: PROMPT IMPROVEMENT
# ================================
def demo6():
    docs = retrieve(QUERY, top_k=3)

    context = "\n".join(docs)

    prompt = f"""
You are an expert assistant.

Instructions:
- Answer ONLY using context
- Be structured
- Cite evidence
- If unsure, say "I don't know"

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 6: PROMPT ENGINEERING ---")
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 7: HALLUCINATION (BAD PROMPT)
# ================================
def demo7():
    docs = retrieve(QUERY, top_k=1)

    context = "\n".join(docs)

    prompt = f"""
Answer the question (you can use general knowledge also).

Context:
{context}

Question: {QUERY}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    print("\n--- DEMO 7: HALLUCINATION ---")
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# DEMO 8: NO DATA FOUND
# ================================
def demo8():
    query = "What quantum computing systems does Walmart use?"

    docs = retrieve(query, top_k=2)

    context = "\n".join(docs)

    prompt = f"""
Answer ONLY from the context.
If not found, say "I don't know".

Context:
{context}

Question: {query}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    print("\n--- DEMO 8: NO DATA ---")
    print("\nQuery:", query)
    print("\nDocs:", docs)
    print("\nAnswer:", response.choices[0].message.content)


# ================================
# RUN ALL DEMOS
# ================================
if __name__ == "__main__":
    index_documents()

    print("Query:", QUERY)
    demo1()
    demo2()
    demo3()
    demo4()
    demo4("Noise")
    demo5()
    demo6()
    demo7()
    demo8()
