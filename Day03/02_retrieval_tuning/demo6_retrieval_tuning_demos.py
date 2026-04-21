# Demo 6:
# -Baseline RAG (Top-k = 4, no tuning)
# -Tune Top-k.
# -Measure Relevance (Scores).
#   - Lower score = more relevant (Chroma distance)
# - Recall vs Precision (Threshold Filtering).
#   - Threshold ↓ → higher precision
#   - Threshold ↑ → higher recall
# - Context Window Optimization.
import time
from common_setup import vectorstore, llm, load_data

# Load documents (only once)
load_data()


def print_divider(title):
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)


# --------------------------------------------------
# Demo 1 — Baseline RAG (No tuning)
# --------------------------------------------------
def demo_1_baseline():
    print_divider("Demo 1 — Baseline RAG (Top-k = 4)")

    query = "What is the leave policy?"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer the question based on the context below:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\nRetrieved Docs:")
    for d in docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 2 — Tune Top-k
# --------------------------------------------------
def demo_2_topk():
    print_divider("Demo 2 — Top-k Tuning (k = 2)")

    query = "What is the leave policy?"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\nRetrieved Docs:")
    for d in docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 3 — Measure Relevance (Scores)
# --------------------------------------------------
def demo_3_scores():
    print_divider("Demo 3 — Relevance Scores")

    query = "What is the leave policy?"

    results = vectorstore.similarity_search_with_score(query, k=4)

    print(f"\nQuery: {query}")
    print("\nRetrieved Docs with Scores:")
    for doc, score in results:
        print(f"Score: {round(score, 4)} | {doc.page_content}")


# --------------------------------------------------
# Demo 4 — Recall vs Precision (Threshold)
# --------------------------------------------------
def demo_4_threshold():
    print_divider("Demo 4 — Threshold Filtering")

    query = "What is the leave policy?"

    results = vectorstore.similarity_search_with_score(query, k=5)

    original_context = ""
    for doc, score in results:
        original_context += doc.page_content + f" (Score: {score})" + "\n"
        
    threshold = 0.5  # Tune this live
    # threshold = 0.95  # Tune this live.

    filtered_docs = [doc for doc, score in results if score < threshold]

    context = "\n".join([doc.page_content for doc in filtered_docs])

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\n--- Original Context ---")
    print(original_context)

    print("\nFiltered Docs:")
    for d in filtered_docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 5 — Context Window Optimization
# --------------------------------------------------
def demo_5_context_limit():
    print_divider("Demo 5 — Context Window Optimization")

    query = "Explain leave rules"

    results = vectorstore.similarity_search_with_score(query, k=5)
    original_docs = ""
    for doc, score in results:
        original_docs += doc.page_content + f" (Score: {score})" + "\n"
        
    # Sort by relevance (lower score = better)
    sorted_docs = sorted(results, key=lambda x: x[1])

    # max_chars = 150 # Tune this.
    max_chars = 50 # Tune this.
    context = ""

    for doc, score in sorted_docs:
        if len(context) + len(doc.page_content) < max_chars:
            context += doc.page_content + "\n"

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\n--- Original Docs ---")
    print(original_docs)

    print("\nOptimized Context:")
    print(context)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Demo 6 — Avoid Over-injection (Metadata Filtering)
# --------------------------------------------------
def demo_6_metadata_filter():
    print_divider("Demo 6 — Metadata Filtering")

    query = "What is the leave policy?"

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"source": "LeavePolicy"}
        }
    )

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
    Answer ONLY from the context:

    {context}

    Question: {query}
    """

    response = llm.invoke(prompt)

    print(f"\nQuery: {query}")
    print("\nFiltered Docs:")
    for d in docs:
        print("-", d.page_content)

    print("\nAnswer:")
    print(response.content)


# --------------------------------------------------
# Run all demos sequentially
# --------------------------------------------------
if __name__ == "__main__":
    demo_1_baseline()
    time.sleep(2)

    demo_2_topk()
    time.sleep(2)

    demo_3_scores()
    time.sleep(2)

    demo_4_threshold()
    time.sleep(2)

    demo_5_context_limit()
    time.sleep(2)

    demo_6_metadata_filter()
