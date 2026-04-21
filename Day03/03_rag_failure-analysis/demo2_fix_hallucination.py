# Add Relevance Threshold + Better Prompt
from demo1_baseline import vectorstore, get_llm, prompt, format_docs

# Use similarity score threshold to filter weak matches
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5, "k": 2}
)

def run_query(question):
    print(f"\n run_query()...")

    print(f"\nRetrieving docs with score...")
    docs_with_scores = vectorstore.similarity_search_with_relevance_scores(
        question, k=2   #, score_threshold=1.0  # Matches your retriever params
    )

    # Print documents and scores
    for doc, score in docs_with_scores:
        print(f"Score: {score:.3f}")
        print(f"Document: {doc.page_content[:200]}...")  # First 200 chars
        print("-" * 50)   

    print(f"\nRetrieving docs with score threshold 0.5...")
    docs = retriever.invoke(question)
    for doc in docs:
        print(f"Document: {doc.page_content[:200]}...")  # First 200 chars
        print("-" * 50)   

    # If no relevant docs found → fail fast
    if not docs:
        return "No relevant information found in knowledge base."

    context = format_docs(docs)

    llm = get_llm()
    response = llm.invoke(prompt.format(context=context, question=question))

    return response.content

query = "What is the maternity leave policy?"
# query = "How many leave days do employees get?"
# # The next query is for Demo3: 
# query = "Explain office culture philosophy"

if __name__ == "__main__":
    print(f"\nQuery: {query}\n")
    print(run_query(query))
