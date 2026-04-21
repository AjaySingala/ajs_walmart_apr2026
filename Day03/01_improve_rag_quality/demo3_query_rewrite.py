from rag_core import index_documents, retrieve, generate_answer
from openai import OpenAI
import os

client = OpenAI()

index_documents()

query = "How does Walmart use AI?"

def rewrite_query(query):
    """Generate expanded query for better retrieval"""
    prompt = f"""
Rewrite the query to improve retrieval in a knowledge base.
Add relevant keywords.

Original query: {query}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content.strip()


# Reformulate query
better_query = rewrite_query(query)

docs = retrieve(better_query, top_k=3)

context = "\n".join(docs)

answer = generate_answer(context, query)

print("Original Query:", query)
print("\nRewritten Query:", better_query)
print("\nAnswer:", answer)
