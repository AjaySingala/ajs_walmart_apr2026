from rag_core import index_documents, retrieve
from openai import OpenAI
import os

client = OpenAI()

index_documents()

query = "How does Walmart use AI?"

docs = retrieve(query, top_k=3)

context = "\n".join(docs)

def generate_better_answer(context, query):
    """Improved prompt for structured and accurate answers"""
    prompt = f"""
You are an AI assistant for Walmart.

Instructions:
- Answer ONLY using the provided context
- Be concise and structured
- If answer is missing, say "I don't know"
- Cite relevant parts

Context:
{context}

Question:
{query}

Answer:
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


answer = generate_better_answer(context, query)


print("Query:", query)
print("\nAnswer:", answer)
