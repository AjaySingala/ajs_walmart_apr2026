def generate_answer_with_structure(query, context_chunks):
    """Structured prompting for higher-quality answers"""

    # Handle both formats: [text] OR [(text, score)]
    processed_chunks = [
        chunk if isinstance(chunk, str) else chunk[0]
        for chunk in context_chunks
    ]

    context = "\n\n".join(processed_chunks)

    prompt = f"""
    You are an enterprise AI assistant.

    Instructions:
    - Answer ONLY from the context
    - Be concise and structured
    - If not found, say "Not available in context"

    Context:
    {context}

    Question:
    {query}

    Provide:
    - Key Answer
    - Supporting Points
    """

    from common_setup import client, MODEL_NAME

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content
