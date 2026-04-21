# Demo 3: Query Reformulation.
from demo1_baseline import vectorstore, get_llm, prompt, format_docs

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Reformulate query before retrieval
def rewrite_query(question):
    llm = get_llm()

    rewrite_prompt = f"Rewrite this question to better match policy documents:\n{question}"
    new_prompt = llm.invoke(rewrite_prompt).content
    print(f"\n Rewritten prompt: {new_prompt} \n")

    return new_prompt

def run_query(question):
    rewritten = rewrite_query(question)  # improve query

    docs = retriever.invoke(rewritten)

    if not docs:
        return "No relevant information found."

    context = format_docs(docs)

    llm = get_llm()
    response = llm.invoke(prompt.format(context=context, question=question))

    return response.content

query = "Tell me about company culture"
# query = "Explain office culture philosophy"

if __name__ == "__main__":
    print(f"\nQuery: {query}\n")
    print(run_query(query))
