# Demo 4: Over-Retrieval / Noisy Context.
# Top-k Tuning + Context Filtering.

from demo1_baseline import get_llm, prompt, get_embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma

# New dataset for this demo only to add noise.
documents = [
    Document(
        page_content="Employees are entitled to 20 days of paid leave annually.",
        metadata={"source": "LeavePolicy"}
    ),
    Document(
        page_content="Travel reimbursement is allowed only for approved business trips.",
        metadata={"source": "TravelPolicy"}
    ),
    Document(
        page_content="Employees must submit timesheets weekly for project tracking.",
        metadata={"source": "OperationsPolicy"}
    ),
    Document(
        page_content="Performance reviews are conducted annually based on employee goals.",
        metadata={"source": "HRPolicy"}
    ),
    Document(
        page_content="Leave requests must be approved by the reporting manager in advance.",
        metadata={"source": "LeavePolicy"}
    ),
    Document(
        page_content="Employees are encouraged to maintain work-life balance through company initiatives.",
        metadata={"source": "WellnessPolicy"}
    ),
    Document(
        page_content="Employees must follow leave approval workflows for all system access requests.",
        metadata={"source": "ITPolicy"}
    ),
    Document(
        page_content="Leave balance dashboards are available in the HR system for reporting purposes.",
        metadata={"source": "AnalyticsPolicy"}
    ),
    Document(
        page_content="Managers are responsible for approving project timelines and resource allocation.",
        metadata={"source": "ProjectPolicy"}
    ),
    Document(
        page_content="Employees should maintain work-life balance through company wellness initiatives.",
        metadata={"source": "WellnessPolicy"}
    )
]

# Force a fresh collection name.
# Every time you create vectorstore, use a unique collection name.
import uuid  # to generate unique collection names

# Create a new vectorstore with a unique name.
collection_name = f"demo_{uuid.uuid4()}"
vectorstore = Chroma(
    collection_name=collection_name,    # ensures clean slate
    embedding_function=get_embeddings()
)

# Delete old data
vectorstore.delete_collection()

# Recreate/Refresh with new data.
vectorstore = Chroma.from_documents(
    documents,
    embedding=get_embeddings(),
    collection_name=collection_name  # ensures clean slate
)
print("Collection created fresh!")

# Reduce retrieved docs
# Force Failure:
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# # Top-k Tuning.
# retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
# Metadata filter
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"source": "LeavePolicy"}
    }
)

# Filter only relevant docs manually
def filter_docs(docs, question):
    print("\n filter_docs()...")
    for d in docs:
        print(f"\nDocument: {d.page_content}")
        print(f"question in doc?: {question.lower() in d.page_content}")

    return [doc for doc in docs if question.lower() in doc.page_content.lower()]

def format_docs(docs):
    print("\n format_docs()...")
    for d in docs:
        print(f"\nDocument: {d.page_content} (source: {d.metadata})")
    return "\n\n".join(doc.page_content for doc in docs)

def run_query(question):
    print("\n run_query()...")
    docs = retriever.invoke(question)

    # filtered_docs = filter_docs(docs, question)  # remove noise

    # if not filtered_docs:
    #     return "No relevant information found."

    # context = format_docs(filtered_docs)
    context = format_docs(docs)
    print(f"\nFormatted Docs (context): {context}")

    print("\n Generating response...")
    llm = get_llm()
    response = llm.invoke(prompt.format(context=context, question=question))

    return response.content

query = "What is the leave policy?"
# query = "What is travel reimbursement policy?"

if __name__ == "__main__":
    print(f"\nQuery: {query}\n")
    print(run_query(query))
