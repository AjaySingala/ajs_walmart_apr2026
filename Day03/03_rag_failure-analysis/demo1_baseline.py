# Demo 1: Baseline RAG (Working System)
# Goal: Establish a working pipeline before breaking it.

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from common_setup import get_llm, get_embeddings

# Sample documents (simulating policies)
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
        page_content="""
The company provides employees with a leave policy.
Employees are eligible for annual leave and other types of leave as per company guidelines.
""",
        metadata={"source": "LeavePolicy"}
    ),
    Document(
        page_content="""
The organization supports employee well-being through various benefits and HR policies.
These include leave programs and employee support initiatives.
""",
        metadata={"source": "HRPolicy"}
    ),
    Document(
        page_content="""
Employees are eligible for structured leave programs.
These programs are designed to support employee needs across different situations.
""",
        metadata={"source": "LeavePolicy"}
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

# Basic retriever (top 2 results)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Not so strict prompt.
# Comment next prompt to test this.
prompt = ChatPromptTemplate.from_template("""
You are an HR expert. Answer confidently.

Context:
{context}

Question:
{question}
""")

# # Prompt enforces strict RAG behavior.
# # Uncomment previous prompt to test this.
# prompt = ChatPromptTemplate.from_template("""
# You are an HR expert.
# Answer ONLY from the context below.
# If answer is not present, say "I don't know".

# Context:
# {context}

# Question:
# {question}
# """)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Run a query through RAG pipeline
def run_query(question):
    docs = retriever.invoke(question)  # retrieve relevant docs
    context = format_docs(docs)

    llm = get_llm()
    response = llm.invoke(prompt.format(context=context, question=question))

    return response.content

query = "How many leave days do employees get?"
# query = "What is the leave policy of Walmart"
# query = "What is the maternity leave policy?"
# query = "What is the maternity leave duration?"
# query =  "What benefits are included in the leave policy?"

if __name__ == "__main__":
    print(f"\nQuery: {query}\n")
    print(run_query(query))
