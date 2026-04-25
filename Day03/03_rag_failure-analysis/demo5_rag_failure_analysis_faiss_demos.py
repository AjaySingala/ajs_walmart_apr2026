import uuid
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# -------------------------------
# COMMON SETUP
# -------------------------------

import sys
import os

folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config


def get_llm():
    print(f"\n get_llm()...")
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0
    )


def get_embeddings():
    print(f"\n get_embedding()...")
    return OpenAIEmbeddings(
        model=os.getenv("TEXT_EMBEDDING_MODEL")
    )


def format_docs(docs):
    print(f"\n format_docs()...")
    context = ""
    for d in docs:
        print(f"\nDocument: {d.page_content.strip()} (source: {d.metadata})")
        context += d.page_content + "\n\n"
    return context


# -------------------------------
# BASE DOCUMENTS
# -------------------------------

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

# =========================================================
# DEMO 1: BASELINE RAG
# =========================================================

def demo1_baseline():
    print("\n===== DEMO 1: BASELINE RAG =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""")

    question = "How many leave days do employees get?"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(prompt.format(context=context, question=question))

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# =========================================================
# DEMO 2: HALLUCINATION
# Always hallucinates.
# =========================================================

def demo2_hallucination():
    print("\n===== DEMO 2: HALLUCINATION =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""
You are an HR expert. Answer confidently.

Context:
{context}

Question:
{question}
""")

    question = "What benefits are included in the leave policy?"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(prompt.format(context=context, question=question))

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# =========================================================
# DEMO 3: LOW RELEVANCE
# Shows empty retrieval
# =========================================================

def demo3_low_relevance():
    print("\n===== DEMO 3: LOW RELEVANCE =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""
Answer ONLY from the context below.
If answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""")

    question = "Explain company culture"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(prompt.format(context=context, question=question))

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# ---------------- FIX: QUERY REWRITE ----------------

def rewrite_query(question):
    print(f"\n rewrite_query()...")
    rewrite_prompt = f"Rewrite this query for HR policy search:\n{question}"
    return get_llm().invoke(rewrite_prompt).content


def demo3_fix():
    print("\n===== DEMO 3 FIX: QUERY REWRITE =====")

    vectorstore = FAISS.from_documents(
        documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    question = "Tell me about company culture"

    rewritten = rewrite_query(question)

    docs = retriever.invoke(rewritten)
    context = format_docs(docs)

    response = get_llm().invoke(f"Context:\n{context}\n\nQuestion:{question}")

    print(f"\n Question: {question}")
    print("\nRewritten Query:", rewritten)
    print("Answer:", response.content)


# =========================================================
# DEMO 4: NOISE
# Clearly shows noisy context failure.
# =========================================================

demo4_documents = [
    Document("Employees are entitled to 20 days of paid leave annually.", metadata={"source": "LeavePolicy"}),
    Document("Leave requests must be approved by the reporting manager.", metadata={"source": "LeavePolicy"}),
    Document("Employees must submit timesheets weekly.", metadata={"source": "OperationsPolicy"}),
    Document("Performance reviews are conducted annually.", metadata={"source": "HRPolicy"}),
    Document("Employees should maintain work-life balance.", metadata={"source": "WellnessPolicy"}),
    Document("Leave balance dashboards are available in HR systems.", metadata={"source": "AnalyticsPolicy"}),
    Document("Managers approve project timelines.", metadata={"source": "ProjectPolicy"}),
]


def demo4_noise():
    print("\n===== DEMO 4: NOISY CONTEXT =====")

    vectorstore = FAISS.from_documents(
        demo4_documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = "Explain the leave policy"

    docs = retriever.invoke(question)
    context = format_docs(docs)

    response = get_llm().invoke(f"Context:\n{context}\n\nQuestion:{question}")

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# ---------------- FIX: MANUAL FILTER ----------------

def demo4_fix():
    print("\n===== DEMO 4 FIX: FILTERED RETRIEVAL =====")

    vectorstore = FAISS.from_documents(
        demo4_documents,
        embedding=get_embeddings()
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    question = "Explain the leave policy"

    docs = retriever.invoke(question)

    # Manual filtering (FAISS workaround)
    docs = [d for d in docs if d.metadata.get("source") == "LeavePolicy"]

    context = format_docs(docs)

    response = get_llm().invoke(f"Context:\n{context}\n\nQuestion:{question}")

    print(f"\n Question: {question}")
    print("\nAnswer:", response.content)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    demo1_baseline()
    demo2_hallucination()
    demo3_low_relevance()
    demo3_fix()

    demo4_noise()
    demo4_fix()
