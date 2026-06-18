# Multi-Step Chatbot.

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from common_setup import documents as docs

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
# --------------------------------------------------
# VECTOR STORE
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# --------------------------------------------------
# STEP 1: INTENT DETECTION
# --------------------------------------------------

def detect_intent(query: str):

    prompt = f"""
Classify the query into one category only:

- Leave
- Travel
- Reimbursement
- Calculation
- Other

Query:
{query}

Return category only.
"""

    return llm.invoke(prompt).content.strip()

# --------------------------------------------------
# STEP 2: RETRIEVE
# --------------------------------------------------

def retrieve(query: str):

    docs = retriever.invoke(query)

    context = "\n".join(
        doc.page_content
        for doc in docs
    )

    return context

# --------------------------------------------------
# STEP 3: QUALITY CHECK
# --------------------------------------------------

def context_is_relevant(query, context):

    prompt = f"""
Question:
{query}

Context:
{context}

Does the context contain enough information
to answer the question?

Answer ONLY:
YES
or
NO
"""

    result = llm.invoke(prompt).content.strip().upper()

    return "YES" in result

# --------------------------------------------------
# STEP 4: QUERY REWRITE
# --------------------------------------------------

def rewrite_query(query):

    prompt = f"""
Rewrite this query so that it is easier
for a company policy search engine
to understand.

Query:
{query}

Return only rewritten query.
"""

    return llm.invoke(prompt).content.strip()

# --------------------------------------------------
# STEP 5: GENERATE ANSWER
# --------------------------------------------------

def generate_answer(query, context):

    prompt = f"""
Answer ONLY from the context.

If the answer is not present,
say "I don't know".

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt).content

# --------------------------------------------------
# STEP 6: VALIDATE ANSWER
# --------------------------------------------------

def validate_answer(query, context, answer):

    prompt = f"""
Question:
{query}

Context:
{context}

Answer:
{answer}

Determine which of the following applies:

1. SUPPORTED
   - The answer is directly supported by the context

2. UNSUPPORTED
   - The answer contains facts not present in the context

3. NOT_FOUND
   - The context does not contain enough information
     to answer the question

Return ONLY one word:
SUPPORTED
UNSUPPORTED
NOT_FOUND
"""

    return llm.invoke(prompt).content.strip()

# --------------------------------------------------
# MAIN WORKFLOW - Orchestrator
# --------------------------------------------------

def run_workflow(query):

    print("\n========== STEP 1 - Intent Detection ==========")

    intent = detect_intent(query)

    print("Intent:", intent)

    print("\n========== STEP 2 - Retrieve Context ==========")

    context = retrieve(query)

    print("Retrieved Context:")
    print(context)

    print("\n========== STEP 3 - Check Context Relevance ==========")

    relevant = context_is_relevant(
        query,
        context
    )

    print("Relevant:", relevant)

    # Retry Retrieval
    if not relevant:

        print("\n========== RETRY - Rewrite Query if not Relevant ==========")

        rewritten_query = rewrite_query(query)

        print("Rewritten Query:")
        print(rewritten_query)

        context = retrieve(rewritten_query)

        print("\nNew Context:")
        print(context)

    print("\n========== STEP 4 - Generate Answer ==========")

    answer = generate_answer(
        query,
        context
    )

    print("Generated Answer:")
    print(answer)

    print("\n========== STEP 5 - Validate Answer ==========")

    validation = validate_answer(
        query,
        context,
        answer
    )

    print("Validation:", validation)

    if validation != "SUPPORTED":
        return "I could not find a reliable answer in the available policies."
    return answer

# --------------------------------------------------
# CHATBOT
# --------------------------------------------------

if __name__ == "__main__":

    print("\nBetter Multi-Step Workflow Demo")
    print("Type 'exit' to quit\n")

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            break

        response = run_workflow(query)

        print("\nBot:", response)


# Intent Detection:
# How many leave days do employees get?
# What is the meal reimbursement limit?
# Can I work from home?
# How many days can i work from home?

# Retrieval Retry:
# Tell me about company spending
# What is the annual bonus percentage?
