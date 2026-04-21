# Common Setup (Used in ALL demos)
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
# Initialize OpenAI client
client = OpenAI()

# Initialize ChromaDB (persistent)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create collection
collection = chroma_client.get_or_create_collection(name="rag_demo")


def embed_text(text):
    """Generate embedding for a given text"""
    response = client.embeddings.create(
        model=os.getenv("TEXT_EMBEDDING_MODEL"),
        input=text
    )
    return response.data[0].embedding


def index_documents():
    """Index richer, overlapping documents to better demonstrate RAG improvements"""

    docs = [
        {
            "id": "1",
            "text": """Walmart uses artificial intelligence to optimize inventory management across stores and warehouses.
            Machine learning models analyze historical sales, regional demand patterns, and real-time purchasing behavior.
            This helps reduce stockouts and overstock situations, ensuring products are available when customers need them.""",
            "category": "AI"
        },
        {
            "id": "2",
            "text": """Demand forecasting is critical in retail supply chains. Walmart applies predictive analytics to forecast
            customer demand using factors like seasonality, promotions, and external events. Accurate forecasting improves
            replenishment planning and reduces logistics costs.""",
            "category": "SupplyChain"
        },
        {
            "id": "3",
            "text": """Walmart leverages generative AI in customer support to automate responses for common queries.
            AI-powered chatbots can resolve issues related to orders, returns, and product information.
            This reduces response time and improves customer satisfaction while lowering operational costs.""",
            "category": "AI"
        },
        {
            "id": "4",
            "text": """Retail sales are heavily influenced by seasonal trends such as holidays, weather changes, and regional events.
            Walmart analyzes these patterns to adjust inventory and marketing strategies.
            For example, winter clothing demand increases in colder regions, while festive seasons drive higher sales overall.""",
            "category": "Retail"
        },
        {
            "id": "5",
            "text": """Walmart's supply chain integrates AI-driven demand forecasting with inventory optimization systems.
            By combining predictive analytics with real-time data, Walmart improves distribution efficiency.
            This enables faster restocking and reduces delays across its logistics network.""",
            "category": "SupplyChain"
        },
        {
            "id": "6",
            "text": """AI in retail is not limited to inventory. Walmart also uses machine learning for pricing optimization,
            fraud detection, and personalized recommendations.
            These applications enhance both operational efficiency and customer experience.""",
            "category": "AI"
        }
    ]

    for doc in docs:
        collection.upsert(
            ids=[doc["id"]],
            documents=[doc["text"]],
            embeddings=[embed_text(doc["text"])],
            metadatas=[{"category": doc["category"]}]
        )

def retrieve(query, top_k=2, where=None):
    """Retrieve relevant documents from ChromaDB"""
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where  # optional metadata filtering
    )

    return results["documents"][0] if results["documents"] else []


def generate_answer(context, query):
    """Strict RAG: answer ONLY from context"""
    prompt = f"""
Answer ONLY from the context below.
If the answer is not present, say "I don't know."

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content
