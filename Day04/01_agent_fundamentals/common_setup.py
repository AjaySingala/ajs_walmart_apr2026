import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
def get_llm():
    """Create LLM instance using env-configured model"""
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        temperature=0
    )

def get_embeddings():
    """Create embedding model from env"""
    return OpenAIEmbeddings(
        model=os.getenv("TEXT_EMBEDDING_MODEL")
    )

# Rich enterprise-like documents (intentionally includes noise + overlaps)
documents = [
    # Leave Policy
    "Employees are entitled to 20 days of paid leave annually.",
    "Leave requests must be submitted at least 2 weeks in advance.",
    "Unused leave cannot be carried forward beyond 30 days.",
    "Sick leave requires medical documentation if more than 3 days.",

    # Travel Policy
    "Travel expenses are reimbursed only for approved business trips.",
    "Employees must use economy class for domestic travel.",
    "Hotel bookings should not exceed the approved budget per city.",
    "Taxi expenses are reimbursable only with receipts.",

    # Expense Policy
    "Meal expenses are reimbursable up to $50 per day.",
    "Entertainment expenses require manager approval.",
    "Office supplies must be procured through approved vendors.",

    # HR / General Policy
    "Employees must adhere to company code of conduct at all times.",
    "Work from home is allowed up to 2 days per week.",
    "All employees must complete mandatory compliance training annually.",

    # Noisy / Irrelevant Content (for failure demos)
    "The company cafeteria menu changes every week.",
    "Parking slots are allocated on a first-come-first-serve basis.",
    "The annual sports event is held in December.",
    "Office WiFi passwords are rotated every 30 days."
]
