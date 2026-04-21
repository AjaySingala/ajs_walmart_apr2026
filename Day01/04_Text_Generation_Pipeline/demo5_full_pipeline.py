"""
GENAI PIPELINE DEMO (OpenAI)

Covers:
1. Text Generation Pipeline
2. Embedding Pipeline
3. Input/Output Handling
4. Model Parameters (high-level abstraction)
"""

from openai import OpenAI
from typing import List, Dict, Any

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
# ================================
# 1. BASE PIPELINE CLASS
# ================================
class BasePipeline:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model


# ================================
# 2. TEXT GENERATION PIPELINE
# ================================
class TextGenerationPipeline(BasePipeline):
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 200
    ):
        super().__init__(model)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        return response.output_text


# ================================
# 3. EMBEDDING PIPELINE
# ================================
class EmbeddingPipeline(BasePipeline):
    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__(model)

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]


# ================================
# 4. INPUT / OUTPUT HANDLER
# ================================
class IOHandler:
    @staticmethod
    def format_prompt(task: str, data: str) -> str:
        return f"""
        Task: {task}

        Input Data:
        {data}

        Provide a clear and concise response.
        """

    @staticmethod
    def format_output(result: str) -> Dict[str, Any]:
        return {
            "status": "success",
            "output": result.strip()
        }


# ================================
# 5. UNIFIED GENAI PIPELINE
# ================================
class GenAIPipeline:
    def __init__(self):
        self.text_pipeline = TextGenerationPipeline()
        self.embedding_pipeline = EmbeddingPipeline()
        self.io_handler = IOHandler()

    def run_text_task(self, task: str, data: str) -> Dict[str, Any]:
        prompt = self.io_handler.format_prompt(task, data)
        result = self.text_pipeline.generate(prompt)
        return self.io_handler.format_output(result)

    def run_embedding_task(self, texts: List[str]) -> Dict[str, Any]:
        embeddings = self.embedding_pipeline.embed(texts)
        return {
            "status": "success",
            "embeddings_count": len(embeddings),
            "vector_length": len(embeddings[0]) if embeddings else 0
        }


# ================================
# 6. DEMO EXECUTION
# ================================
if __name__ == "__main__":
    pipeline = GenAIPipeline()

    # ---- TEXT GENERATION DEMO ----
    print("\n=== TEXT GENERATION ===")
    task = "Analyze retail sales trends"
    data = "Electronics +15%, Apparel -5%, Grocery +8%"

    text_result = pipeline.run_text_task(task, data)
    print(text_result)


    # ---- EMBEDDING DEMO ----
    print("\n=== EMBEDDINGS ===")
    texts = [
        "Walmart uses AI for inventory optimization.",
        "Demand forecasting improves supply chain efficiency."
    ]

    embedding_result = pipeline.run_embedding_task(texts)
    print(embedding_result)
