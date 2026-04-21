from openai import OpenAI

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
class TextGenerationPipeline:
    def __init__(self, model="gpt-4o-mini", max_tokens=200):
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, prompt: str):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_tokens
        )
        return response.output_text


# Create pipeline
generator = TextGenerationPipeline()

# Use like HF pipeline
result = generator("Explain demand forecasting in retail in 3 lines.")
print(result)
