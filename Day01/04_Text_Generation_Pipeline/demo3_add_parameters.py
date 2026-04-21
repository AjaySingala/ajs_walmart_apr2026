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
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def __call__(self, prompt: str, temperature=0.2, max_tokens=200):
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        return response.output_text


generator = TextGenerationPipeline()

print(generator(
    "Generate a product description for a smart refrigerator.",
    temperature=0.9
))
# Notice the truncation.
