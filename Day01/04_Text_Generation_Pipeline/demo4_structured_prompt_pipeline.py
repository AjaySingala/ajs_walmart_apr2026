from openai import OpenAI

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.
class RetailInsightPipeline:
    def __init__(self):
        self.client = OpenAI()

    def __call__(self, sales_data: str):
        prompt = f"""
        You are a retail analyst.

        Analyze the following sales data and provide:
        1. Key insights
        2. Risks
        3. Recommendations

        Data:
        {sales_data}
        """

        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            temperature=0.3
        )

        return response.output_text


pipeline = RetailInsightPipeline()

data = "Electronics +15%, Apparel -5%, Grocery +8%"
print(pipeline(data))
