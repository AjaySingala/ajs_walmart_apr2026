# demo4_llm_wrapper.py

from openai import OpenAI
from dotenv import load_dotenv

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.

class LLMWrapper:
    # Wrapper class to standardize all LLM interactions
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str = "",
        temperature: float = 0.7,
        max_tokens: int = 300,
        top_p: float = 1.0
    ) -> str:
        """
        Main method to call LLM with structured inputs and parameters
        """

        # Combine context with user prompt if provided
        if context:
            user_prompt = f"""
            Context:
            {context}

            Question:
            {user_prompt}
            """

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,  # Controls randomness
            max_tokens=max_tokens,    # Controls response length
            top_p=top_p               # Controls diversity (nucleus sampling)
        )

        return response.choices[0].message.content


# Example usage
if __name__ == "__main__":
    temperatures = [0.2, 0.5, 0.9]
    top_ps = [0.5, 0.9, 1.0]
    i = 0

    for i in range(3):
        llm = LLMWrapper()

        response = llm.generate(
            system_prompt="You are a Walmart business analyst.",
            user_prompt="Summarize the sales trend.",
            # user_prompt="Give a creative explanation of the sales trend with insights and future predictions.",
            context="Sales grew 15% in electronics, declined 5% in apparel.",
            temperature=temperatures[i],
            max_tokens=200,
            top_p=top_ps[i]
        )

        print("\n" + "="*50)
        print(f"Response with temperature={temperatures[i]} and top_p={top_ps[i]}:")
        print(f"{'-'*50}")
        print("LLM Output:", response)
