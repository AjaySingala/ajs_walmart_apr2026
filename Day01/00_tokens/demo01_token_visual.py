# Visualize tokenization & attention

# Set env vars from config.py.
import sys
import os

# Add the folder path (use absolute or relative path)
folder_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.insert(0, folder_path)

import config

# Start.

from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# STEP 1: Tokenization.
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")

text = "Generative AI is transforming enterprise decision making."

tokens = encoding.encode(text)
decoded_tokens = [encoding.decode([t]) for t in tokens]

print("Original Text:\n", text)
print("\nToken IDs:\n", tokens)
print("\nDecoded Tokens:\n", decoded_tokens)

# STEP 2: Ask model how it interprets relationships (attention intuition).
response = client.chat.completions.create(
    model="gpt-4o-mini",  # your deployment name
    messages=[
        {"role": "system", "content": "Explain how words in a sentence relate to each other."},
        {"role": "user", "content": "AI transforms business decision making."}
    ],
    temperature=0
)

print(response.choices[0].message.content)

# STEP 3: Visual attention heatmap.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

tokens = ["AI", "transforms", "business", "decision"]

np.random.seed(42)
embeddings = np.random.rand(len(tokens), 4)

attention_scores = embeddings @ embeddings.T

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

attention_weights = softmax(attention_scores)

plt.figure(figsize=(6, 5))
sns.heatmap(
    attention_weights,
    annot=True,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="Blues"
)

plt.title("Attention Mechanism (Conceptual Visualization)")
plt.xlabel("Words being attended to")
plt.ylabel("Current word")
plt.show()

# STEP 4: Token count → cost awareness.
prompt = "Explain how AI helps in fraud detection in banking."

tokens = encoding.encode(prompt)

print("Prompt:", prompt)
print("Token count:", len(tokens))
for t in tokens:
    print(f"{t} -> '{encoding.decode([t])}'")
