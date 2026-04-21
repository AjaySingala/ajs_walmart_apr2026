# Simple fixed-size chunking (naive approach)

def fixed_chunk(text, chunk_size=100):
    """Split text into fixed-size chunks"""
    chunks = []
    
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    
    return chunks


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    chunks = fixed_chunk(document, chunk_size=80)

    print("Fixed Chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}\n")
        