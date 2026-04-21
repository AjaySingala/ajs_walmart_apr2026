# Overlapping chunks preserve context across boundaries

def overlapping_chunk(text, chunk_size=100, overlap=20):
    """Create chunks with overlap to preserve context"""
    
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)  # move with overlap
    
    return chunks


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    chunks = overlapping_chunk(document, chunk_size=80, overlap=20)

    print("Overlapping Chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}\n")
        