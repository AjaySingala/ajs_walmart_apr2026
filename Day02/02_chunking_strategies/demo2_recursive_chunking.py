# Recursive chunking based on separators (paragraph → sentence → word)

def recursive_chunk(text, max_chunk_size=100):
    """Recursively split text using logical separators"""
    
    separators = ["\n\n", "\n", ".", " "]

    def split_text(text, separators):
        # Base condition
        if len(text) <= max_chunk_size:
            return [text.strip()]
        
        if not separators:
            return [text[:max_chunk_size]]
        
        sep = separators[0]
        parts = text.split(sep)
        
        chunks = []
        current = ""

        for part in parts:
            temp = current + part + sep
            
            if len(temp) <= max_chunk_size:
                current = temp
            else:
                if current:
                    chunks.append(current.strip())
                current = part + sep
        
        if current:
            chunks.append(current.strip())
        
        # If chunks still too big → recurse
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > max_chunk_size:
                final_chunks.extend(split_text(chunk, separators[1:]))
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    return split_text(text, separators)


if __name__ == "__main__":
    document = """Walmart uses AI to optimize inventory management.
    Machine learning helps forecast demand accurately.
    Supply chain efficiency improves with predictive analytics.
    Customer satisfaction increases due to better product availability."""

    chunks = recursive_chunk(document, max_chunk_size=80)

    print("Recursive Chunks:\n")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}\n")
        