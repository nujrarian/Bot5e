def split_text_into_chunks(text, chunk_size=2000, overlap=200):
    sentences = text.split('.')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + '.'
        else:
            chunks.append(current_chunk)
            current_chunk = sentence + '.'
    chunks.append(current_chunk)
    return chunks
