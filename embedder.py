from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(text_chunks):
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return embeddings
