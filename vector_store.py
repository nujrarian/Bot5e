import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_vector_store(embeddings):
    embeddings_np = np.array([embedding.cpu().numpy() for embedding in embeddings])
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)
    return index

def query_vector_store(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = np.array(query_embedding.cpu().numpy())
    distances, indices = index.search(query_embedding_np, top_k)
    results = [chunks[i] for i in indices[0]]
    return results
