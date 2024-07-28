import os
import pickle
import faiss
from langchain_ollama.llms import OllamaLLM
from vector_store import query_vector_store
from sentence_transformers import SentenceTransformer
from pdf_reader import extract_text_from_pdf
from text_splitter import split_text_into_chunks
from embedder import generate_embeddings
from vector_store import create_vector_store

class ChatbotAgent:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.1")
        self.template = """{history}
        Question: {question}
        Answer: Respond in an appropriate and friendly manner."""

    def handle_query(self, question, history):
        formatted_prompt = self.template.format(history=history, question=question)
        return self.model.invoke(formatted_prompt)

class PDFQAAgent:
    def __init__(self, pdf_path, embeddings_path='embeddings.pkl', index_path='index.faiss'):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings_path = embeddings_path
        self.index_path = index_path
        
        if os.path.exists(self.embeddings_path) and os.path.exists(self.index_path):
            with open(self.embeddings_path, 'rb') as f:
                self.chunks, self.embeddings = pickle.load(f)
            self.index = faiss.read_index(self.index_path)
        else:
            pdf_text = extract_text_from_pdf(pdf_path)
            self.chunks = split_text_into_chunks(pdf_text)
            self.embeddings = generate_embeddings(self.chunks)
            self.index = create_vector_store(self.embeddings)
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump((self.chunks, self.embeddings), f)
            faiss.write_index(self.index, self.index_path)
        self.template = """You are a knowledgeable assistant who provides accurate answers based on the given context from the document Do not generate any outside information. Stick to the context exactly.

        {context}
        {history}
        Question: {question}
        Answer:"""

    def handle_query(self, question, history):
        relevant_chunks = query_vector_store(question, self.embedding_model, self.index, self.chunks)
        context = " ".join(relevant_chunks)
        print(context)
        formatted_prompt = self.template.format(context=context, question=question, history=history)
        model = OllamaLLM(model="llama3.1")
        return model.invoke(formatted_prompt)
