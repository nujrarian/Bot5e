import os
import pickle
import faiss
from langchain_ollama.llms import OllamaLLM
from vector_store import query_vector_store
from sentence_transformers import SentenceTransformer
from text_read_split import pdf_read_split
from embedder import generate_embeddings
from vector_store import create_vector_store
from formatter import format_dnd_response
from config import config
from logger import setup_logger

logger = setup_logger(__name__)

class ChatbotAgent:
    def __init__(self):
        logger.info(f"Initializing ChatbotAgent with model: {config.llm_model}")
        self.model = OllamaLLM(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=config.ollama_base_url
        )
        self.template = """{history}
        Question: {question}
        IMPORTANT: You are only for casual greetings and non-D&D conversation. If the user asks ANYTHING about D&D rules, mechanics, classes, spells, or gameplay, respond with: "That's a great D&D question! Let me search the rulebook for you..." and nothing else.

        For casual greetings like "hi", "hello", "how are you", respond friendly and appropriately.

        Answer: """

    def handle_query(self, question, history):
        logger.debug(f"ChatbotAgent handling query: {question[:50]}...")
        try:
            formatted_prompt = self.template.format(history=history, question=question)
            response = self.model.invoke(formatted_prompt)
            return response
        except Exception as e:
            logger.error(f"Error in ChatbotAgent.handle_query: {e}")
            return "I'm sorry, I encountered an error processing your request. Please try again."

class PDFQAAgent:
    def __init__(self, pdf_path=None, embeddings_path=None, index_path=None):
        # Use config defaults if not provided
        pdf_path = pdf_path or config.pdf_path
        self.embeddings_path = embeddings_path or config.embeddings_path
        self.index_path = index_path or config.index_path

        logger.info(f"Initializing PDFQAAgent with model: {config.llm_model}")
        logger.info(f"PDF path: {pdf_path}")

        try:
            # Verify PDF exists
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            # Initialize models
            logger.info(f"Loading embedding model: {config.embedding_model}")
            self.embedding_model = SentenceTransformer(config.embedding_model)

            logger.info(f"Initializing LLM: {config.llm_model}")
            self.model = OllamaLLM(
                model=config.llm_model,
                temperature=config.llm_temperature,
                base_url=config.ollama_base_url
            )

            if os.path.exists(self.embeddings_path) and os.path.exists(self.index_path):
                logger.info("Loading cached embeddings and index")
                try:
                    with open(self.embeddings_path, 'rb') as f:
                        self.chunks, self.embeddings = pickle.load(f)
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded {len(self.chunks)} chunks from cache")
                except Exception as e:
                    logger.error(f"Failed to load cached data: {e}")
                    logger.info("Regenerating embeddings and index")
                    self._generate_embeddings(pdf_path)
            else:
                logger.info("Creating new embeddings and index")
                self._generate_embeddings(pdf_path)

        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PDFQAAgent: {e}")
            raise

    def _generate_embeddings(self, pdf_path):
        """Generate and save embeddings from PDF."""
        try:
            self.chunks = pdf_read_split(pdf_path, config.chunk_size, config.chunk_overlap)
            self.embeddings = generate_embeddings(self.chunks)
            self.index = create_vector_store(self.embeddings)
            logger.info("Saving embeddings and index to disk")
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump((self.chunks, self.embeddings), f)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved {len(self.chunks)} chunks to cache")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise

        self.template = """You are a D&D 5th Edition rules expert. Answer ONLY based on the context provided from the official D&D 5e rulebook.

        CRITICAL RULES:
        1. If the context does not contain information to answer the question, you MUST respond with: "I don't have that specific information in the D&D 5e rulebook I have access to. Could you rephrase your question or ask about something else?"
        2. NEVER make up rules, mechanics, or information that isn't in the context
        3. If you're uncertain, say "I'm not completely sure, but based on the rulebook..."
        4. Stick strictly to what's written in the context below
        5. Format the response with proper spaces and tables if necessary before responding to the user.

        Context from D&D 5e Rulebook:
        {context}

        Conversation History:
        {history}

        Question: {question}

        Answer (based ONLY on the context above):"""

    def handle_query(self, question, history):
        logger.debug(f"PDFQAAgent handling query: {question[:50]}...")

        try:
            relevant_chunks = query_vector_store(
                question,
                self.embedding_model,
                self.index,
                self.chunks,
                top_k=config.top_k
            )

            logger.debug(f"Retrieved {len(relevant_chunks)} relevant chunks")
            context = "\n\n---\n\n".join(relevant_chunks)
            formatted_prompt = self.template.format(context=context, question=question, history=history)
            logger.debug(f"Prompt length: {len(formatted_prompt)} characters")

            response = self.model.invoke(formatted_prompt)

            # Enable formatted response
            try:
                formatted_response = format_dnd_response(response)
                return formatted_response
            except Exception as e:
                logger.warning(f"Failed to format response: {e}. Returning raw response.")
                return response

        except Exception as e:
            logger.error(f"Error in PDFQAAgent.handle_query: {e}")
            return "I'm sorry, I encountered an error searching the rulebook. Please try rephrasing your question."