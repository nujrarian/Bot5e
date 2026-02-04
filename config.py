"""Configuration management for Bot5e application."""
import os
import yaml
from typing import Dict, Any


class Config:
    """Singleton configuration manager."""

    _instance = None
    _config: Dict[str, Any] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from config.yaml file."""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the config value (e.g., 'llm.model')
            default: Default value if key is not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self._config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    @property
    def llm_model(self) -> str:
        """Get LLM model name."""
        return self.get('llm.model', 'llama3.1')

    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature."""
        return self.get('llm.temperature', 0.7)

    @property
    def ollama_base_url(self) -> str:
        """Get Ollama base URL."""
        return self.get('llm.ollama_base_url', 'http://localhost:11434')

    @property
    def pdf_path(self) -> str:
        """Get PDF path."""
        return self.get('documents.pdf_path', 'SRD-OGL_V5.1.pdf')

    @property
    def embeddings_path(self) -> str:
        """Get embeddings path."""
        return self.get('documents.embeddings_path', 'embeddings.pkl')

    @property
    def index_path(self) -> str:
        """Get FAISS index path."""
        return self.get('documents.index_path', 'index.faiss')

    @property
    def chunk_size(self) -> int:
        """Get text chunk size."""
        return self.get('text_processing.chunk_size', 1000)

    @property
    def chunk_overlap(self) -> int:
        """Get text chunk overlap."""
        return self.get('text_processing.chunk_overlap', 200)

    @property
    def embedding_model(self) -> str:
        """Get embedding model name."""
        return self.get('embeddings.model', 'all-MiniLM-L6-v2')

    @property
    def top_k(self) -> int:
        """Get number of top chunks to retrieve."""
        return self.get('vector_store.top_k', 7)

    @property
    def retrieval_mode(self) -> str:
        """Get retrieval mode: baseline, hybrid, or rerank."""
        return self.get('vector_store.retrieval_mode', 'baseline')

    @property
    def reranker_model(self) -> str:
        """Get reranker model name."""
        return self.get('vector_store.reranker_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')

    @property
    def classifier_model(self) -> str:
        """Get classifier model name."""
        return self.get('classifier.model', 'facebook/bart-large-mnli')

    @property
    def confidence_threshold(self) -> float:
        """Get classification confidence threshold."""
        return self.get('classifier.confidence_threshold', 0.6)

    @property
    def classifier_labels(self) -> list:
        """Get classifier labels."""
        return self.get('classifier.labels', [])

    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.get('logging.level', 'INFO')

    @property
    def log_format(self) -> str:
        """Get logging format."""
        return self.get('logging.format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    @property
    def log_file(self) -> str:
        """Get log file path."""
        return self.get('logging.file', 'bot5e.log')

    @property
    def page_title(self) -> str:
        """Get UI page title."""
        return self.get('ui.page_title', 'Bot5e - D&D 5e Assistant')

    @property
    def page_icon(self) -> str:
        """Get UI page icon."""
        return self.get('ui.page_icon', '🐉')

    @property
    def show_agent_by_default(self) -> bool:
        """Get default value for showing agent."""
        return self.get('ui.show_agent_by_default', True)

    @property
    def show_context_by_default(self) -> bool:
        """Get default value for showing context."""
        return self.get('ui.show_context_by_default', False)


# Global config instance
config = Config()
