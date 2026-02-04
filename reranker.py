"""
Cross-encoder reranking.

After initial retrieval (FAISS, BM25, or hybrid), a cross-encoder
scores each (query, chunk) pair jointly — much more accurate than
the bi-encoder used for initial retrieval, but too slow to run
over the full corpus.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~23M params, runs locally, no API needed
  - Trained on MS MARCO passage ranking
  - Returns relevance scores (higher = more relevant)
"""

from sentence_transformers import CrossEncoder
from config import config
from logger import setup_logger

logger = setup_logger(__name__)

def load_reranker() -> CrossEncoder:
    """Load the cross-encoder model."""
    logger.info(f"Loading reranker model: {config.reranker_model}")
    return CrossEncoder(config.reranker_model)


def rerank_chunks(
    query: str,
    candidates: list[str],
    reranker: CrossEncoder,
    top_k: int = 7,
) -> list[str]:
    """
    Rerank candidate chunks using the cross-encoder.

    Args:
        query: user question
        candidates: list of candidate chunk strings from initial retrieval
        reranker: loaded CrossEncoder model
        top_k: number of top chunks to return after reranking

    Returns:
        Top-k chunks sorted by cross-encoder relevance score (highest first)
    """
    if not candidates:
        return []

    # Score each (query, candidate) pair
    pairs = [(query, chunk) for chunk in candidates]
    scores = reranker.predict(pairs)

    logger.debug(f"Reranker scored {len(candidates)} candidates. "
                 f"Top score: {max(scores):.3f}, Min score: {min(scores):.3f}")

    # Sort by score descending, return top_k
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]
