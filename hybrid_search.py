"""
Hybrid search: combines BM25 (lexical) + FAISS (semantic) using
Reciprocal Rank Fusion (RRF) to merge the two ranked lists.

RRF score: score(doc) = sum( 1 / (k + rank_i(doc)) ) across retrievers
Typically k=60 (Cormack et al., 2009). Higher score = more relevant.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from vector_store import query_vector_store
from config import config
from logger import setup_logger

logger = setup_logger(__name__)

RRF_K = 60  # standard constant for reciprocal rank fusion


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    """Build a BM25 index from text chunks."""
    tokenized = [tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized)


def _rrf_merge(faiss_results: list[str], bm25_results: list[str], chunks: list[str]) -> list[str]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns chunks ordered by combined RRF score (highest first).
    """
    scores: dict[str, float] = {}

    for rank, chunk in enumerate(faiss_results):
        scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (RRF_K + rank + 1)

    for rank, chunk in enumerate(bm25_results):
        scores[chunk] = scores.get(chunk, 0.0) + 1.0 / (RRF_K + rank + 1)

    # Sort by RRF score descending
    sorted_chunks = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return sorted_chunks


def hybrid_retrieve(
    query: str,
    embedding_model,
    faiss_index,
    chunks: list[str],
    bm25_index: BM25Okapi,
    top_k: int = 7,
) -> list[str]:
    """
    Retrieve using both FAISS and BM25, then merge with RRF.

    Args:
        query: user question
        embedding_model: SentenceTransformer instance
        faiss_index: FAISS index
        chunks: original text chunks
        bm25_index: pre-built BM25Okapi index
        top_k: number of final results to return

    Returns:
        List of top_k chunk strings ranked by RRF score
    """
    # Retrieve more candidates from each retriever before merging
    fetch_k = top_k * 2

    # FAISS retrieval
    faiss_results = query_vector_store(query, embedding_model, faiss_index, chunks, top_k=fetch_k)
    logger.debug(f"FAISS returned {len(faiss_results)} candidates")

    # BM25 retrieval
    tokenized_query = tokenize(query)
    bm25_scores = bm25_index.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[-fetch_k:][::-1]
    bm25_results = [chunks[i] for i in top_bm25_indices if bm25_scores[i] > 0]
    logger.debug(f"BM25 returned {len(bm25_results)} candidates")

    # Merge with RRF
    merged = _rrf_merge(faiss_results, bm25_results, chunks)

    return merged[:top_k]
