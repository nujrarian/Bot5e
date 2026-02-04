"""
RAGAS evaluation pipeline for Bot5e.

Runs the eval dataset through the retrieval and LLM pipeline,
then scores with faithfulness, answer_relevancy, context_precision,
and context_recall using a local Ollama LLM as judge.

Usage:
    python evaluate.py                        # baseline (FAISS only)
    python evaluate.py --mode hybrid          # hybrid search
    python evaluate.py --mode rerank          # hybrid + reranking
    python evaluate.py --output results.json  # custom output file
"""

import json
import argparse
import pickle
import time
import os
import faiss
import warnings

from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from ragas import evaluate, SingleTurnSample, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmb

from config import config
from logger import setup_logger
from vector_store import query_vector_store

logger = setup_logger(__name__)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# --- Prompt used by PDFQAAgent (must stay in sync) ---
PROMPT_TEMPLATE = """You are a D&D 5th Edition rules expert. Answer ONLY based on the context provided from the official D&D 5e rulebook.

CRITICAL RULES:
1. If the context does not contain information to answer the question, you MUST respond with: "I don't have that specific information in the D&D 5e rulebook I have access to. Could you rephrase your question or ask about something else?"
2. NEVER make up rules, mechanics, or information that isn't in the context
3. If you're uncertain, say "I'm not completely sure, but based on the rulebook..."
4. Stick strictly to what's written in the context below
5. Format the response with proper spaces and tables if necessary before responding to the user.

Context from D&D 5e Rulebook:
{context}

Question: {question}

Answer (based ONLY on the context above):"""


def load_chunks_and_index():
    """Load cached chunks and FAISS index."""
    if not os.path.exists(config.embeddings_path) or not os.path.exists(config.index_path):
        raise FileNotFoundError(
            "Cached embeddings/index not found. Run the app once first to generate them."
        )
    with open(config.embeddings_path, 'rb') as f:
        chunks, _ = pickle.load(f)
    index = faiss.read_index(config.index_path)
    logger.info(f"Loaded {len(chunks)} chunks and FAISS index")
    return chunks, index


def retrieve_baseline(question, embedding_model, index, chunks):
    """Retrieve contexts using FAISS only (baseline)."""
    return query_vector_store(question, embedding_model, index, chunks, top_k=config.top_k)


def retrieve_hybrid(question, embedding_model, index, chunks, bm25_index):
    """Retrieve contexts using hybrid search (BM25 + FAISS with RRF)."""
    from hybrid_search import hybrid_retrieve
    return hybrid_retrieve(question, embedding_model, index, chunks, bm25_index, top_k=config.top_k)


def retrieve_rerank(question, embedding_model, index, chunks, bm25_index, reranker):
    """Retrieve using hybrid search then rerank with cross-encoder."""
    from hybrid_search import hybrid_retrieve
    from reranker import rerank_chunks
    # Retrieve more candidates, then rerank down to top_k
    candidates = hybrid_retrieve(question, embedding_model, index, chunks, bm25_index, top_k=config.top_k * 2)
    return rerank_chunks(question, candidates, reranker, top_k=config.top_k)


def run_evaluation(mode="baseline", output_path="eval_results.json"):
    """
    Run the full evaluation pipeline.

    Args:
        mode: "baseline", "hybrid", or "rerank"
        output_path: where to save results JSON
    """
    # Load eval dataset
    with open("eval_dataset.json", "r") as f:
        eval_data = json.load(f)
    logger.info(f"Loaded {len(eval_data)} evaluation questions")

    # Load models and data
    embedding_model = SentenceTransformer(config.embedding_model)
    chunks, index = load_chunks_and_index()
    llm = ChatOllama(model=config.llm_model, base_url=config.ollama_base_url)

    # Mode-specific setup
    bm25_index = None
    reranker = None
    if mode in ("hybrid", "rerank"):
        from hybrid_search import build_bm25_index
        bm25_index = build_bm25_index(chunks)
        logger.info("BM25 index built")
    if mode == "rerank":
        from reranker import load_reranker
        reranker = load_reranker()
        logger.info("Cross-encoder reranker loaded")

    # --- Phase 1: Collect contexts and answers (cached per mode) ---
    cache_path = f"eval_cache_{mode}.json"
    samples = []

    if os.path.exists(cache_path):
        logger.info(f"Loading Phase 1 cache from {cache_path} (delete to re-run LLM calls)")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        for item in cached:
            samples.append(SingleTurnSample(
                user_input=item["question"],
                retrieved_contexts=item["contexts"],
                response=item["answer"],
                reference=item["ground_truth"],
            ))
    else:
        start = time.time()
        for i, item in enumerate(eval_data):
            question = item["question"]
            ground_truth = item["ground_truth"]

            logger.info(f"[{i+1}/{len(eval_data)}] Retrieving + answering: {question[:60]}...")

            # Retrieve
            if mode == "baseline":
                contexts = retrieve_baseline(question, embedding_model, index, chunks)
            elif mode == "hybrid":
                contexts = retrieve_hybrid(question, embedding_model, index, chunks, bm25_index)
            else:  # rerank
                contexts = retrieve_rerank(question, embedding_model, index, chunks, bm25_index, reranker)

            # Generate answer
            context_text = "\n\n---\n\n".join(contexts)
            prompt = PROMPT_TEMPLATE.format(context=context_text, question=question)
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)

            samples.append(SingleTurnSample(
                user_input=question,
                retrieved_contexts=contexts,
                response=answer,
                reference=ground_truth,
            ))

        elapsed_retrieval = time.time() - start
        logger.info(f"Phase 1 complete: {len(samples)} samples in {elapsed_retrieval:.1f}s")

        # Cache Phase 1 so RAGAS scoring can be re-run without hitting the LLM again
        with open(cache_path, "w") as f:
            json.dump([
                {
                    "question": s.user_input,
                    "contexts": s.retrieved_contexts,
                    "answer": s.response,
                    "ground_truth": s.reference,
                }
                for s in samples
            ], f, indent=2)
        logger.info(f"Phase 1 cached to {cache_path}")

    # --- Phase 2: RAGAS scoring ---
    logger.info("Running RAGAS evaluation (this uses the local LLM as judge)...")
    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(LCHFEmb(model_name=config.embedding_model))

    dataset = EvaluationDataset(samples=samples)

    # max_workers=1: local Ollama can't handle parallel LLM calls without timeouts
    # timeout=300: llama3.1 on CPU can be slow for complex evaluation prompts
    run_cfg = RunConfig(max_workers=1, timeout=300, max_retries=3)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
        raise_exceptions=False,
        show_progress=True,
    )

    # --- Phase 3: Save results ---
    result_df = result.to_pandas()
    per_question = result_df.to_dict(orient="records")

    # Attach categories from eval_data
    for i, record in enumerate(per_question):
        record["category"] = eval_data[i]["category"]

    output = {
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "top_k": config.top_k,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "embedding_model": config.embedding_model,
            "llm_model": config.llm_model,
        },
        "aggregate": {
            "faithfulness": float(result_df["faithfulness"].mean()),
            "answer_relevancy": float(result_df["answer_relevancy"].mean()),
            "context_precision": float(result_df["context_precision"].mean()),
            "context_recall": float(result_df["context_recall"].mean()),
        },
        "per_question": per_question,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to {output_path}")
    logger.info("=== Aggregate Scores ===")
    for metric, score in output["aggregate"].items():
        logger.info(f"  {metric}: {score:.4f}")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation on Bot5e")
    parser.add_argument(
        "--mode",
        choices=["baseline", "hybrid", "rerank"],
        default="baseline",
        help="Retrieval mode to evaluate"
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Output file path for results"
    )
    args = parser.parse_args()

    run_evaluation(mode=args.mode, output_path=args.output)
