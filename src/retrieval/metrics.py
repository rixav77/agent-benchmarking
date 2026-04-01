"""Retrieval quality metrics: Hit-rate@k and Mean Reciprocal Rank (MRR)."""

from typing import List, Tuple

from src.data.schema import QAExample
from src.retrieval.base import BaseRetriever


def hit_rate_at_k(
    retriever: BaseRetriever,
    examples: List[QAExample],
    top_k: int = 5,
) -> float:
    """Compute hit-rate@k: fraction of examples where gold context is in top-k.

    Only evaluates answerable examples (unanswerable have no gold context to find).
    """
    answerable = [ex for ex in examples if not ex.is_unanswerable]
    if not answerable:
        return 0.0

    hits = 0
    for ex in answerable:
        retrieved = retriever.retrieve(ex.question, top_k=top_k)
        retrieved_texts = [ctx for ctx, _ in retrieved]
        if ex.context in retrieved_texts:
            hits += 1

    return hits / len(answerable)


def mean_reciprocal_rank(
    retriever: BaseRetriever,
    examples: List[QAExample],
    top_k: int = 5,
) -> float:
    """Compute MRR: average of 1/rank of gold context across examples.

    If gold context is not in top-k, reciprocal rank is 0 for that example.
    Only evaluates answerable examples.
    """
    answerable = [ex for ex in examples if not ex.is_unanswerable]
    if not answerable:
        return 0.0

    reciprocal_ranks = []
    for ex in answerable:
        retrieved = retriever.retrieve(ex.question, top_k=top_k)
        retrieved_texts = [ctx for ctx, _ in retrieved]
        if ex.context in retrieved_texts:
            rank = retrieved_texts.index(ex.context) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
