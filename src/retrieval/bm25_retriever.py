"""BM25 sparse retriever over context paragraphs."""

from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.data.schema import QAExample
from src.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25 (Okapi) retriever using rank_bm25."""

    def __init__(self):
        self.contexts: List[str] = []
        self.bm25: BM25Okapi | None = None

    def index(self, examples: List[QAExample]) -> None:
        """Build BM25 index from unique context paragraphs."""
        self.contexts = self._extract_unique_contexts(examples)
        # Tokenize by whitespace (simple but effective for BM25)
        tokenized = [ctx.lower().split() for ctx in self.contexts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built: {len(self.contexts)} unique contexts")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k contexts using BM25 scoring."""
        if self.bm25 is None:
            raise RuntimeError("Call index() before retrieve()")

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(self.contexts[i], float(scores[i])) for i in ranked_indices]
