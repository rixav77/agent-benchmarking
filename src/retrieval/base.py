"""Abstract retriever interface.

All retrievers return a list of (context_text, score) tuples,
ranked by relevance (highest score first).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from src.data.schema import QAExample


class BaseRetriever(ABC):
    """Interface that all retrievers must implement."""

    @abstractmethod
    def index(self, examples: List[QAExample]) -> None:
        """Build the retrieval index from a list of QAExamples.

        Extracts unique contexts and indexes them for retrieval.
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k contexts for a query.

        Args:
            query: The search query (typically a question).
            top_k: Number of contexts to return.

        Returns:
            List of (context_text, score) tuples, sorted by score descending.
        """

    def _extract_unique_contexts(self, examples: List[QAExample]) -> List[str]:
        """Extract unique context paragraphs preserving first-seen order."""
        seen = set()
        contexts = []
        for ex in examples:
            if ex.context not in seen:
                seen.add(ex.context)
                contexts.append(ex.context)
        return contexts
