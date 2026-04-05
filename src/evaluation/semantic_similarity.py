"""Tier 2: Semantic similarity evaluation using BGE embeddings.

Computes cosine similarity between predicted and gold answers.
Questions that pass Tier 1 (EM/F1) skip this tier.
"""

import numpy as np
from typing import List, Optional

from sentence_transformers import SentenceTransformer


class SemanticSimilarityScorer:
    """Computes cosine similarity between text pairs using sentence embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None

    def _load_model(self) -> SentenceTransformer:
        if self.model is None:
            print(f"Loading semantic similarity model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device="cpu")
        return self.model

    def score(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts. Returns 0.0-1.0."""
        model = self._load_model()
        embeddings = model.encode(
            [text_a, text_b],
            normalize_embeddings=True,
        )
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))

    def score_batch(self, pairs: List[tuple]) -> List[float]:
        """Compute cosine similarity for a batch of (text_a, text_b) pairs."""
        if not pairs:
            return []

        model = self._load_model()
        texts_a = [a for a, _ in pairs]
        texts_b = [b for _, b in pairs]

        emb_a = model.encode(texts_a, normalize_embeddings=True, batch_size=64)
        emb_b = model.encode(texts_b, normalize_embeddings=True, batch_size=64)

        # Row-wise dot product (cosine sim since normalized)
        scores = np.sum(emb_a * emb_b, axis=1)
        return [max(0.0, min(1.0, float(s))) for s in scores]
