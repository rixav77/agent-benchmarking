"""Dense embedding retriever using BGE (or any sentence-transformers model).

Supports disk caching of embeddings to avoid re-computation.
"""

import hashlib
import numpy as np
from pathlib import Path
from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from src.data.schema import QAExample
from src.retrieval.base import BaseRetriever


class EmbeddingRetriever(BaseRetriever):
    """Dense retriever using sentence-transformers embeddings + cosine similarity."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", cache_dir: str = "data/embedding_cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model: SentenceTransformer | None = None
        self.contexts: List[str] = []
        self.embeddings: np.ndarray | None = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _cache_key(self, contexts: List[str]) -> str:
        """Generate a deterministic cache key from context content."""
        content_hash = hashlib.sha256("|||".join(contexts).encode()).hexdigest()[:16]
        model_short = self.model_name.replace("/", "_")
        return f"{model_short}_{len(contexts)}_{content_hash}"

    def _load_cached_embeddings(self, key: str) -> np.ndarray | None:
        """Load embeddings from disk cache if available."""
        cache_path = self.cache_dir / f"{key}.npy"
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        return None

    def _save_cached_embeddings(self, key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to disk cache."""
        cache_path = self.cache_dir / f"{key}.npy"
        np.save(cache_path, embeddings)
        print(f"Cached embeddings to {cache_path}")

    def index(self, examples: List[QAExample]) -> None:
        """Build embedding index from unique context paragraphs."""
        self.contexts = self._extract_unique_contexts(examples)
        cache_key = self._cache_key(self.contexts)

        # Try loading from cache
        cached = self._load_cached_embeddings(cache_key)
        if cached is not None:
            self.embeddings = cached
            print(f"Embedding index ready: {len(self.contexts)} contexts (from cache)")
            return

        # Compute embeddings
        model = self._load_model()
        print(f"Encoding {len(self.contexts)} contexts...")
        self.embeddings = model.encode(
            self.contexts,
            show_progress_bar=True,
            batch_size=64,
            normalize_embeddings=True,  # For cosine similarity via dot product
        )

        # Cache to disk
        self._save_cached_embeddings(cache_key, self.embeddings)
        print(f"Embedding index built: {len(self.contexts)} contexts")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k contexts using cosine similarity."""
        if self.embeddings is None:
            raise RuntimeError("Call index() before retrieve()")

        model = self._load_model()
        query_embedding = model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        # Cosine similarity = dot product (embeddings are normalized)
        scores = self.embeddings @ query_embedding

        # Top-k indices
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.contexts[i], float(scores[i])) for i in ranked_indices]
