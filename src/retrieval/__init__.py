"""Retrieval module — BM25 and embedding-based retrievers."""

from src.retrieval.base import BaseRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever

__all__ = ["BaseRetriever", "BM25Retriever", "EmbeddingRetriever"]
