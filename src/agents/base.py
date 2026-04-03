"""Abstract agent interface.

All agents take a QAExample and produce a PredictionResult.
The output schema is shared — no agent-specific fields.
"""

from abc import ABC, abstractmethod

from src.data.schema import QAExample, PredictionResult
from src.agents.llm_client import LLMClient
from src.retrieval.base import BaseRetriever


class BaseAgent(ABC):
    """Interface that all RAG agents must implement."""

    def __init__(self, llm_client: LLMClient, retriever: BaseRetriever, config: dict):
        self.llm_client = llm_client
        self.retriever = retriever
        self.config = config

    @abstractmethod
    def answer(self, example: QAExample) -> PredictionResult:
        """Generate an answer for a single QA example.

        Args:
            example: A QAExample with question and context info.

        Returns:
            PredictionResult with all fields populated.
            - reasoning_trace: empty string for Simple RAG, populated for Agentic RAG.
            - Token counts from LLM usage stats.
        """
