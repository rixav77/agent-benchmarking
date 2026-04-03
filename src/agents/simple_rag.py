"""Simple RAG agent: single-pass retrieve → prompt → answer.

No iteration, no query refinement, no verification.
Uses the shared retriever and LLM client.
"""

import time

from src.data.schema import QAExample, PredictionResult
from src.agents.base import BaseAgent
from src.agents.llm_client import LLMClient, strip_think_tags
from src.retrieval.base import BaseRetriever


SYSTEM_PROMPT = """You are a precise question-answering assistant. Answer the question based ONLY on the provided context passages.

Rules:
1. If the answer can be found in the context, provide a concise, direct answer.
2. If the context does not contain enough information to answer the question, respond with exactly: "unanswerable"
3. Do not make up information or use knowledge outside the provided context.
4. Keep your answer brief and to the point — typically a few words or a short sentence."""

USER_PROMPT_TEMPLATE = """Context passages:
{contexts}

Question: {question}

Answer:"""


class SimpleRAGAgent(BaseAgent):
    """Single-pass RAG: retrieve top-k contexts, prompt LLM once, return answer."""

    def __init__(self, llm_client: LLMClient, retriever: BaseRetriever, config: dict):
        super().__init__(llm_client, retriever, config)
        self.top_k = config.get("retriever", {}).get("top_k", 5)
        self.unanswerable_keywords = config.get("eval", {}).get("unanswerable_keywords", ["unanswerable"])

    def answer(self, example: QAExample) -> PredictionResult:
        """Retrieve contexts, prompt LLM, return PredictionResult."""
        start_time = time.perf_counter()

        # Step 1: Retrieve top-k contexts
        retrieved = self.retriever.retrieve(example.question, top_k=self.top_k)
        contexts_used = [ctx for ctx, _ in retrieved]

        # Step 2: Build prompt
        contexts_text = "\n\n---\n\n".join(
            f"[Passage {i+1}]: {ctx}" for i, ctx in enumerate(contexts_used)
        )
        user_prompt = USER_PROMPT_TEMPLATE.format(
            contexts=contexts_text,
            question=example.question,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Step 3: Call LLM
        response = self.llm_client.generate(messages)

        # Step 4: Strip thinking tags (Qwen3 outputs <think>...</think>)
        clean_answer, _ = strip_think_tags(response.content)

        # Step 5: Detect unanswerable
        answer_lower = clean_answer.lower().strip()
        is_unanswerable_pred = any(
            keyword in answer_lower for keyword in self.unanswerable_keywords
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return PredictionResult(
            id=example.id,
            question=example.question,
            predicted_answer=clean_answer,
            is_unanswerable_pred=is_unanswerable_pred,
            contexts_used=contexts_used,
            reasoning_trace="",  # Simple RAG has no reasoning trace
            latency_ms=elapsed_ms,
            agent_type="simple_rag",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
