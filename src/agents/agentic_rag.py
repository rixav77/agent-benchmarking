"""Agentic RAG: multi-step agent with iterative retrieval, relevance assessment,
query rewriting, answerability checking, and self-verified answer generation.

Inspired by CRAG (Corrective RAG) and Self-RAG patterns.
Uses the shared retriever and LLM client — same as Simple RAG.
"""

import time
from typing import List

from src.data.schema import QAExample, PredictionResult
from src.agents.base import BaseAgent
from src.agents.llm_client import LLMClient
from src.agents import tools
from src.retrieval.base import BaseRetriever


class AgenticRAGAgent(BaseAgent):
    """Multi-step agentic RAG with CRAG/Self-RAG inspired loop.

    Agent loop:
        1. Retrieve passages
        2. Assess relevance → if low, rewrite query and re-retrieve (up to max_retries)
        3. Check answerability → if unanswerable, return early
        4. Generate answer with self-verification
    """

    def __init__(self, llm_client: LLMClient, retriever: BaseRetriever, config: dict):
        super().__init__(llm_client, retriever, config)
        self.top_k = config.get("retriever", {}).get("top_k", 5)
        self.max_retries = config.get("agent", {}).get("max_retries", 2)
        self.unanswerable_keywords = config.get("eval", {}).get("unanswerable_keywords", ["unanswerable"])

    def answer(self, example: QAExample) -> PredictionResult:
        """Run the full agentic loop and return a PredictionResult."""
        start_time = time.perf_counter()
        trace = []  # Reasoning trace — records every step
        total_prompt_tokens = 0
        total_completion_tokens = 0
        current_query = example.question
        best_passages: List[str] = []

        # === Step 1: Initial Retrieval ===
        passages, scores = tools.retrieve(self.retriever, current_query, self.top_k)
        best_passages = passages
        trace.append(f"[RETRIEVE] query='{current_query}' → {len(passages)} passages (top score: {scores[0]:.2f})")

        # === Step 2: Assess Relevance + Rewrite Loop ===
        for attempt in range(self.max_retries + 1):
            relevance, rel_resp = tools.assess_relevance(self.llm_client, best_passages, current_query)
            total_prompt_tokens += rel_resp.prompt_tokens
            total_completion_tokens += rel_resp.completion_tokens
            trace.append(f"[ASSESS_RELEVANCE] attempt={attempt+1} → relevance={relevance}")

            if relevance == "high":
                trace.append(f"[DECISION] Relevance is high, proceeding to answerability check")
                break

            if attempt < self.max_retries:
                # Rewrite query and re-retrieve
                rewritten, rw_resp = tools.rewrite_query(self.llm_client, current_query, best_passages)
                total_prompt_tokens += rw_resp.prompt_tokens
                total_completion_tokens += rw_resp.completion_tokens
                trace.append(f"[REWRITE_QUERY] '{current_query}' → '{rewritten}'")

                current_query = rewritten
                passages, scores = tools.retrieve(self.retriever, current_query, self.top_k)
                best_passages = passages
                trace.append(f"[RE-RETRIEVE] query='{current_query}' → {len(passages)} passages (top score: {scores[0]:.2f})")
            else:
                trace.append(f"[DECISION] Max retries reached ({self.max_retries}), proceeding with best available passages")

        # === Step 3: Check Answerability ===
        is_answerable, ans_reason, ans_resp = tools.check_answerability(self.llm_client, best_passages, example.question)
        total_prompt_tokens += ans_resp.prompt_tokens
        total_completion_tokens += ans_resp.completion_tokens
        trace.append(f"[CHECK_ANSWERABILITY] answerable={is_answerable}, reason='{ans_reason}'")

        if not is_answerable:
            trace.append(f"[DECISION] Question deemed unanswerable, returning early")
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            total_tokens = total_prompt_tokens + total_completion_tokens

            return PredictionResult(
                id=example.id,
                question=example.question,
                predicted_answer="unanswerable",
                is_unanswerable_pred=True,
                contexts_used=best_passages,
                reasoning_trace="\n".join(trace),
                latency_ms=elapsed_ms,
                agent_type="agentic_rag",
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
            )

        # === Step 4: Generate Answer with Self-Verification ===
        answer_text, is_supported, gen_resp = tools.generate_answer(self.llm_client, best_passages, example.question)
        total_prompt_tokens += gen_resp.prompt_tokens
        total_completion_tokens += gen_resp.completion_tokens
        trace.append(f"[GENERATE_ANSWER] answer='{answer_text}', supported={is_supported}")

        # If not supported and we haven't exhausted options, flag it
        if not is_supported:
            trace.append(f"[DECISION] Answer not well-supported by passages, but returning best effort")

        # Detect unanswerable in the generated answer
        answer_lower = answer_text.lower().strip()
        is_unanswerable_pred = any(
            keyword in answer_lower for keyword in self.unanswerable_keywords
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        total_tokens = total_prompt_tokens + total_completion_tokens

        return PredictionResult(
            id=example.id,
            question=example.question,
            predicted_answer=answer_text,
            is_unanswerable_pred=is_unanswerable_pred,
            contexts_used=best_passages,
            reasoning_trace="\n".join(trace),
            latency_ms=elapsed_ms,
            agent_type="agentic_rag",
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
        )
