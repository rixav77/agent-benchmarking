"""Agentic RAG tools — each tool is a function that uses the shared LLM client.

Tools:
1. retrieve(query) — BM25 retrieval
2. assess_relevance(passages, query) — LLM scores passage relevance
3. rewrite_query(query, passages) — LLM rewrites query for better retrieval
4. check_answerability(passages, query) — LLM checks if answerable from passages
5. generate_answer(passages, query) — LLM generates answer with self-verification
"""

from typing import List, Tuple

from src.agents.llm_client import LLMClient, LLMResponse
from src.retrieval.base import BaseRetriever


def retrieve(retriever: BaseRetriever, query: str, top_k: int = 5) -> Tuple[List[str], List[float]]:
    """Retrieve top-k passages for a query.

    Returns:
        (passages, scores) tuple.
    """
    results = retriever.retrieve(query, top_k=top_k)
    passages = [ctx for ctx, _ in results]
    scores = [score for _, score in results]
    return passages, scores


def _format_passages(passages: List[str]) -> str:
    """Format passages for LLM prompts."""
    return "\n\n---\n\n".join(
        f"[Passage {i+1}]: {p}" for i, p in enumerate(passages)
    )


def assess_relevance(llm: LLMClient, passages: List[str], query: str) -> Tuple[str, LLMResponse]:
    """LLM assesses if retrieved passages are relevant to the query.

    Returns:
        (relevance_level, llm_response) where relevance_level is "high", "medium", or "low".
    """
    passages_text = _format_passages(passages)
    messages = [
        {"role": "system", "content": "You are a retrieval quality assessor. Be direct and concise. Do NOT explain your reasoning."},
        {"role": "user", "content": f"""Question: {query}

Retrieved passages:
{passages_text}

Rate the relevance of these passages to the question.
- high: At least one passage directly addresses the question
- medium: Passages are topically related but may not directly answer
- low: Passages are mostly irrelevant to the question

Respond with ONLY one word: high, medium, or low. Nothing else."""},
    ]

    response = llm.generate(messages, max_tokens=32)
    level = response.content.lower().strip().strip('"').strip("'")

    # Normalize to valid level
    if "high" in level:
        level = "high"
    elif "medium" in level:
        level = "medium"
    else:
        level = "low"

    return level, response


def rewrite_query(llm: LLMClient, query: str, passages: List[str]) -> Tuple[str, LLMResponse]:
    """LLM rewrites the query for better retrieval based on what was retrieved.

    Returns:
        (rewritten_query, llm_response).
    """
    passages_text = _format_passages(passages[:2])  # Only show first 2 to save tokens
    messages = [
        {"role": "system", "content": "You are a search query optimizer. Be direct. Output ONLY the rewritten question."},
        {"role": "user", "content": f"""This question did not retrieve good results:

Question: {query}

Sample retrieved passages (not relevant enough):
{passages_text}

Rewrite the question using different keywords to find better passages. Output ONLY the rewritten question — one line, no explanation."""},
    ]

    response = llm.generate(messages, max_tokens=128)
    rewritten = response.content
    # Clean up — remove quotes, "Question:" prefix, etc.
    rewritten = rewritten.strip().strip('"').strip("'")
    if rewritten.lower().startswith("question:"):
        rewritten = rewritten[9:].strip()
    # If stripping left nothing useful, fall back to original query with minor variation
    if len(rewritten) < 5 or rewritten == query:
        rewritten = query + " context information details"

    return rewritten, response


def check_answerability(llm: LLMClient, passages: List[str], query: str) -> Tuple[bool, str, LLMResponse]:
    """LLM determines if the question is answerable from the given passages.

    Returns:
        (is_answerable, reason, llm_response).
    """
    passages_text = _format_passages(passages)
    messages = [
        {"role": "system", "content": "You are a reading comprehension analyst. Be direct and concise."},
        {"role": "user", "content": f"""Passages:
{passages_text}

Question: {query}

Can this question be answered based ONLY on the information in the passages above? Be strict — if the passages don't contain a clear, direct answer, say no.

Respond in EXACTLY this format (two lines only):
ANSWERABLE: yes/no
REASON: <one sentence>"""},
    ]

    response = llm.generate(messages, max_tokens=128)

    lines = response.content.strip().split("\n")
    is_answerable = True
    reason = response.content

    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("answerable:"):
            value = line_lower.split(":", 1)[1].strip()
            is_answerable = "yes" in value
        elif line_lower.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return is_answerable, reason, response


def generate_answer(llm: LLMClient, passages: List[str], query: str) -> Tuple[str, bool, LLMResponse]:
    """LLM generates an answer with self-verification.

    The LLM answers AND checks if the answer is supported by the passages.

    Returns:
        (answer, is_supported, llm_response).
    """
    passages_text = _format_passages(passages)
    messages = [
        {"role": "system", "content": "You are a precise question-answering assistant. Answer based ONLY on the provided passages. Be concise."},
        {"role": "user", "content": f"""Passages:
{passages_text}

Question: {query}

Rules: If the answer is in the passages, provide a concise answer. If not answerable from passages, say "unanswerable".

Respond in EXACTLY this format (two lines only):
ANSWER: <your concise answer, or "unanswerable">
SUPPORTED: yes/no"""},
    ]

    response = llm.generate(messages, max_tokens=128)

    lines = response.content.strip().split("\n")
    answer = response.content
    is_supported = True

    for line in lines:
        line_lower = line.lower().strip()
        if line_lower.startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line_lower.startswith("supported:"):
            value = line_lower.split(":", 1)[1].strip()
            is_supported = "yes" in value

    return answer, is_supported, response
