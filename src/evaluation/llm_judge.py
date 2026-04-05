"""Tier 3: LLM-as-a-Judge evaluation.

Two judge functions:
- J_Answerability: Is this question answerable from the given context? (binary 0/1)
- J_Correctness: How correct/complete/faithful is the predicted answer? (structured rubric)

CRITICAL: Judge prompts contain NO agent identifier. Inputs are blinded.
Uses the same LLM as agents (configured in base.yaml), temperature=0.
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.agents.llm_client import LLMClient


@dataclass
class CorrectnessScore:
    """Structured output from J_Correctness."""
    correctness: int  # 0 or 1 (binary: is the answer correct?)
    completeness: int  # 1-5 (how complete is the answer?)
    faithfulness: str  # "faithful", "partial", or "unfaithful"
    overall_score: float  # 0.0-1.0 combined score
    raw_response: str  # Full LLM response for debugging


# ============================================================
# J_Answerability
# ============================================================

_ANSWERABILITY_SYSTEM = """You are an expert reading comprehension evaluator. Your task is to determine whether a question can be answered based ONLY on the provided context passage. Be strict."""

_ANSWERABILITY_USER = """Context:
{context}

Question: {question}

Can this question be fully and correctly answered using ONLY the information in the context above?

Rules:
- Answer 1 if the context contains ALL the information needed to answer the question.
- Answer 0 if any key information is missing, ambiguous, or requires external knowledge.

Respond with ONLY a single digit: 0 or 1"""


def judge_answerability(llm: LLMClient, question: str, context: str) -> int:
    """J_Answerability: Is this question answerable from context alone?

    Returns: 0 (not answerable) or 1 (answerable).
    """
    messages = [
        {"role": "system", "content": _ANSWERABILITY_SYSTEM},
        {"role": "user", "content": _ANSWERABILITY_USER.format(
            context=context, question=question,
        )},
    ]
    response = llm.generate(messages, temperature=0, max_tokens=16)
    text = response.content.strip()

    # Parse binary response
    if "1" in text:
        return 1
    return 0


# ============================================================
# J_Correctness
# ============================================================

_CORRECTNESS_SYSTEM = """You are an expert answer evaluator. Assess the quality of a predicted answer against a gold reference answer, given the source context. Be fair and precise."""

_CORRECTNESS_USER = """Context:
{context}

Question: {question}

Gold answer: {gold_answer}

Predicted answer: {predicted_answer}

Evaluate the predicted answer on these criteria:

1. CORRECTNESS (0 or 1): Is the predicted answer factually correct and does it match the gold answer in meaning?
2. COMPLETENESS (1-5): How complete is the predicted answer?
   1 = Missing most key information
   2 = Missing significant information
   3 = Contains core information but missing some details
   4 = Nearly complete with minor omissions
   5 = Fully complete
3. FAITHFULNESS: Is the answer supported by the context?
   - faithful: Answer is directly supported by the context
   - partial: Answer is partially supported, some claims not in context
   - unfaithful: Answer contains claims contradicting or not in context

Respond in EXACTLY this format (four lines only):
CORRECTNESS: 0 or 1
COMPLETENESS: 1-5
FAITHFULNESS: faithful/partial/unfaithful
OVERALL: 0.0-1.0"""


def judge_correctness(
    llm: LLMClient,
    question: str,
    context: str,
    predicted_answer: str,
    gold_answer: str,
) -> CorrectnessScore:
    """J_Correctness: Evaluate answer quality on correctness, completeness, faithfulness.

    Returns: CorrectnessScore with all rubric dimensions.
    """
    messages = [
        {"role": "system", "content": _CORRECTNESS_SYSTEM},
        {"role": "user", "content": _CORRECTNESS_USER.format(
            context=context,
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
        )},
    ]
    response = llm.generate(messages, temperature=0, max_tokens=64)
    text = response.content.strip()

    # Parse structured response
    correctness = 0
    completeness = 3
    faithfulness = "partial"
    overall = 0.0

    for line in text.split("\n"):
        line_lower = line.lower().strip()
        if line_lower.startswith("correctness:"):
            val = line_lower.split(":", 1)[1].strip()
            correctness = 1 if "1" in val else 0
        elif line_lower.startswith("completeness:"):
            val = line_lower.split(":", 1)[1].strip()
            match = re.search(r"[1-5]", val)
            if match:
                completeness = int(match.group())
        elif line_lower.startswith("faithfulness:"):
            val = line_lower.split(":", 1)[1].strip()
            if "unfaithful" in val:
                faithfulness = "unfaithful"
            elif "partial" in val:
                faithfulness = "partial"
            else:
                faithfulness = "faithful"
        elif line_lower.startswith("overall:"):
            val = line_lower.split(":", 1)[1].strip()
            match = re.search(r"[0-9]*\.?[0-9]+", val)
            if match:
                overall = max(0.0, min(1.0, float(match.group())))

    # If overall wasn't parsed, compute from components
    if overall == 0.0 and correctness == 1:
        faith_score = {"faithful": 1.0, "partial": 0.5, "unfaithful": 0.0}[faithfulness]
        overall = (correctness * 0.4) + (completeness / 5.0 * 0.3) + (faith_score * 0.3)

    return CorrectnessScore(
        correctness=correctness,
        completeness=completeness,
        faithfulness=faithfulness,
        overall_score=overall,
        raw_response=text,
    )
