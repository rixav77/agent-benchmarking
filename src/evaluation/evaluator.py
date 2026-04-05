"""Single evaluation entry point — one codepath for ALL agents.

3-tier cascade:
  Tier 1: EM / Token F1 (fast, exact)
  Tier 2: Semantic Similarity (medium, embedding-based)
  Tier 3: LLM-as-a-Judge (slow, expensive, most nuanced)

Each tier only adds credit. Invariant: EM_acc <= semantic_acc <= combined_acc.

This evaluator is completely agent-agnostic. It takes predictions and examples,
with NO awareness of which agent produced the predictions.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
import json

from src.data.schema import QAExample, PredictionResult
from src.evaluation.metrics import (
    exact_match, token_f1, unanswerable_accuracy, compute_basic_metrics,
)
from src.evaluation.semantic_similarity import SemanticSimilarityScorer
from src.evaluation.llm_judge import (
    judge_answerability, judge_correctness, CorrectnessScore,
)
from src.agents.llm_client import LLMClient


@dataclass
class PerQuestionResult:
    """Detailed per-question evaluation result."""
    id: str
    question: str
    gold_answers: List[str]
    predicted_answer: str
    is_unanswerable: bool
    is_unanswerable_pred: bool
    # Tier 1
    em: float
    f1: float
    unanswerable_correct: float
    # Tier 2
    semantic_similarity: float
    semantic_correct: bool  # Above threshold?
    # Tier 3
    j_answerability: Optional[int] = None
    j_correctness: Optional[CorrectnessScore] = None
    # Combined
    tier_credited: str = ""  # Which tier gave credit: "tier1", "tier2", "tier3", "none"


@dataclass
class EvaluationResult:
    """Complete evaluation result for a set of predictions."""
    # Tier 1 aggregate
    em: float = 0.0
    f1: float = 0.0
    unanswerable_accuracy: float = 0.0
    answerable_em: float = 0.0
    answerable_f1: float = 0.0
    unanswerable_detection_rate: float = 0.0
    # Tier 2 aggregate
    semantic_accuracy: float = 0.0  # Fraction passing semantic threshold
    avg_semantic_similarity: float = 0.0
    # Tier 3 aggregate
    avg_j_correctness: float = 0.0
    avg_j_completeness: float = 0.0
    j_faithfulness_dist: Dict[str, int] = field(default_factory=dict)
    # Combined (3-tier cascade)
    combined_accuracy: float = 0.0  # Fraction correct by any tier
    tier_distribution: Dict[str, int] = field(default_factory=dict)
    # Metadata
    total_questions: int = 0
    answerable_count: int = 0
    unanswerable_count: int = 0
    # Per-question details
    per_question: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove per_question from summary (it's large)
        summary = {k: v for k, v in d.items() if k != "per_question"}
        return summary

    def to_json(self, include_per_question: bool = False) -> str:
        if include_per_question:
            return json.dumps(asdict(self), indent=2, default=str)
        return json.dumps(self.to_dict(), indent=2, default=str)


def evaluate(
    predictions: List[PredictionResult],
    examples: List[QAExample],
    config: dict,
    llm_client: Optional[LLMClient] = None,
    semantic_scorer: Optional[SemanticSimilarityScorer] = None,
    run_judge: bool = True,
) -> EvaluationResult:
    """Run the full 3-tier evaluation cascade.

    Args:
        predictions: Agent predictions (agent-agnostic).
        examples: Gold QA examples.
        config: Shared config dict (eval thresholds, judge params).
        llm_client: LLM client for Tier 3 judge (optional, skips judge if None).
        semantic_scorer: Semantic similarity scorer (optional, created if None).
        run_judge: Whether to run Tier 3 LLM judge.

    Returns:
        EvaluationResult with all metrics and per-question details.
    """
    assert len(predictions) == len(examples), "Predictions and examples must match"

    eval_cfg = config.get("eval", {})
    semantic_threshold = eval_cfg.get("semantic_threshold", 0.85)

    # === Tier 1: Basic Metrics (EM, F1, Unanswerable) ===
    tier1 = compute_basic_metrics(predictions, examples)

    # === Tier 2: Semantic Similarity ===
    if semantic_scorer is None:
        model_name = config.get("retriever", {}).get("embedding_model", "BAAI/bge-large-en-v1.5")
        semantic_scorer = SemanticSimilarityScorer(model_name=model_name)

    # Build pairs for answerable questions where prediction wasn't marked unanswerable
    sem_pairs = []
    sem_indices = []
    for i, (pred, ex) in enumerate(zip(predictions, examples)):
        if not ex.is_unanswerable and not pred.is_unanswerable_pred:
            # Compare predicted answer to best gold answer
            sem_pairs.append((pred.predicted_answer, ex.answers[0]))
            sem_indices.append(i)

    sem_scores_batch = semantic_scorer.score_batch(sem_pairs) if sem_pairs else []
    # Map back to full list
    sem_scores = [0.0] * len(predictions)
    for idx, score in zip(sem_indices, sem_scores_batch):
        sem_scores[idx] = score

    # === Tier 3: LLM Judge (optional) ===
    j_answerability_scores = [None] * len(predictions)
    j_correctness_scores = [None] * len(predictions)

    if run_judge and llm_client is not None:
        print("Running LLM Judge (Tier 3)...")
        for i, (pred, ex) in enumerate(zip(predictions, examples)):
            # J_Answerability — run on all questions
            j_ans = judge_answerability(llm_client, ex.question, ex.context)
            j_answerability_scores[i] = j_ans

            # J_Correctness — only for answerable questions with answers
            if not ex.is_unanswerable and not pred.is_unanswerable_pred:
                j_corr = judge_correctness(
                    llm_client, ex.question, ex.context,
                    pred.predicted_answer, ex.answers[0],
                )
                j_correctness_scores[i] = j_corr

            if (i + 1) % 50 == 0:
                print(f"  Judged {i+1}/{len(predictions)} questions...")

    # === Build Per-Question Results + 3-Tier Cascade ===
    per_question = []
    tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0, "none": 0}
    combined_correct = 0
    semantic_correct_count = 0
    j_correctness_vals = []
    j_completeness_vals = []
    faithfulness_dist = {"faithful": 0, "partial": 0, "unfaithful": 0}

    for i, (pred, ex) in enumerate(zip(predictions, examples)):
        # Compute per-question Tier 1
        if ex.is_unanswerable:
            em = 1.0 if pred.is_unanswerable_pred else 0.0
            f1 = em
        else:
            if pred.is_unanswerable_pred:
                em, f1 = 0.0, 0.0
            else:
                em = exact_match(pred.predicted_answer, ex.answers)
                f1 = token_f1(pred.predicted_answer, ex.answers)

        ua = unanswerable_accuracy(pred, ex)
        sem_sim = sem_scores[i]
        sem_pass = sem_sim >= semantic_threshold

        if sem_pass:
            semantic_correct_count += 1

        # 3-tier cascade: which tier gives credit?
        tier = "none"
        if em >= 1.0 or f1 >= eval_cfg.get("f1_partial_threshold", 0.5):
            tier = "tier1"
            combined_correct += 1
        elif sem_pass:
            tier = "tier2"
            combined_correct += 1
        elif j_correctness_scores[i] is not None and j_correctness_scores[i].correctness == 1:
            tier = "tier3"
            combined_correct += 1
        elif ex.is_unanswerable and pred.is_unanswerable_pred:
            # Correctly identified unanswerable — counts as correct
            tier = "tier1"
            combined_correct += 1

        tier_counts[tier] += 1

        # Collect judge stats
        if j_correctness_scores[i] is not None:
            j_correctness_vals.append(j_correctness_scores[i].overall_score)
            j_completeness_vals.append(j_correctness_scores[i].completeness)
            faithfulness_dist[j_correctness_scores[i].faithfulness] += 1

        pq = PerQuestionResult(
            id=ex.id,
            question=ex.question,
            gold_answers=ex.answers,
            predicted_answer=pred.predicted_answer,
            is_unanswerable=ex.is_unanswerable,
            is_unanswerable_pred=pred.is_unanswerable_pred,
            em=em,
            f1=f1,
            unanswerable_correct=ua,
            semantic_similarity=sem_sim,
            semantic_correct=sem_pass,
            j_answerability=j_answerability_scores[i],
            j_correctness=j_correctness_scores[i],
            tier_credited=tier,
        )
        per_question.append(asdict(pq))

    # === Build Final Result ===
    n = len(predictions)
    result = EvaluationResult(
        # Tier 1
        em=tier1["em"],
        f1=tier1["f1"],
        unanswerable_accuracy=tier1["unanswerable_accuracy"],
        answerable_em=tier1["answerable_em"],
        answerable_f1=tier1["answerable_f1"],
        unanswerable_detection_rate=tier1["unanswerable_detection_rate"],
        # Tier 2
        semantic_accuracy=semantic_correct_count / n if n else 0.0,
        avg_semantic_similarity=sum(sem_scores) / len(sem_scores) if sem_scores else 0.0,
        # Tier 3
        avg_j_correctness=sum(j_correctness_vals) / len(j_correctness_vals) if j_correctness_vals else 0.0,
        avg_j_completeness=sum(j_completeness_vals) / len(j_completeness_vals) if j_completeness_vals else 0.0,
        j_faithfulness_dist=faithfulness_dist,
        # Combined
        combined_accuracy=combined_correct / n if n else 0.0,
        tier_distribution=tier_counts,
        # Metadata
        total_questions=n,
        answerable_count=tier1["answerable_count"],
        unanswerable_count=tier1["unanswerable_count"],
        per_question=per_question,
    )

    return result
