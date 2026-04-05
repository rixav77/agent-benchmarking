"""Core evaluation metrics: Exact Match, Token F1, Unanswerable Accuracy.

All metrics are agent-agnostic — they operate on (predicted, gold) pairs
with no awareness of which agent produced the prediction.
"""

import re
import string
from typing import List, Dict
from collections import Counter

from src.data.schema import QAExample, PredictionResult


def _normalize_answer(text: str) -> str:
    """Normalize answer text for comparison (standard SQuAD normalization)."""
    text = text.lower()
    # Remove punctuation
    text = "".join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


def exact_match(predicted: str, gold_answers: List[str]) -> float:
    """Exact match: 1.0 if normalized prediction matches any gold answer."""
    pred_norm = _normalize_answer(predicted)
    for gold in gold_answers:
        if pred_norm == _normalize_answer(gold):
            return 1.0
    return 0.0


def token_f1(predicted: str, gold_answers: List[str]) -> float:
    """Token-level F1 score (standard SQuAD metric). Returns max F1 across gold answers."""
    pred_tokens = _normalize_answer(predicted).split()

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer(gold).split()

        if not pred_tokens and not gold_tokens:
            best_f1 = max(best_f1, 1.0)
            continue
        if not pred_tokens or not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def unanswerable_accuracy(pred: PredictionResult, example: QAExample) -> float:
    """Binary accuracy for unanswerable detection.

    Returns 1.0 if the prediction correctly identifies answerability status.
    """
    if example.is_unanswerable:
        return 1.0 if pred.is_unanswerable_pred else 0.0
    else:
        return 1.0 if not pred.is_unanswerable_pred else 0.0


def compute_basic_metrics(
    predictions: List[PredictionResult],
    examples: List[QAExample],
) -> Dict[str, float]:
    """Compute Tier 1 metrics (EM, F1, unanswerable accuracy) over a dataset.

    Returns dict with aggregate and per-category metrics.
    """
    assert len(predictions) == len(examples), "Predictions and examples must be same length"

    em_scores = []
    f1_scores = []
    unans_scores = []

    # Per-category tracking
    answerable_em, answerable_f1 = [], []
    unanswerable_acc = []

    for pred, ex in zip(predictions, examples):
        # Unanswerable accuracy (always computed)
        ua = unanswerable_accuracy(pred, ex)
        unans_scores.append(ua)

        if ex.is_unanswerable:
            unanswerable_acc.append(ua)
            # For unanswerable questions, EM/F1 are based on detecting unanswerability
            if pred.is_unanswerable_pred:
                em_scores.append(1.0)
                f1_scores.append(1.0)
            else:
                em_scores.append(0.0)
                f1_scores.append(0.0)
        else:
            # Answerable questions — compare predicted answer to gold
            if pred.is_unanswerable_pred:
                # Predicted unanswerable but was answerable — wrong
                em_scores.append(0.0)
                f1_scores.append(0.0)
                answerable_em.append(0.0)
                answerable_f1.append(0.0)
            else:
                em = exact_match(pred.predicted_answer, ex.answers)
                f1 = token_f1(pred.predicted_answer, ex.answers)
                em_scores.append(em)
                f1_scores.append(f1)
                answerable_em.append(em)
                answerable_f1.append(f1)

    results = {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "unanswerable_accuracy": sum(unans_scores) / len(unans_scores) if unans_scores else 0.0,
        "answerable_em": sum(answerable_em) / len(answerable_em) if answerable_em else 0.0,
        "answerable_f1": sum(answerable_f1) / len(answerable_f1) if answerable_f1 else 0.0,
        "unanswerable_detection_rate": sum(unanswerable_acc) / len(unanswerable_acc) if unanswerable_acc else 0.0,
        "total_questions": len(predictions),
        "answerable_count": sum(1 for e in examples if not e.is_unanswerable),
        "unanswerable_count": sum(1 for e in examples if e.is_unanswerable),
    }
    return results
