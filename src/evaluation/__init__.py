"""Evaluation module — 3-tier cascade: EM/F1 → Semantic → LLM Judge."""

from src.evaluation.evaluator import evaluate, EvaluationResult

__all__ = ["evaluate", "EvaluationResult"]
