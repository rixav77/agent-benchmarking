"""Pipeline comparison — side-by-side analysis of two agents.

Statistical comparison with 95% CIs, per-category breakdown,
cost/latency ratios, and retrieval quality metrics.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from scipy import stats

from src.data.schema import QAExample, PredictionResult
from src.evaluation.evaluator import EvaluationResult
from src.retrieval.base import BaseRetriever
from src.retrieval.metrics import hit_rate_at_k, mean_reciprocal_rank

logger = logging.getLogger(__name__)


# ── Statistical helpers ──────────────────────────────────────────────────────

def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Return (mean, std) for a list of floats."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _confidence_interval_95(values: List[float]) -> Tuple[float, float]:
    """Return (lower, upper) 95% CI using t-distribution."""
    n = len(values)
    if n <= 1:
        m = values[0] if values else 0.0
        return m, m
    arr = np.array(values)
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / math.sqrt(n))
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=stderr)
    return float(ci[0]), float(ci[1])


def _extract_metric_array(per_question: List[Dict], key: str) -> List[float]:
    """Extract a float array from per_question dicts by key."""
    return [float(pq.get(key, 0) or 0) for pq in per_question]


def _split_by_answerability(
    per_question: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Split per_question into (answerable, unanswerable) lists."""
    answerable = [pq for pq in per_question if not pq.get("is_unanswerable", False)]
    unanswerable = [pq for pq in per_question if pq.get("is_unanswerable", False)]
    return answerable, unanswerable


# ── Category comparison ──────────────────────────────────────────────────────

def _compare_metric(
    simple_values: List[float],
    agentic_values: List[float],
) -> Dict:
    """Build comparison dict for a single metric."""
    s_mean, s_std = _mean_std(simple_values)
    a_mean, a_std = _mean_std(agentic_values)
    s_ci = _confidence_interval_95(simple_values)
    a_ci = _confidence_interval_95(agentic_values)
    delta = a_mean - s_mean
    delta_pct = (delta / s_mean * 100) if s_mean != 0 else 0.0

    return {
        "simple": {"mean": s_mean, "std": s_std, "ci_lower": s_ci[0], "ci_upper": s_ci[1]},
        "agentic": {"mean": a_mean, "std": a_std, "ci_lower": a_ci[0], "ci_upper": a_ci[1]},
        "delta": delta,
        "delta_pct": delta_pct,
    }


def compare_category(
    simple_pq: List[Dict],
    agentic_pq: List[Dict],
    metric_keys: List[str],
) -> Dict:
    """Build comparison for each metric within a category (all/answerable/unanswerable)."""
    metrics = {}
    for key in metric_keys:
        simple_vals = _extract_metric_array(simple_pq, key)
        agentic_vals = _extract_metric_array(agentic_pq, key)
        metrics[key] = _compare_metric(simple_vals, agentic_vals)

    return {
        "n": len(simple_pq),
        "metrics": metrics,
    }


# ── Cost / latency ───────────────────────────────────────────────────────────

def compare_cost_latency(
    simple_preds: List[PredictionResult],
    agentic_preds: List[PredictionResult],
) -> Dict:
    """Compare latency and token usage between agents."""
    def _stats(preds: List[PredictionResult], field: str) -> Dict:
        values = [getattr(p, field) for p in preds]
        arr = np.array(values)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "total": float(np.sum(arr)),
        }

    result = {}
    for field in ["latency_ms", "prompt_tokens", "completion_tokens", "total_tokens"]:
        s = _stats(simple_preds, field)
        a = _stats(agentic_preds, field)
        ratio = a["mean"] / s["mean"] if s["mean"] != 0 else 0.0
        result[field] = {
            "simple": s,
            "agentic": a,
            "ratio": ratio,  # agentic / simple
        }

    return result


# ── Retrieval quality ────────────────────────────────────────────────────────

def compute_retrieval_quality(
    retriever: BaseRetriever,
    examples: List[QAExample],
    top_k: int,
) -> Dict:
    """Compute hit_rate@k and MRR once (shared retriever → same for both agents)."""
    answerable = [ex for ex in examples if not ex.is_unanswerable]
    print(f"Computing retrieval metrics on {len(answerable)} answerable questions...")

    hr = hit_rate_at_k(retriever, examples, top_k=top_k)
    mrr = mean_reciprocal_rank(retriever, examples, top_k=top_k)

    return {
        "hit_rate_at_k": hr,
        "mrr": mrr,
        "top_k": top_k,
        "n_answerable": len(answerable),
    }


# ── Top-level comparison ─────────────────────────────────────────────────────

def run_comparison(
    simple_result: EvaluationResult,
    agentic_result: EvaluationResult,
    simple_predictions: List[PredictionResult],
    agentic_predictions: List[PredictionResult],
    examples: List[QAExample],
    retriever: BaseRetriever,
    config: dict,
) -> Tuple[Dict, Path]:
    """Build full side-by-side comparison and save to JSON.

    Returns:
        (comparison_dict, path_to_comparison_json)
    """
    results_dir = Path(config.get("pipeline", {}).get("results_dir", "evaluation/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    top_k = config.get("retriever", {}).get("top_k", 5)

    # Verify both agents answered the same questions
    simple_ids = [p.id for p in simple_predictions]
    agentic_ids = [p.id for p in agentic_predictions]
    if simple_ids != agentic_ids:
        # Find intersection and warn
        common = set(simple_ids) & set(agentic_ids)
        logger.warning(
            f"Question ID mismatch: simple={len(simple_ids)}, agentic={len(agentic_ids)}, "
            f"common={len(common)}. Comparing on intersection only."
        )
        # Filter to common IDs preserving order
        simple_predictions = [p for p in simple_predictions if p.id in common]
        agentic_predictions = [p for p in agentic_predictions if p.id in common]

    metric_keys = ["em", "f1", "semantic_similarity", "unanswerable_correct"]

    # Overall comparison
    print("Computing overall comparison...")
    overall = compare_category(
        simple_result.per_question, agentic_result.per_question, metric_keys
    )

    # Per-category breakdown
    simple_ans, simple_unans = _split_by_answerability(simple_result.per_question)
    agentic_ans, agentic_unans = _split_by_answerability(agentic_result.per_question)

    print("Computing answerable subset comparison...")
    answerable = compare_category(simple_ans, agentic_ans, ["em", "f1", "semantic_similarity"])

    print("Computing unanswerable subset comparison...")
    unanswerable = compare_category(simple_unans, agentic_unans, ["unanswerable_correct"])

    # Cost / latency
    print("Computing cost/latency comparison...")
    cost_latency = compare_cost_latency(simple_predictions, agentic_predictions)

    # Retrieval quality (single retriever, computed once)
    print("Computing retrieval quality...")
    retrieval = compute_retrieval_quality(retriever, examples, top_k)

    # Assemble comparison
    comparison = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "n_questions": len(simple_predictions),
            "retriever_type": config.get("retriever", {}).get("type", "unknown"),
            "llm_model": config.get("llm", {}).get("model", "unknown"),
            "seed": config.get("seed", 42),
        },
        "overall": overall,
        "answerable": answerable,
        "unanswerable": unanswerable,
        "cost_latency": cost_latency,
        "retrieval_quality": retrieval,
        "aggregate_summary": {
            "simple_rag": simple_result.to_dict(),
            "agentic_rag": agentic_result.to_dict(),
        },
    }

    # Save
    out_path = results_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"Comparison saved to {out_path}")

    # Print summary table
    _print_summary_table(simple_result, agentic_result, cost_latency)

    return comparison, out_path


def _print_summary_table(
    simple: EvaluationResult,
    agentic: EvaluationResult,
    cost_latency: Dict,
) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 70)
    print(f"{'Metric':<25} {'Simple RAG':>12} {'Agentic RAG':>12} {'Delta':>10}")
    print("-" * 70)

    rows = [
        ("EM", simple.em, agentic.em),
        ("F1", simple.f1, agentic.f1),
        ("Unanswerable Acc", simple.unanswerable_accuracy, agentic.unanswerable_accuracy),
        ("Semantic Accuracy", simple.semantic_accuracy, agentic.semantic_accuracy),
        ("Combined Accuracy", simple.combined_accuracy, agentic.combined_accuracy),
    ]

    for name, s_val, a_val in rows:
        delta = a_val - s_val
        sign = "+" if delta >= 0 else ""
        print(f"{name:<25} {s_val:>12.3f} {a_val:>12.3f} {sign}{delta:>9.3f}")

    print("-" * 70)

    # Cost/latency rows
    latency_s = cost_latency["latency_ms"]["simple"]["mean"]
    latency_a = cost_latency["latency_ms"]["agentic"]["mean"]
    tokens_s = cost_latency["total_tokens"]["simple"]["mean"]
    tokens_a = cost_latency["total_tokens"]["agentic"]["mean"]

    lat_ratio = latency_a / latency_s if latency_s > 0 else 0
    tok_ratio = tokens_a / tokens_s if tokens_s > 0 else 0

    print(f"{'Avg Latency (ms)':<25} {latency_s:>12.0f} {latency_a:>12.0f} {lat_ratio:>9.1f}x")
    print(f"{'Avg Total Tokens':<25} {tokens_s:>12.0f} {tokens_a:>12.0f} {tok_ratio:>9.1f}x")
    print("=" * 70)

    # Tier distribution
    print(f"\nTier Distribution:")
    print(f"  Simple RAG:  {simple.tier_distribution}")
    print(f"  Agentic RAG: {agentic.tier_distribution}")
