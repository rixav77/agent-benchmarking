"""Pipeline evaluation — load predictions JSONL, run 3-tier evaluation, save results.

Uses the SAME evaluator.evaluate() codepath for both agents.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from src.data.schema import QAExample, PredictionResult
from src.evaluation.evaluator import evaluate, EvaluationResult
from src.evaluation.semantic_similarity import SemanticSimilarityScorer
from src.agents.llm_client import LLMClient

logger = logging.getLogger(__name__)


def load_predictions(predictions_path: Path) -> List[PredictionResult]:
    """Load PredictionResult objects from a JSONL file.

    Skips lines with predicted_answer == "ERROR" (sentinel from runner errors).
    """
    predictions = []
    skipped = 0
    with open(predictions_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pred = PredictionResult.from_json(line)
                if pred.predicted_answer == "ERROR":
                    skipped += 1
                    continue
                predictions.append(pred)
            except Exception:
                logger.warning(f"Skipping malformed line {line_num} in {predictions_path}")
                skipped += 1

    if skipped:
        logger.info(f"Skipped {skipped} error/malformed predictions")
    logger.info(f"Loaded {len(predictions)} predictions from {predictions_path}")
    return predictions


def align_predictions_to_examples(
    predictions: List[PredictionResult],
    examples: List[QAExample],
) -> Tuple[List[PredictionResult], List[QAExample]]:
    """Align predictions to examples by ID in canonical examples order.

    Returns (aligned_predictions, aligned_examples) where both lists
    have the same length and are in the same order.
    """
    pred_map = {p.id: p for p in predictions}
    aligned_preds = []
    aligned_exs = []

    for ex in examples:
        if ex.id in pred_map:
            aligned_preds.append(pred_map[ex.id])
            aligned_exs.append(ex)
        else:
            logger.warning(f"No prediction for question {ex.id}, skipping")

    if not aligned_preds:
        raise ValueError("No predictions matched any examples — check IDs")

    logger.info(f"Aligned {len(aligned_preds)}/{len(examples)} predictions to examples")
    return aligned_preds, aligned_exs


def run_evaluation(
    predictions_path: Path,
    examples: List[QAExample],
    config: dict,
    llm_client: Optional[LLMClient] = None,
    semantic_scorer: Optional[SemanticSimilarityScorer] = None,
    run_judge: bool = True,
) -> Tuple[EvaluationResult, Path]:
    """Load predictions, align with gold, run 3-tier evaluation, save results.

    Args:
        predictions_path: Path to agent's predictions JSONL.
        examples: Canonical ordered gold examples.
        config: Merged config dict.
        llm_client: Shared LLMClient for Tier 3 judge. None to skip.
        semantic_scorer: Shared scorer. Created if None.
        run_judge: Whether to run Tier 3.

    Returns:
        (EvaluationResult, path_to_results_json).
    """
    # Infer agent type from filename
    agent_type = predictions_path.stem.replace("_predictions", "")
    results_dir = Path(config.get("pipeline", {}).get("results_dir", "evaluation/results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{agent_type}] Loading predictions from {predictions_path}")
    predictions = load_predictions(predictions_path)

    print(f"[{agent_type}] Aligning {len(predictions)} predictions to {len(examples)} examples")
    aligned_preds, aligned_exs = align_predictions_to_examples(predictions, examples)

    print(f"[{agent_type}] Running 3-tier evaluation on {len(aligned_preds)} questions...")
    result = evaluate(
        predictions=aligned_preds,
        examples=aligned_exs,
        config=config,
        llm_client=llm_client if run_judge else None,
        semantic_scorer=semantic_scorer,
        run_judge=run_judge,
    )

    # Save full results (with per-question details)
    full_path = results_dir / f"{agent_type}_results.json"
    with open(full_path, "w") as f:
        f.write(result.to_json(include_per_question=True))
    print(f"[{agent_type}] Full results saved to {full_path}")

    # Save summary (without per-question)
    summary_path = results_dir / f"{agent_type}_results_summary.json"
    with open(summary_path, "w") as f:
        f.write(result.to_json(include_per_question=False))
    print(f"[{agent_type}] Summary saved to {summary_path}")

    return result, full_path
