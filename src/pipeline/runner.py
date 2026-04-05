"""Pipeline runner — execute an agent on a dataset with checkpointing.

Writes predictions to JSONL incrementally (crash-safe).
Supports resume: on restart, skips already-answered questions.
"""

import logging
import time
from pathlib import Path
from typing import List, Set

from tqdm import tqdm

from src.data.schema import QAExample, PredictionResult
from src.agents.base import BaseAgent

logger = logging.getLogger(__name__)


def _load_completed_ids(predictions_path: Path) -> Set[str]:
    """Read existing predictions JSONL and return set of already-answered IDs.

    Handles malformed last line (crash mid-write) by skipping parse errors.
    """
    completed = set()
    if not predictions_path.exists():
        return completed

    with open(predictions_path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                pred = PredictionResult.from_json(line)
                completed.add(pred.id)
            except Exception:
                logger.warning(f"Skipping malformed line {line_num} in {predictions_path}")
    return completed


def run_agent(
    agent: BaseAgent,
    examples: List[QAExample],
    config: dict,
    resume: bool = True,
) -> Path:
    """Run agent over every example, writing predictions to JSONL.

    Args:
        agent: Instantiated agent (SimpleRAGAgent or AgenticRAGAgent).
        examples: Canonical ordered list — both agents MUST receive the same list.
        config: Merged config dict with pipeline.predictions_dir and agent.type.
        resume: If True, skip questions already in the predictions file.

    Returns:
        Path to the predictions JSONL file.
    """
    agent_type = config.get("agent", {}).get("type", "unknown")
    predictions_dir = Path(config.get("pipeline", {}).get("predictions_dir", "evaluation/predictions"))
    predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictions_dir / f"{agent_type}_predictions.jsonl"

    # Resume: load already-completed IDs
    completed_ids = _load_completed_ids(out_path) if resume else set()
    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} questions already answered")

    pending = [ex for ex in examples if ex.id not in completed_ids]
    if not pending:
        logger.info(f"All {len(examples)} questions already answered, skipping run")
        print(f"[{agent_type}] All {len(examples)} questions already answered.")
        return out_path

    print(f"[{agent_type}] Running on {len(pending)} questions "
          f"({len(completed_ids)} already done, {len(examples)} total)")

    with open(out_path, "a") as f:
        for ex in tqdm(pending, desc=f"{agent_type}", unit="q"):
            try:
                pred = agent.answer(ex)
            except Exception as e:
                logger.warning(f"Error on question {ex.id}: {e}")
                pred = PredictionResult(
                    id=ex.id,
                    question=ex.question,
                    predicted_answer="ERROR",
                    is_unanswerable_pred=False,
                    contexts_used=[],
                    reasoning_trace=f"ERROR: {e}",
                    latency_ms=0.0,
                    agent_type=agent_type,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

            # Write immediately and flush (crash-safe)
            f.write(pred.to_json() + "\n")
            f.flush()

            # Log summary
            answer_preview = pred.predicted_answer[:60].replace("\n", " ")
            logger.info(
                f"{ex.id} | {agent_type} | {answer_preview} | "
                f"{pred.latency_ms:.0f}ms | {pred.total_tokens} tokens"
            )

    print(f"[{agent_type}] Done. Predictions saved to {out_path}")
    return out_path
