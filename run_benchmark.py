"""CLI entry point for the RAG benchmark pipeline.

Usage:
    python run_benchmark.py --mode all              # Full pipeline: run → evaluate → compare
    python run_benchmark.py --mode run               # Run agents only
    python run_benchmark.py --mode evaluate          # Evaluate existing predictions
    python run_benchmark.py --mode compare           # Compare existing evaluation results
    python run_benchmark.py --mode all --no-judge    # Skip Tier 3 LLM judge (faster)
    python run_benchmark.py --mode run --agents simple_rag  # Run single agent
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

from src.data.schema import QAExample, PredictionResult
from src.data.squad_loader import load_squad_v2, sample_examples, load_from_jsonl, export_to_jsonl
from src.agents.llm_client import LLMClient
from src.agents.simple_rag import SimpleRAGAgent
from src.agents.agentic_rag import AgenticRAGAgent
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedding_retriever import EmbeddingRetriever
from src.evaluation.semantic_similarity import SemanticSimilarityScorer
from src.pipeline.runner import run_agent
from src.pipeline.evaluate import run_evaluation, load_predictions, align_predictions_to_examples
from src.pipeline.compare import run_comparison


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config loading ───────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(base_path: str, experiment_path: str) -> dict:
    """Load base config and deep-merge experiment config on top."""
    with open(base_path) as f:
        base = yaml.safe_load(f)

    with open(experiment_path) as f:
        experiment = yaml.safe_load(f)

    return _deep_merge(base, experiment)


# ── Shared resource builders ─────────────────────────────────────────────────

def build_examples(config: dict):
    """Load or sample examples from dataset."""
    ds_cfg = config["dataset"]
    data_path = Path(ds_cfg.get("output_path", "data/squad_v2_dev.jsonl"))

    if data_path.exists():
        print(f"Loading examples from {data_path}")
        examples = load_from_jsonl(str(data_path))
    else:
        print("Dataset file not found, loading SQuAD 2.0 from HuggingFace...")
        all_examples = load_squad_v2(split=ds_cfg.get("split", "validation"))
        examples = sample_examples(
            all_examples,
            num_questions=ds_cfg["num_questions"],
            answerable_ratio=ds_cfg.get("answerable_ratio", 0.5),
            seed=config.get("seed", 42),
        )
        export_to_jsonl(examples, str(data_path))

    # If we have more examples than needed, subsample
    num_q = ds_cfg.get("num_questions", len(examples))
    if len(examples) > num_q:
        examples = sample_examples(
            examples,
            num_questions=num_q,
            answerable_ratio=ds_cfg.get("answerable_ratio", 0.5),
            seed=config.get("seed", 42),
        )

    answerable = sum(1 for e in examples if not e.is_unanswerable)
    unanswerable = len(examples) - answerable
    print(f"Loaded {len(examples)} examples (answerable: {answerable}, unanswerable: {unanswerable})")
    return examples


def build_retriever(config: dict, examples):
    """Build and index the retriever (shared by both agents)."""
    ret_cfg = config.get("retriever", {})
    ret_type = ret_cfg.get("type", "bm25")

    if ret_type == "bm25":
        retriever = BM25Retriever()
    elif ret_type == "embedding":
        retriever = EmbeddingRetriever(
            model_name=ret_cfg.get("embedding_model", "BAAI/bge-large-en-v1.5"),
            cache_dir=ret_cfg.get("cache_dir", "data/embedding_cache"),
        )
    else:
        raise ValueError(f"Unknown retriever type: {ret_type}")

    print(f"Building {ret_type} index on {len(examples)} examples...")
    retriever.index(examples)
    print(f"Retriever indexed {len(retriever.contexts)} unique contexts")
    return retriever


def build_llm_client(config: dict) -> LLMClient:
    """Build the shared LLM client."""
    llm_cfg = config["llm"]
    client = LLMClient(
        base_url=llm_cfg["base_url"],
        model=llm_cfg["model"],
        temperature=llm_cfg.get("temperature", 0),
        max_tokens=llm_cfg.get("max_tokens", 512),
    )
    print(f"LLM client ready: {llm_cfg['model']} @ {llm_cfg['base_url']}")
    return client


def build_agent(agent_type: str, llm_client: LLMClient, retriever, config: dict):
    """Instantiate the correct agent subclass."""
    # Load agent-specific config and merge
    agent_config_path = Path("configs") / f"{agent_type}.yaml"
    if agent_config_path.exists():
        with open(agent_config_path) as f:
            agent_overrides = yaml.safe_load(f) or {}
        merged = _deep_merge(config, agent_overrides)
    else:
        merged = config.copy()
        merged.setdefault("agent", {})["type"] = agent_type

    if agent_type == "simple_rag":
        return SimpleRAGAgent(llm_client, retriever, merged), merged
    elif agent_type == "agentic_rag":
        return AgenticRAGAgent(llm_client, retriever, merged), merged
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


# ── Pipeline modes ───────────────────────────────────────────────────────────

def mode_run(agents_list, llm_client, retriever, examples, config, resume):
    """Run agents on dataset, save predictions JSONL."""
    prediction_paths = {}
    for agent_type in agents_list:
        print(f"\n{'='*60}")
        print(f"Running agent: {agent_type}")
        print(f"{'='*60}")
        agent, agent_config = build_agent(agent_type, llm_client, retriever, config)
        path = run_agent(agent, examples, agent_config, resume=resume)
        prediction_paths[agent_type] = path
    return prediction_paths


def mode_evaluate(agents_list, examples, config, llm_client, semantic_scorer, run_judge):
    """Evaluate predictions for each agent."""
    predictions_dir = Path(config.get("pipeline", {}).get("predictions_dir", "evaluation/predictions"))
    results = {}
    all_predictions = {}

    for agent_type in agents_list:
        print(f"\n{'='*60}")
        print(f"Evaluating agent: {agent_type}")
        print(f"{'='*60}")
        pred_path = predictions_dir / f"{agent_type}_predictions.jsonl"
        if not pred_path.exists():
            print(f"  Predictions file not found: {pred_path} — skipping")
            continue

        result, result_path = run_evaluation(
            predictions_path=pred_path,
            examples=examples,
            config=config,
            llm_client=llm_client,
            semantic_scorer=semantic_scorer,
            run_judge=run_judge,
        )
        results[agent_type] = result

        # Also keep raw predictions for compare step
        preds = load_predictions(pred_path)
        aligned_preds, _ = align_predictions_to_examples(preds, examples)
        all_predictions[agent_type] = aligned_preds

    return results, all_predictions


def mode_compare(results, all_predictions, examples, retriever, config):
    """Compare two agents side-by-side."""
    if "simple_rag" not in results or "agentic_rag" not in results:
        print("Need both simple_rag and agentic_rag results for comparison. Skipping.")
        return None

    print(f"\n{'='*60}")
    print("Comparing Simple RAG vs Agentic RAG")
    print(f"{'='*60}")

    comparison, comp_path = run_comparison(
        simple_result=results["simple_rag"],
        agentic_result=results["agentic_rag"],
        simple_predictions=all_predictions["simple_rag"],
        agentic_predictions=all_predictions["agentic_rag"],
        examples=examples,
        retriever=retriever,
        config=config,
    )
    return comparison


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark Runner")
    parser.add_argument("--base-config", default="configs/base.yaml",
                        help="Path to base config YAML")
    parser.add_argument("--experiment-config", default="configs/experiment.yaml",
                        help="Path to experiment config YAML")
    parser.add_argument("--mode", choices=["run", "evaluate", "compare", "all"],
                        default="all", help="Pipeline mode")
    parser.add_argument("--agents", nargs="+", choices=["simple_rag", "agentic_rag"],
                        default=None, help="Override which agents to run/evaluate")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip Tier 3 LLM judge (faster evaluation)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoints and start fresh")
    args = parser.parse_args()

    # Load config
    config = load_config(args.base_config, args.experiment_config)
    agents_list = args.agents or config.get("agents", ["simple_rag", "agentic_rag"])
    run_judge = not args.no_judge and config.get("pipeline", {}).get("run_judge", True)
    resume = not args.no_resume and config.get("pipeline", {}).get("resume", True)

    print(f"Mode: {args.mode}")
    print(f"Agents: {agents_list}")
    print(f"Run judge: {run_judge}")
    print(f"Resume: {resume}")
    print(f"Dataset: {config['dataset']['num_questions']} questions, seed={config.get('seed', 42)}")
    print()

    # Build shared resources
    examples = build_examples(config)
    retriever = build_retriever(config, examples)
    llm_client = build_llm_client(config)
    semantic_scorer = SemanticSimilarityScorer(
        model_name=config.get("retriever", {}).get("embedding_model", "BAAI/bge-large-en-v1.5")
    )

    if args.mode in ("run", "all"):
        mode_run(agents_list, llm_client, retriever, examples, config, resume)

    results = {}
    all_predictions = {}
    if args.mode in ("evaluate", "all"):
        results, all_predictions = mode_evaluate(
            agents_list, examples, config, llm_client, semantic_scorer, run_judge
        )

    if args.mode in ("compare", "all"):
        # If we didn't just evaluate, load existing results
        if not results:
            results_dir = Path(config.get("pipeline", {}).get("results_dir", "evaluation/results"))
            predictions_dir = Path(config.get("pipeline", {}).get("predictions_dir", "evaluation/predictions"))
            for agent_type in agents_list:
                result_path = results_dir / f"{agent_type}_results.json"
                pred_path = predictions_dir / f"{agent_type}_predictions.jsonl"
                if result_path.exists():
                    with open(result_path) as f:
                        from src.evaluation.evaluator import EvaluationResult
                        from dataclasses import fields
                        data = json.load(f)
                        results[agent_type] = EvaluationResult(**data)
                if pred_path.exists():
                    preds = load_predictions(pred_path)
                    aligned_preds, _ = align_predictions_to_examples(preds, examples)
                    all_predictions[agent_type] = aligned_preds

        mode_compare(results, all_predictions, examples, retriever, config)

    print("\nDone.")


if __name__ == "__main__":
    main()
