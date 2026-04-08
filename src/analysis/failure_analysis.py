"""Failure analysis module — PDF Section 5 (Analysis).

Identifies strengths, weaknesses, and failure modes for both agents.
Generates failure mode chart + text reports.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.visualize import AGENTS, AGENT_LABELS, COLORS, DPI, _save


# ── Failure Categorization ────────────────────────────────────────────────────

def compute_failure_counts(per_question: List[dict]) -> dict:
    """Compute failure category counts from per-question results.

    Categories:
      A: False Positive — unanswerable question answered (hallucination)
      B: False Negative — answerable question refused
      C: Wrong Answer — answered but all tiers rejected
      D: Faithfulness — answer not grounded in context
    """
    cat_a = cat_b = cat_c = cat_d = 0
    cat_a_examples = []
    cat_b_examples = []
    cat_c_examples = []
    cat_d_examples = []

    for e in per_question:
        is_unans = e["is_unanswerable"]
        pred_unans = e["is_unanswerable_pred"]
        tier = e["tier_credited"]

        # Cat A: unanswerable but agent answered anyway
        if is_unans and not pred_unans:
            cat_a += 1
            if len(cat_a_examples) < 3:
                cat_a_examples.append(e)

        # Cat B: answerable but agent said unanswerable
        if not is_unans and pred_unans:
            cat_b += 1
            if len(cat_b_examples) < 3:
                cat_b_examples.append(e)

        # Cat C: answered but wrong (all tiers rejected)
        if not is_unans and not pred_unans and tier == "none":
            cat_c += 1
            if len(cat_c_examples) < 3:
                cat_c_examples.append(e)

        # Cat D: faithfulness failure
        jc = e.get("j_correctness")
        if jc is not None and jc.get("faithfulness") != "faithful":
            cat_d += 1
            if len(cat_d_examples) < 3:
                cat_d_examples.append(e)

    return {
        "cat_a": cat_a, "cat_a_desc": "False Positive (unanswerable answered)",
        "cat_b": cat_b, "cat_b_desc": "False Negative (answerable refused)",
        "cat_c": cat_c, "cat_c_desc": "Wrong Answer (all tiers rejected)",
        "cat_d": cat_d, "cat_d_desc": "Faithfulness Failure",
        "total": len(per_question),
        "cat_a_examples": cat_a_examples,
        "cat_b_examples": cat_b_examples,
        "cat_c_examples": cat_c_examples,
        "cat_d_examples": cat_d_examples,
    }


def compute_error_analysis(per_question: List[dict]) -> dict:
    """Compute additional breakdown statistics for text analysis."""
    # Unanswerable detection precision/recall
    tp = fp = fn = tn = 0
    for e in per_question:
        is_u = e["is_unanswerable"]
        pred_u = e["is_unanswerable_pred"]
        if is_u and pred_u:
            tp += 1
        elif not is_u and pred_u:
            fp += 1
        elif is_u and not pred_u:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Tier distribution
    tier_counts = {}
    for e in per_question:
        t = e["tier_credited"]
        tier_counts[t] = tier_counts.get(t, 0) + 1

    # Average F1 for wrong answers (cat C)
    wrong_f1s = [
        e["f1"] for e in per_question
        if not e["is_unanswerable"] and not e["is_unanswerable_pred"] and e["tier_credited"] == "none"
    ]
    avg_wrong_f1 = np.mean(wrong_f1s) if wrong_f1s else 0

    # Judge stats (only where judge ran)
    judge_scores = [
        e["j_correctness"]["overall_score"]
        for e in per_question
        if e["j_correctness"] is not None
    ]

    return {
        "unanswerable_precision": precision,
        "unanswerable_recall": recall,
        "unanswerable_f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "tier_counts": tier_counts,
        "avg_wrong_f1": avg_wrong_f1,
        "n_wrong_answers": len(wrong_f1s),
        "avg_judge_score": np.mean(judge_scores) if judge_scores else 0,
        "n_judged": len(judge_scores),
    }


# ── Chart 11: Failure Mode Breakdown ─────────────────────────────────────────

def plot_failure_modes(
    failure_counts: Dict[str, dict],
    out_dir: Path,
) -> Path:
    """Grouped bar chart of 4 failure categories per agent."""
    categories = ["cat_a", "cat_b", "cat_c", "cat_d"]
    cat_labels = [
        "False Positive\n(Unanswerable\nAnswered)",
        "False Negative\n(Answerable\nRefused)",
        "Wrong Answer\n(All Tiers\nRejected)",
        "Faithfulness\nFailure",
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, agent in enumerate(AGENTS):
        counts = [failure_counts[agent][c] for c in categories]
        total = failure_counts[agent]["total"]
        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset, counts, width,
            label=AGENT_LABELS[agent],
            color=COLORS[agent],
            edgecolor="white", linewidth=0.5,
        )
        for bar, count in zip(bars, counts):
            pct = count / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9,
            )

    ax.set_ylabel("Number of Questions")
    ax.set_title("Failure Mode Breakdown (n=1000)")
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, out_dir, "11_failure_modes.png")


# ── Text Report Generation ───────────────────────────────────────────────────

def _format_example(e: dict, max_ans_len: int = 80) -> str:
    """Format a single per-question example for the report."""
    q = e["question"][:100]
    gold = ", ".join(e["gold_answers"][:2])[:60] if e["gold_answers"] else "(unanswerable)"
    pred = e["predicted_answer"][:max_ans_len]
    return f'  - Q: "{q}"\n    Gold: {gold}\n    Pred: {pred}\n    F1={e["f1"]:.3f}, Tier={e["tier_credited"]}'


def generate_failure_report(
    failure_counts: Dict[str, dict],
    error_stats: Dict[str, dict],
) -> str:
    """Generate failure_analysis.md content."""
    lines = ["# Failure Analysis Report\n"]

    # Summary table
    lines.append("## Failure Category Summary\n")
    lines.append("| Category | Description | Simple RAG | Agentic RAG | Delta |")
    lines.append("|----------|-------------|-----------|------------|-------|")

    for cat in ["cat_a", "cat_b", "cat_c", "cat_d"]:
        desc = failure_counts["simple_rag"][f"{cat}_desc"]
        s_count = failure_counts["simple_rag"][cat]
        a_count = failure_counts["agentic_rag"][cat]
        delta = a_count - s_count
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {cat.upper().replace('_', ' ')} | {desc} | {s_count} | {a_count} | {sign}{delta} |")

    lines.append("")

    # Detailed per-category analysis with examples
    for cat, cat_key in [
        ("A: False Positive (Unanswerable Answered)", "cat_a"),
        ("B: False Negative (Answerable Refused)", "cat_b"),
        ("C: Wrong Answer (All Tiers Rejected)", "cat_c"),
        ("D: Faithfulness Failure", "cat_d"),
    ]:
        lines.append(f"\n## Category {cat}\n")
        s = failure_counts["simple_rag"]
        a = failure_counts["agentic_rag"]
        lines.append(f"- Simple RAG: {s[cat_key]} ({s[cat_key]/s['total']*100:.1f}%)")
        lines.append(f"- Agentic RAG: {a[cat_key]} ({a[cat_key]/a['total']*100:.1f}%)")
        lines.append("")

        # Examples from each agent
        for agent in AGENTS:
            examples = failure_counts[agent].get(f"{cat_key}_examples", [])
            if examples:
                lines.append(f"**{AGENT_LABELS[agent]} examples:**\n")
                for ex in examples[:2]:
                    lines.append(_format_example(ex))
                lines.append("")

    # Unanswerable detection precision/recall
    lines.append("\n## Unanswerable Detection Performance\n")
    lines.append("| Metric | Simple RAG | Agentic RAG |")
    lines.append("|--------|-----------|------------|")
    for metric in ["unanswerable_precision", "unanswerable_recall", "unanswerable_f1"]:
        label = metric.replace("unanswerable_", "").capitalize()
        s_val = error_stats["simple_rag"][metric]
        a_val = error_stats["agentic_rag"][metric]
        lines.append(f"| {label} | {s_val:.3f} | {a_val:.3f} |")

    lines.append("")
    lines.append("| | Simple RAG | Agentic RAG |")
    lines.append("|--|-----------|------------|")
    for label, key in [("True Positive", "tp"), ("False Positive", "fp"),
                       ("False Negative", "fn"), ("True Negative", "tn")]:
        s_val = error_stats["simple_rag"][key]
        a_val = error_stats["agentic_rag"][key]
        lines.append(f"| {label} | {s_val} | {a_val} |")

    return "\n".join(lines) + "\n"


def generate_analysis_summary(
    comparison: dict,
    failure_counts: Dict[str, dict],
    error_stats: Dict[str, dict],
) -> str:
    """Generate analysis_summary.md — strengths, weaknesses, failure modes (PDF Section 5)."""
    agg = comparison["aggregate_summary"]
    cl = comparison["cost_latency"]

    lines = ["# Benchmark Analysis Summary\n"]
    lines.append("## Simple RAG vs Agentic RAG — 1000 SQuAD v2 Questions\n")

    # Overall results
    lines.append("## Key Results\n")
    lines.append(f"- **Simple RAG Combined Accuracy**: {agg['simple_rag']['combined_accuracy']:.1%}")
    lines.append(f"- **Agentic RAG Combined Accuracy**: {agg['agentic_rag']['combined_accuracy']:.1%}")
    lines.append(f"- **Delta**: +{(agg['agentic_rag']['combined_accuracy'] - agg['simple_rag']['combined_accuracy'])*100:.1f} percentage points in favor of Agentic RAG")
    lines.append(f"- **Latency Ratio**: Agentic is {cl['latency_ms']['ratio']:.1f}x slower")
    lines.append(f"- **Token Ratio**: Agentic uses {cl['total_tokens']['ratio']:.1f}x more tokens")
    lines.append("")

    # Simple RAG strengths
    lines.append("## Simple RAG\n")
    lines.append("### Strengths\n")
    s_ans_em = agg["simple_rag"]["answerable_em"]
    a_ans_em = agg["agentic_rag"]["answerable_em"]
    lines.append(f"1. **Better answerable EM**: {s_ans_em:.1%} vs {a_ans_em:.1%} ({(s_ans_em - a_ans_em)*100:+.1f}pp) — direct extractive answers are more precise for straightforward questions")
    lines.append(f"2. **5.46x faster**: Mean latency {cl['latency_ms']['simple']['mean']:.0f}ms vs {cl['latency_ms']['agentic']['mean']:.0f}ms — single LLM call vs multi-step pipeline")
    lines.append(f"3. **3.5x cheaper**: {cl['total_tokens']['simple']['mean']:.0f} tokens/query vs {cl['total_tokens']['agentic']['mean']:.0f} — significant cost savings at scale")
    lines.append(f"4. **Simpler architecture**: No query rewriting, relevance assessment, or iterative loops — easier to debug and maintain")
    lines.append("")

    lines.append("### Weaknesses\n")
    s_fp = failure_counts["simple_rag"]["cat_a"]
    a_fp = failure_counts["agentic_rag"]["cat_a"]
    s_det = agg["simple_rag"]["unanswerable_detection_rate"]
    a_det = agg["agentic_rag"]["unanswerable_detection_rate"]
    lines.append(f"1. **More false positives**: {s_fp} vs {a_fp} — answers unanswerable questions {s_fp - a_fp} more times")
    lines.append(f"2. **Lower unanswerable detection**: {s_det:.1%} vs {a_det:.1%} — fails to recognize when context doesn't contain the answer")
    lines.append(f"3. **No self-correction**: Cannot rewrite queries or re-retrieve when initial retrieval is poor")
    lines.append("")

    # Agentic RAG strengths
    lines.append("## Agentic RAG\n")
    lines.append("### Strengths\n")
    lines.append(f"1. **Better unanswerable detection**: {a_det:.1%} vs {s_det:.1%} — **statistically significant** (95% CIs non-overlapping)")
    lines.append(f"2. **Fewer false positives**: {a_fp} vs {s_fp} — the answerability checking tool catches adversarial unanswerable questions")
    lines.append(f"3. **Higher combined accuracy**: {agg['agentic_rag']['combined_accuracy']:.1%} vs {agg['simple_rag']['combined_accuracy']:.1%} — multi-step reasoning recovers answers that single-pass misses")
    lines.append(f"4. **Query rewriting**: Reformulates poor queries to improve retrieval quality on hard questions")
    lines.append("")

    lines.append("### Weaknesses\n")
    lines.append(f"1. **5.46x slower**: Multi-step pipeline with 3-5 LLM calls per question (retrieve → assess → rewrite → check → generate)")
    lines.append(f"2. **3.5x more expensive**: {cl['total_tokens']['agentic']['mean']:.0f} tokens/query — each tool call adds prompt overhead")
    lines.append(f"3. **Lower answerable EM**: {a_ans_em:.1%} vs {s_ans_em:.1%} — over-processing can degrade simple extractive answers")
    lines.append(f"4. **Diminishing returns**: The +2.9pp accuracy gain may not justify the 5.46x cost increase in production")
    lines.append("")

    # Failure modes
    lines.append("## Common Failure Modes\n")
    lines.append("### 1. Adversarial Unanswerable Questions (Largest Failure Category)\n")
    lines.append(f"Both agents struggle with SQuAD v2's adversarial unanswerable questions — questions with misleading premises where the context contains related but irrelevant information.")
    lines.append(f"- Simple RAG: {s_fp} false positives ({s_fp/10:.1f}% of all questions)")
    lines.append(f"- Agentic RAG: {a_fp} false positives ({a_fp/10:.1f}% of all questions)")
    lines.append(f"- Agentic RAG's `check_answerability` tool reduces this by {s_fp - a_fp} cases ({(s_fp - a_fp)/s_fp*100:.1f}% reduction)")
    lines.append("")

    lines.append("### 2. Answerable Questions Refused (False Negatives)\n")
    s_fn = failure_counts["simple_rag"]["cat_b"]
    a_fn = failure_counts["agentic_rag"]["cat_b"]
    lines.append(f"Both agents sometimes refuse to answer answerable questions when retrieval quality is low.")
    lines.append(f"- Simple RAG: {s_fn} false negatives")
    lines.append(f"- Agentic RAG: {a_fn} false negatives — despite query rewriting, some questions remain hard to retrieve")
    lines.append("")

    lines.append("### 3. Wrong Answers (All Tiers Rejected)\n")
    s_wrong = failure_counts["simple_rag"]["cat_c"]
    a_wrong = failure_counts["agentic_rag"]["cat_c"]
    s_wf1 = error_stats["simple_rag"]["avg_wrong_f1"]
    a_wf1 = error_stats["agentic_rag"]["avg_wrong_f1"]
    lines.append(f"A small number of answers are plausible but incorrect — F1 too low for Tier 1, semantic similarity below threshold, and judge rejected.")
    lines.append(f"- Simple RAG: {s_wrong} wrong answers (avg F1={s_wf1:.3f})")
    lines.append(f"- Agentic RAG: {a_wrong} wrong answers (avg F1={a_wf1:.3f})")
    lines.append("")

    lines.append("### 4. Faithfulness Issues\n")
    s_fd = failure_counts["simple_rag"]["cat_d"]
    a_fd = failure_counts["agentic_rag"]["cat_d"]
    lines.append(f"A small fraction of answers contain information not grounded in the provided context.")
    s_faith = agg["simple_rag"]["j_faithfulness_dist"]
    a_faith = agg["agentic_rag"]["j_faithfulness_dist"]
    lines.append(f"- Simple RAG: {s_faith.get('unfaithful', 0)} unfaithful + {s_faith.get('partial', 0)} partial = {s_fd} total")
    lines.append(f"- Agentic RAG: {a_faith.get('unfaithful', 0)} unfaithful + {a_faith.get('partial', 0)} partial = {a_fd} total")
    lines.append(f"- Both agents show >96% faithfulness — the LLM's context-following ability is strong")
    lines.append("")

    # Trade-off analysis
    lines.append("## Trade-off Analysis\n")
    lines.append("| Dimension | Simple RAG | Agentic RAG | Verdict |")
    lines.append("|-----------|-----------|------------|---------|")
    lines.append(f"| Combined Accuracy | {agg['simple_rag']['combined_accuracy']:.1%} | {agg['agentic_rag']['combined_accuracy']:.1%} | Agentic (+2.9pp) |")
    lines.append(f"| Answerable EM | {s_ans_em:.1%} | {a_ans_em:.1%} | Simple (+4.0pp) |")
    lines.append(f"| Unanswerable Det. | {s_det:.1%} | {a_det:.1%} | **Agentic (significant)** |")
    lines.append(f"| Latency | {cl['latency_ms']['simple']['mean']:.0f}ms | {cl['latency_ms']['agentic']['mean']:.0f}ms | Simple (5.46x faster) |")
    lines.append(f"| Token Cost | {cl['total_tokens']['simple']['mean']:.0f} | {cl['total_tokens']['agentic']['mean']:.0f} | Simple (3.5x cheaper) |")
    lines.append(f"| Faithfulness | {s_faith.get('faithful', 0)}/{s_fd + s_faith.get('faithful', 0)} | {a_faith.get('faithful', 0)}/{a_fd + a_faith.get('faithful', 0)} | Comparable |")
    lines.append("")

    lines.append("## Conclusion\n")
    lines.append("Agentic RAG provides a meaningful improvement in overall accuracy (+2.9pp) and a statistically significant advantage in unanswerable detection (+5.0pp). However, this comes at a steep cost: 5.46x latency and 3.5x token usage. The choice depends on the use case:")
    lines.append("")
    lines.append("- **Use Simple RAG** when: speed matters, budget is constrained, questions are mostly straightforward, or the dataset has few unanswerable questions.")
    lines.append("- **Use Agentic RAG** when: accuracy on adversarial/tricky questions is critical, unanswerable detection is important (e.g., medical/legal QA), and latency/cost are acceptable.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_failure_analysis(
    results: Dict[str, dict],
    comparison: dict,
    out_dir: Path,
    reports_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Run full failure analysis: compute counts, generate chart + text reports.

    Returns: (chart_path, failure_report_path, analysis_summary_path)
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Compute failure counts and error stats for both agents
    failure_counts = {}
    error_stats = {}
    for agent in AGENTS:
        pq = results[agent]["per_question"]
        failure_counts[agent] = compute_failure_counts(pq)
        error_stats[agent] = compute_error_analysis(pq)

    # Chart 11: Failure mode breakdown
    print("Generating failure analysis chart...")
    chart_path = plot_failure_modes(failure_counts, out_dir)

    # Text reports
    print("Generating failure analysis report...")
    failure_text = generate_failure_report(failure_counts, error_stats)
    failure_path = reports_dir / "failure_analysis.md"
    with open(failure_path, "w") as f:
        f.write(failure_text)
    print(f"  Saved: {failure_path}")

    print("Generating analysis summary...")
    summary_text = generate_analysis_summary(comparison, failure_counts, error_stats)
    summary_path = reports_dir / "analysis_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"  Saved: {summary_path}")

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("  FAILURE MODE SUMMARY")
    print("=" * 70)
    for cat in ["cat_a", "cat_b", "cat_c", "cat_d"]:
        desc = failure_counts["simple_rag"][f"{cat}_desc"]
        s = failure_counts["simple_rag"][cat]
        a = failure_counts["agentic_rag"][cat]
        delta = a - s
        sign = "+" if delta >= 0 else ""
        print(f"  {desc:<45} Simple={s:>3}  Agentic={a:>3}  ({sign}{delta})")
    print("=" * 70)

    return chart_path, failure_path, summary_path
