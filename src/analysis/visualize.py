"""Visualization module — all benchmark charts for Phase 9.

Generates 10 charts covering PDF Section 6 (Visualization):
score distributions, model comparisons, confusion matrices, cost/latency.
"""

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ── Constants ─────────────────────────────────────────────────────────────────

AGENTS = ["simple_rag", "agentic_rag"]
AGENT_LABELS = {"simple_rag": "Simple RAG", "agentic_rag": "Agentic RAG"}
# Comparison.json uses short keys ("simple"/"agentic") inside metric dicts
AGENT_SHORT = {"simple_rag": "simple", "agentic_rag": "agentic"}
COLORS = {"simple_rag": "#4C72B0", "agentic_rag": "#DD8452"}
DPI = 150

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})


# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_results(agent: str) -> dict:
    """Load full evaluation results (with per_question array)."""
    path = Path(f"evaluation/results/{agent}_results.json")
    with open(path) as f:
        return json.load(f)


def load_predictions(agent: str) -> List[dict]:
    """Load raw predictions from JSONL."""
    path = Path(f"evaluation/predictions/{agent}_predictions.jsonl")
    preds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))
    return preds


def load_comparison() -> dict:
    """Load comparison.json."""
    with open("evaluation/results/comparison.json") as f:
        return json.load(f)


# ── Helper ────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, filename: str) -> Path:
    """Save figure and close."""
    path = out_dir / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ── Chart 1: Metric Comparison with 95% CI Error Bars ────────────────────────

def plot_metric_comparison(comparison: dict, out_dir: Path) -> Path:
    """Overall metrics (EM, F1, Combined Accuracy, Unanswerable Detection) with CIs."""
    overall = comparison["overall"]["metrics"]
    agg = comparison["aggregate_summary"]

    # Metrics to plot — first 3 from overall (have CIs), combined from aggregate (no CI)
    metrics_cfg = [
        ("EM", overall.get("em")),
        ("F1", overall.get("f1")),
        ("Unanswerable\nDetection", overall.get("unanswerable_correct")),
    ]

    labels = [m[0] for m in metrics_cfg]
    labels.append("Combined\nAccuracy")

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, agent in enumerate(AGENTS):
        short = AGENT_SHORT[agent]
        means, err_low, err_high = [], [], []
        for _, metric_data in metrics_cfg:
            a = metric_data[short]
            means.append(a["mean"])
            err_low.append(a["mean"] - a["ci_lower"])
            err_high.append(a["ci_upper"] - a["mean"])

        # Combined accuracy — no CI available
        means.append(agg[agent]["combined_accuracy"])
        err_low.append(0)
        err_high.append(0)

        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=[err_low, err_high],
            label=AGENT_LABELS[agent],
            color=COLORS[agent],
            capsize=4, edgecolor="white", linewidth=0.5,
        )
        # Value labels
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_ylabel("Score")
    ax.set_title("Overall Metric Comparison (with 95% Confidence Intervals)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    return _save(fig, out_dir, "01_metric_comparison.png")


# ── Chart 2: Per-Category Breakdown ──────────────────────────────────────────

def plot_category_breakdown(comparison: dict, out_dir: Path) -> Path:
    """Answerable vs Unanswerable subset metrics with CIs."""
    ans = comparison["answerable"]["metrics"]
    unans = comparison["unanswerable"]["metrics"]

    metrics_cfg = [
        ("Answerable\nEM", ans["em"]),
        ("Answerable\nF1", ans["f1"]),
        ("Unanswerable\nDetection", unans["unanswerable_correct"]),
    ]

    labels = [m[0] for m in metrics_cfg]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, agent in enumerate(AGENTS):
        short = AGENT_SHORT[agent]
        means, err_low, err_high = [], [], []
        for _, metric_data in metrics_cfg:
            a = metric_data[short]
            means.append(a["mean"])
            err_low.append(a["mean"] - a["ci_lower"])
            err_high.append(a["ci_upper"] - a["mean"])

        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset, means, width,
            yerr=[err_low, err_high],
            label=AGENT_LABELS[agent],
            color=COLORS[agent],
            capsize=4, edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_ylabel("Score")
    ax.set_title("Per-Category Breakdown (Answerable vs Unanswerable)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, out_dir, "02_category_breakdown.png")


# ── Chart 3: F1 Score Distribution ───────────────────────────────────────────

def plot_f1_distribution(results: Dict[str, dict], out_dir: Path) -> Path:
    """F1 score histograms with KDE for both agents."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax_idx, (title, filter_fn) in enumerate([
        ("Answerable Questions Only", lambda e: not e["is_unanswerable"]),
        ("All Questions", lambda e: True),
    ]):
        ax = axes[ax_idx]
        for agent in AGENTS:
            pq = results[agent]["per_question"]
            f1_vals = [e["f1"] for e in pq if filter_fn(e)]
            ax.hist(
                f1_vals, bins=30, alpha=0.5, density=True,
                label=AGENT_LABELS[agent], color=COLORS[agent], edgecolor="white",
            )
            # KDE overlay
            if len(f1_vals) > 1:
                sns.kdeplot(f1_vals, ax=ax, color=COLORS[agent], linewidth=2)

        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Density" if ax_idx == 0 else "")
        ax.set_title(title)
        ax.legend()

    fig.suptitle("F1 Score Distribution", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "03_f1_distribution.png")


# ── Chart 4: Tier Distribution ───────────────────────────────────────────────

def plot_tier_distribution(comparison: dict, out_dir: Path) -> Path:
    """Grouped bars showing tier1/tier2/tier3/none counts per agent."""
    agg = comparison["aggregate_summary"]
    tiers = ["tier1", "tier2", "tier3", "none"]
    tier_labels = ["Tier 1\n(EM/F1)", "Tier 2\n(Semantic)", "Tier 3\n(Judge)", "None\n(Incorrect)"]

    x = np.arange(len(tiers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, agent in enumerate(AGENTS):
        td = agg[agent]["tier_distribution"]
        counts = [td.get(t, 0) for t in tiers]
        total = sum(counts)
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
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{count}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9,
            )

    ax.set_ylabel("Number of Questions")
    ax.set_title("3-Tier Evaluation Credit Distribution (n=1000)")
    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return _save(fig, out_dir, "04_tier_distribution.png")


# ── Chart 5: Cost-Accuracy Trade-off ─────────────────────────────────────────

def plot_cost_accuracy_tradeoff(comparison: dict, out_dir: Path) -> Path:
    """Scatter plot: latency vs combined accuracy for both agents."""
    cl = comparison["cost_latency"]
    agg = comparison["aggregate_summary"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for agent in AGENTS:
        short = AGENT_SHORT[agent]
        latency = cl["latency_ms"][short]["mean"]
        accuracy = agg[agent]["combined_accuracy"]
        tokens = cl["total_tokens"][short]["mean"]

        ax.scatter(
            latency, accuracy, s=200, c=COLORS[agent],
            label=AGENT_LABELS[agent], zorder=5, edgecolors="black", linewidth=1,
        )
        ax.annotate(
            f"{AGENT_LABELS[agent]}\nAcc={accuracy:.3f}\nLatency={latency:.0f}ms\nTokens={tokens:.0f}",
            xy=(latency, accuracy),
            xytext=(20, -20 if agent == "simple_rag" else 20),
            textcoords="offset points",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="gray"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

    # Annotate ratio
    lat_ratio = cl["latency_ms"].get("ratio", cl["latency_ms"]["agentic"]["mean"] / cl["latency_ms"]["simple"]["mean"])
    tok_ratio = cl["total_tokens"].get("ratio", cl["total_tokens"]["agentic"]["mean"] / cl["total_tokens"]["simple"]["mean"])
    ax.text(
        0.02, 0.98,
        f"Agentic is {lat_ratio:.1f}x slower, {tok_ratio:.1f}x more tokens\n"
        f"for +{(agg['agentic_rag']['combined_accuracy'] - agg['simple_rag']['combined_accuracy'])*100:.1f}pp accuracy",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Mean Latency (ms)")
    ax.set_ylabel("Combined Accuracy")
    ax.set_title("Cost-Accuracy Trade-off")
    ax.legend(loc="lower right")

    fig.tight_layout()
    return _save(fig, out_dir, "05_cost_accuracy_tradeoff.png")


# ── Chart 6: Latency Distribution ────────────────────────────────────────────

def plot_latency_distribution(preds: Dict[str, list], out_dir: Path) -> Path:
    """Latency histograms with p50/p95 lines."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, agent in zip(axes, AGENTS):
        latencies = [p["latency_ms"] for p in preds[agent]]
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        mean = np.mean(latencies)

        ax.hist(latencies, bins=50, color=COLORS[agent], alpha=0.7, edgecolor="white")
        ax.axvline(p50, color="green", linestyle="--", linewidth=2, label=f"p50={p50:.0f}ms")
        ax.axvline(p95, color="red", linestyle="--", linewidth=2, label=f"p95={p95:.0f}ms")
        ax.axvline(mean, color="black", linestyle="-", linewidth=1.5, label=f"mean={mean:.0f}ms")

        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"{AGENT_LABELS[agent]} Latency Distribution")
        ax.legend(fontsize=9)

    fig.suptitle("Response Latency Distribution (n=1000)", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "06_latency_distribution.png")


# ── Chart 7: Token Usage Comparison ──────────────────────────────────────────

def plot_token_usage(comparison: dict, out_dir: Path) -> Path:
    """Grouped bars for prompt/completion/total tokens with ratio labels."""
    cl = comparison["cost_latency"]
    token_types = ["prompt_tokens", "completion_tokens", "total_tokens"]
    labels = ["Prompt\nTokens", "Completion\nTokens", "Total\nTokens"]

    x = np.arange(len(token_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, agent in enumerate(AGENTS):
        short = AGENT_SHORT[agent]
        means = [cl[t][short]["mean"] for t in token_types]
        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset, means, width,
            label=AGENT_LABELS[agent],
            color=COLORS[agent],
            edgecolor="white", linewidth=0.5,
        )
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9,
            )

    # Ratio annotations
    for j, t in enumerate(token_types):
        ratio = cl[t]["ratio"]
        max_val = max(cl[t]["simple"]["mean"], cl[t]["agentic"]["mean"])
        ax.text(
            j, max_val * 1.12,
            f"{ratio:.1f}x", ha="center", fontsize=10, fontweight="bold", color="gray",
        )

    ax.set_ylabel("Mean Tokens per Query")
    ax.set_title("Token Usage Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper left")

    fig.tight_layout()
    return _save(fig, out_dir, "07_token_usage.png")


# ── Chart 8: Confusion Matrix for Unanswerable Detection ────────────────────

def plot_confusion_matrix(results: Dict[str, dict], out_dir: Path) -> Path:
    """Side-by-side 2x2 confusion matrices for answerability."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels_axis = ["Answerable", "Unanswerable"]

    for ax, agent in zip(axes, AGENTS):
        pq = results[agent]["per_question"]
        y_true = [int(e["is_unanswerable"]) for e in pq]
        y_pred = [int(e["is_unanswerable_pred"]) for e in pq]

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        # Annotate with both count and percentage
        total = cm.sum()
        annot = []
        for row in cm:
            row_annot = []
            for val in row:
                row_annot.append(f"{val}\n({val/total*100:.1f}%)")
            annot.append(row_annot)

        sns.heatmap(
            cm, annot=annot, fmt="", cmap="Blues", ax=ax,
            xticklabels=labels_axis, yticklabels=labels_axis,
            cbar=False, linewidths=1, linecolor="white",
            annot_kws={"fontsize": 12},
        )
        ax.set_ylabel("Ground Truth")
        ax.set_xlabel("Predicted")
        ax.set_title(AGENT_LABELS[agent])

    fig.suptitle("Unanswerable Detection — Confusion Matrix", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "08_confusion_matrix.png")


# ── Chart 9: Faithfulness Distribution ───────────────────────────────────────

def plot_faithfulness(comparison: dict, out_dir: Path) -> Path:
    """Stacked horizontal bars: faithful/partial/unfaithful."""
    agg = comparison["aggregate_summary"]
    categories = ["faithful", "partial", "unfaithful"]
    cat_colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 4))
    y_pos = np.arange(len(AGENTS))

    for agent_idx, agent in enumerate(AGENTS):
        fd = agg[agent]["j_faithfulness_dist"]
        left = 0
        total = sum(fd.get(c, 0) for c in categories)
        for cat, color in zip(categories, cat_colors):
            count = fd.get(cat, 0)
            pct = count / total * 100 if total else 0
            bar = ax.barh(agent_idx, count, left=left, color=color, edgecolor="white", height=0.5)
            if count > 10:  # Only label if bar is wide enough
                ax.text(
                    left + count / 2, agent_idx, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=9, fontweight="bold",
                )
            left += count

    ax.set_yticks(y_pos)
    ax.set_yticklabels([AGENT_LABELS[a] for a in AGENTS])
    ax.set_xlabel("Number of Judged Predictions")
    ax.set_title("LLM Judge — Faithfulness Distribution")

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, label=cat.capitalize()) for cat, c in zip(categories, cat_colors)]
    ax.legend(handles=legend_patches, loc="lower right")

    fig.tight_layout()
    return _save(fig, out_dir, "09_faithfulness.png")


# ── Chart 10: Completeness Score Distribution ────────────────────────────────

def plot_completeness_distribution(results: Dict[str, dict], out_dir: Path) -> Path:
    """Grouped bars for judge completeness scores (1-5)."""
    scores_range = [1, 2, 3, 4, 5]
    x = np.arange(len(scores_range))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))

    for i, agent in enumerate(AGENTS):
        pq = results[agent]["per_question"]
        # Only questions where judge ran
        comp_scores = [
            e["j_correctness"]["completeness"]
            for e in pq
            if e["j_correctness"] is not None
        ]
        counts = [comp_scores.count(s) for s in scores_range]
        offset = -width / 2 + i * width
        bars = ax.bar(
            x + offset, counts, width,
            label=f"{AGENT_LABELS[agent]} (n={len(comp_scores)})",
            color=COLORS[agent],
            edgecolor="white", linewidth=0.5,
        )
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    str(count), ha="center", va="bottom", fontsize=9,
                )

    ax.set_ylabel("Count")
    ax.set_xlabel("Completeness Score")
    ax.set_title("LLM Judge — Completeness Score Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(scores_range)
    ax.legend()

    fig.tight_layout()
    return _save(fig, out_dir, "10_completeness_distribution.png")


# ── Public API ────────────────────────────────────────────────────────────────

def generate_all_charts(
    comparison: dict,
    results: Dict[str, dict],
    preds: Dict[str, list],
    out_dir: Path,
) -> List[Path]:
    """Generate all 10 visualization charts. Returns list of saved paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    print("Generating visualization charts...")
    paths.append(plot_metric_comparison(comparison, out_dir))
    paths.append(plot_category_breakdown(comparison, out_dir))
    paths.append(plot_f1_distribution(results, out_dir))
    paths.append(plot_tier_distribution(comparison, out_dir))
    paths.append(plot_cost_accuracy_tradeoff(comparison, out_dir))
    paths.append(plot_latency_distribution(preds, out_dir))
    paths.append(plot_token_usage(comparison, out_dir))
    paths.append(plot_confusion_matrix(results, out_dir))
    paths.append(plot_faithfulness(comparison, out_dir))
    paths.append(plot_completeness_distribution(results, out_dir))

    print(f"\nGenerated {len(paths)} charts → {out_dir}/")
    return paths
