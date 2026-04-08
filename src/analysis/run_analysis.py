"""CLI runner for Phase 9 — Analysis & Visualization.

Usage:
    python -m src.analysis.run_analysis              # Everything
    python -m src.analysis.run_analysis --charts-only # Charts only
    python -m src.analysis.run_analysis --analysis-only # Failure analysis only
"""

import argparse
from pathlib import Path

from src.analysis.visualize import (
    AGENTS, load_results, load_predictions, load_comparison,
    generate_all_charts,
)
from src.analysis.failure_analysis import run_failure_analysis


FIGURES_DIR = Path("reports/figures")
REPORTS_DIR = Path("reports")


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark Analysis & Visualization")
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR,
                        help="Output directory for charts")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR,
                        help="Output directory for text reports")
    parser.add_argument("--charts-only", action="store_true",
                        help="Only generate charts, skip failure analysis text")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Only run failure analysis, skip visualization charts")
    args = parser.parse_args()

    # Load all data once
    print("Loading data...")
    comparison = load_comparison()
    results = {a: load_results(a) for a in AGENTS}
    preds = {a: load_predictions(a) for a in AGENTS}
    print(f"  Loaded comparison + {len(AGENTS)} agents x (results + predictions)")
    print()

    chart_paths = []

    # Visualization charts (1-10)
    if not args.analysis_only:
        chart_paths = generate_all_charts(comparison, results, preds, args.figures_dir)
        print()

    # Failure analysis (chart 11 + text reports)
    if not args.charts_only:
        chart_path, failure_path, summary_path = run_failure_analysis(
            results, comparison, args.figures_dir, args.reports_dir,
        )
        chart_paths.append(chart_path)
        print()

    # Final summary
    print(f"Done. Generated {len(chart_paths)} charts → {args.figures_dir}/")
    if not args.charts_only:
        print(f"Text reports → {args.reports_dir}/")


if __name__ == "__main__":
    main()
