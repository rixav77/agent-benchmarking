# Agent Benchmarking: Simple RAG vs Agentic RAG

Benchmark comparing two RAG architectures on **SQuAD v2.0** (1,000 questions) with a 3-tier evaluation cascade.

## Key Results

| Metric | Simple RAG | Agentic RAG | Delta |
|--------|-----------|-------------|-------|
| Combined Accuracy | 69.1% | **72.0%** | +2.9pp |
| Unanswerable Detection | 56.2% | **61.2%** | +5.0pp* |
| Mean Latency | **638ms** | 3,483ms | 5.46x |
| Mean Tokens/Query | **1,060** | 3,725 | 3.51x |

*Statistically significant (95% CIs non-overlapping)

## Architecture

- **Simple RAG**: Retrieve top-k passages → single LLM call → extract answer
- **Agentic RAG**: Multi-step pipeline with 5 tools — retrieve, assess relevance, rewrite query, check answerability, generate answer (up to 2 retries)
- **Evaluation**: 3-tier cascade — Exact Match/F1 → Semantic Similarity → LLM-as-Judge

Both agents share the same retriever, LLM, and evaluation pipeline. See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Project Structure

```
src/
  data/          # SQuAD v2 loader, schema
  retrieval/     # BM25 + embedding retrievers
  agents/        # Simple RAG, Agentic RAG, LLM client, tools
  evaluation/    # Metrics, semantic similarity, LLM judge
  pipeline/      # Runner, evaluator, comparison (95% CIs)
  analysis/      # Visualization (11 charts), failure analysis
configs/         # YAML configs (base, experiment, per-agent)
evaluation/      # Results, summaries, comparison JSON
reports/         # Charts, analysis reports,benchmark report
demo/            # Web demo app for side-by-side comparison
```

## Setup

**Prerequisites**: Python 3.11, NVIDIA GPU (30+ GB VRAM)

```bash
# 1. Create environment
conda create --prefix ./env python=3.11 -y && conda activate ./env

# 2. Install dependencies
pip install vllm datasets sentence-transformers rank-bm25 openai \
    numpy pandas matplotlib seaborn pyyaml tqdm scipy scikit-learn

# 3. Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B --port 8000 \
    --gpu-memory-utilization 0.9 --max-model-len 8192
```

## Usage

```bash
# Full pipeline: run both agents → evaluate → compare
python run_benchmark.py --mode all

# Individual steps
python run_benchmark.py --mode run --agents simple_rag
python run_benchmark.py --mode evaluate
python run_benchmark.py --mode compare

# Generate visualizations and failure analysis
python -m src.analysis.run_analysis

# Launch demo web app (no vLLM needed)
python demo/app.py
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen3-14B via vLLM |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Retrieval | BM25 (rank-bm25) |
| Evaluation | 3-tier cascade (EM/F1 → Semantic → LLM Judge) |
| Dataset | SQuAD v2.0 — 1,000 questions (500 answerable, 500 unanswerable) |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — Agent architectures, evaluation design, pipeline flow
- [reports/BENCHMARK_REPORT.md](reports/BENCHMARK_REPORT.md) — Full benchmark report with results, insights, trade-offs
- [reports/analysis_summary.md](reports/analysis_summary.md) — Strengths, weaknesses, failure modes

## License

See [LICENSE](LICENSE) file.
