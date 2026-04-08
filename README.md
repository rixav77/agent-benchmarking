# Agent Benchmarking: Simple RAG vs Agentic RAG

Benchmark comparing two RAG (Retrieval-Augmented Generation) agent architectures on **SQuAD v2.0** (1000 questions) with a rigorous 3-tier evaluation cascade.

## Key Results

| Metric | Simple RAG | Agentic RAG | Delta |
|--------|-----------|-------------|-------|
| Combined Accuracy | 69.1% | **72.0%** | +2.9pp |
| Exact Match | 51.4% | 51.9% | +0.5pp |
| F1 Score | 60.7% | 62.6% | +1.9pp |
| Unanswerable Detection | 56.2% | **61.2%** | +5.0pp* |
| Mean Latency | **638ms** | 3,483ms | 5.46x |
| Mean Tokens/Query | **1,060** | 3,725 | 3.51x |

*Statistically significant (95% CIs non-overlapping)

## Architecture

### Simple RAG
Single-pass pipeline: Retrieve top-k passages → Prompt LLM → Extract answer. One LLM call per question.

### Agentic RAG
Multi-step pipeline with 5 tools:
1. **Retrieve** — BM25 passage retrieval
2. **Assess Relevance** — LLM scores retrieval quality
3. **Rewrite Query** — Reformulates query if relevance is low (up to 2 retries)
4. **Check Answerability** — LLM decides if question is answerable from passages
5. **Generate Answer** — Final answer with self-verification

Both agents share the same retriever, LLM, output schema, and evaluation pipeline.

### 3-Tier Evaluation Cascade
- **Tier 1**: Exact Match / Token F1 (string matching)
- **Tier 2**: Semantic Similarity (BGE-large embeddings, threshold 0.85)
- **Tier 3**: LLM-as-Judge (correctness, completeness, faithfulness)

Each tier only adds credit. Invariant: `EM_accuracy ≤ semantic_accuracy ≤ combined_accuracy`

## Project Structure

```
agent-benchmarking/
├── src/
│   ├── data/               # SQuAD v2 loader, QAExample/PredictionResult schema
│   ├── retrieval/          # BM25 + BGE embedding retrievers, retrieval metrics
│   ├── agents/             # Simple RAG, Agentic RAG, LLM client, tools
│   ├── evaluation/         # EM/F1 metrics, semantic similarity, LLM judge, evaluator
│   ├── pipeline/           # Runner, evaluator, comparison with 95% CIs
│   └── analysis/           # Visualization (11 charts), failure analysis
├── configs/
│   ├── base.yaml           # Shared config (retriever, LLM, eval thresholds)
│   ├── experiment.yaml     # Experiment overrides (deep-merged onto base)
│   ├── simple_rag.yaml     # Simple RAG agent config
│   └── agentic_rag.yaml    # Agentic RAG agent config
├── evaluation/
│   ├── predictions/        # Agent prediction JSONL files (gitignored)
│   └── results/            # Evaluation results, summaries, comparison JSON
├── reports/
│   ├── figures/            # 11 visualization charts (PNG)
│   ├── analysis_summary.md # Strengths, weaknesses, trade-offs
│   └── failure_analysis.md # Failure mode breakdown
├── run_benchmark.py        # CLI entry point for full pipeline
├── ARCHITECTURE.md         # Detailed architecture documentation
└── README.md
```

## Setup

### Prerequisites
- Python 3.11
- NVIDIA GPU (for vLLM serving Qwen3-14B)
- disk storage for model weights

### 1. Create environment

```bash
conda create --prefix ./env python=3.11 -y
conda activate ./env
```

### 2. Install dependencies

```bash
pip install vllm datasets sentence-transformers rank-bm25 openai \
    numpy pandas matplotlib seaborn pyyaml tqdm scipy scikit-learn
```

### 3. Start vLLM server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 8192
```

### 4. Verify setup

```bash
curl http://localhost:8000/v1/models
```

## Usage

### Full pipeline (run + evaluate + compare)

```bash
python run_benchmark.py --mode all
```

### Step by step

```bash
# Run both agents on dataset
python run_benchmark.py --mode run

# Run with specific agent only
python run_benchmark.py --mode run --agents simple_rag

# Evaluate predictions (with LLM judge)
python run_benchmark.py --mode evaluate

# Evaluate without judge (faster, Tier 1+2 only)
python run_benchmark.py --mode evaluate --no-judge

# Compare results
python run_benchmark.py --mode compare
```

### Generate visualizations and analysis

```bash
python -m src.analysis.run_analysis
```

This generates 11 charts in `reports/figures/` and text reports in `reports/`.

## Configuration

All settings are in `configs/base.yaml`:

| Setting | Default | Description |
|---------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `dataset.num_questions` | 1000 | Number of questions to sample |
| `dataset.answerable_ratio` | 0.5 | 50% answerable, 50% unanswerable |
| `retriever.type` | bm25 | Retriever type (bm25 or embedding) |
| `retriever.top_k` | 5 | Number of passages to retrieve |
| `llm.model` | Qwen/Qwen3-14B | LLM model for agents and judge |
| `eval.semantic_threshold` | 0.85 | Tier 2 semantic similarity cutoff |
| `eval.f1_partial_threshold` | 0.5 | Tier 1 F1 threshold for partial credit |

Experiment-specific overrides go in `configs/experiment.yaml` and are deep-merged onto the base config.

## Evaluation Consistency

Both agents are evaluated under identical conditions:
- Same 1000 questions in the same order (seed=42)
- Same BM25 retriever (indexed once, shared)
- Same LLM client (Qwen3-14B via vLLM)
- Same evaluator codepath — agent-agnostic, no branching by agent type
- Same LLM judge — blind evaluation (no agent identifier in prompts)
- Same thresholds from shared config

## Dataset

- **Source**: SQuAD v2.0 (validation split) via HuggingFace Datasets
- **Size**: 1000 questions (500 answerable, 500 unanswerable)
- **Preprocessing**: Sampled with seed=42 and 50/50 answerable ratio
- **Challenge**: Unanswerable questions have adversarial premises with misleading context

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Qwen3-14B via vLLM |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Retrieval | BM25 (rank-bm25) |
| Evaluation | 3-tier cascade (EM/F1 → Semantic → LLM Judge) |
| Statistics | scipy (t-distribution 95% CIs) |
| Visualization | matplotlib + seaborn |

## License

See [LICENSE](LICENSE) file.
