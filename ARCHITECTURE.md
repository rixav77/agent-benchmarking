# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_benchmark.py                         │
│                     (CLI Entry Point)                           │
├─────────────┬──────────────┬──────────────┬────────────────────┤
│  --mode run │ --mode eval  │ --mode compare│ --mode all        │
└──────┬──────┴──────┬───────┴──────┬───────┴────────────────────┘
       │             │              │
       ▼             ▼              ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐
│   Runner    │ │ Evaluate │ │   Compare    │
│ (pipeline/) │ │(pipeline/)│ │ (pipeline/)  │
└──────┬──────┘ └────┬─────┘ └──────┬───────┘
       │             │              │
       ▼             ▼              ▼
┌─────────────┐ ┌──────────┐ ┌──────────────┐
│   Agents    │ │Evaluator │ │  Statistics  │
│ Simple/Agnt │ │ 3-Tier   │ │ 95% CIs     │
└──────┬──────┘ └────┬─────┘ └──────────────┘
       │             │
       ▼             ▼
┌─────────────────────────────┐
│      Shared Resources       │
│  Retriever │ LLM │ Embeddings│
└─────────────────────────────┘
```

## Agent Architectures

### Simple RAG Agent

Single-pass retrieve-and-generate. One LLM call per question.

```
Question
   │
   ▼
┌──────────────────┐
│  BM25 Retrieve   │ → top-5 passages
│  (shared)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Build Prompt    │  System: "Answer from context or say unanswerable"
│  context + query │  User: [passages] + [question]
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Generate    │ → single call
│  (Qwen3-14B)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Parse Response  │ → check for unanswerable keywords
└────────┬─────────┘
         │
         ▼
   PredictionResult
```

**Characteristics:**
- 1 LLM call per question
- ~638ms mean latency, ~1,060 tokens/query
- No self-correction or retry logic
- Better at direct extractive answers (answerable EM: 46.6%)

**Source:** `src/agents/simple_rag.py`

---

### Agentic RAG Agent

Multi-step pipeline with 5 tools, iterative query refinement, and answerability checking. Inspired by CRAG (Corrective RAG) and Self-RAG papers.

```
Question
   │
   ▼
┌──────────────────┐
│ Tool 1: RETRIEVE │ → top-5 passages
└────────┬─────────┘
         │
         ▼
┌──────────────────┐     ┌─────────────────────┐
│ Tool 2: ASSESS   │────▶│ Relevance: HIGH?    │
│ RELEVANCE        │     │                     │
└────────┬─────────┘     └──────┬──────────────┘
         │                      │
    LOW/MEDIUM                 HIGH
         │                      │
         ▼                      │
┌──────────────────┐            │
│ Tool 3: REWRITE  │            │
│ QUERY            │            │
└────────┬─────────┘            │
         │                      │
         ▼                      │
┌──────────────────┐            │
│ Re-RETRIEVE      │            │
│ (up to 2 retries)│            │
└────────┬─────────┘            │
         │                      │
         ▼◄─────────────────────┘
┌──────────────────┐
│ Tool 4: CHECK    │     ┌─────────────────────┐
│ ANSWERABILITY    │────▶│ Answerable?         │
└────────┬─────────┘     └──────┬──────────────┘
         │                      │
     UNANSWERABLE          ANSWERABLE
         │                      │
         ▼                      ▼
   Return               ┌──────────────────┐
   "unanswerable"       │ Tool 5: GENERATE │
                         │ ANSWER           │
                         │ (with self-      │
                         │  verification)   │
                         └────────┬─────────┘
                                  │
                                  ▼
                            PredictionResult
```

**5 Agent Tools:**

| Tool | LLM Calls | Purpose |
|------|-----------|---------|
| `retrieve(query)` | 0 | BM25 retrieval (same retriever as Simple RAG) |
| `assess_relevance(passages, query)` | 1 | Score passage relevance: high/medium/low |
| `rewrite_query(query, passages)` | 1 | Reformulate query for better retrieval |
| `check_answerability(passages, query)` | 1 | Decide if question is answerable from context |
| `generate_answer(passages, query)` | 1 | Generate final answer with self-verification |

**Characteristics:**
- 3-5 LLM calls per question (depending on retry count)
- ~3,483ms mean latency, ~3,725 tokens/query
- Query rewriting recovers from poor initial retrieval
- Answerability checking reduces false positives on unanswerable questions
- Better at unanswerable detection (61.2% vs 56.2%)

**Source:** `src/agents/agentic_rag.py`, `src/agents/tools.py`

---

## Shared Components

### Retriever

Both agents use the **same BM25 retriever instance**, indexed once on all 695 unique contexts from the dataset. This ensures retrieval quality is identical — any performance difference comes purely from agent logic.

- **BM25**: `src/retrieval/bm25_retriever.py` — tokenized whitespace/lowercase, rank-bm25 library
- **Embedding**: `src/retrieval/embedding_retriever.py` — BGE-large-en-v1.5 dense retrieval (alternative)
- **Metrics**: `src/retrieval/metrics.py` — hit_rate@k and MRR

Performance on 500 answerable questions: hit_rate@5 = 0.858, MRR = 0.780

### LLM Client

Thin wrapper around vLLM's OpenAI-compatible API (`src/agents/llm_client.py`):
- Connects to `http://localhost:8000/v1`
- Returns `LLMResponse(content, prompt_tokens, completion_tokens, total_tokens)`
- Strips Qwen3's `<think>` tags when thinking mode is disabled
- Shared by both agents and the LLM judge

### Output Schema

Both agents produce identical `PredictionResult` objects (`src/data/schema.py`):

```
PredictionResult:
  id                  # SQuAD question ID
  question            # Original question text
  predicted_answer    # Agent's answer (or "unanswerable")
  is_unanswerable_pred  # Agent's answerability prediction
  contexts_used       # Retrieved passages
  reasoning_trace     # Step-by-step trace (empty for Simple RAG)
  latency_ms          # End-to-end response time
  agent_type          # "simple_rag" or "agentic_rag"
  prompt_tokens       # Total prompt tokens across all LLM calls
  completion_tokens   # Total completion tokens
  total_tokens        # Sum of prompt + completion
```

---

## Evaluation Architecture

### 3-Tier Cascade

The evaluator (`src/evaluation/evaluator.py`) is completely agent-agnostic. It receives predictions and gold examples with no awareness of which agent produced them.

```
For each question:
   │
   ▼
┌────────────────────────┐
│ TIER 1: EM / Token F1  │  EM=1 or F1≥0.5 ?
│ (string matching)      │──── YES ──→ CORRECT (tier1)
└────────┬───────────────┘
         │ NO
         ▼
┌────────────────────────┐
│ TIER 2: Semantic Sim   │  cosine(pred, gold) ≥ 0.85 ?
│ (BGE embeddings)       │──── YES ──→ CORRECT (tier2)
└────────┬───────────────┘
         │ NO
         ▼
┌────────────────────────┐
│ TIER 3: LLM Judge      │  judge says correct ?
│ (Qwen3-14B)            │──── YES ──→ CORRECT (tier3)
└────────┬───────────────┘
         │ NO
         ▼
      INCORRECT (none)
```

**Invariant:** Each tier only adds credit, never removes it.
`EM_accuracy ≤ semantic_accuracy ≤ combined_accuracy`

### LLM Judge

The judge (`src/evaluation/llm_judge.py`) evaluates two dimensions per question:

1. **Answerability**: Given context, is this question answerable? (0 or 1)
2. **Correctness**: For answered questions — correctness (0/1), completeness (1-5), faithfulness (faithful/partial/unfaithful)

The judge prompt contains **no agent identifier** — evaluation is blind.

### Statistical Comparison

`src/pipeline/compare.py` computes:
- Per-metric 95% confidence intervals via t-distribution (`scipy.stats.t`)
- Per-category breakdown (answerable vs unanswerable subsets)
- Cost/latency statistics (mean, std, p50, p95, total)
- Agentic/simple ratios for latency and tokens
- Retrieval quality (hit_rate@k, MRR) — identical for both agents by construction

---

## Configuration System

Configs use a **deep-merge** strategy:

```
base.yaml          ← shared defaults (retriever, LLM, eval thresholds)
  ↑ deep-merge
experiment.yaml    ← experiment overrides (num_questions, agents list)
  ↑ deep-merge
{agent}.yaml       ← agent-specific overrides (prompt templates, tool params)
```

Override wins at every nesting level. For example, `experiment.yaml` can set `retriever.type: bm25` without losing `retriever.top_k: 5` from base.

---

## Pipeline Flow

```
run_benchmark.py --mode all
  │
  ├── 1. Load configs (base.yaml + experiment.yaml, deep-merged)
  ├── 2. Build shared resources (examples, retriever, LLM client, semantic scorer)
  │
  ├── 3. RUN: For each agent
  │       └── runner.run_agent() → predictions JSONL (crash-safe, resumable)
  │
  ├── 4. EVALUATE: For each agent
  │       └── evaluate.run_evaluation() → results JSON (per-question + summary)
  │
  └── 5. COMPARE
          └── compare.run_comparison() → comparison JSON (with 95% CIs)
```

**Crash safety:** Predictions are written to JSONL with immediate `flush()` after each line. On resume, completed IDs are read and skipped.

---

## Analysis Pipeline

```
python -m src.analysis.run_analysis
  │
  ├── Load: comparison.json + per-question results + predictions
  │
  ├── Visualization (10 charts):
  │   ├── 01: Metric comparison with 95% CI error bars
  │   ├── 02: Per-category breakdown (answerable vs unanswerable)
  │   ├── 03: F1 score distribution (KDE histograms)
  │   ├── 04: Tier distribution
  │   ├── 05: Cost-accuracy trade-off
  │   ├── 06: Latency distribution
  │   ├── 07: Token usage comparison
  │   ├── 08: Confusion matrix (unanswerable detection)
  │   ├── 09: Faithfulness distribution
  │   └── 10: Completeness score distribution
  │
  └── Failure Analysis:
      ├── 11: Failure mode breakdown chart
      ├── failure_analysis.md (category table with examples)
      └── analysis_summary.md (strengths, weaknesses, trade-offs)
```
