# Agent Benchmarking — Session-Based Implementation Plan

Each phase is a self-contained session. Complete one, take a break, start a new session for the next.
At the start of each new session, tell Claude: "Continue from Phase X"

---

## Phase 1: Project Scaffold & Environment ⬜
**Session goal**: Repo structure, conda env, all dependencies installed and verified.

- [ ] Create directory structure (src/data/, src/retrieval/, src/agents/, src/evaluation/, src/pipeline/, src/analysis/, configs/, data/, reports/figures/, evaluation/)
- [ ] Add `__init__.py` files to all src packages
- [ ] Create conda env at `./env` with Python 3.11
- [ ] Install dependencies (vllm, datasets, sentence-transformers, rank-bm25, openai, numpy, pandas, matplotlib, seaborn, pyyaml, tqdm)
- [ ] Create base config YAML schema (configs/base.yaml)
- [ ] Update .gitignore (env/, data/, *.pyc, __pycache__, model caches, etc.)
- [ ] **Verify**: `python -c "import vllm, datasets, sentence_transformers, rank_bm25"` succeeds

**Done when**: All imports work, directory structure exists.

---

## Phase 2: Dataset Handling ⬜
**Session goal**: SQuAD 2.0 loaded, preprocessed, exported to JSONL.

- [ ] Implement `src/data/schema.py` — unified QAExample dataclass
- [ ] Implement `src/data/squad_loader.py` — download SQuAD 2.0 via HuggingFace datasets, preprocess to schema
- [ ] Add sampling utility (N examples, configurable answerable/unanswerable ratio)
- [ ] Export dev set to `data/squad_v2_dev.jsonl`
- [ ] **Verify**: Load 10 examples, print them, confirm schema + unanswerable flags

**Done when**: `data/squad_v2_dev.jsonl` exists with correct schema.

---

## Phase 3: Retrieval Backbone ⬜
**Session goal**: BM25 and embedding retrievers working, gold context found in top-5.

- [ ] Implement `src/retrieval/base.py` — retriever interface
- [ ] Implement `src/retrieval/bm25_retriever.py` — BM25 over all contexts
- [ ] Implement `src/retrieval/embedding_retriever.py` — BGE-large dense retrieval
- [ ] **Verify**: Gold context in top-5 for sample queries with both retrievers

**Done when**: Both retrievers return gold context in top-5 for test queries.

---

## Phase 4: Simple RAG Agent ⬜
**Session goal**: Single-pass RAG agent answering questions via local vLLM.

**Pre-req**: vLLM server running with Llama 3.3 70B (start before this session).

- [ ] Implement `src/agents/llm_client.py` — vLLM OpenAI-compat client wrapper
- [ ] Implement `src/agents/base.py` — agent interface
- [ ] Implement `src/agents/simple_rag.py` — single-pass RAG (retrieve → prompt → answer)
- [ ] Create `configs/simple_rag.yaml`
- [ ] **Verify**: Run on 5 questions (mix answerable + unanswerable), sensible answers

**Done when**: Simple RAG produces correct-looking answers on test questions.

---

## Phase 5: Agentic RAG Agent ⬜
**Session goal**: Multi-step agent with tools (retrieve, refine, verify) working.

**Pre-req**: vLLM server running.

- [ ] Implement `src/agents/tools.py` — retrieve, refine_question, verify_answer tools
- [ ] Implement `src/agents/agentic_rag.py` — multi-step agent loop (plan → act → reflect, max 3 iterations)
- [ ] Create `configs/agentic_rag.yaml`
- [ ] **Verify**: Run on 5 questions, confirm multi-step traces with tool calls

**Done when**: Agentic RAG produces reasoning traces with tool calls.

---

## Phase 6: Evaluation Engine ⬜
**Session goal**: 3-tier evaluation (EM/F1 → semantic similarity → LLM judge) implemented.

- [ ] Implement `src/evaluation/metrics.py` — EM, token F1, unanswerable accuracy
- [ ] Implement `src/evaluation/semantic_similarity.py` — BGE embedding cosine similarity
- [ ] Implement `src/evaluation/llm_judge.py` — correctness, completeness, reasoning, faithfulness
- [ ] Implement 3-tier combined accuracy (EM → semantic → judge, each tier only adds credit)
- [ ] **Verify**: 10 crafted test cases, confirm tier invariant (EM_acc ≤ semantic_acc ≤ combined_acc)

**Done when**: All 3 tiers work, invariant holds on test cases.

---

## Phase 7: Pipeline Runner ⬜
**Session goal**: End-to-end pipeline: run agents → save predictions → compute metrics → compare.

**Pre-req**: vLLM server running.

- [ ] Implement `src/pipeline/runner.py` — run agent on dataset, save JSONL (with checkpointing/resume)
- [ ] Implement `src/pipeline/evaluate.py` — compute all metrics from predictions JSONL
- [ ] Implement `src/pipeline/compare.py` — compare two architectures side by side
- [ ] Create `run_benchmark.py` CLI entry point
- [ ] **Verify**: End-to-end on 50 questions for both agents

**Done when**: Both agents run on 50 questions, metrics JSON produced and compared.

---

## Phase 8: Full Benchmark Run ⬜
**Session goal**: Run both agents on 1000+ questions, full evaluation.

**Pre-req**: vLLM server running (will take a while).

- [ ] Run Simple RAG on full dataset (1000+ questions)
- [ ] Run Agentic RAG on full dataset (1000+ questions)
- [ ] Run 3-tier evaluation on both sets of predictions
- [ ] Run LLM-as-a-judge on both sets
- [ ] Generate comparison results

**Done when**: All evaluation JSONs exist with full results.

---

## Phase 9: Analysis & Visualization ⬜
**Session goal**: Charts, failure analysis, all figures generated.

- [ ] Implement `src/analysis/visualize.py` — bar charts, histograms, confusion matrices
- [ ] Implement `src/analysis/failure_analysis.py` — categorize failure modes
- [ ] Generate all figures to `reports/figures/`
- [ ] **Verify**: Charts render correctly

**Done when**: All charts saved to reports/figures/.

---

## Phase 10: Report & Documentation ⬜
**Session goal**: README, architecture doc, benchmark report, demo script.

- [ ] Write `README.md` — setup + run instructions
- [ ] Write `ARCHITECTURE.md` — agent architecture descriptions
- [ ] Write `reports/BENCHMARK_REPORT.md` — full benchmark report with metrics, charts, insights
- [ ] Create demo script for side-by-side comparison
- [ ] Record demo video

**Done when**: All docs written, demo ready.

---

## Progress Tracker
| Phase | Status | Session Date |
|-------|--------|-------------|
| 1. Scaffold & Env | ⬜ Not started | |
| 2. Dataset | ⬜ Not started | |
| 3. Retrieval | ⬜ Not started | |
| 4. Simple RAG | ⬜ Not started | |
| 5. Agentic RAG | ⬜ Not started | |
| 6. Evaluation | ⬜ Not started | |
| 7. Pipeline | ⬜ Not started | |
| 8. Full Run | ⬜ Not started | |
| 9. Visualization | ⬜ Not started | |
| 10. Report & Docs | ⬜ Not started | |
