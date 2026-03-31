# Agent Benchmarking — Session-Based Implementation Plan

Each phase is a self-contained session. Complete one, take a break, start a new session for the next.
At the start of each new session, tell Claude: "Continue from Phase X"

---

## Evaluation Consistency Contract

Both Simple RAG and Agentic RAG MUST be evaluated under identical conditions:

1. **Same question set**: Both agents answer the exact same questions in the same order (seeded).
2. **Same retriever config**: Both use the same retrieval backbone (BM25/BGE), same k, same index.
3. **Same output schema**: Both produce `PredictionResult` with: id, question, predicted_answer, is_unanswerable_pred, contexts_used, reasoning_trace (empty for Simple RAG), latency_ms.
4. **Same metric functions**: One evaluation codepath for both — never separate eval logic per agent.
5. **Blind LLM judge**: Judge prompt receives (question, context, answer) with NO agent identifier. Predictions shuffled before judging to prevent ordering bias.
6. **Same judge params**: Single judge model, same temperature (0), same prompt template, same rubric.
7. **Same thresholds**: Shared config for semantic similarity cutoff, unanswerable detection keywords, F1 partial match threshold.
8. **Deterministic seeding**: Global random seed (42) for dataset sampling, any stochastic steps.
9. **Same evaluation order**: Metrics computed in same order; 3-tier cascade (EM/F1 → Semantic → Judge) applied identically.

Any violation of these invariants is a bug.

---

## Phase 1: Project Scaffold & Environment ⬜
**Session goal**: Repo structure, conda env, all dependencies installed and verified.

- [ ] Create directory structure:
  - src/data/, src/retrieval/, src/agents/, src/evaluation/, src/pipeline/, src/analysis/
  - configs/, data/, reports/figures/, evaluation/
- [ ] Add `__init__.py` files to all src packages
- [ ] Create conda env at `./env` with Python 3.11
- [ ] Install dependencies (vllm, datasets, sentence-transformers, rank-bm25, openai, numpy, pandas, matplotlib, seaborn, pyyaml, tqdm)
- [ ] Create `configs/base.yaml` — shared config schema including:
  - `seed: 42`
  - `retriever.type`, `retriever.top_k`
  - `eval.semantic_threshold`, `eval.unanswerable_keywords`
  - `judge.temperature: 0`, `judge.model`
  - `dataset.num_questions`, `dataset.answerable_ratio`
- [ ] Update .gitignore (env/, data/, *.pyc, __pycache__, model caches, etc.)
- [ ] **Verify**: `python -c "import vllm, datasets, sentence_transformers, rank_bm25"` succeeds

**Done when**: All imports work, directory structure exists, base config has consistency params.

---

## Phase 2: Dataset Handling ⬜
**Session goal**: SQuAD 2.0 loaded, preprocessed, exported to JSONL.

- [ ] Implement `src/data/schema.py` — unified QAExample dataclass:
  - Fields: id, question, context, answers (list), is_unanswerable, source_doc_id (from SQuAD title)
- [ ] Implement `src/data/schema.py` — PredictionResult dataclass (shared output format for ALL agents):
  - Fields: id, question, predicted_answer, is_unanswerable_pred, contexts_used, reasoning_trace, latency_ms, agent_type
- [ ] Implement `src/data/squad_loader.py` — download SQuAD 2.0 via HuggingFace datasets, preprocess to schema
- [ ] Add sampling utility (N examples, configurable answerable/unanswerable ratio, seed from config)
- [ ] Export dev set to `data/squad_v2_dev.jsonl`
- [ ] **Verify**: Load 10 examples, print them, confirm schema + unanswerable flags + source_doc_id

**Done when**: `data/squad_v2_dev.jsonl` exists with correct schema. PredictionResult defined for both agents.

---

## Phase 3: Retrieval Backbone ⬜
**Session goal**: BM25 and embedding retrievers working, retrieval metrics computed.

- [ ] Implement `src/retrieval/base.py` — retriever interface (returns list of (context, score) tuples)
- [ ] Implement `src/retrieval/bm25_retriever.py` — BM25 over all contexts
- [ ] Implement `src/retrieval/embedding_retriever.py` — BGE-large dense retrieval with embedding cache
- [ ] Implement `src/retrieval/metrics.py` — retrieval quality metrics:
  - Hit-rate@k: does gold context appear in top-k?
  - Mean Reciprocal Rank (MRR): 1/rank of gold context
- [ ] **Verify**: Both retrievers return gold context in top-5 for sample queries. Report hit-rate@5 and MRR.

**Consistency note**: Both agents will use the SAME retriever instance (configured in base.yaml). The retriever is built once, shared by both agents.

**Done when**: Both retrievers work, hit-rate/MRR reported, single retriever config shared.

---

## Phase 4: Simple RAG Agent ⬜
**Session goal**: Single-pass RAG agent answering questions via local vLLM.

**Pre-req**: vLLM server running with Llama 3.3 70B.

- [ ] Implement `src/agents/llm_client.py` — vLLM OpenAI-compat client wrapper
- [ ] Implement `src/agents/base.py` — agent interface requiring:
  - Input: QAExample → Output: PredictionResult (the SHARED schema)
  - Must populate all PredictionResult fields (reasoning_trace="" for simple RAG)
- [ ] Implement `src/agents/simple_rag.py` — single-pass RAG (retrieve → prompt → answer)
- [ ] Create `configs/simple_rag.yaml` (inherits from base.yaml, no agent-specific eval params)
- [ ] **Verify**: Run on 5 questions (mix answerable + unanswerable), produces valid PredictionResult

**Consistency note**: Uses the SAME retriever, SAME LLM client, SAME output schema as Agentic RAG. Only the agent logic differs.

**Done when**: Simple RAG produces PredictionResult objects matching the shared schema.

---

## Phase 5: Agentic RAG Agent ⬜
**Session goal**: Multi-step agent with tools (retrieve, refine, verify) working.

**Pre-req**: vLLM server running.

- [ ] Implement `src/agents/tools.py` — retrieve, refine_question, verify_answer tools
- [ ] Implement `src/agents/agentic_rag.py` — multi-step agent loop (plan → act → reflect, max 3 iterations)
  - Must output PredictionResult (the SAME shared schema as Simple RAG)
  - reasoning_trace populated with tool calls and intermediate steps
- [ ] Create `configs/agentic_rag.yaml` (inherits from base.yaml, no agent-specific eval params)
- [ ] **Verify**: Run on 5 questions, confirm multi-step traces with tool calls, valid PredictionResult

**Consistency note**: Uses the SAME retriever instance, SAME LLM client, SAME output schema. Only the orchestration logic differs.

**Done when**: Agentic RAG produces PredictionResult objects with reasoning traces.

---

## Phase 6: Evaluation Engine ⬜
**Session goal**: Full evaluation suite — one codepath for both agents.

- [ ] Implement `src/evaluation/metrics.py` — EM, token F1, unanswerable accuracy
  - Unanswerable detection uses shared keyword list from config (not agent-specific)
- [ ] Implement `src/evaluation/semantic_similarity.py` — BGE embedding cosine similarity
  - Threshold from shared config (e.g., 0.85)
- [ ] Implement `src/evaluation/llm_judge.py`:
  - **J_Answerability(question, context) → {0, 1}**: is this answerable from context alone?
  - **J_Correctness(question, context, predicted_answer, gold_answer) → score**: correctness + faithfulness
  - Judge prompt takes (question, context, answer) — NO agent name/type anywhere in prompt
  - Temperature fixed at 0 from shared config
  - Rubric: correctness (binary), completeness (1-5), reasoning quality (1-5), faithfulness (faithful/partial/unfaithful)
- [ ] Implement `src/evaluation/evaluator.py` — single evaluation entry point:
  - Takes List[PredictionResult] + List[QAExample] (agent-agnostic)
  - Runs 3-tier cascade: EM/F1 → Semantic Similarity → LLM Judge
  - Each tier only adds credit (invariant: EM_acc ≤ semantic_acc ≤ combined_acc)
  - Returns EvaluationResult with all metrics
- [ ] **Verify**: 10 crafted test cases, confirm:
  - Tier invariant holds
  - Same fake predictions produce identical scores regardless of agent_type label
  - Judge prompt contains no agent identifier

**Consistency enforcement**:
- ONE `evaluator.py` function — both agents go through the exact same code
- Judge receives blinded inputs (no agent name)
- All thresholds read from shared config, never hardcoded per agent

**Done when**: All metrics work, tier invariant holds, blind judging verified.

---

## Phase 7: Pipeline Runner ⬜
**Session goal**: End-to-end pipeline with config-driven experiments.

**Pre-req**: vLLM server running.

- [ ] Implement `src/pipeline/runner.py` — run agent on dataset, save JSONL
  - Checkpointing/resume support
  - Logs: question_id, agent_type, predicted_answer, latency_ms
- [ ] Implement `src/pipeline/evaluate.py` — compute all metrics from predictions JSONL
  - Calls the SAME evaluator.py for both agents
- [ ] Implement `src/pipeline/compare.py` — compare two agents side by side
  - Statistical comparison: mean, std, 95% confidence intervals (t-distribution)
  - Per-category breakdown: answerable vs unanswerable
  - Retrieval quality: hit-rate@k, MRR (shared retriever, so should be identical — assert this)
- [ ] Create `configs/experiment.yaml` — experiment config schema:
  - `task`: rag_eval (extensible to retrieval_eval, k_ablation later)
  - `agents`: [simple_rag, agentic_rag]
  - `retriever`: bm25 | bge (shared)
  - `metrics`: [em, f1, semantic, j_answerability, j_correctness]
  - `dataset.num_questions`, `dataset.seed`
- [ ] Create `run_benchmark.py` — CLI entry point that reads experiment config and dispatches
- [ ] **Verify**: End-to-end on 50 questions for both agents
  - Assert both agents answered the SAME 50 questions
  - Assert evaluation used identical metric functions

**Done when**: Both agents run on 50 questions, metrics JSON produced, comparison shows identical eval conditions.

---

## Phase 8: Full Benchmark Run ⬜
**Session goal**: Run both agents on 1000+ questions, full evaluation.

**Pre-req**: vLLM server running (will take a while).

- [ ] Run Simple RAG on full dataset (1000+ questions)
- [ ] Run Agentic RAG on full dataset (SAME 1000+ questions, same order)
- [ ] Run 3-tier evaluation on both sets of predictions (same evaluator, same config)
- [ ] Run LLM-as-a-judge on both sets (blind, shuffled)
- [ ] Generate comparison results with confidence intervals
- [ ] **Sanity check**: Verify question sets are identical, eval config is identical

**Done when**: All evaluation JSONs exist. Comparison report shows both agents evaluated under identical conditions.

---

## Phase 9: Analysis & Visualization ⬜
**Session goal**: Charts, failure analysis, all figures generated.

- [ ] Implement `src/analysis/visualize.py`:
  - Bar charts comparing both agents on all metrics
  - Histograms of score distributions (side by side)
  - Confusion matrices for unanswerable detection (both agents)
  - Per-category breakdown (answerable vs unanswerable performance)
- [ ] Implement `src/analysis/failure_analysis.py`:
  - LLM-based failure categorization (for both agents):
    - RAG failures: missed context, not extracted, wrong format
    - LLM failures: hallucination, fabrication, logical inconsistency
  - Compare failure distributions between agents
  - Plot failure type counts (side by side)
- [ ] Generate all figures to `reports/figures/`
- [ ] **Verify**: Charts render correctly, both agents shown in every comparison

**Done when**: All charts saved, failure analysis complete for both agents.

---

## Phase 10: Report & Documentation ⬜
**Session goal**: README, architecture doc, benchmark report, demo script.

- [ ] Write `README.md` — setup + run instructions
- [ ] Write `ARCHITECTURE.md` — agent architecture descriptions
- [ ] Write `reports/BENCHMARK_REPORT.md` — full benchmark report with:
  - Evaluation methodology (emphasize identical conditions for both agents)
  - Metrics tables, charts, insights
  - Failure analysis comparison
  - Statistical significance of differences
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
