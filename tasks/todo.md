# Agent Benchmarking — Session-Based Implementation Plan

Each phase is a self-contained session. Complete one, take a break, start a new session for the next.
At the start of each new session, tell Claude: "Continue from Phase X"

---

## Evaluation Consistency Contract

Both Simple RAG and Agentic RAG MUST be evaluated under identical conditions:

1. **Same question set**: Both agents answer the exact same questions in the same order (seeded).
2. **Same retriever config**: Both use the same retrieval backbone (BM25/BGE), same k, same index.
3. **Same output schema**: Both produce `PredictionResult` with: id, question, predicted_answer, is_unanswerable_pred, contexts_used, reasoning_trace (empty for Simple RAG), latency_ms, prompt_tokens, completion_tokens, total_tokens.
4. **Same metric functions**: One evaluation codepath for both — never separate eval logic per agent.
5. **Blind LLM judge**: Judge prompt receives (question, context, answer) with NO agent identifier. Predictions shuffled before judging to prevent ordering bias.
6. **Same judge params**: Single judge model, same temperature (0), same prompt template, same rubric.
7. **Same thresholds**: Shared config for semantic similarity cutoff, unanswerable detection keywords, F1 partial match threshold.
8. **Deterministic seeding**: Global random seed (42) for dataset sampling, any stochastic steps.
9. **Same evaluation order**: Metrics computed in same order; 3-tier cascade (EM/F1 → Semantic → Judge) applied identically.

Any violation of these invariants is a bug.

---

## Phase 1: Project Scaffold & Environment ✅
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

## Phase 2: Dataset Handling ✅
**Session goal**: SQuAD 2.0 loaded, preprocessed, exported to JSONL.

- [ ] Implement `src/data/schema.py` — unified QAExample dataclass:
  - Fields: id, question, context, answers (list), is_unanswerable, source_doc_id (from SQuAD title)
- [ ] Implement `src/data/schema.py` — PredictionResult dataclass (shared output format for ALL agents):
  - Fields: id, question, predicted_answer, is_unanswerable_pred, contexts_used, reasoning_trace, latency_ms, agent_type, prompt_tokens, completion_tokens, total_tokens
- [ ] Implement `src/data/squad_loader.py` — download SQuAD 2.0 via HuggingFace datasets, preprocess to schema
- [ ] Add sampling utility (N examples, configurable answerable/unanswerable ratio, seed from config)
- [ ] Export dev set to `data/squad_v2_dev.jsonl`
- [ ] **Verify**: Load 10 examples, print them, confirm schema + unanswerable flags + source_doc_id

**Done when**: `data/squad_v2_dev.jsonl` exists with correct schema. PredictionResult defined for both agents.

---

## Phase 3: Retrieval Backbone ✅
**Session goal**: BM25 and embedding retrievers working, retrieval metrics computed.

- [ ] Implement `src/retrieval/base.py` — retriever interface (returns list of (context, score) tuples)
- [ ] Implement `src/retrieval/bm25_retriever.py` — BM25 over all contexts
- [ ] Implement `src/retrieval/embedding_retriever.py` — BGE-large dense retrieval with embedding cache
- [ ] Implement `src/retrieval/metrics.py` — retrieval quality metrics:
  - Hit-rate@k: does gold context appear in top-k?
  - Mean Reciprocal Rank (MRR): 1/rank of gold context
- [ ] **Verify**: Both retrievers return gold context in top-5 for sample queries. Report hit-rate@5 and MRR.
  - BM25: hit-rate@5 = 0.920, MRR = 0.806 (50 answerable samples)
  - BGE Embedding: hit-rate@5 = 0.940, MRR = 0.848 (50 answerable samples)
  - Both index same 695 unique contexts, embedding cache on disk

**Consistency note**: Both agents will use the SAME retriever instance (configured in base.yaml). The retriever is built once, shared by both agents.

**Done when**: Both retrievers work, hit-rate/MRR reported, single retriever config shared.

---

## Phase 4: Simple RAG Agent ✅
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

## Phase 5: Agentic RAG Agent ✅
**Session goal**: Multi-step CRAG/Self-RAG inspired agent with 5 tools and iterative refinement.

**Pre-req**: vLLM server running.

**Design** (inspired by CRAG + Self-RAG papers):
- Unlike Simple RAG (single-pass, no decision-making), the Agentic RAG agent makes explicit decisions at each step: is the retrieval good enough? should I rewrite the query? is this answerable? is my answer supported?
- Agent loop: Retrieve → Assess relevance → (Rewrite & re-retrieve if poor) → Check answerability → Generate with self-verification
- Max 3 iterations for query refinement, then fallback

**5 Agent Tools** (all use the shared LLM client):
1. `retrieve(query)` — BM25 retrieval (same retriever as Simple RAG)
2. `assess_relevance(passages, query)` — LLM scores if passages are relevant (high/medium/low)
3. `rewrite_query(query, passages)` — LLM rewrites query when retrieval quality is poor
4. `check_answerability(passages, query)` — LLM decides if question is answerable from passages
5. `generate_answer(passages, query)` — Final answer generation with self-verification

- [ ] Implement `src/agents/tools.py` — 5 tool functions (retrieve, assess_relevance, rewrite_query, check_answerability, generate_answer)
- [ ] Implement `src/agents/agentic_rag.py` — agent loop:
  - Step 1: Retrieve top-k passages
  - Step 2: Assess relevance → if low, rewrite query and re-retrieve (up to 2 retries)
  - Step 3: Check answerability → if unanswerable, return "unanswerable" early
  - Step 4: Generate answer with self-verification (is answer supported by passages?)
  - Must output PredictionResult (the SAME shared schema as Simple RAG)
  - reasoning_trace populated with all tool calls, decisions, and intermediate steps
  - Token counts aggregated across all LLM calls in the loop
- [ ] Create `configs/agentic_rag.yaml` (inherits from base.yaml, no agent-specific eval params)
- [ ] **Verify**: Run on 5 questions (mix answerable + unanswerable), confirm:
  - Multi-step reasoning traces showing tool calls and decisions
  - Query rewriting triggered on at least 1 hard question
  - Unanswerable detection working
  - Valid PredictionResult with aggregated token counts

**Consistency note**: Uses the SAME retriever instance, SAME LLM client, SAME output schema. Only the orchestration logic differs.

**Done when**: Agentic RAG produces PredictionResult objects with rich reasoning traces showing multi-step decision-making.

---

## Phase 6: Evaluation Engine ✅
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

## Phase 7: Pipeline Runner ✅
**Session goal**: End-to-end pipeline with config-driven experiments.

**Pre-req**: vLLM server running.

- [x] Implement `src/pipeline/runner.py` — run agent on dataset, save JSONL
  - Checkpointing/resume support (crash-safe append+flush)
  - Logs: question_id, agent_type, predicted_answer, latency_ms
- [x] Implement `src/pipeline/evaluate.py` — compute all metrics from predictions JSONL
  - Calls the SAME evaluator.py for both agents
- [x] Implement `src/pipeline/compare.py` — compare two agents side by side
  - Statistical comparison: mean, std, 95% confidence intervals (t-distribution)
  - Per-category breakdown: answerable vs unanswerable
  - Retrieval quality: hit-rate@k, MRR (shared retriever, so should be identical — assert this)
  - Cost/latency ratio: avg tokens per query (prompt + completion), avg latency, agentic/simple ratio
- [x] Create `configs/experiment.yaml` — experiment config schema
- [x] Create `run_benchmark.py` — CLI entry point (--mode run/evaluate/compare/all)
- [x] **Verify**: End-to-end on 50 questions for both agents
  - Both agents answered the SAME 50 questions (same IDs, same order) ✅
  - Evaluation used identical metric functions (single codepath) ✅
  - Resume works (re-run skips completed questions) ✅
  - Tier invariant holds (EM ≤ Combined for both agents) ✅
  - Comparison JSON has all expected keys ✅

**Results (50-question test, no judge):**
  - Simple RAG: EM=0.520, F1=0.660, Combined=0.680
  - Agentic RAG: EM=0.520, F1=0.665, Combined=0.700
  - Agentic uses 4.5x latency, 3.3x tokens vs Simple
  - Retrieval: hit_rate@5=1.000, MRR=0.960

**Done**: Both agents run on 50 questions, metrics JSON produced, comparison shows identical eval conditions.

---

## Phase 8: Full Benchmark Run 🔶 (Partially Complete)
**Session goal**: Run both agents on 1000+ questions, full evaluation.

**Pre-req**: vLLM server running (will take a while).

- [x] Run Simple RAG on full dataset (1000 questions)
- [x] Run Agentic RAG on full dataset (SAME 1000 questions, same order)
- [x] Run 3-tier evaluation on both sets of predictions (same evaluator, same config)
- [x] Run LLM-as-a-judge on both sets (blind, same judge config)
- [ ] Generate comparison results with confidence intervals
- [x] **Sanity check**: Verify question sets are identical, eval config is identical

**Results (1000 questions, full 3-tier with judge):**
  - Simple RAG: EM=0.514, F1=0.607, Combined=0.691, Unanswerable Detection=56.2%
  - Agentic RAG: EM=0.519, F1=0.626, Combined=0.720, Unanswerable Detection=61.2%
  - Tier distribution (Simple): tier1=623, tier2=4, tier3=64, none=309
  - Tier distribution (Agentic): tier1=641, tier2=5, tier3=74, none=280
  - Judge faithfulness: Simple=422 faithful, Agentic=424 faithful
  - Agentic wins on combined accuracy (+2.9%) and unanswerable detection (+5.0%)
  - Simple wins on answerable EM (+4.0%) and answerable F1 (+1.4%)

**Files saved (gitignored, on disk):**
  - evaluation/predictions/simple_rag_predictions.jsonl (1000 predictions)
  - evaluation/predictions/agentic_rag_predictions.jsonl (1000 predictions)
  - evaluation/results/simple_rag_results.json (full per-question results)
  - evaluation/results/agentic_rag_results.json (full per-question results)
  - evaluation/results/*_summary.json (aggregate summaries)

**Remaining**: Run comparison phase (statistics, CIs — quick, no LLM calls needed).

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
  - Performance vs Cost section: is agentic RAG worth the extra tokens/latency?
- [ ] Create demo script for side-by-side comparison
- [ ] Record demo video

**Done when**: All docs written, demo ready.

---

## Progress Tracker
| Phase | Status | Session Date |
|-------|--------|-------------|
| 1. Scaffold & Env | ✅ Complete | 2026-04-01 |
| 2. Dataset | ✅ Complete | 2026-04-01 |
| 3. Retrieval | ✅ Complete | 2026-04-03 |
| 4. Simple RAG | ✅ Complete | 2026-04-03 |
| 5. Agentic RAG | ✅ Complete | 2026-04-04 |
| 6. Evaluation | ✅ Complete | 2026-04-05 |
| 7. Pipeline | ✅ Complete | 2026-04-06 |
| 8. Full Run | 🔶 Partial (comparison pending) | 2026-04-07 |
| 9. Visualization | ⬜ Not started | |
| 10. Report & Docs | ⬜ Not started | |
