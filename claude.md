# Agent Benchmarking Plan – Simple RAG vs Agentic RAG on SQuAD 2.0

## 1. High‑Level Objective

Benchmark and compare two QA agent architectures—Simple RAG and Agentic RAG—on the SQuAD 2.0 dataset (answerable + unanswerable questions), measuring accuracy and faithfulness, with both quantitative metrics and LLM-as-a-judge evaluation.

Target outputs:
- Reproducible benchmark pipeline.
- Clear comparison of architectures: strengths, weaknesses, failure modes.
- Report with metrics, analysis, and example cases.
- Demo showing example queries and side‑by‑side outputs.

## 2. Benchmark Scope

- Benchmarking option: **A – Compare two agent architectures**.
- Architectures:
  - **Simple RAG**: single-pass retrieval + answer generation.
  - **Agentic RAG**: multi-step agent with planning / tool use (e.g., question refinement, iterative retrieval, answer verification).
- Task: extractive / grounded QA with abstention on unanswerable questions.

### Dataset: SQuAD 2.0 (Know What You Don’t Know)

- Source: "Know What You Don’t Know: Unanswerable Questions for SQuAD" (Rajpurkar et al., 2018).
- Key idea: adds 53,775 hard, human-written unanswerable questions so models must abstain when the answer is not in the paragraph.
- Split sizes (approx.):
  - Train: 130,319 total (43,498 unanswerable; ~2:1 answerable:unanswerable).
  - Dev: 11,873 total (5,945 unanswerable; ~1:1).
  - Test: 8,862 total (4,332 unanswerable; ~1:1).
- Input format per example (illustrative):
  - `id`: unique identifier.
  - `title`: article title (e.g., "Super_Bowl_50").
  - `context`: paragraph text.
  - `question`: natural-language question.
  - `answers`: for answerable questions, one or more answer spans with `text` and `answer_start`; for unanswerable, empty list / special flag.
- Why this dataset:
  - Requires handling both answerable and unanswerable questions.
  - Unanswerable questions are adversarial but plausible (negation, entity swaps, etc.).
  - Standard benchmark with known human and model baselines.

## 3. Agent Definitions

### 3.1 Simple RAG

**Goal:** Strong but straightforward retrieval-augmented QA baseline.

- Components:
  - Document store / index over SQuAD contexts.
  - Retriever (e.g., BM25 or embedding-based) returning top‑k passages per question.
  - Single LLM call that takes:
    - The question.
    - Retrieved context passages (concatenated or structured).
  - Prompt instructs model to:
    - Answer concisely if the answer is supported by the context.
    - Explicitly say that the question is unanswerable based on the given context when evidence is missing.
- Behavior:
  - One-shot, no iterative reasoning.
  - No explicit verification beyond the base prompt.

### 3.2 Agentic RAG

**Goal:** More sophisticated agent that explicitly plans, uses tools, and verifies answers to improve faithfulness and abstention behavior.

- Components (example design):
  - Same underlying document store / retriever interface.
  - Agent controller with tools, such as:
    - `retrieve_passages(query, k)`: retrieve top‑k context passages.
    - `refine_question(previous_question, context)`: optionally rewrite / clarify question.
    - `verify_answer(question, context, draft_answer)`: judge whether the draft answer is grounded.
  - Multi-step reasoning loop:
    1. **Plan:** interpret the question, decide whether more retrieval is needed.
    2. **Act:** call tools (retrieve, refine, etc.).
    3. **Reflect:** verify draft answer; if ungrounded, either re-retrieve/re-answer or abstain.
- Behavior:
  - Produces an explicit reasoning trace (tool calls, intermediate thoughts).
  - Designed to reduce hallucinations and better handle unanswerable questions.

*(Implementation can be done in Python with any LLM framework; this spec is architecture-agnostic.)*

## 4. Evaluation Design

### 4.1 Metrics

- **Accuracy (primary):**
  - For answerable questions:
    - Exact match / F1 between predicted answer and any ground-truth span.
  - For unanswerable questions:
    - Binary correctness of abstention (did the model correctly indicate that the question is unanswerable from the context?).
  - Report:
    - EM/F1 on answerable subset.
    - Unanswerable accuracy.
    - Combined accuracy across all questions.

- **Faithfulness:**
  - Definition: the answer should be strictly supported by the provided context with no hallucinated facts.
  - Measured via **LLM-as-a-judge**:
    - Inputs: question, context passage(s), model answer.
    - Output: faithfulness label or score, e.g.:
      - Faithful / Partially Faithful / Unfaithful, or
      - 1–5 rating.
  - Report average faithfulness and hallucination rate per architecture.

- *(Optional but nice-to-have)* **Latency & Cost:**
  - Average per-question inference time.
  - Approximate token / API cost per 1k questions.

### 4.2 Evaluation Types

- **Quantitative:**
  - Exact match / F1 on answerable questions.
  - Unanswerable detection accuracy.
  - Overall accuracy.

- **LLM-as-a-judge (qualitative + semi-quantitative):**
  - Rubric dimensions:
    - Correctness (is the answer factually correct?).
    - Completeness (does it cover all parts of the question?).
    - Reasoning quality (coherent, uses context logically).
    - Faithfulness (no unsupported statements relative to the context).
  - Use the same judge prompts for both architectures.
  - Optionally evaluate on a sampled subset (e.g., 500–1,000 examples) for cost control.

## 5. Experimentation Framework

**Overall pipeline: Run → Collect → Evaluate → Compare**

- **Dataset loader:**
  - Load SQuAD 2.0 splits.
  - Standardize each example into a schema like:
    - `id`, `title`, `context`, `question`, `answers`, `is_unanswerable`.

- **Runner:**
  - For each architecture (Simple RAG, Agentic RAG) and config:
    - Iterate over dataset split (e.g., dev or a 1k+ subset).
    - Call the agent to produce outputs:
      - `predicted_answer` (may be an explicit "unanswerable" response).
      - Optional reasoning trace / tool logs.
      - Latency and token stats.
    - Log all inputs/outputs and metadata to disk (e.g., JSONL/CSV).

- **Evaluator:**
  - Quantitative metrics:
    - Compute EM/F1 for answerable questions.
    - Compute unanswerable accuracy.
    - Aggregate metrics by split and architecture.
  - LLM-as-a-judge:
    - For a selected subset:
      - Call judge model with rubric.
      - Record scores/labels for correctness, completeness, reasoning, and faithfulness.

- **Comparer & analysis:**
  - Produce summary tables for each architecture.
  - Identify where Agentic RAG > Simple RAG and vice versa.
  - Analyze systematic failure modes (e.g., negation, entity swaps, number confusions).

- **Reproducibility:**
  - Use configuration files (YAML/JSON) to specify:
    - Model names.
    - Retrieval settings (top‑k, embedding model, etc.).
    - Prompt templates.
    - Dataset split and sampling strategy.

## 6. Repository Structure (Target)

Planned structure for this project (can evolve as needed):

- repo/
  - src/
    - Core agent implementations, retrieval stack, and pipeline scripts.
  - data/
    - Downloaded / preprocessed SQuAD 2.0 or scripts to fetch it.
  - evaluation/
    - Metric implementations, LLM-judge scripts, and analysis notebooks/scripts.
  - reports/ (optional)
    - Generated metrics, plots, and the final benchmark report.
  - README.md
    - High-level description and setup / run instructions.
  - ARCHITECTURE.md
    - Detailed description of Simple RAG and Agentic RAG implementations.
  - DEMO_VIDEO
    - Link or file for demo showing example outputs and comparisons.
  - requirements.txt or pyproject.toml
    - Python dependencies and environment setup.

## 7. Concrete Implementation Steps / TODO

1. **Project setup**
   - Initialize repository structure.
   - Add environment configuration (requirements, etc.).

2. **Dataset handling**
   - Download SQuAD 2.0.
   - Implement loader + preprocessing to the unified schema.

3. **Retrieval backbone**
   - Build context index over SQuAD paragraphs.
   - Implement retriever interface with configurable backend (BM25 / embeddings).

4. **Simple RAG agent**
   - Implement single-pass RAG pipeline.
   - Define prompts, retrieval settings, and configuration.

5. **Agentic RAG agent**
   - Design tool schema and reasoning loop.
   - Implement planner/controller that orchestrates retrieval, refinement, and verification.

6. **Evaluation**
   - Implement EM/F1 and unanswerable accuracy.
   - Implement LLM-as-a-judge evaluation for correctness, completeness, reasoning, and faithfulness.

7. **Experiment scripts**
   - Scripts/CLI to:
     - Run each agent on the chosen split.
     - Save outputs (JSONL/CSV) with predictions and traces.
     - Run evaluation and save metrics.

8. **Analysis & visualization**
   - Generate tables and charts comparing both architectures.
   - Collect qualitative examples of successes/failures.

9. **Report & demo**
   - Write the benchmark report summarizing setup, metrics, and insights.
   - Record a short demo video of the system and comparisons.

## 8. Success Criteria

- Experiments run on at least 1,000 SQuAD 2.0 questions (ideally full dev/test split).
- Both agents implemented and configurable via code / configs.
- Quantitative and LLM-judge results clearly show differences between Simple RAG and Agentic RAG.
- Repository is reproducible from the README instructions.
- Benchmark report clearly explains:
  - Setup and design choices.
  - Key findings and trade-offs between Simple RAG and Agentic RAG.
