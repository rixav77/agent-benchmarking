# Benchmark Analysis Summary

## Simple RAG vs Agentic RAG — 1000 SQuAD v2 Questions

## Key Results

- **Simple RAG Combined Accuracy**: 69.1%
- **Agentic RAG Combined Accuracy**: 72.0%
- **Delta**: +2.9 percentage points in favor of Agentic RAG
- **Latency Ratio**: Agentic is 5.5x slower
- **Token Ratio**: Agentic uses 3.5x more tokens

## Simple RAG

### Strengths

1. **Better answerable EM**: 46.6% vs 42.6% (+4.0pp) — direct extractive answers are more precise for straightforward questions
2. **5.46x faster**: Mean latency 638ms vs 3483ms — single LLM call vs multi-step pipeline
3. **3.5x cheaper**: 1060 tokens/query vs 3725 — significant cost savings at scale
4. **Simpler architecture**: No query rewriting, relevance assessment, or iterative loops — easier to debug and maintain

### Weaknesses

1. **More false positives**: 219 vs 194 — answers unanswerable questions 25 more times
2. **Lower unanswerable detection**: 56.2% vs 61.2% — fails to recognize when context doesn't contain the answer
3. **No self-correction**: Cannot rewrite queries or re-retrieve when initial retrieval is poor

## Agentic RAG

### Strengths

1. **Better unanswerable detection**: 61.2% vs 56.2% — **statistically significant** (95% CIs non-overlapping)
2. **Fewer false positives**: 194 vs 219 — the answerability checking tool catches adversarial unanswerable questions
3. **Higher combined accuracy**: 72.0% vs 69.1% — multi-step reasoning recovers answers that single-pass misses
4. **Query rewriting**: Reformulates poor queries to improve retrieval quality on hard questions

### Weaknesses

1. **5.46x slower**: Multi-step pipeline with 3-5 LLM calls per question (retrieve → assess → rewrite → check → generate)
2. **3.5x more expensive**: 3725 tokens/query — each tool call adds prompt overhead
3. **Lower answerable EM**: 42.6% vs 46.6% — over-processing can degrade simple extractive answers
4. **Diminishing returns**: The +2.9pp accuracy gain may not justify the 5.46x cost increase in production

## Common Failure Modes

### 1. Adversarial Unanswerable Questions (Largest Failure Category)

Both agents struggle with SQuAD v2's adversarial unanswerable questions — questions with misleading premises where the context contains related but irrelevant information.
- Simple RAG: 219 false positives (21.9% of all questions)
- Agentic RAG: 194 false positives (19.4% of all questions)
- Agentic RAG's `check_answerability` tool reduces this by 25 cases (11.4% reduction)

### 2. Answerable Questions Refused (False Negatives)

Both agents sometimes refuse to answer answerable questions when retrieval quality is low.
- Simple RAG: 63 false negatives
- Agentic RAG: 61 false negatives — despite query rewriting, some questions remain hard to retrieve

### 3. Wrong Answers (All Tiers Rejected)

A small number of answers are plausible but incorrect — F1 too low for Tier 1, semantic similarity below threshold, and judge rejected.
- Simple RAG: 27 wrong answers (avg F1=0.085)
- Agentic RAG: 25 wrong answers (avg F1=0.087)

### 4. Faithfulness Issues

A small fraction of answers contain information not grounded in the provided context.
- Simple RAG: 11 unfaithful + 4 partial = 15 total
- Agentic RAG: 9 unfaithful + 6 partial = 15 total
- Both agents show >96% faithfulness — the LLM's context-following ability is strong

## Trade-off Analysis

| Dimension | Simple RAG | Agentic RAG | Verdict |
|-----------|-----------|------------|---------|
| Combined Accuracy | 69.1% | 72.0% | Agentic (+2.9pp) |
| Answerable EM | 46.6% | 42.6% | Simple (+4.0pp) |
| Unanswerable Det. | 56.2% | 61.2% | **Agentic (significant)** |
| Latency | 638ms | 3483ms | Simple (5.46x faster) |
| Token Cost | 1060 | 3725 | Simple (3.5x cheaper) |
| Faithfulness | 422/437 | 424/439 | Comparable |

## Conclusion

Agentic RAG provides a meaningful improvement in overall accuracy (+2.9pp) and a statistically significant advantage in unanswerable detection (+5.0pp). However, this comes at a steep cost: 5.46x latency and 3.5x token usage. The choice depends on the use case:

- **Use Simple RAG** when: speed matters, budget is constrained, questions are mostly straightforward, or the dataset has few unanswerable questions.
- **Use Agentic RAG** when: accuracy on adversarial/tricky questions is critical, unanswerable detection is important (e.g., medical/legal QA), and latency/cost are acceptable.

