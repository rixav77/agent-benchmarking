#!/usr/bin/env python3
"""
Find 5-6 curated demo questions that best showcase Simple RAG vs Agentic RAG differences.

Categories:
1. Unanswerable where Agentic caught it but Simple hallucinated
2. Hard answerable where Agentic's query rewriting helped (agentic correct, simple wrong)
3. Easy answerable where both succeed
4. Unanswerable where BOTH failed
5. Answerable where Simple was right but Agentic was wrong (honest trade-off)
"""

import json
import sys
from pathlib import Path

BASE = Path("/home/shivank_g/agent-benchmarking")

# --- Load data ---
with open(BASE / "evaluation/results/simple_rag_results.json") as f:
    simple_results = {q["id"]: q for q in json.load(f)["per_question"]}

with open(BASE / "evaluation/results/agentic_rag_results.json") as f:
    agentic_results = {q["id"]: q for q in json.load(f)["per_question"]}

agentic_traces = {}
with open(BASE / "evaluation/predictions/agentic_rag_predictions.jsonl") as f:
    for line in f:
        row = json.loads(line)
        agentic_traces[row["id"]] = row["reasoning_trace"]

simple_answers = {}
with open(BASE / "evaluation/predictions/simple_rag_predictions.jsonl") as f:
    for line in f:
        row = json.loads(line)
        simple_answers[row["id"]] = row

# All question IDs that exist in both
all_ids = set(simple_results) & set(agentic_results)


TIER_MAP = {"none": 0, "tier1": 1, "tier2": 2, "tier3": 3}

def tier(q):
    """Return numeric tier value: none=0, tier1=1, tier2=2, tier3=3."""
    t = q.get("tier_credited", "none")
    if isinstance(t, bool):
        return 3 if t else 0
    if isinstance(t, (int, float)):
        return int(t)
    return TIER_MAP.get(str(t).lower(), 0)


def score(q):
    """Composite score: semantic_correct * 3 + tier_credited + f1."""
    return (
        (1 if q.get("semantic_correct") else 0) * 3
        + tier(q)
        + q.get("f1", 0)
    )


def gold_answer_str(q):
    answers = q.get("gold_answers", [])
    if not answers:
        return "unanswerable"
    # SQuAD 2.0: if first answer is empty string it's unanswerable
    if all(a == "" for a in answers):
        return "unanswerable"
    # Return unique answers
    seen = []
    for a in answers:
        if a not in seen:
            seen.append(a)
    return " / ".join(seen)


def make_entry(qid, category_label):
    s = simple_results[qid]
    a = agentic_results[qid]
    gold = gold_answer_str(s)
    return {
        "category": category_label,
        "question_id": qid,
        "question": s["question"],
        "gold_answer": gold,
        "simple_rag": {
            "predicted_answer": s["predicted_answer"],
            "tier_credited": s.get("tier_credited", "none"),
            "em": s.get("em", 0),
            "f1": round(s.get("f1", 0), 3),
            "semantic_correct": s.get("semantic_correct", False),
            "is_unanswerable_pred": s.get("is_unanswerable_pred", False),
        },
        "agentic_rag": {
            "predicted_answer": a["predicted_answer"],
            "tier_credited": a.get("tier_credited", "none"),
            "em": a.get("em", 0),
            "f1": round(a.get("f1", 0), 3),
            "semantic_correct": a.get("semantic_correct", False),
            "is_unanswerable_pred": a.get("is_unanswerable_pred", False),
            "reasoning_trace": agentic_traces.get(qid, ""),
        },
    }



# -----------------------------------------------------------------------
# Category 1: Unanswerable where Agentic caught it, Simple hallucinated
# Criteria:
#   - is_unanswerable == True (gold unanswerable)
#   - agentic predicted unanswerable (is_unanswerable_pred == True)
#   - simple predicted answerable (is_unanswerable_pred == False) and gave a non-empty answer
# Prefer cases where simple gave a confident, longer hallucinated answer
# -----------------------------------------------------------------------
cat1_candidates = []
for qid in all_ids:
    s = simple_results[qid]
    a = agentic_results[qid]
    if (
        s.get("is_unanswerable") == True
        and a.get("is_unanswerable_pred") == True
        and s.get("is_unanswerable_pred") == False
        and len(s.get("predicted_answer", "")) > 20
    ):
        # Score: longer simple hallucination = more dramatic
        cat1_candidates.append((len(s["predicted_answer"]), qid))

cat1_candidates.sort(reverse=True)

# -----------------------------------------------------------------------
# Category 2: Hard answerable where Agentic succeeded but Simple failed
# Criteria:
#   - is_unanswerable == False
#   - agentic tier >= 2 (full credit)
#   - simple tier <= 1 (partial or wrong)
#   - agentic reasoning_trace contains query rewriting or multi-step signals
# -----------------------------------------------------------------------
cat2_candidates = []
for qid in all_ids:
    s = simple_results[qid]
    a = agentic_results[qid]
    trace = agentic_traces.get(qid, "")
    if (
        s.get("is_unanswerable") == False
        and tier(a) >= 2
        and tier(s) <= 1
    ):
        # Prefer traces that show reformulation/rewrite
        has_rewrite = any(kw in trace.lower() for kw in ["refor", "rewrite", "requery", "retry", "attempt=2", "attempt=3", "reformulat"])
        # Score: agentic score - simple score (bigger gap = more dramatic)
        gap = score(a) - score(s)
        cat2_candidates.append((int(has_rewrite) * 10 + gap, qid))

cat2_candidates.sort(reverse=True)

# -----------------------------------------------------------------------
# Category 3: Easy answerable where BOTH succeed
# Criteria:
#   - is_unanswerable == False
#   - both tier >= 2
#   - both semantic_correct
# Prefer a question with a clean, concise answer
# -----------------------------------------------------------------------
cat3_candidates = []
for qid in all_ids:
    s = simple_results[qid]
    a = agentic_results[qid]
    if (
        s.get("is_unanswerable") == False
        and tier(s) >= 2
        and tier(a) >= 2
        and s.get("semantic_correct") == True
        and a.get("semantic_correct") == True
    ):
        # Prefer shorter, crisper questions and answers
        q_len = len(s["question"])
        a_len = len(s.get("predicted_answer", ""))
        cat3_candidates.append((-(q_len + a_len), qid))

cat3_candidates.sort(reverse=True)

# -----------------------------------------------------------------------
# Category 4: Unanswerable where BOTH failed (both said answerable)
# Criteria:
#   - is_unanswerable == True
#   - simple is_unanswerable_pred == False
#   - agentic is_unanswerable_pred == False
# Prefer cases where both gave long, confident-sounding hallucinations
# -----------------------------------------------------------------------
cat4_candidates = []
for qid in all_ids:
    s = simple_results[qid]
    a = agentic_results[qid]
    if (
        s.get("is_unanswerable") == True
        and s.get("is_unanswerable_pred") == False
        and a.get("is_unanswerable_pred") == False
    ):
        combined_len = len(s.get("predicted_answer", "")) + len(a.get("predicted_answer", ""))
        cat4_candidates.append((combined_len, qid))

cat4_candidates.sort(reverse=True)

# -----------------------------------------------------------------------
# Category 5: Answerable where Simple was right but Agentic was wrong
# Criteria:
#   - is_unanswerable == False
#   - simple tier >= 2
#   - agentic tier <= 1
# -----------------------------------------------------------------------
cat5_candidates = []
for qid in all_ids:
    s = simple_results[qid]
    a = agentic_results[qid]
    if (
        s.get("is_unanswerable") == False
        and tier(s) >= 2
        and tier(a) <= 1
        and s.get("semantic_correct") == True
    ):
        # Prefer bigger gap
        gap = score(s) - score(a)
        cat5_candidates.append((gap, qid))

cat5_candidates.sort(reverse=True)


# --- Pick top candidate for each category, avoiding overlaps ---
used_ids = set()
demo_questions = []

def pick_best(candidates, label, n=1):
    picked = 0
    for _, qid in candidates:
        if qid not in used_ids:
            used_ids.add(qid)
            demo_questions.append(make_entry(qid, label))
            picked += 1
            if picked >= n:
                break
    if picked == 0:
        demo_questions.append({"category": label, "error": "No candidates found"})


pick_best(cat1_candidates, "1. Unanswerable: Agentic caught it, Simple hallucinated")
pick_best(cat2_candidates, "2. Hard answerable: Agentic succeeded via query rewriting, Simple failed")
pick_best(cat3_candidates, "3. Easy answerable: Both succeed (baseline comparison)")
pick_best(cat4_candidates, "4. Unanswerable: Both failed (honest limitations)")
pick_best(cat5_candidates, "5. Answerable: Simple right, Agentic wrong (trade-off)")

# Print summary stats for each category
stats = {
    "cat1_pool_size": len(cat1_candidates),
    "cat2_pool_size": len(cat2_candidates),
    "cat3_pool_size": len(cat3_candidates),
    "cat4_pool_size": len(cat4_candidates),
    "cat5_pool_size": len(cat5_candidates),
}

output = {
    "stats": stats,
    "demo_questions": demo_questions
}

print(json.dumps(output, indent=2))
