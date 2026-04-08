"""Demo web app for RAG Benchmark — designed for video recording.

Shows side-by-side comparison of Simple RAG vs Agentic RAG with
step-by-step reasoning traces, metrics, and curated examples.

Usage:
    python demo/app.py
    # Open http://localhost:8501 in browser
"""

import json
import os
import sys
from pathlib import Path


from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# ── Load data ─────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent

def load_examples():
    """Load curated demo examples from actual benchmark results."""
    simple_preds = {}
    agentic_preds = {}
    with open(ROOT / "evaluation/predictions/simple_rag_predictions.jsonl") as f:
        for line in f:
            p = json.loads(line)
            simple_preds[p["id"]] = p
    with open(ROOT / "evaluation/predictions/agentic_rag_predictions.jsonl") as f:
        for line in f:
            p = json.loads(line)
            agentic_preds[p["id"]] = p

    with open(ROOT / "evaluation/results/simple_rag_results.json") as f:
        simple_pq = {e["id"]: e for e in json.load(f)["per_question"]}
    with open(ROOT / "evaluation/results/agentic_rag_results.json") as f:
        agentic_pq = {e["id"]: e for e in json.load(f)["per_question"]}

    # Curated question IDs (found from actual results)
    curated = [
        {
            "id": "5a838a2ee60761001a2eb794",
            "category": "Agentic Wins: Catches Unanswerable",
            "story": "This question is unanswerable — the context talks about ctenophores but doesn't describe how the aboral organ moves. Simple RAG fabricates an answer. Agentic RAG's answerability check catches it.",
        },
        {
            "id": "570d4030fed7b91900d45da1",
            "category": "Agentic Wins: Query Rewriting Helps",
            "story": "Simple RAG gives up on this question because the initial retrieval doesn't find the right passage. Agentic RAG rewrites the query and re-retrieves, finding the answer.",
        },
        {
            "id": "5725c071271a42140099d128",
            "category": "Both Succeed: Baseline Comparison",
            "story": "A straightforward factoid question. Both agents find the answer easily — this shows the system works. Notice the latency and token difference.",
        },
        {
            "id": "5a25e8d5ef59cd001a623d18",
            "category": "Both Fail: Honest Limitation",
            "story": "An adversarial unanswerable question. The context discusses procurement in construction, which tricks both agents into answering. This shows the hardest challenge in SQuAD v2.",
        },
        {
            "id": "571cde695efbb31900334e1a",
            "category": "Simple Wins: Agentic Over-thinks",
            "story": "Simple RAG directly extracts 'Bone' from context. Agentic RAG over-rewrites the query 3 times, loses the original intent, and declares unanswerable. Shows the cost of over-processing.",
        },
    ]

    examples = []
    for c in curated:
        qid = c["id"]
        if qid not in simple_pq or qid not in agentic_pq:
            continue
        s = simple_pq[qid]
        a = agentic_pq[qid]
        sp = simple_preds.get(qid, {})
        ap = agentic_preds.get(qid, {})

        examples.append({
            "id": qid,
            "category": c["category"],
            "story": c["story"],
            "question": s["question"],
            "gold_answers": s["gold_answers"] if s["gold_answers"] else ["(unanswerable)"],
            "is_unanswerable": s["is_unanswerable"],
            "context": (sp.get("contexts_used") or [""])[0][:600],
            "simple": {
                "answer": sp.get("predicted_answer", ""),
                "tier": s["tier_credited"],
                "correct": s["tier_credited"] != "none",
                "em": s["em"],
                "f1": s["f1"],
                "latency_ms": sp.get("latency_ms", 0),
                "total_tokens": sp.get("total_tokens", 0),
                "prompt_tokens": sp.get("prompt_tokens", 0),
                "completion_tokens": sp.get("completion_tokens", 0),
            },
            "agentic": {
                "answer": ap.get("predicted_answer", ""),
                "tier": a["tier_credited"],
                "correct": a["tier_credited"] != "none",
                "em": a["em"],
                "f1": a["f1"],
                "latency_ms": ap.get("latency_ms", 0),
                "total_tokens": ap.get("total_tokens", 0),
                "prompt_tokens": ap.get("prompt_tokens", 0),
                "completion_tokens": ap.get("completion_tokens", 0),
                "reasoning_trace": ap.get("reasoning_trace", ""),
            },
        })

    return examples


def load_summary():
    """Load benchmark summary stats."""
    with open(ROOT / "evaluation/results/comparison.json") as f:
        return json.load(f)


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="RAG Benchmark Demo")

# Serve chart images
app.mount("/figures", StaticFiles(directory=str(ROOT / "reports/figures")), name="figures")

EXAMPLES = load_examples()
COMPARISON = load_summary()


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/api/examples")
async def get_examples():
    return EXAMPLES


@app.get("/api/summary")
async def get_summary():
    agg = COMPARISON["aggregate_summary"]
    cl = COMPARISON["cost_latency"]
    return {
        "simple": {
            "combined_accuracy": agg["simple_rag"]["combined_accuracy"],
            "em": agg["simple_rag"]["em"],
            "f1": agg["simple_rag"]["f1"],
            "unanswerable_detection": agg["simple_rag"]["unanswerable_detection_rate"],
            "mean_latency": cl["latency_ms"]["simple"]["mean"],
            "mean_tokens": cl["total_tokens"]["simple"]["mean"],
        },
        "agentic": {
            "combined_accuracy": agg["agentic_rag"]["combined_accuracy"],
            "em": agg["agentic_rag"]["em"],
            "f1": agg["agentic_rag"]["f1"],
            "unanswerable_detection": agg["agentic_rag"]["unanswerable_detection_rate"],
            "mean_latency": cl["latency_ms"]["agentic"]["mean"],
            "mean_tokens": cl["total_tokens"]["agentic"]["mean"],
        },
        "n_questions": 1000,
    }


# ── HTML Template ─────────────────────────────────────────────────────────────

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Benchmark Demo — Simple vs Agentic</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; }

  .header { background: linear-gradient(135deg, #1a1a2e, #16213e); padding: 24px 40px; border-bottom: 1px solid #333; }
  .header h1 { font-size: 28px; font-weight: 700; color: #fff; }
  .header p { color: #999; margin-top: 4px; font-size: 14px; }

  .tabs { display: flex; gap: 0; background: #1a1a2e; padding: 0 40px; border-bottom: 1px solid #333; }
  .tab { padding: 12px 24px; cursor: pointer; color: #888; font-size: 14px; font-weight: 600;
         border-bottom: 3px solid transparent; transition: all 0.2s; }
  .tab:hover { color: #ccc; }
  .tab.active { color: #60a5fa; border-bottom-color: #60a5fa; }

  .container { max-width: 1400px; margin: 0 auto; padding: 24px 40px; }

  /* Dashboard */
  .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
  .stat-card { background: #1e1e2e; border-radius: 12px; padding: 20px; border: 1px solid #333; }
  .stat-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .stat-card .values { display: flex; justify-content: space-between; margin-top: 12px; }
  .stat-card .agent-val { text-align: center; }
  .stat-card .agent-val .num { font-size: 28px; font-weight: 700; }
  .stat-card .agent-val .name { font-size: 11px; color: #888; margin-top: 4px; }
  .simple-color { color: #60a5fa; }
  .agentic-color { color: #f59e0b; }

  .charts-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 16px; }
  .chart-card { background: #1e1e2e; border-radius: 12px; padding: 16px; border: 1px solid #333; }
  .chart-card img { width: 100%; border-radius: 8px; }
  .chart-card .title { font-size: 13px; color: #888; margin-bottom: 8px; }

  /* Examples */
  .example-nav { display: flex; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
  .example-btn { padding: 8px 16px; background: #1e1e2e; border: 1px solid #333; border-radius: 8px;
                 color: #ccc; cursor: pointer; font-size: 13px; transition: all 0.2s; }
  .example-btn:hover { border-color: #60a5fa; color: #fff; }
  .example-btn.active { background: #1a365d; border-color: #60a5fa; color: #60a5fa; }

  .example-container { display: none; }
  .example-container.active { display: block; }

  .story-box { background: #1a2332; border-left: 4px solid #60a5fa; padding: 16px 20px;
               border-radius: 0 8px 8px 0; margin-bottom: 20px; font-size: 14px; line-height: 1.6; color: #ccc; }

  .question-box { background: #1e1e2e; border-radius: 12px; padding: 20px; border: 1px solid #333; margin-bottom: 20px; }
  .question-box .q-label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .question-box .q-text { font-size: 20px; font-weight: 600; color: #fff; margin: 8px 0; }
  .question-box .gold { font-size: 14px; color: #4ade80; }
  .question-box .gold.unanswerable { color: #f87171; }
  .question-box .context { font-size: 13px; color: #888; margin-top: 12px; line-height: 1.5;
                           max-height: 100px; overflow-y: auto; padding: 10px; background: #161622; border-radius: 6px; }

  .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 20px; }
  .agent-card { background: #1e1e2e; border-radius: 12px; padding: 20px; border: 2px solid #333; }
  .agent-card.simple { border-color: #1e40af; }
  .agent-card.agentic { border-color: #92400e; }
  .agent-card .agent-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; }
  .agent-card .agent-name { font-size: 18px; font-weight: 700; }
  .agent-card .verdict { padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 700; }
  .verdict.correct { background: #064e3b; color: #4ade80; }
  .verdict.wrong { background: #450a0a; color: #f87171; }

  .answer-box { background: #161622; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
  .answer-box .ans-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; }
  .answer-box .ans-text { font-size: 16px; margin-top: 6px; line-height: 1.5; }

  .metrics-row { display: flex; gap: 12px; flex-wrap: wrap; }
  .metric-pill { background: #161622; border-radius: 6px; padding: 8px 12px; font-size: 12px; }
  .metric-pill .m-label { color: #888; }
  .metric-pill .m-value { color: #fff; font-weight: 600; margin-left: 4px; }

  /* Reasoning trace */
  .trace-box { margin-top: 16px; }
  .trace-box .trace-label { font-size: 12px; color: #f59e0b; text-transform: uppercase; letter-spacing: 1px;
                            cursor: pointer; user-select: none; }
  .trace-box .trace-label:hover { color: #fbbf24; }
  .trace-steps { margin-top: 10px; display: none; }
  .trace-steps.open { display: block; }
  .trace-step { display: flex; gap: 12px; padding: 10px 0; border-bottom: 1px solid #222; font-size: 13px; }
  .trace-step:last-child { border-bottom: none; }
  .step-icon { width: 32px; height: 32px; border-radius: 8px; display: flex; align-items: center;
               justify-content: center; font-size: 14px; flex-shrink: 0; }
  .step-retrieve { background: #1e3a5f; }
  .step-assess { background: #3b2f63; }
  .step-rewrite { background: #5f3a1e; }
  .step-check { background: #1e5f3a; }
  .step-generate { background: #5f1e3a; }
  .step-decision { background: #3a3a3a; }
  .step-content .step-name { font-weight: 600; color: #ccc; }
  .step-content .step-detail { color: #888; margin-top: 2px; line-height: 1.4; }

  .hidden { display: none; }
</style>
</head>
<body>

<div class="header">
  <h1>Simple RAG vs Agentic RAG — Benchmark Demo</h1>
  <p>1,000 SQuAD v2 Questions | Qwen3-14B | 3-Tier Evaluation | BM25 Retrieval</p>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('dashboard')">Dashboard</div>
  <div class="tab" onclick="showTab('examples')">Live Examples</div>
  <div class="tab" onclick="showTab('charts')">Charts</div>
</div>

<div id="tab-dashboard" class="container tab-content">
  <div class="stats-grid" id="stats-grid"></div>
</div>

<div id="tab-examples" class="container tab-content hidden">
  <div class="example-nav" id="example-nav"></div>
  <div id="examples-container"></div>
</div>

<div id="tab-charts" class="container tab-content hidden">
  <div class="charts-grid">
    <div class="chart-card"><div class="title">Overall Metric Comparison</div><img src="/figures/01_metric_comparison.png"></div>
    <div class="chart-card"><div class="title">Per-Category Breakdown</div><img src="/figures/02_category_breakdown.png"></div>
    <div class="chart-card"><div class="title">Tier Distribution</div><img src="/figures/04_tier_distribution.png"></div>
    <div class="chart-card"><div class="title">F1 Score Distribution</div><img src="/figures/03_f1_distribution.png"></div>
    <div class="chart-card"><div class="title">Confusion Matrix</div><img src="/figures/08_confusion_matrix.png"></div>
    <div class="chart-card"><div class="title">Cost-Accuracy Trade-off</div><img src="/figures/05_cost_accuracy_tradeoff.png"></div>
    <div class="chart-card"><div class="title">Latency Distribution</div><img src="/figures/06_latency_distribution.png"></div>
    <div class="chart-card"><div class="title">Token Usage</div><img src="/figures/07_token_usage.png"></div>
    <div class="chart-card"><div class="title">Failure Modes</div><img src="/figures/11_failure_modes.png"></div>
    <div class="chart-card"><div class="title">Faithfulness</div><img src="/figures/09_faithfulness.png"></div>
  </div>
</div>

<script>
let examples = [];
let summary = {};

// Tab switching
function showTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.remove('hidden');
  event.target.classList.add('active');
}

// Parse reasoning trace into steps
function parseTrace(trace) {
  if (!trace) return [];
  const steps = [];
  const regex = /\[([A-Z_]+)\]\s*([\s\S]*?)(?=\[|$)/g;
  let match;
  while ((match = regex.exec(trace)) !== null) {
    steps.push({ name: match[1], detail: match[2].trim().substring(0, 200) });
  }
  return steps;
}

function stepClass(name) {
  const map = { 'RETRIEVE': 'retrieve', 'ASSESS_RELEVANCE': 'assess', 'REWRITE_QUERY': 'rewrite',
                'CHECK_ANSWERABILITY': 'check', 'GENERATE_ANSWER': 'generate', 'DECISION': 'decision' };
  return map[name] || 'decision';
}

function stepIcon(name) {
  const map = { 'RETRIEVE': '&#128269;', 'ASSESS_RELEVANCE': '&#9878;', 'REWRITE_QUERY': '&#9997;',
                'CHECK_ANSWERABILITY': '&#10067;', 'GENERATE_ANSWER': '&#9889;', 'DECISION': '&#10148;' };
  return map[name] || '&#8226;';
}

function toggleTrace(id) {
  const el = document.getElementById('trace-' + id);
  el.classList.toggle('open');
}

// Build dashboard
function buildDashboard(s) {
  const metrics = [
    { label: 'Combined Accuracy', simple: s.simple.combined_accuracy, agentic: s.agentic.combined_accuracy, fmt: v => (v*100).toFixed(1)+'%' },
    { label: 'Exact Match', simple: s.simple.em, agentic: s.agentic.em, fmt: v => (v*100).toFixed(1)+'%' },
    { label: 'Mean Latency', simple: s.simple.mean_latency, agentic: s.agentic.mean_latency, fmt: v => v.toFixed(0)+'ms' },
    { label: 'Mean Tokens/Query', simple: s.simple.mean_tokens, agentic: s.agentic.mean_tokens, fmt: v => v.toFixed(0) },
  ];
  const grid = document.getElementById('stats-grid');
  grid.innerHTML = metrics.map(m => `
    <div class="stat-card">
      <div class="label">${m.label}</div>
      <div class="values">
        <div class="agent-val"><div class="num simple-color">${m.fmt(m.simple)}</div><div class="name">Simple RAG</div></div>
        <div class="agent-val"><div class="num agentic-color">${m.fmt(m.agentic)}</div><div class="name">Agentic RAG</div></div>
      </div>
    </div>
  `).join('');
}

// Build examples
function buildExamples(exs) {
  const nav = document.getElementById('example-nav');
  const container = document.getElementById('examples-container');

  nav.innerHTML = exs.map((ex, i) => `
    <div class="example-btn ${i===0?'active':''}" onclick="showExample(${i})">${ex.category}</div>
  `).join('');

  container.innerHTML = exs.map((ex, i) => {
    const traceSteps = parseTrace(ex.agentic.reasoning_trace);
    const goldText = ex.is_unanswerable ? 'UNANSWERABLE' : ex.gold_answers.join(' / ');
    const goldClass = ex.is_unanswerable ? 'unanswerable' : '';

    return `
    <div class="example-container ${i===0?'active':''}" id="example-${i}">
      <div class="story-box">${ex.story}</div>

      <div class="question-box">
        <div class="q-label">Question</div>
        <div class="q-text">${ex.question}</div>
        <div class="gold ${goldClass}">Gold Answer: ${goldText}</div>
        <div class="context"><strong>Retrieved Context (top-1):</strong><br>${ex.context}</div>
      </div>

      <div class="comparison">
        <div class="agent-card simple">
          <div class="agent-header">
            <span class="agent-name simple-color">Simple RAG</span>
            <span class="verdict ${ex.simple.correct?'correct':'wrong'}">${ex.simple.correct?'CORRECT':'WRONG'}</span>
          </div>
          <div class="answer-box">
            <div class="ans-label">Answer</div>
            <div class="ans-text">${ex.simple.answer || '(empty)'}</div>
          </div>
          <div class="metrics-row">
            <div class="metric-pill"><span class="m-label">Latency:</span><span class="m-value">${ex.simple.latency_ms.toFixed(0)}ms</span></div>
            <div class="metric-pill"><span class="m-label">Tokens:</span><span class="m-value">${ex.simple.total_tokens}</span></div>
            <div class="metric-pill"><span class="m-label">F1:</span><span class="m-value">${ex.simple.f1.toFixed(3)}</span></div>
            <div class="metric-pill"><span class="m-label">Tier:</span><span class="m-value">${ex.simple.tier}</span></div>
          </div>
        </div>

        <div class="agent-card agentic">
          <div class="agent-header">
            <span class="agent-name agentic-color">Agentic RAG</span>
            <span class="verdict ${ex.agentic.correct?'correct':'wrong'}">${ex.agentic.correct?'CORRECT':'WRONG'}</span>
          </div>
          <div class="answer-box">
            <div class="ans-label">Answer</div>
            <div class="ans-text">${ex.agentic.answer || '(empty)'}</div>
          </div>
          <div class="metrics-row">
            <div class="metric-pill"><span class="m-label">Latency:</span><span class="m-value">${ex.agentic.latency_ms.toFixed(0)}ms</span></div>
            <div class="metric-pill"><span class="m-label">Tokens:</span><span class="m-value">${ex.agentic.total_tokens}</span></div>
            <div class="metric-pill"><span class="m-label">F1:</span><span class="m-value">${ex.agentic.f1.toFixed(3)}</span></div>
            <div class="metric-pill"><span class="m-label">Tier:</span><span class="m-value">${ex.agentic.tier}</span></div>
          </div>
          ${traceSteps.length > 0 ? `
          <div class="trace-box">
            <div class="trace-label" onclick="toggleTrace(${i})">&#9660; Reasoning Trace (${traceSteps.length} steps) — click to expand</div>
            <div class="trace-steps" id="trace-${i}">
              ${traceSteps.map(s => `
                <div class="trace-step">
                  <div class="step-icon step-${stepClass(s.name)}">${stepIcon(s.name)}</div>
                  <div class="step-content">
                    <div class="step-name">${s.name.replace(/_/g, ' ')}</div>
                    <div class="step-detail">${s.detail}</div>
                  </div>
                </div>
              `).join('')}
            </div>
          </div>
          ` : ''}
        </div>
      </div>
    </div>`;
  }).join('');
}

function showExample(idx) {
  document.querySelectorAll('.example-btn').forEach((el, i) => el.classList.toggle('active', i === idx));
  document.querySelectorAll('.example-container').forEach((el, i) => el.classList.toggle('active', i === idx));
}

// Init
async function init() {
  const [exRes, sumRes] = await Promise.all([
    fetch('/api/examples').then(r => r.json()),
    fetch('/api/summary').then(r => r.json()),
  ]);
  examples = exRes;
  summary = sumRes;
  buildDashboard(summary);
  buildExamples(examples);
}

init();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    print("\n  RAG Benchmark Demo")
    print("  Open http://localhost:8501 in your browser\n")
    uvicorn.run(app, host="0.0.0.0", port=8501, log_level="warning")
