"""Demo: Agentic RAG vs Simple RAG — side by side comparison."""

import yaml
from src.data.squad_loader import load_from_jsonl
from src.retrieval.bm25_retriever import BM25Retriever
from src.agents.llm_client import LLMClient
from src.agents.simple_rag import SimpleRAGAgent
from src.agents.agentic_rag import AgenticRAGAgent


def main():
    with open("configs/base.yaml") as f:
        config = yaml.safe_load(f)
    config["agent"] = {"max_retries": 2}

    print("\n Loading dataset...")
    examples = load_from_jsonl(config["dataset"]["output_path"])

    print(" Building BM25 index...")
    bm25 = BM25Retriever()
    bm25.index(examples)

    print(" Connecting to LLM (Qwen3-14B via vLLM)...")
    llm = LLMClient(
        base_url=config["llm"]["base_url"],
        model=config["llm"]["model"],
        temperature=config["llm"]["temperature"],
        max_tokens=config["llm"]["max_tokens"],
    )

    simple = SimpleRAGAgent(llm_client=llm, retriever=bm25, config=config)
    agentic = AgenticRAGAgent(llm_client=llm, retriever=bm25, config=config)
    print(" Ready!\n")

    # Pick demo questions
    answerable = [e for e in examples if not e.is_unanswerable]
    unanswerable = [e for e in examples if e.is_unanswerable]
    demo = [answerable[0], answerable[5], unanswerable[0]]

    sep = "=" * 70

    print(sep)
    print("  SIMPLE RAG vs AGENTIC RAG  --  Live Demo")
    print("  Model: Qwen3-14B | Retriever: BM25 (top-5) | Dataset: SQuAD 2.0")
    print(sep)

    for i, ex in enumerate(demo):
        label = "ANSWERABLE" if not ex.is_unanswerable else "UNANSWERABLE"
        gold = ex.answers[0] if ex.answers else "[no answer possible]"

        print(f"\n{'~'*70}")
        print(f"  Question {i+1}/3  [{label}]")
        print(f"{'~'*70}")
        print(f"\n  Q: {ex.question}")
        print(f"  Gold Answer: {gold}")

        # --- Simple RAG ---
        print(f"\n  --- SIMPLE RAG ---")
        print(f"  Running...", end="", flush=True)
        s_pred = simple.answer(ex)
        print(f" done! ({s_pred.latency_ms:.0f}ms)")
        print(f"  Answer: {s_pred.predicted_answer}")
        print(f"  Tokens: {s_pred.total_tokens}")

        # --- Agentic RAG ---
        print(f"\n  --- AGENTIC RAG ---")
        print(f"  Running...", end="", flush=True)
        a_pred = agentic.answer(ex)
        print(f" done! ({a_pred.latency_ms:.0f}ms)")
        print(f"  Answer: {a_pred.predicted_answer}")
        print(f"  Tokens: {a_pred.total_tokens}")
        print(f"\n  Reasoning trace:")
        for line in a_pred.reasoning_trace.split("\n"):
            print(f"    {line}")

    print(f"\n{sep}")
    print("  DEMO COMPLETE")
    print(sep + "\n")


if __name__ == "__main__":
    main()
