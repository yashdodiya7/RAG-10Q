#!/usr/bin/env python
"""
A/B test: metadata vector RAG vs. GraphRAG on a gold question set.

Arms (all use the SAME passages, embeddings and brand filter):
  vector@4   baseline: Chroma similarity search, top 4, filtered by brand metadata
  vector@10  same, top 10  -> same passage budget as the graph arm (fairness check)
  graph-add  vector@4 + graph expansion appended (naive GraphRAG, ~9 passages)
  graph-rrf  vector top-20 + question->concept links + graph expansion, fused with
             reciprocal rank fusion, cut to 10 passages (same budget as vector@10)

Metric: section recall. Each question lists groups of gold sections
('Section title@page'); a group is hit when a retrieved passage belongs to one of
them. recall = hit groups / groups, complete = all groups hit.

    python scripts/compare_retrievers.py --brand BMW
    python scripts/compare_retrievers.py --brand BMW --answers 3     # also generate answers
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.graph.schemas import KnowledgeGraph, slugify  # noqa: E402

ANSWER_PROMPT = """You answer questions about a vehicle using ONLY the numbered manual excerpts below.
Cite the excerpts you use like [1], [3]. If the excerpts do not contain the answer, say so.

Excerpts:
{context}

Question: {question}

Answer:"""


def matches(doc, target: str) -> bool:
    """target 'Section title@page': same section title and the passage covers that page."""
    title, _, page = target.rpartition("@")
    start, end = doc.metadata.get("page_start"), doc.metadata.get("page_end")
    if start is None or doc.metadata.get("section", "").strip().lower() != title.strip().lower():
        return False
    return int(start) <= int(page) <= int(end if end is not None else start)


def score(gold_groups, docs) -> dict:
    hits = [any(matches(d, t) for d in docs for t in group) for group in gold_groups]
    return {"recall": sum(hits) / len(hits), "complete": all(hits), "hits": hits}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", required=True)
    parser.add_argument("--questions", default=None, help="Default: data/eval/<brand>_questions.json")
    parser.add_argument("--rebuild-index", action="store_true", help="Re-embed all passages into Chroma")
    parser.add_argument("--answers", type=int, default=0, help="Also generate answers for the first N multi-hop questions")
    args = parser.parse_args()

    settings = get_settings()
    key = slugify(args.brand)
    questions_path = Path(args.questions or PROJECT_ROOT / "data" / "eval" / f"{key}_questions.json")
    gold = json.loads(questions_path.read_text(encoding="utf-8"))
    kg = KnowledgeGraph.from_dict(json.loads((settings.KG_OUTPUT_DIR / f"{key}_kg.json").read_text(encoding="utf-8")))

    from src.graph.graph_rag import (
        GraphRAGRetriever,
        build_passage_index,
        format_docs_with_citations,
        make_vector_retriever,
    )
    from src.graph.neo4j_store import Neo4jStore

    print("Building / loading passage index ...")
    index = build_passage_index(kg, settings.CHROMA_DB_DIR, f"{key}_passages", rebuild=args.rebuild_index)
    print(f"  {index._collection.count()} passages in Chroma collection '{key}_passages'")

    store = Neo4jStore()
    arms = {
        "vector@4": make_vector_retriever(index, 4, args.brand),
        "vector@10": make_vector_retriever(index, 10, args.brand),
        "graph-add": GraphRAGRetriever(
            seed_retriever=make_vector_retriever(index, 4, args.brand), store=store, brand=args.brand, fusion=False
        ),
        "graph-rrf": GraphRAGRetriever(
            seed_retriever=make_vector_retriever(index, 20, args.brand), store=store, brand=args.brand, max_docs=10
        ),
    }

    results = []
    for q in gold["questions"]:
        row = {"id": q["id"], "question": q["question"], "multi_hop": q["multi_hop"], "style": q.get("style", ""), "gold": q["gold"], "arms": {}}
        for name, retriever in arms.items():
            docs = retriever.invoke(q["question"])
            s = score(q["gold"], docs)
            s["n_passages"] = sum(1 for d in docs if d.metadata.get("page_start") is not None)
            s["retrieved"] = [
                {"citation": d.metadata.get("citation"), "via": d.metadata.get("retrieved_via", "vector")} for d in docs
            ]
            row["arms"][name] = s
            row.setdefault("_docs", {})[name] = docs
        results.append(row)

    # ---------------------------------------------------------------- summary
    def agg(rows, arm):
        return (
            mean(r["arms"][arm]["recall"] for r in rows),
            sum(r["arms"][arm]["complete"] for r in rows),
            mean(r["arms"][arm]["n_passages"] for r in rows),
        )

    print(f"\n{'question':<6} {'style':<7} {'hop':<6} " + " ".join(f"{a:>10}" for a in arms))
    for r in results:
        cells = " ".join(f"{r['arms'][a]['recall']:>10.0%}" for a in arms)
        print(f"{r['id']:<6} {r['style']:<7} {'multi' if r['multi_hop'] else 'single':<6} {cells}")

    summary = {}
    for label, rows in (
        ("all", results),
        ("single-hop", [r for r in results if not r["multi_hop"]]),
        ("multi-hop", [r for r in results if r["multi_hop"]]),
        ("manual wording", [r for r in results if r["style"] == "manual"]),
        ("user wording", [r for r in results if r["style"] == "user"]),
    ):
        if not rows:
            continue
        summary[label] = {}
        print(f"\n{label} ({len(rows)} questions)")
        for arm in arms:
            recall, complete, n = agg(rows, arm)
            summary[label][arm] = {"recall": round(recall, 3), "complete": complete, "questions": len(rows), "avg_passages": round(n, 1)}
            print(f"  {arm:<10} recall {recall:6.1%}   complete {complete:>2}/{len(rows)}   avg passages {n:4.1f}")

    # ---------------------------------------------------------------- answers
    if args.answers:
        from langchain_core.prompts import PromptTemplate

        from src.agents.llm_factory import LLMFactory

        chain = PromptTemplate.from_template(ANSWER_PROMPT) | LLMFactory.get_primary_llm()
        for r in [r for r in results if r["multi_hop"]][: args.answers]:
            r["answers"] = {}
            for arm in ("vector@4", "vector@10", "graph-rrf"):
                out = chain.invoke({"context": format_docs_with_citations(r["_docs"][arm]), "question": r["question"]})
                r["answers"][arm] = getattr(out, "content", str(out))

    out_dir = PROJECT_ROOT / "data" / "eval" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}_retrieval_{date.today().isoformat()}.json"
    for r in results:
        r.pop("_docs", None)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nReport: {out_path}")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
