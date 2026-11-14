#!/usr/bin/env python
"""
Build the automotive knowledge graph from MinerU structure output.

Pipeline:
  1. Structure layer   (rules, free)     Document -> Chapter -> Section -> Passage
  2. Semantic layer    (LLM, optional)   ontology-typed entities + relations, cached per passage
  3. Export            RDF/Turtle (always) and Neo4j (if --neo4j)

Examples:
  python scripts/build_knowledge_graph.py --brand BMW \
      --structure data/automotive_corpus/parsed/BMW/BMW/auto/BMW_structure_detected.json

  python scripts/build_knowledge_graph.py --brand BMW --structure <...> \
      --extract 20 --chapter "Wheels and tires" --neo4j
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402
from src.graph import KnowledgeGraph, RDFStore, StructureGraphBuilder  # noqa: E402
from src.graph.schemas import slugify  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--brand", required=True, help="Brand / document key, e.g. BMW")
    parser.add_argument("--structure", required=True, help="Path to <name>_structure_detected.json")
    parser.add_argument("--document-type", default="owner_manual")
    parser.add_argument("--extract", type=int, default=0, help="Number of passages to run LLM extraction on")
    parser.add_argument(
        "--chapter",
        action="append",
        default=None,
        help="Only extract passages from chapters containing this text (repeatable: --chapter A --chapter B)",
    )
    parser.add_argument("--neo4j", action="store_true", help="Also write the graph to Neo4j")
    parser.add_argument(
        "--reset-neo4j",
        action="store_true",
        help="With --neo4j: first delete this brand's existing nodes (removes stale entities from older runs)",
    )
    parser.add_argument("--resolve", action="store_true", help="Merge duplicate concepts after extraction")
    parser.add_argument(
        "--resolve-threshold",
        type=float,
        default=0.75,
        help="Embedding similarity above which an LLM judges whether two concepts are the same",
    )
    parser.add_argument("--output-dir", default=None, help="Default: settings.KG_OUTPUT_DIR")
    return parser


def run_extraction(kg: KnowledgeGraph, limit: int, chapters: list[str] | None, cache_path: Path) -> None:
    from src.graph.entity_extractor import EntityExtractor, select_passages

    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    passage_ids: list[str] = []
    for chapter in chapters or [None]:
        for pid in select_passages(kg, limit=limit, chapter_filter=chapter):
            if pid not in passage_ids and len(passage_ids) < limit:
                passage_ids.append(pid)
    print(f"Selected {len(passage_ids)} passages")

    def progress(passage_id: str, cached: bool) -> None:
        print(f"  {'cache' if cached else 'llm  '}  {passage_id}")
        if not cached:  # persist after every paid call
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")

    counts = EntityExtractor().enrich(kg, passage_ids, cache=cache, on_progress=progress)
    print(f"Extraction: {counts}")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    settings = get_settings()
    out_dir = Path(args.output_dir) if args.output_dir else settings.KG_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    key = slugify(args.brand)

    print("=== 1. Structure layer ===")
    kg = StructureGraphBuilder().build_from_file(
        Path(args.structure), brand=args.brand, document_type=args.document_type
    )
    print(json.dumps(kg.stats(), indent=1))

    if args.extract > 0:
        print("\n=== 2. Semantic layer (LLM extraction) ===")
        run_extraction(kg, args.extract, args.chapter, out_dir / f"{key}_extraction_cache.json")
        print(json.dumps(kg.stats(), indent=1))

    if args.resolve:
        from src.graph.entity_resolution import EntityResolver

        print("\n=== 2b. Entity resolution (merge duplicate concepts) ===")
        overrides_path = Path(__file__).resolve().parent.parent / "ontology" / "entity_overrides.json"
        resolver = EntityResolver(
            threshold=args.resolve_threshold,
            cache_path=out_dir / "entity_resolution_cache.json",  # brand-independent verdicts
            overrides=EntityResolver.load_overrides(overrides_path),
        )
        kg, report = resolver.resolve(kg)
        print(json.dumps(report.summary(), indent=1))
        merges_path = out_dir / f"{key}_entity_merges.json"
        merges_path.write_text(json.dumps(report.groups, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Merged groups ({len(report.groups)}) for review: {merges_path}")

    print("\n=== 3. Export ===")
    kg_json = out_dir / f"{key}_kg.json"
    kg_json.write_text(json.dumps(kg.to_dict(), ensure_ascii=False), encoding="utf-8")
    print(f"Graph JSON : {kg_json}")

    rdf = RDFStore()
    triples = rdf.add_knowledge_graph(kg)
    ttl_path = rdf.save(out_dir / f"{key}_kg.ttl")
    print(f"RDF/Turtle : {ttl_path} ({triples} triples)")

    if args.neo4j:
        from src.graph.neo4j_store import Neo4jStore

        with Neo4jStore() as store:
            if args.reset_neo4j:
                print(f"Neo4j reset: {store.clear_brand(key)}")
            written = store.write_graph(kg)
        print(f"Neo4j      : {written} -> {settings.NEO4J_URI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
