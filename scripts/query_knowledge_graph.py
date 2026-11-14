#!/usr/bin/env python
"""
Query the knowledge graph.

  SPARQL over the RDF export (no server needed):
    python scripts/query_knowledge_graph.py sparql ontology/queries/chapter_overview.rq --brand BMW

  GraphRAG search in Neo4j:
    python scripts/query_knowledge_graph.py search "tire pressure" --brand BMW

  Concept lookup across manuals (Neo4j):
    python scripts/query_knowledge_graph.py concept "tire"

  Read-only Cypher (Neo4j):
    python scripts/query_knowledge_graph.py cypher "MATCH (c:Chapter) RETURN c.title LIMIT 5"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_settings  # noqa: E402


def _print(rows) -> None:
    print(json.dumps(rows, ensure_ascii=False, indent=1, default=str))
    print(f"\n{len(rows)} row(s)")


def cmd_sparql(args) -> None:
    from src.graph import RDFStore

    settings = get_settings()
    store = RDFStore(ontology_path=settings.ONTOLOGY_PATH)
    ttl_files = sorted(settings.KG_OUTPUT_DIR.glob("*_kg.ttl"))
    if not ttl_files:
        raise SystemExit(f"No *_kg.ttl in {settings.KG_OUTPUT_DIR}; run build_knowledge_graph.py first")
    for path in ttl_files:
        store.load(path)
    query = Path(args.query).read_text(encoding="utf-8") if Path(args.query).exists() else args.query
    if args.brand:
        query = query.replace("$brand", args.brand)
    _print(store.sparql(query))


def cmd_neo4j(args) -> None:
    from src.graph.neo4j_store import Neo4jStore

    with Neo4jStore() as store:
        if args.command == "search":
            rows = store.search_passages(args.query, k=args.k, brand=args.brand)
            for row in rows:
                row["text"] = row["text"][:300] + ("..." if len(row["text"]) > 300 else "")
        elif args.command == "concept":
            rows = store.entity_context(args.query)
        else:
            rows = store.run_read_query(args.query)
    _print(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["sparql", "search", "concept", "cypher"])
    parser.add_argument("query", help="SPARQL file/text, search text, concept name or Cypher")
    parser.add_argument("--brand", default=None)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "sparql":
        cmd_sparql(args)
    else:
        cmd_neo4j(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
