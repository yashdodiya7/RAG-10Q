#!/usr/bin/env python
"""
Ask the knowledge-graph agent a question about a vehicle manual.

    python scripts/ask_agent.py "My tyre light came on, what now?" --brand BMW
    python scripts/ask_agent.py "I got new tires fitted, anything to do?" --brand BMW --trace

Needs Neo4j running (docker compose up -d neo4j), a built graph
(scripts/build_knowledge_graph.py ... --neo4j) and GEMINI_API_KEY in .env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--brand", default=None, help="Restrict searches to one manual, e.g. BMW")
    parser.add_argument("--trace", action="store_true", help="Show the agent's tool calls")
    args = parser.parse_args()

    from src.agents.graph_agent import GraphAgent

    agent = GraphAgent.from_settings(brand=args.brand)
    try:
        result = agent.ask(args.question)
    finally:
        agent.store.close()

    if args.trace:
        print("=== Tool calls ===")
        print(result.pretty_trace())
        print()
    print("=== Answer ===")
    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
