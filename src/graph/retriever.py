"""
GraphRAG retriever and agent tools on top of Neo4j.

``GraphRetriever`` is a standard LangChain retriever, so it plugs into the
existing ``RAGChain(retriever=...)`` or an ``EnsembleRetriever`` with Chroma.
Each returned Document carries a citation (brand > chapter > section, page)
and the ontology entities mentioned in the passage.
"""

from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from pydantic import ConfigDict

from src.graph.neo4j_store import Neo4jStore


def _citation(row: dict) -> str:
    pages = f"p. {row['page_start']}" if row["page_start"] == row["page_end"] else (
        f"pp. {row['page_start']}-{row['page_end']}"
    )
    return f"{row['brand']} manual > {row['chapter']} > {row['section']}, {pages}"


class GraphRetriever(BaseRetriever):
    """Full-text + structure-aware retrieval from the knowledge graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    store: Neo4jStore
    k: int = 4
    brand: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        rows = self.store.search_passages(query, k=self.k, brand=self.brand)
        return [
            Document(
                page_content=row["text"],
                metadata={
                    "source": "knowledge_graph",
                    "passage_id": row["passage_id"],
                    "brand": row["brand"],
                    "chapter": row["chapter"],
                    "section": row["section"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "citation": _citation(row),
                    "entities": row["entities"],
                    "score": row["score"],
                },
            )
            for row in rows
        ]


def build_graph_tools(store: Neo4jStore) -> list:
    """LangChain tools for an agent: passage search and concept lookup."""

    @tool
    def search_manuals(query: str, brand: Optional[str] = None) -> str:
        """Search vehicle owner manuals. Returns passages with citations
        (brand, chapter, section, page) and the concepts they mention.
        Use `brand` (e.g. 'BMW', 'Volvo') to restrict to one manual."""
        rows = store.search_passages(query, k=4, brand=brand)
        return json.dumps(
            [
                {"citation": _citation(r), "text": r["text"][:1200], "entities": r["entities"]}
                for r in rows
            ],
            ensure_ascii=False,
        )

    @tool
    def explore_concept(name: str) -> str:
        """Look up a vehicle concept (component, feature, warning light, procedure...)
        in the knowledge graph: its relations to other concepts and every manual,
        section and page where it is mentioned. Good for cross-brand comparisons."""
        return json.dumps(store.entity_context(name), ensure_ascii=False, default=str)

    return [search_manuals, explore_concept]
