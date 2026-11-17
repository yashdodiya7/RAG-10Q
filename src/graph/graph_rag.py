"""
GraphRAG: vector search for entry points, knowledge graph for the rest.

    question ──► vector search (Chroma, metadata-filtered) ──► seed passages
                                                                 │
                       Neo4j expansion ◄─────────────────────────┘
                       ├─ passages sharing specific concepts (cross-chapter)
                       ├─ previous / next passage (text cut by chunking)
                       └─ concept relations as short facts

``passage_documents`` turns the knowledge graph's passages into LangChain
Documents with citation metadata, so the vector baseline and the graph use the
SAME chunks; a comparison then measures the graph, not a different chunking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, PrivateAttr

from src.graph.neo4j_store import Neo4jStore
from src.graph.schemas import KnowledgeGraph


def citation(meta: Dict[str, Any]) -> str:
    start, end = meta.get("page_start"), meta.get("page_end")
    pages = f"p. {start}" if start == end or end is None else f"pp. {start}-{end}"
    return f"{meta.get('brand')} manual > {meta.get('chapter')} > {meta.get('section')}, {pages}"


def passage_documents(kg: KnowledgeGraph) -> List[Document]:
    """One Document per Passage node, with brand/chapter/section/page metadata."""
    section_of = {r.target: r.source for r in kg.relations.values() if r.type == "HAS_PASSAGE"}
    docs = []
    for node in kg.nodes_by_label("Passage"):
        section = kg.nodes.get(section_of.get(node.id, ""))
        sp = section.properties if section else {}
        meta = {
            "passage_id": node.id,
            "brand": node.properties.get("brand", ""),
            "chapter": sp.get("chapter", ""),
            "section": sp.get("title", ""),
            "page_start": node.properties.get("pageStart"),
            "page_end": node.properties.get("pageEnd"),
        }
        meta["citation"] = citation(meta)
        docs.append(Document(page_content=node.properties.get("text", ""), metadata=meta))
    return docs


def build_passage_index(kg: KnowledgeGraph, persist_dir: Path, collection: str, rebuild: bool = False):
    """Chroma index of the graph's passages (MiniLM embeddings, persisted)."""
    from langchain_chroma import Chroma

    from src.rag.embeddings import EmbeddingsFactory

    embeddings = EmbeddingsFactory.get_default_embeddings()
    store = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=str(persist_dir))
    if rebuild:
        store.delete_collection()
        store = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=str(persist_dir))
    if store._collection.count() == 0:
        docs = passage_documents(kg)
        for start in range(0, len(docs), 256):
            batch = docs[start : start + 256]
            store.add_documents(batch, ids=[d.metadata["passage_id"] for d in batch])
    return store


def _row_to_document(row: Dict[str, Any], retrieved_via: str) -> Document:
    meta = {k: row[k] for k in ("passage_id", "brand", "chapter", "section", "page_start", "page_end")}
    meta["citation"] = citation(meta)
    meta["retrieved_via"] = retrieved_via
    return Document(page_content=row["text"], metadata=meta)


class GraphRAGRetriever(BaseRetriever):
    """Vector search + knowledge graph, combined with reciprocal rank fusion.

    Ranked lists that are fused (RRF, weight / (60 + rank)):
      vector      the seed retriever's ranking (use a larger k, e.g. 20, as a pool)
      concepts    question -> matching concept names/aliases -> passages that mention them
      expansion   passages sharing specific concepts with the top vector seeds
      neighbors   previous / next passage of the top seeds (same chapter)
    A passage found by several lists rises; a passage only the noisy expansion
    found rarely makes the cut. ``fusion=False`` gives the naive variant:
    top seeds + expansion appended (kept for comparison).

    Documents carry ``retrieved_via`` (which lists found them); concept facts are
    appended as one extra Document (``retrieved_via='graph facts'``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    seed_retriever: BaseRetriever
    store: Neo4jStore
    embeddings: Any = None          # for concept linking; default MiniLM
    brand: Optional[str] = None
    fusion: bool = True
    max_docs: int = 10
    seed_n: int = 4                 # top vector hits used as expansion seeds
    concept_k: int = 4
    neighbor_k: int = 2
    hub_max_mentions: int = 8       # concepts in more passages are too generic to follow
    link_k: int = 3                 # concepts linked from the question
    link_min_similarity: float = 0.5
    weights: Dict[str, float] = {"vector": 1.0, "concepts": 1.0, "expansion": 0.5, "neighbors": 0.5}
    include_facts: bool = True

    _catalog: Optional[List[Dict[str, Any]]] = PrivateAttr(default=None)
    _catalog_vectors: Any = PrivateAttr(default=None)

    # -- concept linking ------------------------------------------------------
    def _embedder(self):
        if self.embeddings is None:
            from src.rag.embeddings import EmbeddingsFactory

            self.embeddings = EmbeddingsFactory.get_default_embeddings()
        return self.embeddings

    def link_concepts(self, query: str) -> List[Dict[str, Any]]:
        """Concepts whose name or alias is semantically close to the question."""
        import numpy as np

        if self._catalog is None:
            self._catalog = [c for c in self.store.concept_catalog() if 0 < c["df"] <= self.hub_max_mentions]
            forms = [(i, form) for i, c in enumerate(self._catalog) for form in [c["name"], *c["aliases"]]]
            vectors = np.asarray(self._embedder().embed_documents([f for _, f in forms]), dtype=float)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
            self._catalog_vectors = (np.array([i for i, _ in forms]), vectors)
        if not self._catalog:
            return []
        owner, vectors = self._catalog_vectors
        q = np.asarray(self._embedder().embed_query(query), dtype=float)
        sims = vectors @ (q / (np.linalg.norm(q) + 1e-12))
        best: Dict[int, float] = {}
        for idx, sim in zip(owner, sims):
            best[int(idx)] = max(best.get(int(idx), -1.0), float(sim))
        ranked = sorted(best.items(), key=lambda x: -x[1])[: self.link_k]
        return [{**self._catalog[i], "similarity": s} for i, s in ranked if s >= self.link_min_similarity]

    def _concept_list(self, query: str) -> List[Document]:
        linked = self.link_concepts(query)
        if not linked:
            return []
        weight = {c["id"]: c["similarity"] / max(c["df"], 1) ** 0.5 for c in linked}
        scores: Dict[str, float] = {}
        rows: Dict[str, Dict[str, Any]] = {}
        via: Dict[str, List[str]] = {}
        for row in self.store.passages_for_concepts(list(weight), brand=self.brand):
            pid = row["passage_id"]
            scores[pid] = scores.get(pid, 0.0) + weight[row["concept_id"]]
            rows[pid] = row
            via.setdefault(pid, []).append(row["concept"])
        order = sorted(scores, key=lambda pid: -scores[pid])
        return [_row_to_document(rows[pid], "question concept: " + ", ".join(via[pid])) for pid in order]

    # -- retrieval ------------------------------------------------------------
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        pool = self.seed_retriever.invoke(query)
        for doc in pool:
            doc.metadata.setdefault("retrieved_via", "vector")
        seed_ids = [d.metadata["passage_id"] for d in pool[: self.seed_n] if d.metadata.get("passage_id")]
        expansion = self.store.expand_passages(
            seed_ids, concept_k=self.concept_k, neighbor_k=self.neighbor_k, hub_max_mentions=self.hub_max_mentions
        )
        expanded = [_row_to_document(r, "shared concept: " + ", ".join(r["via"])) for r in expansion["concepts"]]
        neighbors = [_row_to_document(r, r["via"][0]) for r in expansion["neighbors"]]

        if self.fusion:
            docs = self._fuse(
                {
                    "vector": pool,
                    "concepts": self._concept_list(query),
                    "expansion": expanded,
                    "neighbors": neighbors,
                }
            )
        else:
            docs = pool[: self.seed_n] + expanded + neighbors

        if self.include_facts and expansion["facts"]:
            lines = [f"- {f['source']} {f['relation'].lower().replace('_', ' ')} {f['target']}" for f in expansion["facts"]]
            docs.append(
                Document(
                    page_content="Known relations between concepts in these passages:\n" + "\n".join(lines),
                    metadata={"retrieved_via": "graph facts", "citation": "knowledge graph"},
                )
            )
        return docs

    def _fuse(self, lists: Dict[str, List[Document]], k: int = 60) -> List[Document]:
        scores: Dict[str, float] = {}
        docs: Dict[str, Document] = {}
        sources: Dict[str, List[str]] = {}
        for name, ranked in lists.items():
            for rank, doc in enumerate(ranked, 1):
                pid = doc.metadata.get("passage_id")
                if not pid:
                    continue
                scores[pid] = scores.get(pid, 0.0) + self.weights.get(name, 1.0) / (k + rank)
                docs.setdefault(pid, doc)
                sources.setdefault(pid, []).append(
                    f"{name}#{rank}" if name == "vector" else doc.metadata.get("retrieved_via", name)
                )
        top = sorted(scores, key=lambda pid: -scores[pid])[: self.max_docs]
        out = []
        for pid in top:
            doc = docs[pid]
            doc.metadata["retrieved_via"] = " + ".join(sources[pid])
            out.append(doc)
        return out


def format_docs_with_citations(docs: List[Document]) -> str:
    """Like src.data.format_docs, but keeps the source so the LLM can cite pages."""
    return "\n\n".join(f"[{i}] ({d.metadata.get('citation', 'unknown source')})\n{d.page_content}" for i, d in enumerate(docs, 1))


def make_graph_rag_retriever(
    store: Neo4jStore, brand: Optional[str] = None, settings=None, pool_k: int = 20, max_docs: int = 10
) -> "GraphRAGRetriever":
    """The best-performing setup from the benchmark (graph-rrf), ready to use.

    Uses the Chroma collection ``<brand>_passages``; builds it from
    ``<brand>_kg.json`` on first use. Without a brand, the BMW index is used
    (the only manual loaded so far) and no brand filter is applied.
    """
    import json

    from src.graph.schemas import slugify

    if settings is None:
        from src.config import get_settings

        settings = get_settings()
    key = slugify(brand or "BMW")
    kg_path = settings.KG_OUTPUT_DIR / f"{key}_kg.json"

    from langchain_chroma import Chroma

    from src.rag.embeddings import EmbeddingsFactory

    embeddings = EmbeddingsFactory.get_default_embeddings()
    index = Chroma(collection_name=f"{key}_passages", embedding_function=embeddings,
                   persist_directory=str(settings.CHROMA_DB_DIR))
    if index._collection.count() == 0:
        if not kg_path.exists():
            raise FileNotFoundError(f"{kg_path} not found - run scripts/build_knowledge_graph.py first")
        kg = KnowledgeGraph.from_dict(json.loads(kg_path.read_text(encoding="utf-8")))
        index = build_passage_index(kg, settings.CHROMA_DB_DIR, f"{key}_passages")
    return GraphRAGRetriever(
        seed_retriever=make_vector_retriever(index, pool_k, brand),
        store=store,
        embeddings=embeddings,
        brand=brand,
        max_docs=max_docs,
    )


def make_vector_retriever(store, k: int, brand: Optional[str] = None):
    kwargs: Dict[str, Any] = {"k": k}
    if brand:
        kwargs["filter"] = {"brand": brand}
    return store.as_retriever(search_kwargs=kwargs)
