"""
Neo4j persistence and graph queries (the store GraphRAG retrieval runs on).

All writes use MERGE on a shared ``:KGNode(id)`` key, so re-running ingestion
is idempotent. Labels and relationship types cannot be Cypher parameters, so
they are checked against the ontology before being placed in a query.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from neo4j import Driver, GraphDatabase

from src.config import get_settings
from src.graph.ontology import validate_label, validate_relation
from src.graph.schemas import KnowledgeGraph, chunked

FULLTEXT_INDEX = "kg_text"
BATCH_SIZE = 500

_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
_WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH)\b|CALL\s+(apoc|n10s|db\.create|dbms)",
    re.IGNORECASE,
)


def escape_lucene(text: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", text)


class Neo4jStore:
    """Thin wrapper around the Neo4j driver for the knowledge graph."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        driver: Optional[Driver] = None,
    ):
        settings = get_settings()
        self.database = database or settings.NEO4J_DATABASE
        self.driver = driver or GraphDatabase.driver(
            uri or settings.NEO4J_URI,
            auth=(user or settings.NEO4J_USER, password or settings.NEO4J_PASSWORD),
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ schema
    def ensure_schema(self) -> None:
        statements = [
            "CREATE CONSTRAINT kg_node_id IF NOT EXISTS FOR (n:KGNode) REQUIRE n.id IS UNIQUE",
            f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
            "FOR (n:Passage|Section|Chapter) ON EACH [n.text, n.title]",
            "CREATE INDEX kg_entity_name IF NOT EXISTS FOR (n:KGNode) ON (n.name)",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    # ------------------------------------------------------------------ writes
    def clear_brand(self, brand_key: str) -> Dict[str, int]:
        """Delete one brand's document nodes (ids "<brand>:...") and any concept
        nodes left without a MENTIONS link. write_graph only MERGEs, so without
        this, entities from older extraction runs stay in the database."""
        prefix = f"{brand_key}:"
        with self.driver.session(database=self.database) as session:
            docs = self._delete_prefix(session, prefix)
            orphans = session.execute_write(
                lambda tx: tx.run(
                    "MATCH (e:KGNode) WHERE NOT ()-[:MENTIONS]->(e) "
                    "AND NOT e:Document AND NOT e:Chapter AND NOT e:Section AND NOT e:Passage "
                    "DETACH DELETE e"
                ).consume().counters.nodes_deleted
            )
        return {"document_nodes": docs, "orphan_concepts": orphans}

    @staticmethod
    def _delete_prefix(session: Any, prefix: str) -> int:
        deleted = 0
        while True:  # batch to keep transactions small
            n = session.execute_write(
                lambda tx: tx.run(
                    "MATCH (n:KGNode) WHERE n.id STARTS WITH $p WITH n LIMIT 1000 DETACH DELETE n",
                    p=prefix,
                ).consume().counters.nodes_deleted
            )
            deleted += n
            if n == 0:
                return deleted

    def write_graph(self, kg: KnowledgeGraph) -> Dict[str, int]:
        self.ensure_schema()
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for node in kg.nodes.values():
            by_label.setdefault(validate_label(node.label), []).append(
                {"id": node.id, "props": {**node.properties, "id": node.id}}
            )
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for rel in kg.relations.values():
            by_type.setdefault(validate_relation(rel.type), []).append(
                {"source": rel.source, "target": rel.target, "props": rel.properties}
            )

        with self.driver.session(database=self.database) as session:
            for label, rows in by_label.items():
                query = (
                    "UNWIND $rows AS row "
                    f"MERGE (n:KGNode {{id: row.id}}) SET n:`{label}`, n += row.props"
                )
                for batch in chunked(rows, BATCH_SIZE):
                    session.execute_write(lambda tx, q=query, b=batch: tx.run(q, rows=b).consume())
            for rel_type, rows in by_type.items():
                query = (
                    "UNWIND $rows AS row "
                    "MATCH (a:KGNode {id: row.source}), (b:KGNode {id: row.target}) "
                    f"MERGE (a)-[r:`{rel_type}`]->(b) SET r += row.props"
                )
                for batch in chunked(rows, BATCH_SIZE):
                    session.execute_write(lambda tx, q=query, b=batch: tx.run(q, rows=b).consume())
        return {"nodes": len(kg.nodes), "relations": len(kg.relations)}

    # ------------------------------------------------------------------ reads
    def _read(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(lambda tx: [r.data() for r in tx.run(query, **params)])

    def search_passages(self, query: str, k: int = 5, brand: Optional[str] = None) -> List[Dict[str, Any]]:
        """Full-text search, then expand each hit with its section path and linked entities."""
        cypher = f"""
        CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $q) YIELD node, score
        // A hit on a Chapter/Section title expands to its passages, discounted by distance.
        MATCH path = (node)-[:HAS_SECTION|HAS_PASSAGE*0..2]->(p:Passage)
        WITH p, max(score / (1 + length(path))) AS score
        WHERE $brand IS NULL OR toLower(p.brand) = toLower($brand)
        ORDER BY score DESC
        LIMIT $k
        MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:HAS_PASSAGE]->(p)
        OPTIONAL MATCH (p)-[:MENTIONS]->(e)
        WITH p, d, c, s, score, collect(DISTINCT e) AS ents
        RETURN p.id AS passage_id, p.text AS text, p.pageStart AS page_start, p.pageEnd AS page_end,
               d.brand AS brand, c.title AS chapter, s.title AS section, score,
               [x IN ents | [l IN labels(x) WHERE l <> 'KGNode'][0] + ':' + x.name] AS entities
        ORDER BY score DESC
        LIMIT $k
        """
        return self._read(cypher, q=escape_lucene(query), k=k, brand=brand)

    _PASSAGE_FIELDS = """
        MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:HAS_PASSAGE]->(p)
        RETURN p.id AS passage_id, p.text AS text, p.pageStart AS page_start, p.pageEnd AS page_end,
               d.brand AS brand, c.title AS chapter, s.title AS section, score, via
        ORDER BY score DESC
    """

    def expand_passages(
        self,
        seed_ids: List[str],
        concept_k: int = 4,
        neighbor_k: int = 2,
        hub_max_mentions: int = 8,
        fact_limit: int = 15,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """GraphRAG expansion around passages found by another retriever.

        - ``concepts``: other passages sharing concepts with the seeds, scored by
          sum(1 / mentions) so rare, specific concepts count more. Concepts
          mentioned in more than ``hub_max_mentions`` passages (e.g. 'tire') are
          too generic to follow and are skipped.
        - ``neighbors``: passage directly before/after the seeds (text cut by chunking).
        - ``facts``: ontology relations between the seeds' (non-hub) concepts.
        """
        if not seed_ids:
            return {"concepts": [], "neighbors": [], "facts": []}
        concepts = self._read(
            """
            MATCH (seed:Passage)-[:MENTIONS]->(e) WHERE seed.id IN $ids
            WITH DISTINCT e, size([(x:Passage)-[:MENTIONS]->(e) | x]) AS df
            WHERE df <= $hub_max
            MATCH (p:Passage)-[:MENTIONS]->(e) WHERE NOT p.id IN $ids
            WITH p, sum(1.0 / df) AS score, collect(DISTINCT e.name) AS via
            ORDER BY score DESC LIMIT $k
            """ + self._PASSAGE_FIELDS,
            ids=seed_ids, hub_max=hub_max_mentions, k=concept_k,
        )
        taken = set(seed_ids) | {row["passage_id"] for row in concepts}
        neighbors = self._read(
            """
            UNWIND range(0, size($ids) - 1) AS rank
            MATCH (seed:Passage {id: $ids[rank]})-[r:NEXT_PASSAGE]-(p:Passage)
            WHERE NOT p.id IN $taken
            // stay inside the chapter: the passage after a chapter's last one is unrelated
            MATCH (c:Chapter)-[:HAS_SECTION]->(:Section)-[:HAS_PASSAGE]->(seed),
                  (c)-[:HAS_SECTION]->(:Section)-[:HAS_PASSAGE]->(p)
            WITH p, min(rank) AS rank,
                 head(collect(CASE WHEN startNode(r) = p THEN 'previous passage' ELSE 'next passage' END)) AS via
            WITH p, 1.0 / (1 + rank) AS score, [via] AS via
            ORDER BY score DESC LIMIT $k
            """ + self._PASSAGE_FIELDS,
            ids=seed_ids, taken=list(taken), k=neighbor_k,
        )
        facts = self._read(
            """
            MATCH (seed:Passage)-[:MENTIONS]->(e) WHERE seed.id IN $ids
            WITH collect(DISTINCT e) AS es
            UNWIND es AS a
            MATCH (a)-[r:PART_OF|INDICATES|APPLIES_TO|RELATED_TO]->(b) WHERE b IN es
            WITH a, r, b, size([(x:Passage)-[:MENTIONS]->(a) | x]) + size([(x:Passage)-[:MENTIONS]->(b) | x]) AS df
            WHERE df <= 2 * $hub_max
            RETURN DISTINCT a.name AS source, type(r) AS relation, b.name AS target
            LIMIT $limit
            """,
            ids=seed_ids, hub_max=hub_max_mentions, limit=fact_limit,
        )
        return {"concepts": concepts, "neighbors": neighbors, "facts": facts}

    def concept_catalog(self) -> List[Dict[str, Any]]:
        """Every domain concept with its aliases and how many passages mention it."""
        return self._read(
            """
            MATCH (e:KGNode) WHERE e.name IS NOT NULL AND NOT e:Document AND NOT e:Chapter
                  AND NOT e:Section AND NOT e:Passage
            RETURN e.id AS id, e.name AS name, coalesce(e.aliases, []) AS aliases,
                   size([(p:Passage)-[:MENTIONS]->(e) | p]) AS df
            """
        )

    def passages_for_concepts(self, concept_ids: List[str], brand: Optional[str] = None) -> List[Dict[str, Any]]:
        """Passages mentioning any of the concepts (one row per passage/concept pair)."""
        return self._read(
            """
            MATCH (p:Passage)-[:MENTIONS]->(e:KGNode) WHERE e.id IN $ids
                  AND ($brand IS NULL OR toLower(p.brand) = toLower($brand))
            MATCH (d:Document)-[:HAS_CHAPTER]->(c:Chapter)-[:HAS_SECTION]->(s:Section)-[:HAS_PASSAGE]->(p)
            RETURN e.id AS concept_id, e.name AS concept, p.id AS passage_id, p.text AS text,
                   p.pageStart AS page_start, p.pageEnd AS page_end,
                   d.brand AS brand, c.title AS chapter, s.title AS section
            """,
            ids=concept_ids, brand=brand,
        )

    def entity_context(self, name: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Relations of an entity plus where (brand, section, page) it is mentioned."""
        cypher = """
        MATCH (e:KGNode) WHERE e.name IS NOT NULL AND (
            toLower(e.name) CONTAINS toLower($name)
            OR any(a IN coalesce(e.aliases, []) WHERE toLower(a) CONTAINS toLower($name)))
        OPTIONAL MATCH (e)-[r]-(other:KGNode) WHERE other.name IS NOT NULL
        WITH e, collect(DISTINCT {relation: type(r),
                                  direction: CASE WHEN startNode(r) = e THEN 'out' ELSE 'in' END,
                                  other: other.name}) AS relations
        OPTIONAL MATCH (s:Section)-[:HAS_PASSAGE]->(p:Passage)-[:MENTIONS]->(e)
        WITH e, relations, collect(DISTINCT {brand: p.brand, section: s.title, page: p.pageStart}) AS mentions
        RETURN e.name AS entity, [l IN labels(e) WHERE l <> 'KGNode'][0] AS type,
               [x IN relations WHERE x.other IS NOT NULL] AS relations,
               [m IN mentions WHERE m.brand IS NOT NULL] AS mentioned_in
        LIMIT $limit
        """
        return self._read(cypher, name=name, limit=limit)

    def run_read_query(self, cypher: str, **params: Any) -> List[Dict[str, Any]]:
        """Run user/LLM-supplied Cypher in a READ transaction, with a write-clause guard."""
        if _WRITE_CLAUSES.search(cypher):
            raise ValueError("Only read-only Cypher is allowed")
        return self._read(cypher, **params)

    def stats(self) -> List[Dict[str, Any]]:
        return self._read(
            "MATCH (n:KGNode) RETURN [l IN labels(n) WHERE l <> 'KGNode'][0] AS label, count(*) AS count "
            "ORDER BY count DESC"
        )
