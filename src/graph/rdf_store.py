"""
RDF/OWL export and SPARQL querying of the knowledge graph (rdflib, in-process).

The same ``KnowledgeGraph`` that goes to Neo4j is serialised as RDF triples
using the OWL ontology's classes and properties, so it can be queried with
SPARQL, validated against the ontology, or loaded into any triple store
(GraphDB, Fuseki, or Neo4j via the n10s plugin).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

from rdflib import RDF, RDFS, Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, SKOS, XSD

from src.graph.ontology import DATA_NAMESPACE, DATATYPE_PROPERTIES, NAMESPACE, RELATION_TO_PROPERTY
from src.graph.schemas import KnowledgeGraph

AUTO = Namespace(NAMESPACE)
DATA = Namespace(DATA_NAMESPACE)

_INTEGER_PROPERTIES = {"pageStart", "pageEnd", "order"}


def node_uri(node_id: str) -> URIRef:
    return DATA[quote(node_id, safe="")]


class RDFStore:
    """Hold the ontology plus instance data in one rdflib graph."""

    def __init__(self, ontology_path: Optional[Path] = None):
        self.graph = Graph()
        self.graph.bind("auto", AUTO)
        self.graph.bind("data", DATA)
        self.graph.bind("owl", OWL)
        if ontology_path is not None:
            self.graph.parse(str(ontology_path), format="turtle")

    def add_knowledge_graph(self, kg: KnowledgeGraph) -> int:
        """Convert nodes/relations to triples. Returns the number of triples added."""
        before = len(self.graph)
        for node in kg.nodes.values():
            subject = node_uri(node.id)
            self.graph.add((subject, RDF.type, AUTO[node.label]))
            label = node.properties.get("title") or node.properties.get("name")
            if label:
                self.graph.add((subject, RDFS.label, Literal(label)))
            for alias in node.properties.get("aliases") or []:  # merged duplicate names
                self.graph.add((subject, SKOS.altLabel, Literal(alias)))
            for key, value in node.properties.items():
                if key not in DATATYPE_PROPERTIES or value in (None, ""):
                    continue
                datatype = XSD.integer if key in _INTEGER_PROPERTIES else None
                self.graph.add((subject, AUTO[key], Literal(value, datatype=datatype)))

        for rel in kg.relations.values():
            predicate = AUTO[RELATION_TO_PROPERTY[rel.type]]
            self.graph.add((node_uri(rel.source), predicate, node_uri(rel.target)))
        return len(self.graph) - before

    def load(self, path: Path) -> None:
        self.graph.parse(str(path), format="turtle")

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.graph.serialize(destination=str(path), format="turtle", encoding="utf-8")
        return path

    def sparql(self, query: str) -> List[Dict[str, str]]:
        """Run a SPARQL SELECT and return rows as plain dicts."""
        result = self.graph.query(query, initNs={"auto": AUTO, "rdfs": RDFS, "owl": OWL})
        variables = [str(v) for v in result.vars]
        rows = []
        for row in result:
            rows.append({var: (str(row[var]) if row[var] is not None else None) for var in variables})
        return rows
