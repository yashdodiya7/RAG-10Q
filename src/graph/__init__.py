"""Knowledge graph package: OWL ontology, Neo4j property graph, RDF/SPARQL export, GraphRAG."""

from .ontology import NAMESPACE, NODE_LABELS, RELATION_TYPES
from .schemas import (
    ExtractionResult,
    GraphNode,
    GraphRelation,
    KnowledgeGraph,
)
from .structure_builder import StructureGraphBuilder
from .rdf_store import RDFStore

__all__ = [
    "NAMESPACE",
    "NODE_LABELS",
    "RELATION_TYPES",
    "ExtractionResult",
    "GraphNode",
    "GraphRelation",
    "KnowledgeGraph",
    "StructureGraphBuilder",
    "RDFStore",
]

# Neo4j / LLM pieces are imported lazily by callers:
#   from src.graph.neo4j_store import Neo4jStore
#   from src.graph.entity_extractor import EntityExtractor
#   from src.graph.retriever import GraphRetriever, build_graph_tools
