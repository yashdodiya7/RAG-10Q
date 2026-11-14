"""
Shared data model for the knowledge graph.

``GraphNode`` / ``GraphRelation`` / ``KnowledgeGraph`` are store-agnostic: the
same object is written to Neo4j and exported to RDF.
``ExtractionResult`` is the structured output schema the LLM must return.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Literal

from pydantic import BaseModel, Field, field_validator

from src.graph.ontology import validate_label, validate_relation


def slugify(value: str) -> str:
    """Stable, URL-safe id fragment (used for canonical entity ids)."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unnamed"


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _label_in_ontology(cls, value: str) -> str:
        return validate_label(value)


class GraphRelation(BaseModel):
    source: str
    target: str
    type: str
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _type_in_ontology(cls, value: str) -> str:
        return validate_relation(value)

    @property
    def key(self) -> tuple:
        return (self.source, self.type, self.target)


class KnowledgeGraph(BaseModel):
    """In-memory graph with id-based de-duplication."""

    nodes: Dict[str, GraphNode] = Field(default_factory=dict)
    relations: Dict[tuple, GraphRelation] = Field(default_factory=dict)

    def add_node(self, node: GraphNode) -> GraphNode:
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        # Merge: keep existing values, fill in missing ones.
        for key, value in node.properties.items():
            existing.properties.setdefault(key, value)
        return existing

    def add_relation(self, relation: GraphRelation) -> None:
        if relation.source not in self.nodes or relation.target not in self.nodes:
            raise ValueError(f"Relation endpoints must exist: {relation.key}")
        self.relations.setdefault(relation.key, relation)

    def merge(self, other: "KnowledgeGraph") -> None:
        for node in other.nodes.values():
            self.add_node(node)
        for relation in other.relations.values():
            self.add_relation(relation)

    def nodes_by_label(self, label: str) -> List[GraphNode]:
        return [n for n in self.nodes.values() if n.label == label]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "relations": [r.model_dump() for r in self.relations.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeGraph":
        graph = cls()
        for node in data.get("nodes", []):
            graph.add_node(GraphNode.model_validate(node))
        for rel in data.get("relations", []):
            graph.add_relation(GraphRelation.model_validate(rel))
        return graph

    def stats(self) -> Dict[str, Dict[str, int]]:
        node_counts: Dict[str, int] = {}
        for node in self.nodes.values():
            node_counts[node.label] = node_counts.get(node.label, 0) + 1
        rel_counts: Dict[str, int] = {}
        for rel in self.relations.values():
            rel_counts[rel.type] = rel_counts.get(rel.type, 0) + 1
        return {"nodes": node_counts, "relations": rel_counts}


# ---------------------------------------------------------------------------
# LLM structured-output schema (ontology-constrained)
# ---------------------------------------------------------------------------

EntityType = Literal[
    "VehicleSystem",
    "Component",
    "Feature",
    "WarningIndicator",
    "Procedure",
    "MaintenanceTask",
    "SafetyInstruction",
    "Specification",
]

RelationType = Literal["PART_OF", "INDICATES", "APPLIES_TO", "RELATED_TO"]


class ExtractedEntity(BaseModel):
    name: str = Field(description="Short canonical name in English, singular, e.g. 'tire pressure monitor'.")
    type: EntityType = Field(description="Ontology class of the entity.")
    description: str = Field(default="", description="One sentence grounded in the passage.")


class ExtractedRelation(BaseModel):
    source: str = Field(description="Name of the source entity (must match an extracted entity name).")
    target: str = Field(description="Name of the target entity (must match an extracted entity name).")
    type: RelationType


class ExtractionResult(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)


def entity_id(entity_type: str, name: str) -> str:
    """Canonical, brand-independent id so the same concept links across manuals."""
    return f"{entity_type}:{slugify(name)}"


def chunked(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
