"""
Python view of the OWL ontology in ``ontology/automotive.ttl``.

The TTL file is the formal schema; this module is the single place the code
reads it from. ``tests/test_graph.py`` checks that both stay in sync.

Naming convention:
- Neo4j node label  == OWL class local name      (``Component`` <-> ``auto:Component``)
- Neo4j rel type    == UPPER_SNAKE of property   (``PART_OF``   <-> ``auto:partOf``)
"""

from typing import Dict, FrozenSet

NAMESPACE = "https://w3id.org/rag10q/auto#"
DATA_NAMESPACE = "https://w3id.org/rag10q/data/"

STRUCTURE_LABELS: FrozenSet[str] = frozenset({"Document", "Chapter", "Section", "Passage"})

DOMAIN_LABELS: FrozenSet[str] = frozenset(
    {
        "VehicleSystem",
        "Component",
        "Feature",
        "WarningIndicator",
        "Procedure",
        "MaintenanceTask",
        "SafetyInstruction",
        "Specification",
    }
)

NODE_LABELS: FrozenSet[str] = STRUCTURE_LABELS | DOMAIN_LABELS

# Neo4j relationship type -> OWL object property local name
RELATION_TO_PROPERTY: Dict[str, str] = {
    "HAS_CHAPTER": "hasChapter",
    "HAS_SECTION": "hasSection",
    "HAS_PASSAGE": "hasPassage",
    "NEXT_PASSAGE": "nextPassage",
    "MENTIONS": "mentions",
    "PART_OF": "partOf",
    "INDICATES": "indicates",
    "APPLIES_TO": "appliesTo",
    "RELATED_TO": "relatedTo",
}

RELATION_TYPES: FrozenSet[str] = frozenset(RELATION_TO_PROPERTY)

# Relations an LLM is allowed to produce between domain concepts.
DOMAIN_RELATION_TYPES: FrozenSet[str] = frozenset(
    {"PART_OF", "INDICATES", "APPLIES_TO", "RELATED_TO"}
)

# Node/relationship properties exported as OWL datatype properties.
DATATYPE_PROPERTIES: FrozenSet[str] = frozenset(
    {
        "name",
        "title",
        "text",
        "description",
        "brand",
        "sourceFile",
        "documentType",
        "pageStart",
        "pageEnd",
        "order",
    }
)


def validate_label(label: str) -> str:
    """Return the label if it is part of the ontology, else raise."""
    if label not in NODE_LABELS:
        raise ValueError(f"Label '{label}' is not defined in the ontology")
    return label


def validate_relation(rel_type: str) -> str:
    """Return the relationship type if it is part of the ontology, else raise."""
    if rel_type not in RELATION_TYPES:
        raise ValueError(f"Relation '{rel_type}' is not defined in the ontology")
    return rel_type
