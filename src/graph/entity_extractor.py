"""
LLM-based entity/relation extraction into the ontology (the semantic layer).

Each passage is sent to a chat model with ``with_structured_output`` bound to
``ExtractionResult``, so the model can only return ontology classes and
relation types. Entities get canonical, brand-independent ids
(``Component:tire-pressure-monitor``) so the same concept found in the BMW and
Volvo manuals becomes ONE node with two ``MENTIONS`` edges.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Callable, Dict, Iterable, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from src.graph.schemas import (
    ExtractionResult,
    GraphNode,
    GraphRelation,
    KnowledgeGraph,
    entity_id,
)

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You build a knowledge graph from vehicle owner manuals.
Extract only concepts that are explicitly stated in the passage.

Entity types (use exactly these):
- VehicleSystem: functional system (brakes, transmission, climate control, drivetrain)
- Component: physical part (tire, battery, brake pad, wiper blade)
- Feature: driver-facing function (Auto Hold, cruise control, Park Distance Control)
- WarningIndicator: warning lamp, symbol or check-control message
- Procedure: steps the driver performs (resetting tire pressure monitor)
- MaintenanceTask: servicing work (oil change, replacing wiper blades)
- SafetyInstruction: a warning or safety rule
- Specification: a technical value (tire pressure 2.4 bar, fuel tank 67 l)

Relation types (use exactly these):
- PART_OF: Component -> VehicleSystem, or Component -> Component
- INDICATES: WarningIndicator -> the system/component/condition it signals
- APPLIES_TO: Procedure/MaintenanceTask/SafetyInstruction/Specification -> what it concerns
- RELATED_TO: any other clear association

Rules:
- Use short, generic English names, singular, lowercase, no brand names
  (e.g. 'tire pressure monitor', not 'BMW RDC').
- Relation source/target must exactly match names in your entity list.
- Return empty lists if the passage has no domain content (e.g. table of contents)."""

EXTRACTION_HUMAN_PROMPT = """Brand: {brand}
Chapter: {chapter}
Section: {section}

Passage:
{text}"""


class EntityExtractor:
    """Extract ontology-typed entities and relations from passages."""

    def __init__(self, llm: Optional[BaseChatModel] = None):
        if llm is None:
            from src.agents.llm_factory import LLMFactory

            llm = LLMFactory.get_primary_llm()
        if not hasattr(llm, "with_structured_output"):
            raise TypeError(
                f"{type(llm).__name__} does not support structured output; use a chat model."
            )
        prompt = ChatPromptTemplate.from_messages(
            [("system", EXTRACTION_SYSTEM_PROMPT), ("human", EXTRACTION_HUMAN_PROMPT)]
        )
        self.chain = prompt | llm.with_structured_output(ExtractionResult)

    def extract(self, text: str, brand: str, chapter: str = "", section: str = "") -> ExtractionResult:
        result = self.chain.invoke(
            {"text": text, "brand": brand, "chapter": chapter, "section": section}
        )
        return result if isinstance(result, ExtractionResult) else ExtractionResult.model_validate(result)

    def enrich(
        self,
        graph: KnowledgeGraph,
        passage_ids: Iterable[str],
        cache: Optional[Dict[str, dict]] = None,
        on_progress: Optional[Callable[[str, bool], None]] = None,
    ) -> Dict[str, int]:
        """Extract entities for the given passages and add them to ``graph``.

        ``cache`` maps a hash of the passage text to a stored ExtractionResult, so
        re-running the pipeline never pays for the same passage twice.
        """
        section_of = {
            rel.target: rel.source for rel in graph.relations.values() if rel.type == "HAS_PASSAGE"
        }
        counts = {"llm_calls": 0, "cache_hits": 0, "failures": 0}

        for passage_id in passage_ids:
            passage = graph.nodes[passage_id]
            text = passage.properties["text"]
            key = text_hash(text)
            if cache is not None and key in cache:
                result = ExtractionResult.model_validate(cache[key])
                counts["cache_hits"] += 1
                cached = True
            else:
                section = graph.nodes.get(section_of.get(passage_id, ""))
                try:
                    result = self.extract(
                        text=text,
                        brand=passage.properties.get("brand", ""),
                        chapter=(section.properties.get("chapter", "") if section else ""),
                        section=(section.properties.get("title", "") if section else ""),
                    )
                except Exception as exc:  # one bad passage must not stop a batch
                    logger.warning("Extraction failed for %s: %s", passage_id, exc)
                    counts["failures"] += 1
                    continue
                counts["llm_calls"] += 1
                cached = False
                if cache is not None:
                    cache[key] = result.model_dump()
            add_extraction(graph, passage_id, result)
            if on_progress:
                on_progress(passage_id, cached)
        return counts



def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def add_extraction(graph: KnowledgeGraph, passage_id: str, result: ExtractionResult) -> None:
    """Add extracted entities (canonical ids) + MENTIONS provenance + relations."""
    name_to_id = {}
    for entity in result.entities:
        name = entity.name.strip().lower()
        if not name:
            continue
        node_id = entity_id(entity.type, name)
        graph.add_node(
            GraphNode(
                id=node_id,
                label=entity.type,
                properties={"name": name, "description": entity.description},
            )
        )
        graph.add_relation(GraphRelation(source=passage_id, target=node_id, type="MENTIONS"))
        name_to_id[name] = node_id

    for relation in result.relations:
        source = name_to_id.get(relation.source.strip().lower())
        target = name_to_id.get(relation.target.strip().lower())
        if source and target and source != target:
            graph.add_relation(GraphRelation(source=source, target=target, type=relation.type))


def select_passages(
    graph: KnowledgeGraph,
    limit: int,
    chapter_filter: Optional[str] = None,
    min_chars: int = 200,
) -> List[str]:
    """Pick passages worth extracting (skips tiny ones, optional chapter filter)."""
    section_of = {rel.target: rel.source for rel in graph.relations.values() if rel.type == "HAS_PASSAGE"}
    chosen: List[str] = []
    for node in sorted(graph.nodes_by_label("Passage"), key=lambda n: n.properties.get("order", 0)):
        if len(node.properties.get("text", "")) < min_chars:
            continue
        if chapter_filter:
            section = graph.nodes.get(section_of.get(node.id, ""))
            chapter = (section.properties.get("chapter", "") if section else "").lower()
            if chapter_filter.lower() not in chapter:
                continue
        chosen.append(node.id)
        if len(chosen) >= limit:
            break
    return chosen
