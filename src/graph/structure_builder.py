"""
Build the document-structure layer of the knowledge graph from MinerU output.

Input:  ``<name>_structure_detected.json`` (block index -> {type, level, page_idx, text})
Output: ``KnowledgeGraph`` with Document -> Chapter -> Section -> Passage.

No LLM is needed for this layer: hierarchy comes from the detected heading
levels, with filters for the noise MinerU produces (page numbers, callout
digits, button labels, numbered steps).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.graph.schemas import GraphNode, GraphRelation, KnowledgeGraph, slugify

SKIP_BLOCK_TYPES = {"image", "footer", "header", "page_number"}
MAX_HEADING_CHARS = 120
DEFAULT_MAX_PASSAGE_CHARS = 1500

_SPACES = re.compile(r"[     ]+")
_SOFT_HYPHEN_BREAK = re.compile(r"(\w)[‐­]\s+(\w)")  # "Con‐ trol" -> "Control"
_STARTS_WITH_DIGIT = re.compile(r"^\s*\d")
_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = _SPACES.sub(" ", text or "")
    text = _SOFT_HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace("\\*", "").replace("\\$", "$")
    return _WHITESPACE.sub(" ", text).strip()


def is_heading_text(text: str) -> bool:
    """Reject callout numbers ('10'), button codes ('AUTOH') and steps ('1. Turn.')."""
    return (
        4 <= len(text) <= MAX_HEADING_CHARS
        and re.search(r"[a-z]", text) is not None
        and _STARTS_WITH_DIGIT.match(text) is None
    )


@dataclass
class _PassageBuffer:
    parts: List[str] = field(default_factory=list)
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    size: int = 0

    def add(self, text: str, page: int) -> None:
        self.parts.append(text)
        self.size += len(text) + 1
        self.page_start = page if self.page_start is None else self.page_start
        self.page_end = page

    def is_empty(self) -> bool:
        return not self.parts


class StructureGraphBuilder:
    """Convert MinerU structure metadata into a structural knowledge graph."""

    def __init__(self, max_passage_chars: int = DEFAULT_MAX_PASSAGE_CHARS):
        self.max_passage_chars = max_passage_chars

    def build_from_file(
        self,
        structure_path: Path,
        brand: str,
        document_type: str = "owner_manual",
        title: Optional[str] = None,
    ) -> KnowledgeGraph:
        with open(structure_path, "r", encoding="utf-8") as handle:
            blocks: Dict[str, dict] = json.load(handle)
        return self.build(
            blocks,
            brand=brand,
            document_type=document_type,
            title=title or f"{brand} {document_type.replace('_', ' ')}",
            source_file=Path(structure_path).name,
        )

    def build(
        self,
        blocks: Dict[str, dict],
        brand: str,
        document_type: str,
        title: str,
        source_file: str,
    ) -> KnowledgeGraph:
        graph = KnowledgeGraph()
        brand_key = slugify(brand)
        doc_id = f"{brand_key}:doc"
        graph.add_node(
            GraphNode(
                id=doc_id,
                label="Document",
                properties={
                    "title": title,
                    "brand": brand,
                    "documentType": document_type,
                    "sourceFile": source_file,
                },
            )
        )

        state = {
            "chapter_id": None,
            "chapter_title": None,
            "section_id": None,
            "chapter_n": 0,
            "section_n": 0,
            "passage_n": 0,
            "prev_passage_id": None,
        }
        buffer = _PassageBuffer()

        def flush() -> None:
            nonlocal buffer
            if buffer.is_empty() or state["section_id"] is None:
                buffer = _PassageBuffer()
                return
            state["passage_n"] += 1
            passage_id = f"{brand_key}:p:{state['passage_n']}"
            graph.add_node(
                GraphNode(
                    id=passage_id,
                    label="Passage",
                    properties={
                        "text": " ".join(buffer.parts),
                        "brand": brand,
                        "pageStart": buffer.page_start + 1,  # 1-based like the printed manual
                        "pageEnd": buffer.page_end + 1,
                        "order": state["passage_n"],
                    },
                )
            )
            graph.add_relation(GraphRelation(source=state["section_id"], target=passage_id, type="HAS_PASSAGE"))
            if state["prev_passage_id"]:
                graph.add_relation(
                    GraphRelation(source=state["prev_passage_id"], target=passage_id, type="NEXT_PASSAGE")
                )
            state["prev_passage_id"] = passage_id
            buffer = _PassageBuffer()

        def open_chapter(chapter_title: str, page: int) -> None:
            flush()
            state["chapter_n"] += 1
            chapter_id = f"{brand_key}:ch:{state['chapter_n']}"
            graph.add_node(
                GraphNode(
                    id=chapter_id,
                    label="Chapter",
                    properties={
                        "title": chapter_title,
                        "brand": brand,
                        "pageStart": page + 1,
                        "order": state["chapter_n"],
                    },
                )
            )
            graph.add_relation(GraphRelation(source=doc_id, target=chapter_id, type="HAS_CHAPTER"))
            state.update(chapter_id=chapter_id, chapter_title=chapter_title, section_id=None)

        def open_section(section_title: str, page: int) -> None:
            flush()
            if state["chapter_id"] is None:
                open_chapter("Front matter", page)
            state["section_n"] += 1
            section_id = f"{brand_key}:sec:{state['section_n']}"
            graph.add_node(
                GraphNode(
                    id=section_id,
                    label="Section",
                    properties={
                        "title": section_title,
                        "brand": brand,
                        "chapter": state["chapter_title"],
                        "pageStart": page + 1,
                        "order": state["section_n"],
                    },
                )
            )
            graph.add_relation(
                GraphRelation(source=state["chapter_id"], target=section_id, type="HAS_SECTION")
            )
            state["section_id"] = section_id

        for _, block in sorted(blocks.items(), key=lambda item: int(item[0])):
            block_type = block.get("type", "text")
            if block_type in SKIP_BLOCK_TYPES:
                continue
            text = clean_text(block.get("text", ""))
            if not text:
                continue
            level = int(block.get("level", 4))
            page = int(block.get("page_idx", 0))

            if block_type == "text" and level == 1 and is_heading_text(text):
                open_chapter(text, page)
                continue
            if block_type == "text" and level == 2 and is_heading_text(text):
                open_section(text, page)
                continue
            if len(text) < 3:
                continue  # stray callout markers

            if state["section_id"] is None:
                open_section(state["chapter_title"] or "Overview", page)
            if buffer.size + len(text) > self.max_passage_chars and not buffer.is_empty():
                flush()
            buffer.add(text, page)

        flush()
        return graph
