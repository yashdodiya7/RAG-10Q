"""
MinerU-backed document parser.
Uses MinerU CLI to convert PDFs to Markdown/JSON before chunking.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from langchain_core.documents import Document
except ModuleNotFoundError:  # pragma: no cover - metadata-only flows do not need langchain
    Document = Any

from src.config import get_settings


VIN_PATTERN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
MAKE_MODEL_PATTERN = re.compile(
    r"\b(Toyota|Honda|Ford|BMW|Mercedes|Tesla)\s+([A-Za-z0-9\-]+)",
    re.IGNORECASE,
)
DATE_PATTERN = re.compile(
    r"\b(\d{1,2})\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{4})\b",
    re.IGNORECASE,
)
MILEAGE_PATTERN = re.compile(r"(?:mileage)[:\s]*(\d{1,6}(?:,\d{3})*)", re.IGNORECASE)


SYSTEM_KEYWORDS: Dict[str, List[str]] = {
    "transmission": ["shift", "slipping", "jerky", "hard shift", "won't shift"],
    "engine": ["misfire", "knocking", "overheating", "rough idle", "stalling"],
    "brakes": ["soft pedal", "pulling", "grinding", "pulsing"],
    "electrical": ["battery", "alternator", "voltage", "charging"],
    "safety": ["airbag", "seat belt", "warning", "caution", "attention"],
}

IMPORTANT_SECTION_HINTS = {
    "safety",
    "warning",
    "driving",
    "maintenance",
    "engine",
    "wheels",
    "tires",
    "brakes",
    "breakdown",
    "refueling",
    "fuel",
    "emergency",
}


def _month_to_number(month_name: str) -> int:
    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    return months.get(month_name.lower(), 1)


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()

    
class MinerUParser:
    """Parse PDFs with MinerU and convert outputs into LangChain Documents."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def parse_pdf(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        backend: Optional[str] = None,
    ) -> List[Any]:
        """
        Parse a PDF or directory of PDFs using MinerU CLI.

        MinerU supports Markdown/JSON output and CPU fallback via `pipeline`.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"MinerU input not found: {input_path}")

        output_dir = output_dir or self.settings.AUTOMOTIVE_PARSED_DIR / input_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        self._run_mineru(input_path=input_path, output_dir=output_dir, backend=backend)
        return self._load_parsed_documents(output_dir, source_path=input_path)

    def step1_extract_text_blocks(
        self,
        pdf_path: str,
        output_json_path: Optional[str] = None,
        backend: Optional[str] = None,
    ) -> List[dict]:
        """
        Step 1: Extract text blocks from a PDF using MinerU.

        This mirrors the original pdfplumber-based step, but uses MinerU to
        parse the PDF first and then converts the parsed output into a JSON list
        of text blocks.
        """
        input_path = Path(pdf_path)
        if not input_path.exists():
            raise FileNotFoundError(f"MinerU input not found: {input_path}")

        output_dir = self.settings.AUTOMOTIVE_PARSED_DIR / input_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        self._run_mineru(input_path=input_path, output_dir=output_dir, backend=backend)

        blocks = self._load_text_blocks(output_dir=output_dir, source_path=input_path)

        resolved_output_path = (
            Path(output_json_path)
            if output_json_path
            else output_dir / f"{input_path.stem}_content_list.json"
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            json.dumps(blocks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return blocks

    @staticmethod
    def detect_heading_level(text: str) -> int:
        """
        Detect heading level from plain text.

        Levels:

        1: Most important title
        2: Major section heading
        3: Subsection heading
        4: Body text / regular content
        """
        normalized = (text or "").strip()
        if not normalized:
            return 4

        # Rule 1: short all-caps lines are likely top-level headings.
        if len(normalized) < 100 and normalized.upper() == normalized:
            return 1

        # Rule 2: keyword-based section starts.
        section_keywords = ("vehicle", "service", "symptoms", "parts")
        lower_text = normalized.lower()
        if any(lower_text.startswith(keyword) for keyword in section_keywords):
            return 2

        # Rule 3: numbered heading format, e.g., "1. Item".
        if normalized[0].isdigit() and "." in normalized[:5]:
            return 3

        return 4

    def build_structure_metadata(self, blocks: List[dict]) -> dict:
        """
        Build structure metadata from extracted content blocks.

        The output uses block index as key, which avoids collisions when
        multiple entries share the same page_idx.
        """
        structure: dict = {}

        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue

            text = str(block.get("text", ""))
            inferred_level = self.detect_heading_level(text)

            # Respect parser-provided text_level when available.
            existing_level = block.get("text_level")
            if isinstance(existing_level, int) and 1 <= existing_level <= 4:
                level = min(inferred_level, existing_level)
            else:
                level = inferred_level

            structure[str(idx)] = {
                "type": block.get("type", "unknown"),
                "level": level,
                "page_idx": block.get("page_idx"),
                "text": text,
            }

        return structure

    def step2_generate_structure_metadata(
        self,
        content_json_path: str,
        output_json_path: Optional[str] = None,
    ) -> dict:
        """
        Step 2: Generate structure metadata from a content list JSON.

        Args:
            content_json_path: Path to a *_content_list.json file.
            output_json_path: Optional explicit output path.

        Returns:
            A dictionary keyed by block index with heading level metadata.
        """
        content_path = Path(content_json_path)
        if not content_path.exists():
            raise FileNotFoundError(f"Content list not found: {content_path}")

        with content_path.open("r", encoding="utf-8") as file_obj:
            blocks = json.load(file_obj)

        if not isinstance(blocks, list):
            raise ValueError(
                f"Expected a JSON list in {content_path}, got: {type(blocks).__name__}"
            )

        structure = self.build_structure_metadata(blocks=blocks)

        resolved_output_path = (
            Path(output_json_path)
            if output_json_path
            else content_path.with_name(
                content_path.name.replace("_content_list.json", "_structure_detected.json")
            )
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            json.dumps(structure, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return structure

    def build_document_metadata(
        self,
        structure: Dict[str, Dict[str, Any]],
        source_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build document-level metadata from structure metadata.

        This is intentionally metadata-only and does not create chunks.
        """
        entries: List[Dict[str, Any]] = []
        for key in sorted(structure.keys(), key=lambda raw: int(raw) if raw.isdigit() else raw):
            value = structure[key]
            if isinstance(value, dict):
                text = _compact_text(str(value.get("text", "")))
                entries.append(
                    {
                        "idx": int(key) if str(key).isdigit() else key,
                        "type": value.get("type", "unknown"),
                        "level": int(value.get("level", 4)),
                        "page_idx": value.get("page_idx"),
                        "text": text,
                    }
                )

        page_values = [e["page_idx"] for e in entries if isinstance(e.get("page_idx"), int)]
        page_count = (max(page_values) + 1) if page_values else 0

        text_entries = [e for e in entries if e["type"] in {"text", "header"} and e["text"]]
        heading_entries = [e for e in text_entries if e["level"] <= 2]

        combined_text = " ".join(e["text"] for e in text_entries)
        text_lower = combined_text.lower()

        vin_match = VIN_PATTERN.search(combined_text)
        vin = vin_match.group(0) if vin_match else None

        make_model_match = MAKE_MODEL_PATTERN.search(combined_text)
        make = make_model_match.group(1).lower() if make_model_match else None
        model = make_model_match.group(2).lower() if make_model_match else None

        date_match = DATE_PATTERN.search(combined_text)
        service_date = None
        if date_match:
            day_raw, month_raw, year_raw = date_match.groups()
            service_date = f"{int(year_raw):04d}-{_month_to_number(month_raw):02d}-{int(day_raw):02d}"

        mileage_match = MILEAGE_PATTERN.search(combined_text)
        mileage = int(mileage_match.group(1).replace(",", "")) if mileage_match else None

        detected_systems: Dict[str, List[str]] = {}
        for system, keywords in SYSTEM_KEYWORDS.items():
            hits = [keyword for keyword in keywords if keyword in text_lower]
            if hits:
                detected_systems[system] = sorted(set(hits))

        block_type_counts = Counter(e["type"] for e in entries)
        level_counts = Counter(e["level"] for e in entries)

        heading_candidates: List[Dict[str, Any]] = []
        for entry in heading_entries:
            score = 0
            if entry["level"] == 1:
                score += 100
            elif entry["level"] == 2:
                score += 70
            if len(entry["text"]) <= 120:
                score += 10
            lowered = entry["text"].lower()
            if any(hint in lowered for hint in IMPORTANT_SECTION_HINTS):
                score += 25

            heading_candidates.append(
                {
                    "text": entry["text"],
                    "page_idx": entry["page_idx"],
                    "level": entry["level"],
                    "importance_score": score,
                }
            )

        heading_candidates.sort(key=lambda item: item["importance_score"], reverse=True)
        important_points = heading_candidates[:20]

        top_sections = [
            {
                "text": item["text"],
                "page_idx": item["page_idx"],
                "level": item["level"],
            }
            for item in heading_candidates[:12]
        ]

        doc_kind = "owner_manual" if "owner" in text_lower and "manual" in text_lower else "automotive_document"

        metadata = {
            "document_info": {
                "source_file": source_file,
                "document_type": doc_kind,
                "generated_on": str(date.today()),
            },
            "vehicle_id": {
                "vin": vin,
                "make": make,
                "model": model,
            },
            "service_info": {
                "service_date": service_date,
                "mileage": mileage,
            },
            "issue_summary": {
                "systems_detected": detected_systems,
                "symptoms": sorted({hit for values in detected_systems.values() for hit in values}),
            },
            "document_statistics": {
                "page_count": page_count,
                "block_count": len(entries),
                "text_block_count": len(text_entries),
                "heading_count": len(heading_entries),
                "block_type_distribution": dict(block_type_counts),
                "level_distribution": dict(level_counts),
            },
            "important_points": important_points,
            "top_sections": top_sections,
        }
        return metadata

    def step3_generate_document_metadata(
        self,
        structure_json_path: str,
        output_json_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Step 3: Generate document metadata from structure metadata JSON.

        Args:
            structure_json_path: Path to *_structure_detected.json.
            output_json_path: Optional explicit output path.

        Returns:
            Generated metadata dictionary.
        """
        structure_path = Path(structure_json_path)
        if not structure_path.exists():
            raise FileNotFoundError(f"Structure metadata not found: {structure_path}")

        with structure_path.open("r", encoding="utf-8") as file_obj:
            structure = json.load(file_obj)

        if not isinstance(structure, dict):
            raise ValueError(
                f"Expected a JSON object in {structure_path}, got: {type(structure).__name__}"
            )

        metadata = self.build_document_metadata(
            structure=structure,
            source_file=str(structure_path),
        )

        resolved_output_path = (
            Path(output_json_path)
            if output_json_path
            else structure_path.with_name(
                structure_path.name.replace("_structure_detected.json", "_metadata.json")
            )
        )
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_output_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return metadata

    def _run_mineru(
        self,
        input_path: Path,
        output_dir: Path,
        backend: Optional[str] = None,
    ) -> None:
        """Invoke MinerU CLI for a PDF or a directory of PDFs."""
        command = ["mineru", "-p", str(input_path), "-o", str(output_dir)]

        resolved_backend = backend or self.settings.MINERU_BACKEND
        if resolved_backend == "pipeline":
            command.extend(["-b", "pipeline"])

        subprocess.run(command, check=True)

    def _load_parsed_documents(self, output_dir: Path, source_path: Path) -> List[Any]:
        """Load MinerU output files into LangChain Documents."""
        text_blocks = self._load_text_blocks(output_dir=output_dir, source_path=source_path)
        documents: List[Any] = []

        for block in text_blocks:
            documents.append(
                Document(
                    page_content=block["text"],
                    metadata={
                        "source": block["source"],
                        "parsed_by": block["parsed_by"],
                        "output_file": block["output_file"],
                        "doc_type": block["doc_type"],
                        "format": block["format"],
                        "page_idx": block["page_idx"],
                        "block_idx": block["block_idx"],
                    },
                )
            )

        return documents

    def _load_text_blocks(self, output_dir: Path, source_path: Path) -> List[dict]:
        """Load MinerU output files and convert them into text blocks."""
        blocks: List[dict] = []
        source_name = source_path.stem.lower()
        block_index = 0

        for file_path in output_dir.rglob("*"):
            if file_path.is_dir():
                continue

            suffix = file_path.suffix.lower()
            if suffix not in {".md", ".markdown", ".json"}:
                continue

            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".json":
                try:
                    payload = json.loads(text)
                    text = payload.get("markdown") or payload.get("text") or text
                except json.JSONDecodeError:
                    pass

            if not text.strip():
                continue

            normalized_text = re.sub(r"\n{3,}", "\n\n", text).strip()
            for paragraph in [chunk.strip() for chunk in normalized_text.split("\n\n") if chunk.strip()]:
                blocks.append(
                    {
                        "type": "text",
                        "text": paragraph,
                        "page_idx": block_index,
                        "block_idx": block_index,
                        "source": str(source_path),
                        "parsed_by": "mineru",
                        "output_file": str(file_path),
                        "doc_type": source_name,
                        "format": suffix.lstrip("."),
                    }
                )
                block_index += 1

        return blocks
