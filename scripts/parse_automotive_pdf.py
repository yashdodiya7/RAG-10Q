#!/usr/bin/env python
"""
MinerU-backed PDF parsing entry point.

Usage:
    python scripts/parse_automotive_pdf.py --pdf data/automotive_corpus/incoming_pdfs/sample.pdf
    python scripts/parse_automotive_pdf.py --pdf data/automotive_corpus/incoming_pdfs/sample.pdf --output data/automotive_corpus/parsed/sample_content_list.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DocumentLoader


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Parse a PDF with MinerU and save extracted text blocks.")
    parser.add_argument(
        "--pdf",
        required=True,
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path. Defaults to data/automotive_corpus/parsed/<pdf_stem>_content_list.json",
    )
    return parser


def main() -> int:
    """Run MinerU parsing for a single PDF."""
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf)
    output_path = Path(args.output) if args.output else None

    loader = DocumentLoader()

    print(f"\n=== STEP 1: Extracting text from {pdf_path} with MinerU ===\n")
    blocks = loader.step1_extract_text_blocks(
        pdf_path=str(pdf_path),
        output_json_path=str(output_path) if output_path else None,
    )

    resolved_output = (
        output_path
        if output_path
        else loader.settings.AUTOMOTIVE_PARSED_DIR / pdf_path.stem / f"{pdf_path.stem}_content_list.json"
    )

    print(f"\nResult: {len(blocks)} blocks extracted")
    print(f"✓ Saved: {resolved_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
