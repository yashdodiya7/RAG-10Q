#!/usr/bin/env python
"""Generate document metadata JSON from a structure metadata JSON file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MinerUParser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate document-level metadata from structure metadata."
    )
    parser.add_argument(
        "--structure",
        required=True,
        help="Path to input structure JSON (for example: BMW_structure_detected.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: <input>_metadata.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    structure_path = Path(args.structure)
    output_path = Path(args.output) if args.output else None

    parser = MinerUParser()

    print("\n=== STEP 3: Generating document metadata ===\n")
    metadata = parser.step3_generate_document_metadata(
        structure_json_path=str(structure_path),
        output_json_path=str(output_path) if output_path else None,
    )

    resolved_output = (
        output_path
        if output_path
        else structure_path.with_name(
            structure_path.name.replace("_structure_detected.json", "_metadata.json")
        )
    )

    print(f"Result: metadata generated with {len(metadata.get('important_points', []))} important points")
    print(f"Saved: {resolved_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
