#!/usr/bin/env python
"""Generate structure metadata JSON from a MinerU content list JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import MinerUParser


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="Generate heading-level structure metadata from a content list JSON file."
    )
    parser.add_argument(
        "--content",
        required=True,
        help="Path to input content list JSON (for example: BMW_content_list.json).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: <input>_structure_detected.json",
    )
    return parser


def main() -> int:
    """CLI entrypoint."""
    args = build_parser().parse_args()
    content_path = Path(args.content)
    output_path = Path(args.output) if args.output else None

    parser = MinerUParser()

    print("\n=== STEP 2: Generating structure metadata ===\n")
    structure = parser.step2_generate_structure_metadata(
        content_json_path=str(content_path),
        output_json_path=str(output_path) if output_path else None,
    )

    resolved_output = (
        output_path
        if output_path
        else content_path.with_name(
            content_path.name.replace("_content_list.json", "_structure_detected.json")
        )
    )

    print(f"Result: {len(structure)} structure entries generated")
    print(f"Saved: {resolved_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
