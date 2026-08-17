#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eczema_rag.config import PipelineConfig, load_resources, resolve_path
from eczema_rag.pipeline import IngestionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse, section-chunk, embed, index, and validate the fixed seven-PDF "
            "eczema/contact-dermatitis guideline corpus."
        )
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.json",
        help="Pipeline configuration path relative to the repository root.",
    )
    parser.add_argument("--resources", help="Override the resource manifest path.")
    parser.add_argument("--questions", help="Override the retrieval questions path.")
    parser.add_argument("--output-dir", help="Override the output directory.")
    parser.add_argument("--collection", help="Override the collection name.")
    parser.add_argument("--log-level", help="Override the configured logging level.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate configuration, PDF presence, and SHA-256 values without parsing.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print the final pipeline summary as JSON.",
    )
    return parser.parse_args()


def apply_overrides(config: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    changes: dict[str, object] = {}
    if args.resources:
        changes["resources_file"] = resolve_path(ROOT, args.resources)
    if args.questions:
        changes["questions_file"] = resolve_path(ROOT, args.questions)
    if args.output_dir:
        changes["output_directory"] = resolve_path(ROOT, args.output_dir)
    if args.collection:
        changes["collection_name"] = args.collection
    if args.log_level:
        logging_config = dict(config.logging)
        logging_config["level"] = args.log_level.upper()
        changes["logging"] = logging_config
    return replace(config, **changes) if changes else config


def main() -> int:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    config = PipelineConfig.load(ROOT, config_path)
    config = apply_overrides(config, args)

    if args.validate_only:
        resources, metadata = load_resources(ROOT, config.resources_file)
        print(
            f"Fixed corpus valid: {len(resources)} PDFs; "
            f"corpus={metadata['corpus_name']}; collection={config.collection_name}."
        )
        return 0

    result = IngestionPipeline(config).run()
    if args.json_summary:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print("Day 1 ingestion pipeline complete.")
        print(f"Documents processed: {result.resources_processed}")
        print(f"Pages processed: {result.pages_processed}")
        print(f"Sections detected: {result.sections_detected}")
        print(f"Chunks created: {result.chunks_created}")
        print(f"Vectors indexed: {result.vectors_indexed}")
        print(f"Tables extracted: {result.tables_extracted}")
        print(f"Figure references: {result.figure_references}")
        print(f"Pages flagged for OCR: {result.pages_needing_ocr}")
        print("See outputs/ingestion_report.md and outputs/retrieval_results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
