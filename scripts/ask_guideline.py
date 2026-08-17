#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eczema_rag.config import PipelineConfig, resolve_path
from eczema_rag.generator import GroundedAnswerGenerator
from eczema_rag.retriever import GuidelineRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a clinical question using retrieved guideline evidence."
    )
    parser.add_argument(
        "query",
        help="Clinical question to answer from the indexed guidelines.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.json",
        help="Pipeline configuration path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Override the number of retrieved chunks.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON result.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PipelineConfig.load(
        ROOT,
        resolve_path(ROOT, args.config),
    )

    retriever = GuidelineRetriever(
        vector_store_path=Path(config.vector_store["path"]),
        collection_name=config.collection_name,
        dimension=int(config.embedding["dimension"]),
        top_k=int(config.retrieval["top_k"]),
        minimum_score=float(config.retrieval["minimum_score"]),
    )

    hits = retriever.search(
        args.query,
        top_k=args.top_k,
    )

    generator = GroundedAnswerGenerator(config.generation)
    result = generator.generate(args.query, hits)

    if args.json:
        print(
            json.dumps(
                {
                    "status": result.status,
                    "answer": result.answer,
                    "citations": result.citations,
                    "retrieval_scores": result.retrieval_scores,
                    "refusal_reason": result.refusal_reason,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Status: {result.status}\n")
    print(result.answer)

    if result.citations:
        print("\nVerified retrieved citations:")
        for citation in result.citations:
            print(f"- {citation}")

    if result.refusal_reason:
        print(f"\nRefusal reason: {result.refusal_reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())