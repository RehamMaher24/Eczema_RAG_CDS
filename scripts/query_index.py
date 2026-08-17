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
from eczema_rag.retriever import GuidelineRetriever, citation_for_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the retrieval-ready eczema guideline vector collection."
    )
    parser.add_argument("query", help="Clinical guideline retrieval question.")
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--minimum-score", type=float)
    parser.add_argument(
        "--doc-id",
        action="append",
        dest="doc_ids",
        help="Optional document filter; may be supplied multiple times.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig.load(ROOT, resolve_path(ROOT, args.config))
    retriever = GuidelineRetriever(
        vector_store_path=Path(config.vector_store["path"]),
        collection_name=config.collection_name,
        dimension=int(config.embedding["dimension"]),
        top_k=int(config.retrieval.get("top_k", 5)),
        minimum_score=float(config.retrieval.get("minimum_score", 0.05)),
    )
    hits = retriever.search(
        args.query,
        top_k=args.top_k,
        minimum_score=args.minimum_score,
        doc_ids=args.doc_ids,
    )
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "rank": hit.rank,
                        "score": hit.score,
                        "citation": citation_for_hit(hit),
                        **hit.chunk.to_dict(),
                    }
                    for hit in hits
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    if not hits:
        print("No chunk met the configured retrieval threshold.")
        return 0

    for hit in hits:
        print(f"\n[{hit.rank}] score={hit.score:.4f}")
        print(citation_for_hit(hit))
        print(hit.chunk.text[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
