#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eczema_rag.config import PipelineConfig, load_json, resolve_path
from eczema_rag.retriever import GuidelineRetriever, citation_for_hit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate document- and exact-chunk retrieval quality."
    )
    parser.add_argument(
        "--dataset",
        default="config/eval_questions.json",
        help="Path to the evaluation dataset JSON file.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline.json",
        help="Path to the pipeline configuration JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of chunks to retrieve per question; must be at least 5.",
    )
    parser.add_argument(
        "--output",
        default="outputs/evaluations/local_hashing_chunk_eval.json",
        help="Path for the evaluation-results JSON file.",
    )
    return parser.parse_args()


def first_rank_for_documents(hits, expected_doc_ids: set[str]) -> int | None:
    for hit in hits:
        if hit.chunk.doc_id in expected_doc_ids:
            return hit.rank
    return None


def first_rank_for_chunks(hits, expected_chunk_ids: set[str]) -> int | None:
    for hit in hits:
        if hit.chunk.chunk_id in expected_chunk_ids:
            return hit.rank
    return None


def main() -> int:
    args = parse_args()

    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5.")

    config = PipelineConfig.load(ROOT, resolve_path(ROOT, args.config))
    dataset = load_json(resolve_path(ROOT, args.dataset))
    queries = dataset.get("queries", [])

    if not isinstance(queries, list) or not queries:
        raise ValueError("Dataset must contain a non-empty 'queries' list.")

    retriever = GuidelineRetriever(
        vector_store_path=Path(config.vector_store["path"]),
        collection_name=config.collection_name,
        dimension=int(config.embedding["dimension"]),
        top_k=args.top_k,
        minimum_score=float(config.retrieval["minimum_score"]),
    )

    results = []

    for item in queries:
        query_id = str(item["id"])
        question = str(item["query"]).strip()

        expected_doc_ids = set(item.get("expected_doc_ids", []))
        if not expected_doc_ids and item.get("expected_doc"):
            expected_doc_ids = {str(item["expected_doc"])}

        evidence_blocks = item.get("expected_evidence", [])
        expected_chunk_groups = [
            set(block.get("acceptable_chunk_ids", []))
            for block in evidence_blocks
            if block.get("acceptable_chunk_ids")
        ]
        expected_chunk_ids = set().union(*expected_chunk_groups) if expected_chunk_groups else set()

        started = time.perf_counter()
        hits = retriever.search(question, top_k=args.top_k)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        first_document_rank = first_rank_for_documents(hits, expected_doc_ids)
        first_chunk_rank = first_rank_for_chunks(hits, expected_chunk_ids)

        retrieved_top_1 = {hit.chunk.chunk_id for hit in hits[:1]}
        retrieved_top_3 = {hit.chunk.chunk_id for hit in hits[:3]}
        retrieved_top_5 = {hit.chunk.chunk_id for hit in hits[:5]}

        evidence_group_hits_at_5 = [
            bool(group & retrieved_top_5)
            for group in expected_chunk_groups
        ]

        results.append(
            {
                "id": query_id,
                "query": question,
                "query_style": item.get("query_style", "unknown"),
                "topic": item.get("topic", "unknown"),
                "difficulty": item.get("difficulty", "unknown"),
                "annotation_status": item.get("annotation_status", "unknown"),
                "expected_doc_ids": sorted(expected_doc_ids),
                "expected_chunk_ids": sorted(expected_chunk_ids),

                "first_correct_document_rank": first_document_rank,
                "document_hit_at_1": first_document_rank == 1,
                "document_hit_at_3": (
                    first_document_rank is not None
                    and first_document_rank <= 3
                ),
                "document_hit_at_5": (
                    first_document_rank is not None
                    and first_document_rank <= 5
                ),

                "first_correct_chunk_rank": first_chunk_rank,
                "exact_chunk_hit_at_1": first_chunk_rank == 1,
                "exact_chunk_hit_at_3": (
                    first_chunk_rank is not None
                    and first_chunk_rank <= 3
                ),
                "exact_chunk_hit_at_5": (
                    first_chunk_rank is not None
                    and first_chunk_rank <= 5
                ),

                "evidence_group_count": len(expected_chunk_groups),
                "evidence_groups_found_at_5": sum(evidence_group_hits_at_5),
                "all_evidence_groups_found_at_5": (
                    bool(expected_chunk_groups)
                    and all(evidence_group_hits_at_5)
                ),

                "query_latency_ms": latency_ms,
                "retrieved_hits": [
                    {
                        "rank": hit.rank,
                        "score": hit.score,
                        "chunk_id": hit.chunk.chunk_id,
                        "doc_id": hit.chunk.doc_id,
                        "section_path": hit.chunk.section_path,
                        "page_start": hit.chunk.page_start,
                        "page_end": hit.chunk.page_end,
                        "citation": citation_for_hit(hit),
                    }
                    for hit in hits
                ],
            }
        )

    total = len(results)
    annotated_results = [
        row for row in results if row["expected_chunk_ids"]
    ]

    summary = {
        "query_count": total,
        "annotated_chunk_query_count": len(annotated_results),

        "document_recall_at_1": round(
            sum(row["document_hit_at_1"] for row in results) / total, 4
        ),
        "document_recall_at_3": round(
            sum(row["document_hit_at_3"] for row in results) / total, 4
        ),
        "document_recall_at_5": round(
            sum(row["document_hit_at_5"] for row in results) / total, 4
        ),
        "document_mrr": round(
            sum(
                1 / row["first_correct_document_rank"]
                for row in results
                if row["first_correct_document_rank"] is not None
            ) / total,
            4,
        ),

        "exact_chunk_recall_at_1": round(
            sum(row["exact_chunk_hit_at_1"] for row in annotated_results)
            / len(annotated_results),
            4,
        ) if annotated_results else None,
        "exact_chunk_recall_at_3": round(
            sum(row["exact_chunk_hit_at_3"] for row in annotated_results)
            / len(annotated_results),
            4,
        ) if annotated_results else None,
        "exact_chunk_recall_at_5": round(
            sum(row["exact_chunk_hit_at_5"] for row in annotated_results)
            / len(annotated_results),
            4,
        ) if annotated_results else None,
        "exact_chunk_mrr": round(
            sum(
                1 / row["first_correct_chunk_rank"]
                for row in annotated_results
                if row["first_correct_chunk_rank"] is not None
            ) / len(annotated_results),
            4,
        ) if annotated_results else None,

        "complete_evidence_recall_at_5": round(
            sum(row["all_evidence_groups_found_at_5"] for row in annotated_results)
            / len(annotated_results),
            4,
        ) if annotated_results else None,

        "mean_query_latency_ms": round(
            sum(row["query_latency_ms"] for row in results) / total,
            2,
        ),
    }

    output_path = resolve_path(ROOT, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": str(args.dataset),
        "embedding": config.embedding,
        "collection_name": config.collection_name,
        "top_k": args.top_k,
        "summary": summary,
        "per_query": results,
    }

    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"\nDetailed results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())