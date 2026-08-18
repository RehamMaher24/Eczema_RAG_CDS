#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eczema_rag.config import PipelineConfig, resolve_path
from eczema_rag.generator import GroundedAnswerGenerator
from eczema_rag.judge import GroundedAnswerJudge
from eczema_rag.retriever import GuidelineRetriever, citation_for_hit
from eczema_rag.router import route_question, route_question_with_scores


DEFAULT_QUESTION_FILE = "config/questions.json"
DEFAULT_REPORT_FILE = "outputs/ingestion_report.json"
DEFAULT_SUMMARY_FILE = "outputs/rag_evaluation_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a focused evaluation of the eczema RAG pipeline: routing, retrieval, "
            "optional answer generation, and a compact summary of the current system performance."
        )
    )
    parser.add_argument("--config", default="config/pipeline.json")
    parser.add_argument("--questions", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_FILE)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip Groq answer generation and judge review, only run routing + retrieval.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("questions", [])
    else:
        items = payload
    return [str(item).strip() for item in items if str(item).strip()]


def load_ingestion_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def evaluate_retrieval(
    retriever: GuidelineRetriever,
    questions: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for question in questions:
        route = route_question(question)
        weights = route_question_with_scores(question)
        hits = retriever.search(question, top_k=3, minimum_score=0.05)

        result = {
            "question": question,
            "route": route,
            "weights": {key: round(value, 4) for key, value in weights.items()},
            "top_expert": route[0] if route else "unknown",
            "hits": [],
            "top_score": None,
            "mean_score": None,
        }

        if hits:
            scores = [hit.score for hit in hits]
            result["top_score"] = round(max(scores), 6)
            result["mean_score"] = round(sum(scores) / len(scores), 6)

        for hit in hits:
            item = {
                "rank": hit.rank,
                "score": round(hit.score, 6),
                "document_name": hit.chunk.document_name,
                "doc_id": hit.chunk.doc_id,
                "citation": citation_for_hit(hit),
                "section_path": hit.chunk.section_path,
                "chunk_id": hit.chunk.chunk_id,
            }
            result["hits"].append(item)

        results.append(result)

    return results


def evaluate_generation(
    retriever: GuidelineRetriever,
    questions: list[str],
) -> list[dict[str, Any]]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return []

    generator = GroundedAnswerGenerator({
        "model": "openai/gpt-oss-120b",
        "temperature": 0.0,
        "max_tokens": 700,
        "evidence_top_k": 4,
        "minimum_retrieval_score": 0.12,
    })
    judge = GroundedAnswerJudge({
        "model": "openai/gpt-oss-20b",
        "temperature": 0.0,
        "max_tokens": 500,
    })

    results: list[dict[str, Any]] = []
    for question in questions:
        hits = retriever.search(question, top_k=4, minimum_score=0.05)
        answer = generator.generate(question, hits)
        review = judge.review(question, answer.answer, hits)

        results.append(
            {
                "question": question,
                "status": answer.status,
                "answer": answer.answer,
                "citations": answer.citations,
                "retrieval_scores": answer.retrieval_scores,
                "judge_decision": review.decision,
                "judge_grounded": review.grounded,
                "judge_reason": review.reason,
            }
        )

    return results


def build_summary(
    ingestion_summary: dict[str, Any],
    question_results: list[dict[str, Any]],
    generation_results: list[dict[str, Any]],
) -> dict[str, Any]:
    route_count: dict[str, int] = {}
    top_expert_count: dict[str, int] = {}
    top_scores: list[float] = []
    mean_scores: list[float] = []

    for result in question_results:
        for expert in result["route"]:
            route_count[expert] = route_count.get(expert, 0) + 1
        top_expert_count[result["top_expert"]] = (
            top_expert_count.get(result["top_expert"], 0) + 1
        )
        if result["top_score"] is not None:
            top_scores.append(result["top_score"])
        if result["mean_score"] is not None:
            mean_scores.append(result["mean_score"])

    summary = {
        "project": "eczema_rag",
        "ingestion_summary": ingestion_summary,
        "router_summary": {
            "expert_route_counts": dict(sorted(route_count.items())),
            "top_expert_counts": dict(sorted(top_expert_count.items())),
        },
        "retrieval_summary": {
            "questions_evaluated": len(question_results),
            "avg_top_score": round(sum(top_scores) / len(top_scores), 6) if top_scores else None,
            "avg_mean_score": round(sum(mean_scores) / len(mean_scores), 6) if mean_scores else None,
            "questions_with_hits": sum(1 for item in question_results if item["hits"]),
        },
        "question_results": question_results,
    }

    if generation_results:
        summary["generation_summary"] = {
            "questions_answered": sum(1 for item in generation_results if item["status"] == "answered"),
            "questions_with_insufficient_evidence": sum(
                1 for item in generation_results if item["status"] == "insufficient_evidence"
            ),
            "judge_approved": sum(
                1 for item in generation_results if item["judge_decision"] == "approved"
            ),
            "judge_revise": sum(
                1 for item in generation_results if item["judge_decision"] == "revise"
            ),
            "judge_refuse": sum(
                1 for item in generation_results if item["judge_decision"] == "refuse"
            ),
        }
        summary["generation_results"] = generation_results

    return summary


def main() -> int:
    args = parse_args()
    config_path = resolve_path(ROOT, args.config)
    config = PipelineConfig.load(ROOT, config_path)

    questions_path = resolve_path(ROOT, args.questions)
    questions = load_questions(questions_path)
    retriever = GuidelineRetriever(
        vector_store_path=Path(config.vector_store["path"]),
        collection_name=config.collection_name,
        dimension=int(config.embedding["dimension"]),
        top_k=int(config.retrieval.get("top_k", 5)),
        minimum_score=float(config.retrieval.get("minimum_score", 0.05)),
    )

    ingestion_summary = load_ingestion_summary(ROOT / DEFAULT_REPORT_FILE)
    question_results = evaluate_retrieval(retriever, questions)
    generation_results: list[dict[str, Any]] = []
    if not args.skip_generation:
        generation_results = evaluate_generation(retriever, questions)

    summary = build_summary(ingestion_summary, question_results, generation_results)
    out_path = resolve_path(ROOT, args.summary_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("RAG evaluation summary")
    print(f"Questions tested: {len(question_results)}")
    print(f"Avg top retrieval score: {summary['retrieval_summary']['avg_top_score']}")
    print(f"Questions with retrieval hits: {summary['retrieval_summary']['questions_with_hits']}")
    print(f"Router top experts: {summary['router_summary']['top_expert_counts']}")
    if generation_results:
        print(f"Generated answers: {summary['generation_summary']['questions_answered']}")
        print(f"Judge approved: {summary['generation_summary']['judge_approved']}")
    print(f"Summary written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
