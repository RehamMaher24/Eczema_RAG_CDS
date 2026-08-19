from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

import requests

from dotenv import load_dotenv

from eczema_rag.config import PipelineConfig
from eczema_rag.Gemini_Scope_Checker import GeminiScopeChecker, ScopeDecision
from eczema_rag.generator import GroundedAnswerGenerator
from eczema_rag.judge import GroundedAnswerJudge, JudgeReview
from eczema_rag.retriever import GuidelineRetriever, citation_for_hit, clean_section_path
from eczema_rag.router import route_question_with_scores
from eczema_rag.vector_store import SQLiteVectorStore, VectorStoreError

from .schemas import EvidenceItem, GroundingReview, ImagePrediction, RoutingResponse, ScopeCheckResponse, Timings
#here is the updated code
logger = logging.getLogger(__name__)


class ImageClassifier(Protocol):
    def classify(self, image_bytes: bytes, filename: str) -> ImagePrediction: ...


class NotConfiguredImageClassifier:
    def classify(self, image_bytes: bytes, filename: str) -> ImagePrediction:
        return ImagePrediction()


class RemoteImageClassifier:
    """Adapter for the separately running skin_api FastAPI service."""

    def __init__(self, base_url: str, timeout_seconds: float = 20.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def classify(self, image_bytes: bytes, filename: str) -> ImagePrediction:
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                files={"file": (filename or "image.jpg", image_bytes, _mime_for_filename(filename))},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise ServiceError(503, "The skin classifier service is unavailable.") from exc
        except ValueError as exc:
            raise ServiceError(503, "The skin classifier returned malformed data.") from exc

        status = str(payload.get("status", "uncertain"))
        confidence = payload.get("confidence")
        return ImagePrediction(
            status="available" if status == "usable_as_retrieval_hint" else status,
            predicted_type=payload.get("predicted_class"),
            confidence=confidence,
            alternatives=[str(label) for label in (payload.get("probabilities") or {}).keys() if label != payload.get("predicted_class")],
        )


class ScopeChecker(Protocol):
    def check(self, question: str) -> ScopeDecision: ...


class UnavailableScopeChecker:
    """Fail closed when the scope model cannot be configured."""

    def check(self, question: str) -> ScopeDecision:
        return ScopeDecision(
            in_scope=False,
            confidence=0.0,
            reason="Scope verification is unavailable; the request was stopped safely.",
            status="scope_check_unavailable",
        )


class ServiceError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class AppSettings:
    root: Path
    config_path: Path
    max_image_bytes: int
    allowed_image_types: frozenset[str]
    generator_timeout_seconds: float
    judge_timeout_seconds: float
    cors_origins: tuple[str, ...]
    skin_classifier_api_url: str | None

    @classmethod
    def from_environment(cls, root: Path) -> "AppSettings":
        load_dotenv(root / ".env")
        return cls(
            root=root,
            config_path=root / os.getenv("RAG_PIPELINE_CONFIG", "config/pipeline_gemini.json"),
            max_image_bytes=int(os.getenv("MAX_IMAGE_BYTES", str(5 * 1024 * 1024))),
            allowed_image_types=frozenset(os.getenv("ALLOWED_IMAGE_MIME_TYPES", "image/jpeg,image/png").split(",")),
            generator_timeout_seconds=float(os.getenv("GENERATOR_TIMEOUT_SECONDS", "30")),
            judge_timeout_seconds=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "15")),
            cors_origins=tuple(item.strip() for item in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8501").split(",") if item.strip()),
            skin_classifier_api_url=os.getenv("SKIN_CLASSIFIER_API_URL") or os.getenv("SKIN_MODEL_API_URL"),
        )


class ClinicalRagService:
    """Thin API wrapper that only opens the existing SQLite collection for queries."""

    def __init__(self, settings: AppSettings, image_classifier: ImageClassifier, scope_checker: ScopeChecker | None = None) -> None:
        self.settings = settings
        self.config = PipelineConfig.load(settings.root, settings.config_path)
        self.image_classifier = image_classifier
        if scope_checker is not None:
            self.scope_checker = scope_checker
        else:
            try:
                self.scope_checker = GeminiScopeChecker()
            except RuntimeError:
                self.scope_checker = UnavailableScopeChecker()
        self.retriever = GuidelineRetriever(
            Path(self.config.vector_store["path"]), self.config.collection_name,
            int(self.config.embedding["dimension"]), int(self.config.retrieval["top_k"]),
            float(self.config.retrieval["minimum_score"]),
        )

    def health(self) -> dict[str, object]:
        # Read-only metadata check: no embedding or collection replacement occurs here.
        try:
            with SQLiteVectorStore(self.retriever.vector_store_path, self.retriever.dimension) as store:
                store.get_model_state(self.config.collection_name)
        except VectorStoreError as exc:
            raise ServiceError(503, "Configured vector collection is unavailable.") from exc
        return {"status": "ok", "collection": self.config.collection_name,
                "embedding_model": str(self.config.embedding["model"]),
                "image_classifier_available": not isinstance(self.image_classifier, NotConfiguredImageClassifier),
                "scope_checker_available": not isinstance(self.scope_checker, UnavailableScopeChecker)}

    def retrieve(self, question: str, top_k: int | None = None, document_filters: list[str] | None = None) -> tuple[ScopeCheckResponse, RoutingResponse, list[EvidenceItem], Timings]:
        question = self._validate_question(question)
        start = time.perf_counter()
        scope, scope_ms = self._scope_check(question)
        if not scope.in_scope:
            return scope, RoutingResponse(experts=[], weights={}), [], Timings(scope_check=scope_ms, total=elapsed_ms(start))
        routing_started = time.perf_counter()
        weights = route_question_with_scores(question)
        routing_ms = elapsed_ms(routing_started)
        retrieval_started = time.perf_counter()
        try:
            hits = self.retriever.search(question, top_k=top_k, doc_ids=document_filters)
        except VectorStoreError as exc:
            raise ServiceError(503, "Configured vector collection is unavailable.") from exc
        except RuntimeError as exc:
            raise ServiceError(503, "Query embedding is unavailable. Check the embedding service configuration.") from exc
        retrieval_ms = elapsed_ms(retrieval_started)
        experts = [expert for expert, weight in weights.items() if weight > 0]
        evidence = [self._evidence_item(hit) for hit in hits]
        logger.info("retrieval question_length=%d route_weights=%s chunk_ids=%s scores=%s", len(question), weights, [hit.chunk.chunk_id for hit in hits], [hit.score for hit in hits])
        return scope, RoutingResponse(experts=experts, weights=weights), evidence, Timings(scope_check=scope_ms, routing=routing_ms, retrieval=retrieval_ms, total=elapsed_ms(start))

    def chat(self, question: str, image_bytes: bytes | None = None, filename: str = "", mime_type: str | None = None, top_k: int | None = None) -> dict[str, object]:
        started = time.perf_counter()
        question = self._validate_question(question)
        scope, scope_ms = self._scope_check(question)
        if not scope.in_scope:
            return self._out_of_scope_chat(question, scope, scope_ms, elapsed_ms(started))
        image_started = time.perf_counter()
        prediction = self._classify_image(image_bytes, filename, mime_type)
        image_ms = elapsed_ms(image_started)
        routing, evidence, retrieval_timings, hits = self._retrieve_with_hits(question, top_k, prediction)
        warnings: list[str] = []
        generation_started = time.perf_counter()
        try:
            generated = GroundedAnswerGenerator(self.config.generation).generate(question, hits)
            answer, answer_status = generated.answer, generated.status
        except Exception:
            answer = "The evidence was retrieved, but grounded answer generation is currently unavailable."
            answer_status = "generation_error"
            warnings.append("Grounded answer generation was unavailable; inspect the evidence and citations.")
        generation_ms = elapsed_ms(generation_started)
        judging_started = time.perf_counter()
        review = self._review(question, answer, hits, answer_status, warnings)
        judging_ms = elapsed_ms(judging_started)
        timings = Timings(scope_check=scope_ms, image_classification=image_ms, routing=retrieval_timings.routing, retrieval=retrieval_timings.retrieval, generation=generation_ms, judging=judging_ms, total=elapsed_ms(started))
        return {"request_id": str(uuid4()), "question": question.strip(), "scope_check": scope, "image_prediction": prediction,
                "routing": routing, "evidence": evidence, "answer": answer, "answer_status": answer_status,
                "grounding_review": review, "warnings": warnings, "timings_ms": timings}

    def _review(self, question: str, answer: str, hits: list, answer_status: str, warnings: list[str]) -> GroundingReview:
        if answer_status == "insufficient_evidence":
            return GroundingReview(status="insufficient_evidence", grounded=None, citation_valid=None, reason="Retrieved evidence did not meet the configured threshold.")
        if answer_status == "generation_error":
            return GroundingReview(status="judge_error", grounded=None, citation_valid=None, reason="No generated answer was available to review.")
        for _ in range(2):
            try:
                result: JudgeReview = GroundedAnswerJudge(self.config.judge).review(question, answer, hits)
                if result.reason.startswith("Judge response parsing failed"):
                    raise RuntimeError("Malformed judge response")
                status = result.decision if result.decision in {"approved", "revise", "refuse"} else "revise"
                return GroundingReview(status=status, grounded=result.grounded, citation_valid=result.citation_valid, reason=result.reason or "Automated review completed.")
            except Exception:
                continue
        warnings.append("Automated grounding verification was unavailable; retrieved evidence is still shown.")
        return GroundingReview(status="judge_error", grounded=None, citation_valid=None, reason="Automated verification did not return a usable response.")

    def _classify_image(self, image_bytes: bytes | None, filename: str, mime_type: str | None) -> ImagePrediction:
        if image_bytes is None:
            return ImagePrediction()
        if mime_type not in self.settings.allowed_image_types:
            raise ServiceError(415, "Unsupported image type. Upload a JPG, JPEG, or PNG image.")
        if len(image_bytes) > self.settings.max_image_bytes:
            raise ServiceError(413, "Image exceeds the configured maximum size.")
        return self.image_classifier.classify(image_bytes, filename)

    @staticmethod
    def _validate_question(question: str) -> str:
        if not question or not question.strip():
            raise ServiceError(422, "Question must not be empty.")
        return question.strip()

    def _scope_check(self, question: str) -> tuple[ScopeCheckResponse, int]:
        started = time.perf_counter()
        decision = self.scope_checker.check(question)
        scope = ScopeCheckResponse(**decision.to_dict())
        logger.info("scope_check question_length=%d in_scope=%s confidence=%.2f status=%s", len(question), scope.in_scope, scope.confidence, scope.status)
        return scope, elapsed_ms(started)

    @staticmethod
    def _out_of_scope_chat(question: str, scope: ScopeCheckResponse, scope_ms: int, total_ms: int) -> dict[str, object]:
        return {
            "request_id": str(uuid4()), "question": question, "scope_check": scope,
            "image_prediction": ImagePrediction(), "routing": RoutingResponse(experts=[], weights={}), "evidence": [],
            "answer": "I can only help with guideline evidence questions about eczema and dermatitis.",
            "answer_status": "out_of_scope",
            "grounding_review": GroundingReview(status="insufficient_evidence", grounded=None, citation_valid=None, reason="The question was stopped by the scope check before retrieval."),
            "warnings": ["No image classification, embedding, retrieval, generation, or judging was performed."],
            "timings_ms": Timings(scope_check=scope_ms, total=total_ms),
        }

    def _retrieve_with_hits(self, question: str, top_k: int | None, image_prediction: ImagePrediction | None = None):
        question = self._validate_question(question)
        routing_started = time.perf_counter()
        weights = route_question_with_scores(question)
        routing_ms = elapsed_ms(routing_started)
        retrieval_started = time.perf_counter()
        try:
            hits = self.retriever.search(question, top_k=top_k)
        except VectorStoreError as exc:
            raise ServiceError(503, "Configured vector collection is unavailable.") from exc
        except RuntimeError as exc:
            raise ServiceError(503, "Query embedding is unavailable. Check the embedding service configuration.") from exc
        evidence = [self._evidence_item(hit) for hit in hits]
        return (RoutingResponse(experts=[expert for expert, weight in weights.items() if weight > 0], weights=weights), evidence,
                Timings(routing=routing_ms, retrieval=elapsed_ms(retrieval_started)), hits)

    @staticmethod
    def _evidence_item(hit) -> EvidenceItem:
        chunk = hit.chunk
        return EvidenceItem(rank=hit.rank, score=hit.score, raw_score=hit.raw_score if hit.raw_score is not None else hit.score, text=chunk.text,
            citation=citation_for_hit(hit), document=chunk.document_name,
            section=" > ".join(clean_section_path(chunk.section_path)),
            pdf_page_start=chunk.page_start, pdf_page_end=chunk.page_end, chunk_id=chunk.chunk_id)


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _mime_for_filename(filename: str) -> str:
    return "image/png" if filename.lower().endswith(".png") else "image/jpeg"
