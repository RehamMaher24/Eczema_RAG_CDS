from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImagePrediction(BaseModel):
    status: Literal["not_available", "available"] = "not_available"
    predicted_type: str | None = None
    confidence: float | None = None
    alternatives: list[str] = Field(default_factory=list)


class ScopeCheckResponse(BaseModel):
    in_scope: bool
    confidence: float
    reason: str
    status: str


class RoutingResponse(BaseModel):
    experts: list[str]
    weights: dict[str, float]


class EvidenceItem(BaseModel):
    rank: int
    score: float
    raw_score: float
    text: str
    citation: str
    document: str
    section: str
    pdf_page_start: int
    pdf_page_end: int
    chunk_id: str


class GroundingReview(BaseModel):
    status: Literal["approved", "revise", "refuse", "judge_error", "insufficient_evidence"]
    grounded: bool | None = None
    citation_valid: bool | None = None
    reason: str


class Timings(BaseModel):
    scope_check: int = 0
    image_classification: int = 0
    routing: int = 0
    retrieval: int = 0
    generation: int = 0
    judging: int = 0
    total: int = 0


class ChatResponse(BaseModel):
    request_id: str
    question: str
    scope_check: ScopeCheckResponse
    image_prediction: ImagePrediction
    routing: RoutingResponse
    evidence: list[EvidenceItem]
    answer: str
    answer_status: str
    grounding_review: GroundingReview
    warnings: list[str] = Field(default_factory=list)
    timings_ms: Timings


class RetrieveRequest(BaseModel):
    question: str
    image_metadata: ImagePrediction | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_filters: list[str] | None = None


class RetrieveResponse(BaseModel):
    question: str
    scope_check: ScopeCheckResponse
    routing: RoutingResponse
    evidence: list[EvidenceItem]
    timings_ms: Timings
