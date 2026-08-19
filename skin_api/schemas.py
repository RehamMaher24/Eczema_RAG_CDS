from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_backbone: str | None = None
    device: str | None = None
    labels: list[str] = Field(default_factory=list)
    message: str


class PredictionResponse(BaseModel):
    predicted_class: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    status: str
    message: str
    is_retrieval_hint: bool
    disclaimer: str = (
        "AI image output is a research retrieval hint, not a confirmed diagnosis "
        "or substitute for a qualified clinician."
    )


class ErrorResponse(BaseModel):
    detail: str
    error_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)
