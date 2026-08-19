from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .services import AppSettings, ClinicalRagService, NotConfiguredImageClassifier


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.from_environment(Path(__file__).resolve().parents[1])


@lru_cache
def get_service() -> ClinicalRagService:
    settings = get_settings()
    return ClinicalRagService(settings, image_classifier=NotConfiguredImageClassifier())
