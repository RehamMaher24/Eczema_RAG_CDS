from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .services import AppSettings, ClinicalRagService, NotConfiguredImageClassifier, RemoteImageClassifier


@lru_cache
def get_settings() -> AppSettings:
    return AppSettings.from_environment(Path(__file__).resolve().parents[1])


@lru_cache
def get_service() -> ClinicalRagService:
    settings = get_settings()
    classifier = (RemoteImageClassifier(settings.skin_classifier_api_url)
                  if settings.skin_classifier_api_url else NotConfiguredImageClassifier())
    return ClinicalRagService(settings, image_classifier=classifier)
