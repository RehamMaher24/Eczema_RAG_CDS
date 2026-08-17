from __future__ import annotations

import json
from pathlib import Path

import pytest

from eczema_rag.config import (
    CorpusIntegrityError,
    PipelineConfig,
    load_resources,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_corpus_manifest_validates_all_seven_pdfs() -> None:
    resources, metadata = load_resources(ROOT, ROOT / "config" / "resources.json")

    assert len(resources) == 7
    assert metadata["corpus_name"] == "eczema_guidelines_fixed_corpus"
    assert len({resource.doc_id for resource in resources}) == 7
    assert all((ROOT / resource.path).is_file() for resource in resources)


def test_hash_mismatch_is_rejected_without_modifying_source(tmp_path: Path) -> None:
    original = json.loads(
        (ROOT / "config" / "resources.json").read_text(encoding="utf-8")
    )
    original["resources"] = [dict(original["resources"][0])]
    original["resources"][0]["sha256"] = "0" * 64
    manifest = tmp_path / "resources.json"
    manifest.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(CorpusIntegrityError, match="SHA-256 mismatch"):
        load_resources(ROOT, manifest)


def test_pipeline_configuration_loads_with_supported_backends() -> None:
    config = PipelineConfig.load(ROOT, ROOT / "config" / "pipeline.json")

    assert config.embedding["provider"] == "local_hashing"
    assert config.vector_store["provider"] == "sqlite"
    assert config.collection_name == "eczema_guidelines_v1"
