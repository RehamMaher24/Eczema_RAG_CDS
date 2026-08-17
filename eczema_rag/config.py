from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Resource


class ConfigurationError(ValueError):
    """Raised when pipeline configuration or the fixed corpus manifest is invalid."""


class CorpusIntegrityError(RuntimeError):
    """Raised when a source PDF is missing or does not match its manifest hash."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    root: Path
    input_directory: Path
    resources_file: Path
    questions_file: Path
    output_directory: Path
    collection_name: str
    parser: dict[str, Any]
    chunking: dict[str, Any]
    embedding: dict[str, Any]
    vector_store: dict[str, Any]
    retrieval: dict[str, Any]
    logging: dict[str, Any]

    @classmethod
    def load(cls, root: Path, path: Path) -> "PipelineConfig":
        data = load_json(path)
        required = {
            "input_directory",
            "resources_file",
            "questions_file",
            "output_directory",
            "collection_name",
            "parser",
            "chunking",
            "embedding",
            "vector_store",
            "retrieval",
            "logging",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ConfigurationError(
                f"Pipeline configuration is missing required keys: {', '.join(missing)}"
            )

        vector_store = dict(data["vector_store"])
        logging_config = dict(data["logging"])
        vector_store["path"] = str(resolve_path(root, vector_store["path"]))
        logging_config["file"] = str(resolve_path(root, logging_config["file"]))

        config = cls(
            root=root,
            input_directory=resolve_path(root, data["input_directory"]),
            resources_file=resolve_path(root, data["resources_file"]),
            questions_file=resolve_path(root, data["questions_file"]),
            output_directory=resolve_path(root, data["output_directory"]),
            collection_name=str(data["collection_name"]),
            parser=dict(data["parser"]),
            chunking=dict(data["chunking"]),
            embedding=dict(data["embedding"]),
            vector_store=vector_store,
            retrieval=dict(data["retrieval"]),
            logging=logging_config,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.collection_name.strip():
            raise ConfigurationError("collection_name must not be empty")
        if int(self.chunking.get("target_words", 0)) <= 0:
            raise ConfigurationError("chunking.target_words must be positive")
        if int(self.chunking.get("maximum_words", 0)) < int(
            self.chunking.get("target_words", 0)
        ):
            raise ConfigurationError(
                "chunking.maximum_words must be greater than or equal to target_words"
            )
        if int(self.chunking.get("overlap_words", -1)) < 0:
            raise ConfigurationError("chunking.overlap_words must be non-negative")
        if int(self.embedding.get("dimension", 0)) <= 0:
            raise ConfigurationError("embedding.dimension must be positive")
        if self.embedding.get("provider") != "local_hashing":
            raise ConfigurationError(
                "This Day 1 implementation supports embedding.provider='local_hashing' only"
            )
        if self.vector_store.get("provider") != "sqlite":
            raise ConfigurationError(
                "This Day 1 implementation supports vector_store.provider='sqlite' only"
            )


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"Required configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Invalid JSON in {path}: {exc}") from exc


def load_resources(root: Path, path: Path) -> tuple[list[Resource], dict[str, Any]]:
    data = load_json(path)
    resource_rows = data.get("resources")
    if not isinstance(resource_rows, list) or not resource_rows:
        raise ConfigurationError("resources.json must contain a non-empty resources list")

    resources: list[Resource] = []
    seen_doc_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, row in enumerate(resource_rows, start=1):
        if not isinstance(row, dict):
            raise ConfigurationError(f"Resource {index} must be an object")
        try:
            resource = Resource.from_dict(row)
        except KeyError as exc:
            raise ConfigurationError(
                f"Resource {index} is missing required field {exc.args[0]!r}"
            ) from exc

        if resource.doc_id in seen_doc_ids:
            raise ConfigurationError(f"Duplicate doc_id in resources: {resource.doc_id}")
        if resource.path in seen_paths:
            raise ConfigurationError(f"Duplicate source path in resources: {resource.path}")
        seen_doc_ids.add(resource.doc_id)
        seen_paths.add(resource.path)

        pdf_path = resolve_path(root, resource.path)
        if not pdf_path.exists():
            raise CorpusIntegrityError(f"Source PDF is missing: {pdf_path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise CorpusIntegrityError(f"Source is not a PDF: {pdf_path}")
        actual_hash = sha256_file(pdf_path)
        if actual_hash.lower() != resource.sha256.lower():
            raise CorpusIntegrityError(
                f"SHA-256 mismatch for {pdf_path.name}: expected {resource.sha256}, "
                f"found {actual_hash}. The fixed corpus must not be altered."
            )
        resources.append(resource)

    metadata = {
        "schema_version": data.get("schema_version", 1),
        "corpus_name": data.get("corpus_name", "unknown"),
        "corpus_policy": data.get("corpus_policy", ""),
    }
    return resources, metadata


def load_questions(path: Path) -> list[str]:
    data = load_json(path)
    rows = data.get("questions")
    if not isinstance(rows, list) or not rows:
        raise ConfigurationError("questions.json must contain a non-empty questions list")
    questions = [str(item).strip() for item in rows if str(item).strip()]
    if not questions:
        raise ConfigurationError("questions.json contains no usable questions")
    return questions


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()
