from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

from .text_utils import tokenize

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "what",
    "when",
    "which",
    "with",
}


@dataclass(slots=True)
class LocalHashingEmbedder:
    dimension: int = 768
    model: str = "blake2b-feature-hashing-v1"
    normalize: bool = True
    batch_size: int = 64
    document_count: int = 0
    idf: dict[str, float] = field(default_factory=dict)

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        features = [Counter(self._features(text)) for text in texts]
        document_frequency: Counter[str] = Counter()
        for row in features:
            document_frequency.update(row.keys())

        self.document_count = len(texts)
        self.idf = {
            feature: math.log((self.document_count + 1) / (frequency + 1)) + 1.0
            for feature, frequency in document_frequency.items()
        }
        return [self._vectorize(row) for row in features]

    def transform(self, texts: Iterable[str]) -> list[list[float]]:
        if self.document_count <= 0 or not self.idf:
            raise RuntimeError("Embedder must be fitted or restored before transform()")
        return [self._vectorize(Counter(self._features(text))) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.transform([text])[0]

    def _features(self, text: str) -> list[str]:
        tokens = [
            token
            for token in tokenize(text)
            if token not in STOP_WORDS and len(token) >= 2
        ]
        features = [f"u:{token}" for token in tokens]
        features.extend(
            f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:])
        )
        for token in tokens:
            if len(token) >= 6:
                features.extend(
                    f"c:{token[index:index + 4]}"
                    for index in range(0, len(token) - 3)
                )
        return features

    def _vectorize(self, features: Counter[str]) -> list[float]:
        vector = [0.0] * self.dimension
        default_idf = math.log((self.document_count + 1) / 1.0) + 1.0
        for feature, count in features.items():
            weight = (1.0 + math.log(count)) * self.idf.get(feature, default_idf)
            index, sign = self._hash_feature(feature)
            vector[index] += sign * weight
        if self.normalize:
            length = math.sqrt(sum(value * value for value in vector))
            if length:
                vector = [value / length for value in vector]
        return vector

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:8], "big") % self.dimension
        sign = 1.0 if digest[8] & 1 else -1.0
        return index, sign

    def to_state(self) -> dict[str, Any]:
        return {
            "provider": "local_hashing",
            "model": self.model,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "document_count": self.document_count,
            "idf": self.idf,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "LocalHashingEmbedder":
        embedder = cls(
            dimension=int(state["dimension"]),
            model=str(state.get("model", "blake2b-feature-hashing-v1")),
            normalize=bool(state.get("normalize", True)),
            batch_size=int(state.get("batch_size", 64)),
        )
        embedder.document_count = int(state["document_count"])
        embedder.idf = {
            str(key): float(value) for key, value in state["idf"].items()
        }
        return embedder


def create_embedder(config: dict[str, Any]) -> Any:
    provider = str(config.get("provider", ""))
    if provider == "local_hashing":
        return LocalHashingEmbedder(
            dimension=int(config["dimension"]),
            model=str(config["model"]),
            normalize=bool(config.get("normalize", True)),
            batch_size=int(config.get("batch_size", 64)),
        )
    if provider == "gemini":
        from .GeminiEmbedder import GeminiEmbedder

        return GeminiEmbedder(
            dimension=int(config["dimension"]),
            model=str(config.get("model", "gemini-embedding-001")),
            normalize=bool(config.get("normalize", True)),
            batch_size=int(config.get("batch_size", 32)),
            document_task_type=str(
                config.get("document_task_type", "RETRIEVAL_DOCUMENT")
            ),
            query_task_type=str(config.get("query_task_type", "RETRIEVAL_QUERY")),
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def embedder_from_state(state: dict[str, Any]) -> Any:
    provider = str(state.get("provider", ""))
    if provider == "local_hashing":
        return LocalHashingEmbedder.from_state(state)
    if provider == "gemini":
        from .GeminiEmbedder import GeminiEmbedder

        return GeminiEmbedder.from_state(state)
    raise ValueError(f"Unsupported stored embedding provider: {provider}")


