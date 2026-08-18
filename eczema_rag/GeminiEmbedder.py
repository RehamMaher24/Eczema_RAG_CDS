from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from dotenv import load_dotenv
from google import genai
from google.genai import types
import time  # <-- Add this import

@dataclass(slots=True)
class GeminiEmbedder:
    """Gemini retrieval embeddings with separate document and query task types."""

    dimension: int = 768
    model: str = "gemini-embedding-001"
    normalize: bool = True
    batch_size: int = 32
    document_task_type: str = "RETRIEVAL_DOCUMENT"
    query_task_type: str = "RETRIEVAL_QUERY"
    client: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment")
        self.client = genai.Client(api_key=api_key)

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, self.document_task_type)

    def transform(self, texts: Iterable[str]) -> list[list[float]]:
        return self._embed(list(texts), self.document_task_type)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], self.query_task_type)[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            
            # --- START OF NEW RETRY LOGIC ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type=task_type,
                            output_dimensionality=self.dimension,
                        ),
                    )
                    break  # Success! Break out of the retry loop
                except Exception as exc:
                    # Check if it's a rate limit error (429) and we haven't run out of retries
                    if "429" in str(exc) and attempt < max_retries - 1:
                        print(f"Rate limit hit at batch {start}. Waiting 60 seconds before retry...")
                        time.sleep(60)  # Pause the script for 60 seconds
                    else:
                        raise RuntimeError(
                            f"Gemini embedding request failed for batch starting at {start}: {exc}"
                        ) from exc
            # --- END OF NEW RETRY LOGIC ---

            batch_vectors = [
                [float(value) for value in embedding.values]
                for embedding in response.embeddings
            ]
            if len(batch_vectors) != len(batch):
                raise RuntimeError(
                    "Gemini returned a different number of embeddings than inputs"
                )
            for vector in batch_vectors:
                if len(vector) != self.dimension:
                    raise RuntimeError(
                        f"Gemini returned {len(vector)} dimensions; expected {self.dimension}"
                    )
                vectors.append(self._normalize(vector))
        return vectors

    def _normalize(self, vector: list[float]) -> list[float]:
        if not self.normalize:
            return vector
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector] if length else vector

    def to_state(self) -> dict[str, Any]:
        return {
            "provider": "gemini",
            "model": self.model,
            "dimension": self.dimension,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
            "document_task_type": self.document_task_type,
            "query_task_type": self.query_task_type,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "GeminiEmbedder":
        return cls(
            dimension=int(state["dimension"]),
            model=str(state.get("model", "gemini-embedding-001")),
            normalize=bool(state.get("normalize", True)),
            batch_size=int(state.get("batch_size", 32)),
            document_task_type=str(
                state.get("document_task_type", "RETRIEVAL_DOCUMENT")
            ),
            query_task_type=str(state.get("query_task_type", "RETRIEVAL_QUERY")),
        )