from __future__ import annotations

import array
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .models import Chunk, RetrievalHit


class VectorStoreError(RuntimeError):
    """Raised when the local vector store cannot be initialized or queried."""


class SQLiteVectorStore:
    def __init__(self, path: Path, dimension: int) -> None:
        self.path = Path(path)
        self.dimension = int(dimension)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = NORMAL")
        self._initialize_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteVectorStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        try:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    embedding_dimension INTEGER NOT NULL,
                    model_state_json TEXT NOT NULL,
                    corpus_manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    collection_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    chunk_hash TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_norm REAL NOT NULL,
                    PRIMARY KEY (collection_name, chunk_id),
                    FOREIGN KEY (collection_name)
                        REFERENCES collections(name) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_collection_doc
                    ON chunks(collection_name, doc_id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_collection_hash
                    ON chunks(collection_name, chunk_hash);
                """
            )
            self.connection.commit()
        except sqlite3.Error as exc:
            raise VectorStoreError(f"Failed to initialize vector store: {exc}") from exc

    def replace_collection(
        self,
        collection_name: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
        model_state: dict[str, Any],
        corpus_manifest: dict[str, Any],
    ) -> int:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk and vector counts do not match")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise VectorStoreError(
                    f"Embedding dimension mismatch: expected {self.dimension}, "
                    f"received {len(vector)}"
                )

        now = datetime.now(timezone.utc).isoformat()
        try:
            with self.connection:
                existing = self.connection.execute(
                    "SELECT created_at FROM collections WHERE name = ?",
                    (collection_name,),
                ).fetchone()
                created_at = existing["created_at"] if existing else now
                self.connection.execute(
                    """
                    INSERT INTO collections (
                        name, embedding_dimension, model_state_json,
                        corpus_manifest_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET
                        embedding_dimension = excluded.embedding_dimension,
                        model_state_json = excluded.model_state_json,
                        corpus_manifest_json = excluded.corpus_manifest_json,
                        updated_at = excluded.updated_at
                    """,
                    (
                        collection_name,
                        self.dimension,
                        json.dumps(model_state, ensure_ascii=False, sort_keys=True),
                        json.dumps(corpus_manifest, ensure_ascii=False, sort_keys=True),
                        created_at,
                        now,
                    ),
                )
                self.connection.execute(
                    "DELETE FROM chunks WHERE collection_name = ?",
                    (collection_name,),
                )
                self.connection.executemany(
                    """
                    INSERT INTO chunks (
                        collection_name, chunk_id, doc_id, chunk_hash,
                        text, metadata_json, vector, vector_norm
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        self._chunk_row(collection_name, chunk, vector)
                        for chunk, vector in zip(chunks, vectors)
                    ],
                )
        except sqlite3.IntegrityError as exc:
            raise VectorStoreError(
                f"Duplicate chunk content or identifier detected: {exc}"
            ) from exc
        except sqlite3.Error as exc:
            raise VectorStoreError(f"Failed to replace collection: {exc}") from exc
        return len(chunks)

    def _chunk_row(
        self, collection_name: str, chunk: Chunk, vector: list[float]
    ) -> tuple[Any, ...]:
        metadata = chunk.to_dict()
        metadata.pop("text", None)
        norm = math.sqrt(sum(value * value for value in vector))
        return (
            collection_name,
            chunk.chunk_id,
            chunk.doc_id,
            chunk.chunk_hash,
            chunk.text,
            json.dumps(metadata, ensure_ascii=False, sort_keys=True),
            serialize_vector(vector),
            norm,
        )

    def collection_count(self, collection_name: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM chunks WHERE collection_name = ?",
            (collection_name,),
        ).fetchone()
        return int(row["count"])

    def document_counts(self, collection_name: str) -> dict[str, int]:
        rows = self.connection.execute(
            """
            SELECT doc_id, COUNT(*) AS count
            FROM chunks
            WHERE collection_name = ?
            GROUP BY doc_id
            ORDER BY doc_id
            """,
            (collection_name,),
        ).fetchall()
        return {str(row["doc_id"]): int(row["count"]) for row in rows}

    def get_model_state(self, collection_name: str) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT embedding_dimension, model_state_json
            FROM collections WHERE name = ?
            """,
            (collection_name,),
        ).fetchone()
        if row is None:
            raise VectorStoreError(f"Collection not found: {collection_name}")
        if int(row["embedding_dimension"]) != self.dimension:
            raise VectorStoreError(
                "Configured embedding dimension does not match the stored collection"
            )
        return json.loads(row["model_state_json"])

    def get_corpus_manifest(self, collection_name: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT corpus_manifest_json FROM collections WHERE name = ?",
            (collection_name,),
        ).fetchone()
        if row is None:
            raise VectorStoreError(f"Collection not found: {collection_name}")
        return json.loads(row["corpus_manifest_json"])

    def similarity_search(
        self,
        collection_name: str,
        query_vector: list[float],
        top_k: int = 5,
        minimum_score: float = 0.0,
        doc_ids: Iterable[str] | None = None,
    ) -> list[RetrievalHit]:
        if len(query_vector) != self.dimension:
            raise VectorStoreError(
                f"Query dimension mismatch: expected {self.dimension}, "
                f"received {len(query_vector)}"
            )

        parameters: list[Any] = [collection_name]
        sql = (
            "SELECT chunk_id, doc_id, text, metadata_json, vector, vector_norm "
            "FROM chunks WHERE collection_name = ?"
        )
        selected_doc_ids = sorted(set(doc_ids or []))
        if selected_doc_ids:
            placeholders = ",".join("?" for _ in selected_doc_ids)
            sql += f" AND doc_id IN ({placeholders})"
            parameters.extend(selected_doc_ids)

        rows = self.connection.execute(sql, parameters).fetchall()
        query_norm = math.sqrt(sum(value * value for value in query_vector)) or 1.0
        scored: list[tuple[float, Chunk]] = []
        for row in rows:
            vector = deserialize_vector(row["vector"])
            denominator = query_norm * (float(row["vector_norm"]) or 1.0)
            score = dot_product(query_vector, vector) / denominator
            if score < minimum_score:
                continue
            metadata = json.loads(row["metadata_json"])
            metadata["text"] = row["text"]
            chunk = Chunk(**metadata)
            scored.append((score, chunk))

        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            RetrievalHit(rank=rank, score=round(score, 6), chunk=chunk)
            for rank, (score, chunk) in enumerate(scored[:top_k], start=1)
        ]


def serialize_vector(vector: list[float]) -> bytes:
    values = array.array("f", vector)
    return values.tobytes()


def deserialize_vector(payload: bytes) -> list[float]:
    values = array.array("f")
    values.frombytes(payload)
    return values.tolist()


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
