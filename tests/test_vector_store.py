from __future__ import annotations

from pathlib import Path

from eczema_rag.embedder import LocalHashingEmbedder
from eczema_rag.models import Chunk
from eczema_rag.vector_store import SQLiteVectorStore


def make_chunk(index: int, text: str, topic: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc:chunk:{index:04d}:hash{index}",
        chunk_hash=f"hash-{index}",
        doc_id="doc",
        document_name="Test Guideline",
        publisher="Test Publisher",
        publication_year=2026,
        document_type="clinical_guideline",
        topic=topic,
        scope="Test scope",
        source_path="data/raw/test.pdf",
        source_reference="https://example.org/test",
        source_sha256="a" * 64,
        section="Treatment",
        section_path=["Treatment"],
        page_start=index,
        page_end=index,
        printed_page_start=str(index),
        printed_page_end=str(index),
        chunk_index=index,
        text=text,
        word_count=len(text.split()),
        table_count=0,
        figure_reference_count=0,
        content_types=["text"],
    )


def test_local_embeddings_are_deterministic_and_dimension_safe() -> None:
    texts = [
        "topical corticosteroid treatment for atopic dermatitis",
        "patch testing for allergic contact dermatitis",
    ]
    first = LocalHashingEmbedder(dimension=64)
    second = LocalHashingEmbedder(dimension=64)

    vectors_one = first.fit_transform(texts)
    vectors_two = second.fit_transform(texts)

    assert vectors_one == vectors_two
    assert all(len(vector) == 64 for vector in vectors_one)
    assert first.embed_query("patch testing") == second.embed_query("patch testing")


def test_sqlite_collection_replacement_is_idempotent_and_retrievable(
    tmp_path: Path,
) -> None:
    chunks = [
        make_chunk(1, "topical corticosteroid eczema treatment", "atopic dermatitis"),
        make_chunk(2, "patch testing allergic contact dermatitis", "contact dermatitis"),
    ]
    embedder = LocalHashingEmbedder(dimension=128)
    vectors = embedder.fit_transform([chunk.text for chunk in chunks])
    database = tmp_path / "vectors.sqlite3"

    with SQLiteVectorStore(database, 128) as store:
        inserted = store.replace_collection(
            "test",
            chunks,
            vectors,
            embedder.to_state(),
            {"corpus_name": "test"},
        )
        assert inserted == 2
        assert store.collection_count("test") == 2

        store.replace_collection(
            "test",
            chunks,
            vectors,
            embedder.to_state(),
            {"corpus_name": "test"},
        )
        assert store.collection_count("test") == 2

        query = embedder.embed_query("allergic contact patch test")
        hits = store.similarity_search("test", query, top_k=1)

    assert hits
    assert hits[0].chunk.chunk_id == chunks[1].chunk_id
    assert hits[0].chunk.page_start == 2
    assert hits[0].score > 0
