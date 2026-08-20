import pytest
from pydantic import ValidationError

from eczema_rag.generator import ClaimEvidence, GeneratedClaim
from eczema_rag.grounding import validate_claims
from eczema_rag.models import Chunk, RetrievalHit


def make_hit() -> RetrievalHit:
    chunk = Chunk(
        chunk_id="test-doc:chunk:0001:abc123",
        chunk_hash="abc123",
        doc_id="test-doc",
        document_name="Test Guideline",
        publisher="Test Publisher",
        publication_year=2026,
        document_type="clinical_guideline",
        topic="eczema",
        scope="Test scope",
        source_path="data/raw/test.pdf",
        source_reference="https://example.org/test",
        source_sha256="a" * 64,
        section="Treatment",
        section_path=["Treatment"],
        page_start=1,
        page_end=1,
        printed_page_start="1",
        printed_page_end="1",
        chunk_index=1,
        text=(
            "Systemic corticosteroids should generally be avoided because "
            "of short-term and long-term adverse effects."
        ),
        word_count=12,
        table_count=0,
        figure_reference_count=0,
        content_types=["text"],
    )
    return RetrievalHit(rank=1, score=0.8, chunk=chunk)


def test_generated_claim_requires_at_least_one_evidence_item() -> None:
    with pytest.raises(ValidationError):
        GeneratedClaim(
            claim="Systemic corticosteroids are safe.",
            evidence=[],
        )


def test_rejects_claim_with_unretrieved_chunk_id() -> None:
    hits = [make_hit()]

    claims = [
        GeneratedClaim(
            claim="Long-term oral corticosteroids are the safest treatment.",
            evidence=[
                ClaimEvidence(
                    chunk_id="missing:chunk:9999",
                    quote="This fabricated quote is long enough.",
                )
            ],
        )
    ]

    result = validate_claims(claims, hits)

    assert result.valid is False
    assert result.unsupported_claims
    assert result.citation_errors


def test_rejects_quote_not_found_in_retrieved_chunk() -> None:
    hits = [make_hit()]

    claims = [
        GeneratedClaim(
            claim="A fabricated clinical claim.",
            evidence=[
                ClaimEvidence(
                    chunk_id=hits[0].chunk.chunk_id,
                    quote="This quote does not appear in the guideline text.",
                )
            ],
        )
    ]

    result = validate_claims(claims, hits)

    assert result.valid is False
    assert result.unsupported_claims
    assert result.citation_errors


def test_accepts_exact_quote_from_retrieved_chunk() -> None:
    hits = [make_hit()]

    claims = [
        GeneratedClaim(
            claim="Systemic corticosteroids should generally be avoided.",
            evidence=[
                ClaimEvidence(
                    chunk_id=hits[0].chunk.chunk_id,
                    quote="Systemic corticosteroids should generally be avoided",
                )
            ],
        )
    ]

    result = validate_claims(claims, hits)

    assert result.valid is True
    assert result.unsupported_claims == []
    assert result.citation_errors == []