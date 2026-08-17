from __future__ import annotations

from eczema_rag.models import Chunk, RetrievalHit
from eczema_rag.retriever import (
    canonical_term,
    clean_section_path,
    is_reference_chunk,
    retrieval_adjustment,
)
from eczema_rag.text_utils import heading_level, update_section_path


def make_hit(section_path: list[str], text: str, word_count: int = 120) -> RetrievalHit:
    chunk = Chunk(
        chunk_id="doc:chunk:0001:test",
        chunk_hash="test-hash",
        doc_id="doc",
        document_name="Test Guideline",
        publisher="Test Publisher",
        publication_year=2026,
        document_type="clinical_guideline",
        topic="eczema",
        scope="test",
        source_path="data/raw/test.pdf",
        source_reference="https://example.org/test",
        source_sha256="a" * 64,
        section=section_path[-1],
        section_path=section_path,
        page_start=1,
        page_end=1,
        printed_page_start="1",
        printed_page_end="1",
        chunk_index=1,
        text=text,
        word_count=word_count,
        table_count=0,
        figure_reference_count=0,
        content_types=["text"],
    )
    return RetrievalHit(rank=1, score=0.5, chunk=chunk)


def test_numbered_bibliography_entry_is_not_a_heading() -> None:
    line = (
        "131 Cramer C, Link E, Bauer CP, Hoffmann U, von Berg A, et al. "
        "Allergy 2011;66:68-75."
    )
    assert heading_level(line) is None


def test_numbered_guideline_heading_is_preserved() -> None:
    assert heading_level("1.2 Assessing severity and quality of life") == (
        2,
        "1.2 Assessing severity and quality of life",
    )
    assert heading_level("1.4.1.12 Explain trigger factors") == (
        3,
        "1.4.1.12 Explain trigger factors",
    )
    assert heading_level("6.0 Diagnostic tests") == (1, "6.0 Diagnostic tests")


def test_false_numbered_and_journal_headings_are_rejected() -> None:
    assert heading_level("116 Drake LA, Fallon JD, Sober A. Relief of pruritus") is None
    assert heading_level("19 September 2016 tact dermatitis") is None
    assert heading_level("FEBRUARY 2014") is None
    assert heading_level("Recommendations ........................ 6") is None


def test_section_path_does_not_duplicate_existing_heading() -> None:
    path = ["1 Diagnosis", "1.2 Assessment"]
    assert update_section_path(path, 2, "1.2 Assessment") == path


def test_reference_section_is_filtered() -> None:
    hit = make_hit(
        ["References"],
        "Smith A, et al. J Dermatol 2014;70:338-351. Jones B, et al. "
        "Allergy 2015;3:10-20. doi: 10.1000/test.",
    )
    assert is_reference_chunk(hit)


def test_clinical_term_normalization() -> None:
    assert canonical_term("children") == "child"
    assert canonical_term("referred") == "refer"
    assert canonical_term("testing") == "test"
    assert canonical_term("flares") == "flare"


def test_hybrid_reranking_prefers_matching_clinical_domain() -> None:
    atopic = make_hit(["Diagnosis"], "Atopic eczema diagnostic criteria and assessment")
    atopic.chunk.topic = "Atopic dermatitis diagnosis"
    atopic.chunk.document_name = "Atopic dermatitis guideline"
    contact = make_hit(["Diagnosis"], "Contact dermatitis diagnostic tests")
    contact.chunk.topic = "Contact dermatitis"
    contact.chunk.document_name = "Contact dermatitis guideline"

    assert retrieval_adjustment("diagnostic criteria for atopic eczema", atopic) > retrieval_adjustment(
        "diagnostic criteria for atopic eczema", contact
    )


def test_clean_section_path_removes_toc_and_journal_headers() -> None:
    path = [
        "5 Education and adherence ................................ 27",
        "Volume 71",
        "1.2 Assessment",
        "1.2 Assessment",
    ]
    assert clean_section_path(path) == ["1.2 Assessment"]
