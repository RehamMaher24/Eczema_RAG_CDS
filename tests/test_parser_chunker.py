from __future__ import annotations

from pathlib import Path

from eczema_rag.chunker import SectionAwareChunker
from eczema_rag.config import PipelineConfig, load_resources
from eczema_rag.pdf_parser import StructuredPDFParser

ROOT = Path(__file__).resolve().parents[1]


def test_nice_pdf_parsing_preserves_page_and_source_provenance() -> None:
    config = PipelineConfig.load(ROOT, ROOT / "config" / "pipeline.json")
    resources, _ = load_resources(ROOT, config.resources_file)
    resource = next(item for item in resources if item.doc_id.startswith("nice_cg57"))

    pages = StructuredPDFParser(ROOT, config.parser).parse(resource)

    assert len(pages) == resource.page_count == 31
    assert sum(bool(page.text.strip()) for page in pages) >= 30
    assert all(page.source_sha256 == resource.sha256 for page in pages)
    assert all(page.pdf_page_number >= 1 for page in pages)
    assert any(page.section_path for page in pages)


def test_section_aware_chunks_have_traceable_metadata() -> None:
    config = PipelineConfig.load(ROOT, ROOT / "config" / "pipeline.json")
    resources, _ = load_resources(ROOT, config.resources_file)
    resource = next(item for item in resources if item.doc_id.startswith("nice_cg57"))
    parser = StructuredPDFParser(ROOT, config.parser)
    pages = parser.parse(resource)

    chunks = SectionAwareChunker(config.chunking).create_chunks(resource, pages)

    assert chunks
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert len({chunk.chunk_hash for chunk in chunks}) == len(chunks)
    assert all(chunk.doc_id == resource.doc_id for chunk in chunks)
    assert all(chunk.source_sha256 == resource.sha256 for chunk in chunks)
    assert all(chunk.page_start <= chunk.page_end for chunk in chunks)
    assert all(chunk.section_path for chunk in chunks)
    assert all(chunk.text.startswith("Section:") for chunk in chunks)
    assert max(chunk.word_count for chunk in chunks) <= 850
