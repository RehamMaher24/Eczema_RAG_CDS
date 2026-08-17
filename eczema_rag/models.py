from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Resource:
    doc_id: str
    title: str
    publisher: str
    publication_year: int
    document_type: str
    topic: str
    scope: str
    path: str
    source_reference: str
    rights_note: str
    page_count: int
    layout_hint: str
    sha256: str
    corpus_status: str
    why_credible: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Resource":
        return cls(**{key: data[key] for key in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TableRecord:
    table_index: int
    page_number: int
    rows: list[list[str]]
    markdown: str
    extraction_status: str = "extracted"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FigureReference:
    figure_index: int
    page_number: int
    object_type: str
    bbox: tuple[float, float, float, float] | None = None
    note: str = "Image/figure object detected; semantic interpretation deferred."

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.bbox is not None:
            data["bbox"] = list(self.bbox)
        return data


@dataclass(slots=True)
class ParsedPage:
    doc_id: str
    document_title: str
    publisher: str
    source_path: str
    source_reference: str
    source_sha256: str
    pdf_page_number: int
    printed_page_label: str | None
    section_path: list[str]
    text: str
    raw_text: str
    tables: list[TableRecord] = field(default_factory=list)
    figures: list[FigureReference] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extraction_method: str = "pdfplumber"

    @property
    def section(self) -> str:
        return self.section_path[-1] if self.section_path else "Front matter"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["section"] = self.section
        return data


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    chunk_hash: str
    doc_id: str
    document_name: str
    publisher: str
    publication_year: int
    document_type: str
    topic: str
    scope: str
    source_path: str
    source_reference: str
    source_sha256: str
    section: str
    section_path: list[str]
    page_start: int
    page_end: int
    printed_page_start: str | None
    printed_page_end: str | None
    chunk_index: int
    text: str
    word_count: int
    table_count: int
    figure_reference_count: int
    content_types: list[str]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalHit:
    rank: int
    score: float
    chunk: Chunk

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "chunk": self.chunk.to_dict(),
        }


@dataclass(slots=True)
class DocumentStats:
    doc_id: str
    document_name: str
    source_sha256: str
    pages_total: int = 0
    pages_with_text: int = 0
    pages_needing_ocr: int = 0
    tables_extracted: int = 0
    figure_references: int = 0
    sections_detected: int = 0
    chunks_created: int = 0
    vectors_upserted: int = 0
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PipelineResult:
    collection_name: str
    resources_processed: int
    pages_processed: int
    sections_detected: int
    chunks_created: int
    vectors_indexed: int
    tables_extracted: int
    figure_references: int
    pages_needing_ocr: int
    document_stats: list[DocumentStats]
    output_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["document_stats"] = [item.to_dict() for item in self.document_stats]
        return data
