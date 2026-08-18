from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .chunker import SectionAwareChunker
from .config import PipelineConfig, load_questions, load_resources
from .embedder import create_embedder
from .models import Chunk, DocumentStats, ParsedPage, PipelineResult, Resource
from .pdf_parser import StructuredPDFParser
from .retriever import GuidelineRetriever, citation_for_hit
from .vector_store import SQLiteVectorStore

LOGGER = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.resources, self.corpus_metadata = load_resources(
            config.root, config.resources_file
        )
        self.questions = load_questions(config.questions_file)
        self.parser = StructuredPDFParser(config.root, config.parser)
        self.chunker = SectionAwareChunker(config.chunking)
        self.embedder = create_embedder(config.embedding)

    def run(self) -> PipelineResult:
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        configure_logging(self.config)
        LOGGER.info(
            "Starting fixed-corpus ingestion for %s resources",
            len(self.resources),
        )

        all_pages: list[ParsedPage] = []
        all_chunks: list[Chunk] = []
        document_stats: list[DocumentStats] = []

        for resource in self.resources:
            LOGGER.info("Parsing %s", resource.path)
            pages = self.parser.parse(resource)
            chunks = self.chunker.create_chunks(resource, pages)
            stats = build_document_stats(resource, pages, chunks)
            all_pages.extend(pages)
            all_chunks.extend(chunks)
            document_stats.append(stats)
            LOGGER.info(
                "Parsed %s pages and created %s chunks for %s",
                len(pages),
                len(chunks),
                resource.doc_id,
            )

        if not all_chunks:
            raise RuntimeError("No chunks were generated; ingestion cannot continue")

        vectors = self.embedder.fit_transform([chunk.text for chunk in all_chunks])
        vector_store_path = Path(self.config.vector_store["path"])
        corpus_manifest = {
            **self.corpus_metadata,
            "collection_name": self.config.collection_name,
            "resources": [resource.to_dict() for resource in self.resources],
        }
        with SQLiteVectorStore(
            vector_store_path, int(self.config.embedding["dimension"])
        ) as store:
            indexed = store.replace_collection(
                self.config.collection_name,
                all_chunks,
                vectors,
                self.embedder.to_state(),
                corpus_manifest,
            )
            counts_by_document = store.document_counts(self.config.collection_name)

        for stats in document_stats:
            stats.vectors_upserted = counts_by_document.get(stats.doc_id, 0)

        output_files = self._write_outputs(
            all_pages,
            all_chunks,
            document_stats,
            corpus_manifest,
            indexed,
        )
        retrieval_files = self._run_retrieval_checks(vector_store_path)
        output_files.extend(retrieval_files)

        result = PipelineResult(
            collection_name=self.config.collection_name,
            resources_processed=len(self.resources),
            pages_processed=len(all_pages),
            sections_detected=sum(item.sections_detected for item in document_stats),
            chunks_created=len(all_chunks),
            vectors_indexed=indexed,
            tables_extracted=sum(item.tables_extracted for item in document_stats),
            figure_references=sum(item.figure_references for item in document_stats),
            pages_needing_ocr=sum(item.pages_needing_ocr for item in document_stats),
            document_stats=document_stats,
            output_files=sorted(set(output_files)),
        )
        write_json(
            self.config.output_directory / "ingestion_report.json", result.to_dict()
        )
        write_ingestion_report_markdown(
            self.config.output_directory / "ingestion_report.md", result
        )
        LOGGER.info(
            "Ingestion complete: %s pages, %s chunks, %s vectors",
            result.pages_processed,
            result.chunks_created,
            result.vectors_indexed,
        )
        return result

    def _write_outputs(
        self,
        pages: list[ParsedPage],
        chunks: list[Chunk],
        document_stats: list[DocumentStats],
        corpus_manifest: dict[str, Any],
        indexed: int,
    ) -> list[str]:
        output = self.config.output_directory
        structured_pages_path = output / "structured_pages.jsonl"
        chunks_path = output / "chunks.jsonl"
        embedding_state_path = output / "embedding_model.json"
        index_manifest_path = output / "index_manifest.json"
        source_credibility_path = output / "source_credibility.md"
        sample_pages_path = output / "sample_pages.md"

        write_jsonl(structured_pages_path, [page.to_dict() for page in pages])
        write_jsonl(chunks_path, [chunk.to_dict() for chunk in chunks])
        write_json(embedding_state_path, self.embedder.to_state())
        write_json(
            index_manifest_path,
            {
                "collection_name": self.config.collection_name,
                "vector_store": self.config.vector_store,
                "embedding": {
                    key: value
                    for key, value in self.config.embedding.items()
                    if key != "api_key"
                },
                "corpus_manifest": corpus_manifest,
                "document_chunk_counts": {
                    item.doc_id: item.chunks_created for item in document_stats
                },
                "vectors_indexed": indexed,
            },
        )
        write_source_credibility(source_credibility_path, self.resources)
        write_sample_pages(sample_pages_path, pages)
        return [
            str(structured_pages_path.relative_to(self.config.root)),
            str(chunks_path.relative_to(self.config.root)),
            str(embedding_state_path.relative_to(self.config.root)),
            str(index_manifest_path.relative_to(self.config.root)),
            str(source_credibility_path.relative_to(self.config.root)),
            str(sample_pages_path.relative_to(self.config.root)),
            str(Path(self.config.vector_store["path"]).relative_to(self.config.root)),
        ]

    def _run_retrieval_checks(self, vector_store_path: Path) -> list[str]:
        retriever = GuidelineRetriever(
            vector_store_path=vector_store_path,
            collection_name=self.config.collection_name,
            dimension=int(self.config.embedding["dimension"]),
            top_k=int(self.config.retrieval.get("top_k", 5)),
            minimum_score=float(self.config.retrieval.get("minimum_score", 0.05)),
        )
        results: list[dict[str, Any]] = []
        for question in self.questions:
            hits = retriever.search(question)
            results.append(
                {
                    "question": question,
                    "top_chunks": [
                        {
                            "rank": hit.rank,
                            "score": hit.score,
                            "citation": citation_for_hit(hit),
                            **hit.chunk.to_dict(),
                        }
                        for hit in hits
                    ],
                }
            )

        json_path = self.config.output_directory / "retrieval_results.json"
        markdown_path = self.config.output_directory / "retrieval_results.md"
        write_json(json_path, results)
        write_retrieval_results_markdown(markdown_path, results)
        return [
            str(json_path.relative_to(self.config.root)),
            str(markdown_path.relative_to(self.config.root)),
        ]


def build_document_stats(
    resource: Resource, pages: list[ParsedPage], chunks: list[Chunk]
) -> DocumentStats:
    section_paths = {tuple(chunk.section_path) for chunk in chunks}
    warnings = sorted(
        {warning for page in pages for warning in page.warnings if warning}
    )
    return DocumentStats(
        doc_id=resource.doc_id,
        document_name=resource.title,
        source_sha256=resource.sha256,
        pages_total=len(pages),
        pages_with_text=sum(bool(page.text.strip()) for page in pages),
        pages_needing_ocr=sum(
            any(warning.startswith("ocr_candidate") for warning in page.warnings)
            for page in pages
        ),
        tables_extracted=sum(len(page.tables) for page in pages),
        figure_references=sum(len(page.figures) for page in pages),
        sections_detected=len(section_paths),
        chunks_created=len(chunks),
        warnings=warnings,
    )


def configure_logging(config: PipelineConfig) -> None:
    log_path = Path(config.logging["file"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level_name = str(config.logging.get("level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in root_logger.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    if not any(
        isinstance(handler, logging.StreamHandler)
        and not isinstance(handler, logging.FileHandler)
        for handler in root_logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_source_credibility(path: Path, resources: list[Resource]) -> None:
    lines = [
        "# Fixed Corpus Provenance",
        "",
        "The pipeline indexes the seven supplied PDFs exactly as provided. It verifies each file against the SHA-256 value in `config/resources.json` and does not download, replace, or edit source documents.",
        "",
    ]
    for resource in resources:
        lines.extend(
            [
                f"## {resource.title}",
                "",
                f"- Document ID: `{resource.doc_id}`",
                f"- File: `{resource.path}`",
                f"- Publisher stated in the PDF: {resource.publisher}",
                f"- Publication year: {resource.publication_year}",
                f"- Source reference: {resource.source_reference}",
                f"- SHA-256: `{resource.sha256}`",
                f"- Corpus status: `{resource.corpus_status}`",
                f"- Provenance note: {resource.why_credible}",
                f"- Rights note: {resource.rights_note}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sample_pages(path: Path, pages: list[ParsedPage]) -> None:
    lines = ["# Parsed Page Samples", ""]
    seen: set[str] = set()
    for page in pages:
        if page.doc_id in seen or len(page.text.split()) < 80:
            continue
        seen.add(page.doc_id)
        lines.extend(
            [
                f"## {page.document_title}",
                "",
                f"- PDF page: {page.pdf_page_number}",
                f"- Printed page: {page.printed_page_label or 'not detected'}",
                f"- Section: {' > '.join(page.section_path)}",
                f"- Tables detected: {len(page.tables)}",
                f"- Figure references: {len(page.figures)}",
                "",
                "```text",
                page.text[:1600],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_retrieval_results_markdown(
    path: Path, results: list[dict[str, Any]]
) -> None:
    lines = ["# Retrieval Validation", ""]
    for result in results:
        lines.extend([f"## {result['question']}", ""])
        if not result["top_chunks"]:
            lines.extend(["> No chunk met the configured score threshold.", ""])
            continue
        for chunk in result["top_chunks"]:
            lines.extend(
                [
                    f"### Rank {chunk['rank']} — score {chunk['score']}",
                    "",
                    f"- Citation: {chunk['citation']}",
                    f"- Document ID: `{chunk['doc_id']}`",
                    f"- Section: `{' > '.join(chunk['section_path'])}`",
                    f"- PDF pages: {chunk['page_start']}-{chunk['page_end']}",
                    "",
                    "```text",
                    chunk["text"][:1000],
                    "```",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ingestion_report_markdown(path: Path, result: PipelineResult) -> None:
    lines = [
        "# Ingestion Validation Report",
        "",
        f"Collection: `{result.collection_name}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Documents processed | {result.resources_processed} |",
        f"| Pages processed | {result.pages_processed} |",
        f"| Sections detected | {result.sections_detected} |",
        f"| Chunks created | {result.chunks_created} |",
        f"| Vectors indexed | {result.vectors_indexed} |",
        f"| Tables extracted | {result.tables_extracted} |",
        f"| Figure references recorded | {result.figure_references} |",
        f"| Pages flagged for OCR | {result.pages_needing_ocr} |",
        "",
        "## Per-document results",
        "",
        "| Document ID | Pages | Sections | Chunks | Vectors | Tables | Figures | OCR candidates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in result.document_stats:
        lines.append(
            f"| `{item.doc_id}` | {item.pages_total} | {item.sections_detected} | "
            f"{item.chunks_created} | {item.vectors_upserted} | "
            f"{item.tables_extracted} | {item.figure_references} | "
            f"{item.pages_needing_ocr} |"
        )
    lines.extend(
        [
            "",
            "## Known Day 1 limitations",
            "",
            "The parser detects pages that may need OCR but does not perform OCR. Image objects are recorded as page-level references but are not semantically interpreted. Tables are extracted heuristically and preserved as Markdown when detection succeeds; complex multi-column or spanning tables may need manual review. The local hashing embedding model is deterministic and dependency-light, but a production deployment may replace it through the embedding abstraction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
