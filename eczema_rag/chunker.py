from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .models import Chunk, ParsedPage, Resource
from .text_utils import heading_level, split_paragraphs, update_section_path, word_count


@dataclass(slots=True)
class ContentUnit:
    text: str
    page_number: int
    printed_page_label: str | None
    section_path: list[str]
    content_type: str
    table_count: int = 0
    figure_reference_count: int = 0
    warnings: list[str] = field(default_factory=list)


class SectionAwareChunker:
    def __init__(self, config: dict[str, Any]) -> None:
        self.target_words = int(config.get("target_words", 500))
        self.maximum_words = int(config.get("maximum_words", 800))
        self.minimum_words = int(config.get("minimum_words", 80))
        self.overlap_words = int(config.get("overlap_words", 80))
        self.preserve_tables = bool(config.get("preserve_tables", True))
        self.include_section_context = bool(
            config.get("include_section_context", True)
        )
        if self.maximum_words < self.target_words:
            raise ValueError("maximum_words must be >= target_words")
        if self.overlap_words >= self.maximum_words:
            raise ValueError("overlap_words must be smaller than maximum_words")

    def create_chunks(
        self, resource: Resource, pages: list[ParsedPage]
    ) -> list[Chunk]:
        raw_units = self._build_units(pages)
        units = [piece for unit in raw_units for piece in self._split_oversized_unit(unit)]
        chunks: list[Chunk] = []
        current_units: list[ContentUnit] = []
        current_section: tuple[str, ...] | None = None

        for unit in units:
            unit_section = tuple(unit.section_path)
            if current_units and unit_section != current_section:
                self._flush(resource, chunks, current_units)
                current_units = []

            current_section = unit_section
            projected = word_count(self._render_units(current_units + [unit]))
            if current_units and projected > self.maximum_words:
                previous_units = list(current_units)
                self._flush(resource, chunks, current_units)
                current_units = self._overlap_units(previous_units)

            current_units.append(unit)
            current_words = word_count(self._render_units(current_units))
            if current_words >= self.target_words:
                self._flush(resource, chunks, current_units)
                current_units = self._overlap_units(current_units)

        if current_units:
            self._flush(resource, chunks, current_units)

        return self._merge_short_tail_chunks(chunks)

    def _build_units(self, pages: list[ParsedPage]) -> list[ContentUnit]:
        units: list[ContentUnit] = []
        section_path = ["Front matter"]
        for page in pages:
            page_has_text_unit = False
            for paragraph in split_paragraphs(page.text):
                detected = heading_level(paragraph)
                if detected:
                    level, heading = detected
                    section_path = update_section_path(section_path, level, heading)
                    continue
                if not paragraph.strip():
                    continue
                units.append(
                    ContentUnit(
                        text=paragraph.strip(),
                        page_number=page.pdf_page_number,
                        printed_page_label=page.printed_page_label,
                        section_path=list(section_path),
                        content_type="text",
                        figure_reference_count=len(page.figures),
                        warnings=list(page.warnings),
                    )
                )
                page_has_text_unit = True

            if self.preserve_tables:
                for table in page.tables:
                    if not table.markdown:
                        continue
                    units.append(
                        ContentUnit(
                            text=(
                                f"[Table {table.table_index} on PDF page "
                                f"{page.pdf_page_number}]\n{table.markdown}"
                            ),
                            page_number=page.pdf_page_number,
                            printed_page_label=page.printed_page_label,
                            section_path=list(section_path),
                            content_type="table",
                            table_count=1,
                            figure_reference_count=len(page.figures),
                            warnings=list(page.warnings),
                        )
                    )

            if not page_has_text_unit and not page.tables and page.figures:
                units.append(
                    ContentUnit(
                        text=(
                            f"[Figure reference on PDF page {page.pdf_page_number}; "
                            "image interpretation deferred]"
                        ),
                        page_number=page.pdf_page_number,
                        printed_page_label=page.printed_page_label,
                        section_path=list(section_path),
                        content_type="figure_reference",
                        figure_reference_count=len(page.figures),
                        warnings=list(page.warnings),
                    )
                )
        return units

    def _split_oversized_unit(self, unit: ContentUnit) -> list[ContentUnit]:
        context_words = (
            word_count("Section: " + " > ".join(unit.section_path))
            if self.include_section_context
            else 0
        )
        body_limit = max(50, self.maximum_words - context_words - 5)
        if word_count(unit.text) <= body_limit:
            return [unit]
        words = unit.text.split()
        step = max(1, body_limit - min(self.overlap_words, body_limit // 2))
        pieces: list[ContentUnit] = []
        for start in range(0, len(words), step):
            part = words[start : start + body_limit]
            if not part:
                continue
            pieces.append(
                ContentUnit(
                    text=" ".join(part),
                    page_number=unit.page_number,
                    printed_page_label=unit.printed_page_label,
                    section_path=list(unit.section_path),
                    content_type=unit.content_type,
                    table_count=unit.table_count if start == 0 else 0,
                    figure_reference_count=unit.figure_reference_count,
                    warnings=sorted(set(unit.warnings + ["oversized_unit_split"])),
                )
            )
            if start + body_limit >= len(words):
                break
        return pieces

    def _flush(
        self,
        resource: Resource,
        chunks: list[Chunk],
        units: list[ContentUnit],
    ) -> None:
        if not units:
            return
        text = self._render_units(units)
        if not text.strip():
            return
        index = len(chunks) + 1
        page_numbers = [unit.page_number for unit in units]
        page_provenance = ",".join(str(number) for number in page_numbers)
        digest = hashlib.sha256(
            (
                resource.sha256
                + "\n"
                + str(index)
                + "\n"
                + page_provenance
                + "\n"
                + " > ".join(units[-1].section_path)
                + "\n"
                + text
            ).encode("utf-8")
        ).hexdigest()
        chunk_id = f"{resource.doc_id}:chunk:{index:04d}:{digest[:12]}"
        printed_labels = [
            unit.printed_page_label
            for unit in units
            if unit.printed_page_label is not None
        ]
        warnings = sorted(
            {
                warning
                for unit in units
                for warning in unit.warnings
                if warning
            }
        )
        content_types = sorted({unit.content_type for unit in units})
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                chunk_hash=digest,
                doc_id=resource.doc_id,
                document_name=resource.title,
                publisher=resource.publisher,
                publication_year=resource.publication_year,
                document_type=resource.document_type,
                topic=resource.topic,
                scope=resource.scope,
                source_path=resource.path,
                source_reference=resource.source_reference,
                source_sha256=resource.sha256,
                section=units[-1].section_path[-1],
                section_path=list(units[-1].section_path),
                page_start=min(page_numbers),
                page_end=max(page_numbers),
                printed_page_start=printed_labels[0] if printed_labels else None,
                printed_page_end=printed_labels[-1] if printed_labels else None,
                chunk_index=index,
                text=text,
                word_count=word_count(text),
                table_count=sum(unit.table_count for unit in units),
                figure_reference_count=max(
                    (unit.figure_reference_count for unit in units), default=0
                ),
                content_types=content_types,
                warnings=warnings,
            )
        )

    def _render_units(self, units: list[ContentUnit]) -> str:
        if not units:
            return ""
        body = "\n\n".join(unit.text.strip() for unit in units if unit.text.strip())
        if not self.include_section_context:
            return body
        section_context = " > ".join(units[-1].section_path)
        return f"Section: {section_context}\n\n{body}".strip()

    def _overlap_units(self, units: list[ContentUnit]) -> list[ContentUnit]:
        if self.overlap_words <= 0 or not units:
            return []
        words: list[str] = []
        for unit in units:
            words.extend(unit.text.split())
        overlap_text = " ".join(words[-self.overlap_words :]).strip()
        if not overlap_text:
            return []
        last = units[-1]
        return [
            ContentUnit(
                text=overlap_text,
                page_number=last.page_number,
                printed_page_label=last.printed_page_label,
                section_path=list(last.section_path),
                content_type="context_overlap",
                warnings=list(last.warnings),
            )
        ]

    def _merge_short_tail_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        if len(chunks) < 2:
            return chunks
        result: list[Chunk] = []
        for chunk in chunks:
            if (
                result
                and chunk.word_count < self.minimum_words
                and result[-1].doc_id == chunk.doc_id
                and result[-1].section_path == chunk.section_path
                and result[-1].word_count + chunk.word_count <= self.maximum_words
            ):
                previous = result.pop()
                merged_text = f"{previous.text}\n\n{chunk.text}"
                digest = hashlib.sha256(
                    (
                        previous.source_sha256
                        + "\n"
                        + str(previous.chunk_index)
                        + "\n"
                        + f"{previous.page_start}-{chunk.page_end}"
                        + "\n"
                        + merged_text
                    ).encode("utf-8")
                ).hexdigest()
                previous.text = merged_text
                previous.chunk_hash = digest
                previous.chunk_id = (
                    f"{previous.doc_id}:chunk:{previous.chunk_index:04d}:{digest[:12]}"
                )
                previous.page_end = chunk.page_end
                previous.printed_page_end = chunk.printed_page_end
                previous.word_count = word_count(merged_text)
                previous.table_count += chunk.table_count
                previous.figure_reference_count = max(
                    previous.figure_reference_count, chunk.figure_reference_count
                )
                previous.content_types = sorted(
                    set(previous.content_types + chunk.content_types)
                )
                previous.warnings = sorted(set(previous.warnings + chunk.warnings))
                result.append(previous)
            else:
                result.append(chunk)
        for index, chunk in enumerate(result, start=1):
            chunk.chunk_index = index
        return result
