from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import pdfplumber
from pdfplumber.pdf import PDF

from .config import resolve_path, sha256_file
from .models import FigureReference, ParsedPage, Resource, TableRecord
from .text_utils import (
    detect_printed_page_label,
    detect_repeated_margin_lines,
    heading_level,
    normalize_line,
    normalize_text,
    strip_repeated_margin_lines,
    update_section_path,
)

LOGGER = logging.getLogger(__name__)
TABLE_CAPTION_RE = re.compile(r"\btable\s+(?:[ivxlcdm]+|\d+)\b", re.IGNORECASE)


class PDFParsingError(RuntimeError):
    """Raised when a source PDF cannot be parsed safely."""


class StructuredPDFParser:
    def __init__(self, root: Path, config: dict[str, Any]) -> None:
        self.root = root
        self.extract_tables = bool(config.get("extract_tables", True))
        self.record_image_references = bool(
            config.get("record_image_references", True)
        )
        self.ocr_mode = str(config.get("ocr_mode", "detect_only"))
        self.minimum_page_text_characters = int(
            config.get("minimum_page_text_characters", 80)
        )
        self.header_footer_repeat_ratio = float(
            config.get("header_footer_repeat_ratio", 0.6)
        )
        self.header_footer_margin_lines = int(
            config.get("header_footer_margin_lines", 3)
        )

    def parse(self, resource: Resource) -> list[ParsedPage]:
        pdf_path = resolve_path(self.root, resource.path)
        self._validate_source(pdf_path, resource)

        try:
            with pdfplumber.open(pdf_path) as pdf:
                return self._parse_open_pdf(pdf, pdf_path, resource)
        except PDFParsingError:
            raise
        except Exception as exc:  # pragma: no cover - library-specific failures
            raise PDFParsingError(f"Failed to parse {pdf_path}: {exc}") from exc

    def _parse_open_pdf(
        self, pdf: PDF, pdf_path: Path, resource: Resource
    ) -> list[ParsedPage]:
        if not pdf.pages:
            raise PDFParsingError(f"PDF contains no pages: {pdf_path}")

        page_count_warning = None
        if resource.page_count and len(pdf.pages) != resource.page_count:
            page_count_warning = (
                f"Manifest page_count={resource.page_count}, actual={len(pdf.pages)}"
            )
            LOGGER.warning("%s: %s", resource.doc_id, page_count_warning)

        raw_texts = [self._extract_page_text(page, resource) for page in pdf.pages]
        repeated_lines = detect_repeated_margin_lines(
            raw_texts,
            margin_lines=self.header_footer_margin_lines,
            repeat_ratio=self.header_footer_repeat_ratio,
        )

        parsed_pages: list[ParsedPage] = []
        section_path = ["Front matter"]
        for page_number, (page, raw_text) in enumerate(
            zip(pdf.pages, raw_texts), start=1
        ):
            cleaned_text = strip_repeated_margin_lines(raw_text, repeated_lines)
            section_path = self._update_sections(cleaned_text, section_path)
            warnings: list[str] = []
            if page_count_warning:
                warnings.append(page_count_warning)
            if len(cleaned_text) < 20 and (not cleaned_text or bool(page.images)):
                warnings.append(
                    "ocr_candidate: insufficient native text extracted from this page"
                )
            elif len(cleaned_text) < self.minimum_page_text_characters:
                warnings.append("short_text_page: review if content appears incomplete")
            if not cleaned_text:
                warnings.append("empty_page_text")

            table_caption_detected = bool(TABLE_CAPTION_RE.search(raw_text))
            tables = self._extract_tables(page, page_number, raw_text, resource)
            if table_caption_detected and not tables:
                warnings.append(
                    "table_caption_detected_but_structure_not_extracted"
                )
            figures = self._extract_figures(page, page_number)
            if figures:
                warnings.append(
                    "figure_semantics_deferred: references recorded without image interpretation"
                )

            parsed_pages.append(
                ParsedPage(
                    doc_id=resource.doc_id,
                    document_title=resource.title,
                    publisher=resource.publisher,
                    source_path=resource.path,
                    source_reference=resource.source_reference,
                    source_sha256=resource.sha256,
                    pdf_page_number=page_number,
                    printed_page_label=detect_printed_page_label(raw_text),
                    section_path=list(section_path),
                    text=cleaned_text,
                    raw_text=normalize_text(raw_text),
                    tables=tables,
                    figures=figures,
                    warnings=warnings,
                    extraction_method=(
                        "pdfplumber-layout"
                        if resource.layout_hint == "two_column"
                        else "pdfplumber"
                    ),
                )
            )
        return parsed_pages

    def _validate_source(self, pdf_path: Path, resource: Resource) -> None:
        if not pdf_path.exists():
            raise PDFParsingError(f"Source PDF is missing: {pdf_path}")
        if pdf_path.stat().st_size == 0:
            raise PDFParsingError(f"Source PDF is empty: {pdf_path}")
        actual_hash = sha256_file(pdf_path)
        if actual_hash.lower() != resource.sha256.lower():
            raise PDFParsingError(
                f"Source hash mismatch for {pdf_path.name}; expected {resource.sha256}, "
                f"found {actual_hash}. The fixed corpus must remain unchanged."
            )

    @staticmethod
    def _extract_page_text(page: Any, resource: Resource) -> str:
        if resource.layout_hint == "two_column":
            text = page.extract_text(
                x_tolerance=2,
                y_tolerance=3,
                layout=True,
                x_density=7.25,
                y_density=13,
            )
            if text and len(text.strip()) >= 40:
                return text
        return page.extract_text(x_tolerance=2, y_tolerance=3) or ""

    @staticmethod
    def _update_sections(text: str, current_path: list[str]) -> list[str]:
        section_path = list(current_path)
        for line in text.splitlines():
            detected = heading_level(line)
            if detected:
                level, heading = detected
                section_path = update_section_path(section_path, level, heading)
        return section_path

    def _extract_tables(
        self,
        page: Any,
        page_number: int,
        raw_text: str,
        resource: Resource,
    ) -> list[TableRecord]:
        if not self.extract_tables or not TABLE_CAPTION_RE.search(raw_text):
            return []
        records: list[TableRecord] = []
        try:
            tables = page.extract_tables(
                {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                }
            )
            if not tables and resource.layout_hint == "two_column":
                tables = page.extract_tables(
                    {
                        "vertical_strategy": "text",
                        "horizontal_strategy": "text",
                        "min_words_vertical": 3,
                        "min_words_horizontal": 1,
                    }
                )
        except Exception as exc:  # pragma: no cover - malformed table geometry
            LOGGER.warning("Table extraction failed on PDF page %s: %s", page_number, exc)
            return [
                TableRecord(
                    table_index=1,
                    page_number=page_number,
                    rows=[],
                    markdown="",
                    extraction_status=f"failed: {exc}",
                )
            ]

        for table_index, table in enumerate(tables or [], start=1):
            rows = [
                [normalize_line(cell or "") for cell in row]
                for row in table
                if row and any(normalize_line(cell or "") for cell in row)
            ]
            if not is_meaningful_table(rows):
                continue
            records.append(
                TableRecord(
                    table_index=table_index,
                    page_number=page_number,
                    rows=rows,
                    markdown=table_to_markdown(rows),
                )
            )
        return records

    def _extract_figures(self, page: Any, page_number: int) -> list[FigureReference]:
        if not self.record_image_references:
            return []
        figures: list[FigureReference] = []
        for figure_index, image in enumerate(page.images or [], start=1):
            bbox = None
            coordinates = [image.get(key) for key in ("x0", "top", "x1", "bottom")]
            if all(value is not None for value in coordinates):
                bbox = tuple(float(value) for value in coordinates)  # type: ignore[arg-type]
            figures.append(
                FigureReference(
                    figure_index=figure_index,
                    page_number=page_number,
                    object_type=str(image.get("object_type", "image")),
                    bbox=bbox,
                )
            )
        return figures


def is_meaningful_table(rows: list[list[str]]) -> bool:
    if len(rows) < 2:
        return False
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    normalized = [row + [""] * (width - len(row)) for row in rows]
    populated_by_row = [sum(bool(cell.strip()) for cell in row) for row in normalized]
    populated_by_column = [
        sum(bool(row[index].strip()) for row in normalized) for index in range(width)
    ]
    if sum(count >= 2 for count in populated_by_column) < 2:
        return False
    if sum(count >= 2 for count in populated_by_row) < 2:
        return False
    populated_cells = sum(populated_by_row)
    if populated_cells < 4:
        return False
    average_cell_length = sum(
        len(cell) for row in normalized for cell in row if cell
    ) / populated_cells
    if average_cell_length > 500:
        return False
    return True


def table_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = [cell or f"Column {index + 1}" for index, cell in enumerate(normalized_rows[0])]
    body = normalized_rows[1:]
    lines = [
        "| " + " | ".join(escape_markdown_cell(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append(
            "| " + " | ".join(escape_markdown_cell(cell) for cell in row) + " |"
        )
    return "\n".join(lines)


def escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()
