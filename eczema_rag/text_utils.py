from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

WHITESPACE_RE = re.compile(r"[\t\f\v ]+")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
NUMBER_RE = re.compile(r"\d+")
PRINTED_PAGE_PATTERNS = (
    re.compile(r"\bPage\s+(\d+)\s+of\s+\d+\b", re.IGNORECASE),
    re.compile(r"^\s*([SRA]?\d{1,4})\s*$", re.IGNORECASE),
)
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?P<number>\d+(?:\.\d+){0,4})\.?\s+(?P<title>[A-Za-z][^\n]{2,180})\s*$"
)
SECTION_HEADING_RE = re.compile(
    r"^\s*(?P<prefix>Section\s+\d+(?:\.\d+)*)[.:]?\s*(?P<title>[^\n]{3,180})$",
    re.IGNORECASE,
)
SUMMARY_HEADING_RE = re.compile(
    r"^\s*(?P<label>(?:Summary\s+Statement|Recommendation|Recommendations|Table|Box|Appendix)\s*[A-Za-z0-9.-]*)\s*[:.-]?\s*(?P<title>[^\n]{0,160})$",
    re.IGNORECASE,
)
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")
REFERENCE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
REFERENCE_VOLUME_RE = re.compile(r"\b\d{1,4}\s*:\s*\d{1,4}(?:[-–]\d{1,4})?\b")
JOURNAL_HEADER_RE = re.compile(
    r"^(?:J\s+AM\s+ACAD\s+DERMATOL|VOLUME\s+\d+|(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+(?:19|20)\d{2}|[A-Z .'-]+\s+ET\s+AL\.?$)",
    re.IGNORECASE,
)


def normalize_line(line: str) -> str:
    line = line.replace("\u00ad", "")
    line = line.replace("\u2022", "-")
    line = line.replace("\uf0b7", "-")
    line = line.replace("\u2013", "-")
    line = line.replace("\u2014", "-")
    line = line.replace("\u00a0", " ")
    return WHITESPACE_RE.sub(" ", line).strip()


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    lines = [normalize_line(line) for line in text.replace("\r\n", "\n").split("\n")]
    cleaned: list[str] = []
    for line in lines:
        if line:
            cleaned.append(line)
        elif cleaned and cleaned[-1] != "":
            cleaned.append("")
    return MULTI_BLANK_RE.sub("\n\n", "\n".join(cleaned)).strip()


def normalize_repeated_line(line: str) -> str:
    normalized = normalize_line(line).lower()
    normalized = NUMBER_RE.sub("#", normalized)
    normalized = re.sub(r"https?://\S+", "<url>", normalized)
    return normalized


def nonempty_lines(text: str) -> list[str]:
    return [line for line in (normalize_line(item) for item in text.splitlines()) if line]


def detect_repeated_margin_lines(
    page_texts: Iterable[str], margin_lines: int = 3, repeat_ratio: float = 0.6
) -> set[str]:
    pages = [nonempty_lines(text) for text in page_texts]
    pages = [lines for lines in pages if lines]
    if len(pages) < 3:
        return set()

    counts: Counter[str] = Counter()
    for lines in pages:
        candidates = lines[:margin_lines] + lines[-margin_lines:]
        counts.update(set(normalize_repeated_line(line) for line in candidates if line))

    threshold = max(3, int(len(pages) * repeat_ratio + 0.999))
    return {
        line
        for line, count in counts.items()
        if count >= threshold and len(line) >= 3
    }


def strip_repeated_margin_lines(text: str, repeated_lines: set[str]) -> str:
    if not repeated_lines:
        return normalize_text(text)
    kept: list[str] = []
    for line in text.splitlines():
        normalized = normalize_line(line)
        if normalized and normalize_repeated_line(normalized) in repeated_lines:
            continue
        kept.append(normalized)
    return normalize_text("\n".join(kept))


def detect_printed_page_label(text: str) -> str | None:
    lines = nonempty_lines(text)
    candidates = lines[:4] + lines[-4:]
    for pattern in PRINTED_PAGE_PATTERNS:
        for line in candidates:
            match = pattern.search(line)
            if match:
                return match.group(1)
    return None


def heading_level(line: str) -> tuple[int, str] | None:
    candidate = normalize_line(line)
    if not candidate or len(candidate) > 190:
        return None

    section_match = SECTION_HEADING_RE.match(candidate)
    if section_match:
        prefix = section_match.group("prefix")
        level = prefix.count(".") + 1
        return level, f"{prefix}: {section_match.group('title').strip()}"

    numbered_match = NUMBERED_HEADING_RE.match(candidate)
    if numbered_match:
        title = numbered_match.group("title").strip()
        number = numbered_match.group("number")
        if looks_like_reference_entry(candidate) or "." * 3 in candidate:
            return None
        if looks_like_sentence(title) and len(title.split()) > 18:
            return None
        if "." not in number:
            if int(number) > 9 or len(title.split()) > 14 or not title[0].isupper():
                return None
        level = 1 if re.fullmatch(r"\d+\.0", number) else min(number.count(".") + 1, 3)
        return level, f"{number} {title}"

    summary_match = SUMMARY_HEADING_RE.match(candidate)
    if summary_match and len(candidate.split()) <= 18 and "..." not in candidate:
        return 2, candidate

    if JOURNAL_HEADER_RE.match(candidate):
        return None

    letters = [character for character in candidate if character.isalpha()]
    if (
        letters
        and len(candidate.split()) <= 14
        and len(candidate) >= 4
        and all(character.isupper() for character in letters)
        and not candidate.startswith(("DOI", "ISSN", "HTTP"))
    ):
        return 1, candidate.title() if candidate.isupper() else candidate

    return None


def looks_like_sentence(text: str) -> bool:
    return text.endswith((".", "?", "!", ";"))


def looks_like_reference_entry(text: str) -> bool:
    lowered = text.lower()
    citation_signals = 0
    citation_signals += int(" et al" in lowered)
    citation_signals += int(bool(REFERENCE_YEAR_RE.search(text)))
    citation_signals += int(bool(REFERENCE_VOLUME_RE.search(text)))
    citation_signals += int(text.count(",") >= 2)
    citation_signals += int(" doi" in lowered or "http" in lowered)
    citation_signals += int(any(term in lowered for term in (" dermatol", " allergy", " immunol", " journal ")))
    return citation_signals >= 2


def update_section_path(section_path: list[str], level: int, heading: str) -> list[str]:
    level = max(1, min(level, 6))
    if section_path and section_path[-1].casefold() == heading.casefold():
        return list(section_path)
    for index, existing in enumerate(section_path):
        if existing.casefold() == heading.casefold():
            return list(section_path[: index + 1])
    base_path = [] if section_path == ["Front matter"] else list(section_path)
    keep = min(level - 1, len(base_path))
    new_path = list(base_path[:keep])
    new_path.append(heading)
    return new_path


def split_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        normalized = normalize_line(line)
        if not normalized:
            if current:
                paragraphs.append(join_wrapped_lines(current))
                current = []
            continue
        if heading_level(normalized):
            if current:
                paragraphs.append(join_wrapped_lines(current))
                current = []
            paragraphs.append(normalized)
        else:
            current.append(normalized)
    if current:
        paragraphs.append(join_wrapped_lines(current))
    return [paragraph for paragraph in paragraphs if paragraph]


def join_wrapped_lines(lines: list[str]) -> str:
    text = ""
    for line in lines:
        if not text:
            text = line
        elif text.endswith("-") and line and line[0].islower():
            text = text[:-1] + line
        else:
            text += " " + line
    return WHITESPACE_RE.sub(" ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]
