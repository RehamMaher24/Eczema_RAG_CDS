import json
import math
import re
from collections import Counter
from pathlib import Path

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
RESOURCES_FILE = ROOT / "config" / "resources.json"
QUESTIONS_FILE = ROOT / "config" / "questions.json"
OUTPUT_DIR = ROOT / "outputs"

MAX_CHUNK_WORDS = 180

SECTION_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s+([A-Z][A-Za-z ,/()-]{3,120})$")
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9']+")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "atopic",
    "at",
    "be",
    "by",
    "child",
    "children",
    "eczema",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "what",
    "when",
    "with",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clean_text(text):
    if not text:
        return ""

    text = text.replace("\u2022", "-")
    text = text.replace("\u00e2\u20ac\u00a2", "-")
    text = text.replace("\u00ef\u00bc\u0686", "-")

    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            cleaned_lines.append("")
            continue
        lowered = line.lower()
        if "notice-of-rights" in lowered:
            continue
        if lowered.startswith("page ") or lowered.startswith("copyright"):
            continue
        if lowered.startswith("\u00a9 nice"):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def find_section(text, current_section):
    for line in text.splitlines():
        match = SECTION_PATTERN.match(line.strip())
        if match:
            current_section = f"{match.group(1)} {match.group(2)}"
    return current_section


def split_into_paragraphs(text):
    paragraphs = []
    current = []

    for line in text.splitlines():
        if not line.strip():
            if current:
                paragraphs.append(" ".join(current))
                current = []
        else:
            current.append(line.strip())

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def extract_pages(resources):
    all_pages = []

    for resource in resources:
        pdf_path = ROOT / resource["path"]
        current_section = "Front matter"

        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = clean_text(page.extract_text())
                current_section = find_section(text, current_section)

                all_pages.append(
                    {
                        "doc_id": resource["doc_id"],
                        "title": resource["title"],
                        "page": page_number,
                        "section": current_section,
                        "text": text,
                    }
                )

    return all_pages


def create_chunks(pages):
    chunks = []
    chunk_numbers = {}

    for page in pages:
        paragraphs = split_into_paragraphs(page["text"])
        current_text = []
        start_page = page["page"]

        for paragraph in paragraphs:
            next_word_count = len(" ".join(current_text + [paragraph]).split())
            if current_text and next_word_count > MAX_CHUNK_WORDS:
                add_chunk(chunks, chunk_numbers, page, start_page, current_text)
                current_text = []
                start_page = page["page"]

            current_text.append(paragraph)

        if current_text:
            add_chunk(chunks, chunk_numbers, page, start_page, current_text)

    return chunks


def add_chunk(chunks, chunk_numbers, page, start_page, text_parts):
    doc_id = page["doc_id"]
    chunk_numbers[doc_id] = chunk_numbers.get(doc_id, 0) + 1

    chunks.append(
        {
            "chunk_id": f"{doc_id}_chunk_{chunk_numbers[doc_id]:03d}",
            "doc_id": doc_id,
            "document": page["title"],
            "section": page["section"],
            "page_start": start_page,
            "page_end": page["page"],
            "text": " ".join(text_parts).strip(),
        }
    )


def normalize_token(token):
    token = token.lower()
    if token in {"diagnosis", "diagnosed", "diagnosing", "diagnostic"}:
        return "diagnose"
    if token in {"criteria", "criterion"}:
        return "criterion"
    if token in {"referred", "referral", "referring"}:
        return "refer"
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 5 and token.endswith("ing"):
        token = token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]
    return token


def tokenize(text):
    tokens = []
    for token in TOKEN_PATTERN.findall(text):
        token = normalize_token(token)
        if token not in STOP_WORDS and len(token) > 2:
            tokens.append(token)
    return tokens


def build_embeddings(chunks):
    chunk_tokens = []
    for chunk in chunks:
        text_for_index = chunk["section"] + " " + chunk["text"]
        text_for_index = text_for_index.replace(chunk["document"], " ")
        chunk_tokens.append(Counter(tokenize(text_for_index)))

    document_frequency = Counter()

    for tokens in chunk_tokens:
        document_frequency.update(tokens.keys())

    total_chunks = len(chunks)
    idf = {
        word: math.log((total_chunks + 1) / (count + 1)) + 1
        for word, count in document_frequency.items()
    }

    embeddings = []
    for chunk, tokens in zip(chunks, chunk_tokens):
        vector = {}
        for word, count in tokens.items():
            vector[word] = count * idf[word]

        length = math.sqrt(sum(value * value for value in vector.values())) or 1
        vector = {word: round(value / length, 6) for word, value in vector.items()}

        embeddings.append(
            {
                "chunk_id": chunk["chunk_id"],
                "vector": vector,
            }
        )

    index = {
        "method": "simple TF-IDF",
        "chunk_count": len(chunks),
        "vocabulary_size": len(idf),
        "idf": idf,
    }

    return embeddings, index


def vectorize_question(question, idf):
    tokens = Counter(tokenize(question))
    vector = {}

    for word, count in tokens.items():
        if word in idf:
            vector[word] = count * idf[word]

    length = math.sqrt(sum(value * value for value in vector.values())) or 1
    return {word: value / length for word, value in vector.items()}


def score_vectors(question_vector, chunk_vector):
    return sum(question_vector.get(word, 0) * chunk_vector.get(word, 0) for word in question_vector)


def check_questions(questions, chunks, embeddings, index):
    results = []
    embedding_lookup = {item["chunk_id"]: item["vector"] for item in embeddings}

    for question in questions:
        question_vector = vectorize_question(question, index["idf"])
        scores = []

        for chunk in chunks:
            chunk_vector = embedding_lookup[chunk["chunk_id"]]
            score = score_vectors(question_vector, chunk_vector)
            if score > 0:
                scores.append((score, chunk))

        scores.sort(reverse=True, key=lambda item: item[0])
        top_chunks = []

        for score, chunk in scores[:3]:
            top_chunks.append(
                {
                    "score": round(score, 4),
                    "chunk_id": chunk["chunk_id"],
                    "doc_id": chunk["doc_id"],
                    "section": chunk["section"],
                    "pages": f"{chunk['page_start']}-{chunk['page_end']}",
                    "text": chunk["text"],
                }
            )

        results.append({"question": question, "top_chunks": top_chunks})

    return results


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_source_credibility(resources):
    lines = ["# Source Credibility", ""]
    lines.append("Clinical topic: childhood eczema and related food allergy assessment.")
    lines.append("")

    for resource in resources:
        lines.append(f"## {resource['title']}")
        lines.append("")
        lines.append(f"- File: `{resource['path']}`")
        lines.append(f"- Publisher: {resource['publisher']}")
        lines.append(f"- Why credible: {resource['why_credible']}")
        lines.append("")

    (OUTPUT_DIR / "source_credibility.md").write_text("\n".join(lines), encoding="utf-8")


def write_sample_pages(pages):
    lines = ["# Sample Pages", ""]
    used_docs = set()

    for page in pages:
        if page["doc_id"] in used_docs:
            continue
        if len(page["text"].split()) < 80:
            continue

        used_docs.add(page["doc_id"])
        lines.append(f"## {page['title']} - page {page['page']}")
        lines.append("")
        lines.append(f"Section: `{page['section']}`")
        lines.append("")
        lines.append("```text")
        lines.append(page["text"][:1200])
        lines.append("```")
        lines.append("")

    (OUTPUT_DIR / "sample_pages.md").write_text("\n".join(lines), encoding="utf-8")


def write_retrieval_results(results):
    lines = ["# Retrieval Results", ""]

    for result in results:
        lines.append(f"## Question: {result['question']}")
        lines.append("")

        for chunk in result["top_chunks"]:
            lines.append(f"### {chunk['chunk_id']} - score {chunk['score']}")
            lines.append("")
            lines.append(f"- Document ID: `{chunk['doc_id']}`")
            lines.append(f"- Section: `{chunk['section']}`")
            lines.append(f"- Pages: {chunk['pages']}")
            lines.append("")
            lines.append("```text")
            lines.append(chunk["text"][:800])
            lines.append("```")
            lines.append("")

    (OUTPUT_DIR / "retrieval_results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    resources = load_json(RESOURCES_FILE)["resources"]
    questions = load_json(QUESTIONS_FILE)["questions"]

    pages = extract_pages(resources)
    chunks = create_chunks(pages)
    embeddings, index = build_embeddings(chunks)
    retrieval_results = check_questions(questions, chunks, embeddings, index)

    write_source_credibility(resources)
    write_sample_pages(pages)
    write_jsonl(OUTPUT_DIR / "chunks.jsonl", chunks)
    write_json(OUTPUT_DIR / "embeddings.json", embeddings)
    write_json(OUTPUT_DIR / "index.json", index)
    write_retrieval_results(retrieval_results)

    print("Day 1 pipeline complete.")
    print(f"Pages extracted: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Questions tested: {len(questions)}")
    print("Open outputs/retrieval_results.md to inspect retrieved chunks.")


if __name__ == "__main__":
    main()
