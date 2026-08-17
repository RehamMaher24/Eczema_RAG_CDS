# Eczema RAG Clinical Decision-Support Corpus

This repository implements a **citation-preserving clinical RAG ingestion and retrieval pipeline** for the seven supplied eczema and contact-dermatitis guideline PDFs. The source PDFs are treated as an immutable corpus: the pipeline neither downloads replacements nor edits their contents. Every run verifies each document against the SHA-256 values in `config/resources.json` and `config/corpus_sha256.txt` before parsing.

> **Clinical-use boundary:** This is a technical evidence-retrieval prototype, not a diagnostic or treatment system. Retrieved passages must be reviewed by an appropriately qualified clinician before clinical use.

## Release status

| Validation item | Result |
| --- | ---: |
| Source PDFs | 7 |
| Parsed PDF pages | 153 |
| Section-aware chunks | 510 |
| Indexed vectors | 510 |
| Structured tables extracted | 35 |
| Figure references recorded | 3 |
| Pages requiring OCR | 0 |
| Automated tests at release | 14 passed |
| Index integrity checks | All passed |

The release artifacts are in `outputs/`. The detailed ingestion report is available in `outputs/ingestion_report.md`, while `outputs/retrieval_results.md` records the evidence returned for the configured evaluation questions.

## Fixed corpus

| Document | Issuing body stated in PDF | Year/version | Repository file | Source reference |
| --- | --- | ---: | --- | --- |
| *Atopic eczema in under 12s: diagnosis and management* | NICE | Updated 2025 | `data/raw/atopic-eczema-under-12.pdf` | [NICE CG57][1] |
| *Contact Dermatitis: A Practice Parameter-Update 2015* | AAAAI/ACAAI Joint Task Force | 2015 | `data/raw/Contact Dermatitis.pdf` | [DOI][2] |
| *Atopic dermatitis: Section 1, diagnosis and assessment* | American Academy of Dermatology | 2014 | `data/raw/Diagnosis and assessment of atopic dermatitis.pdf` | [DOI][3] |
| *British Association of Dermatologists’ guidelines for contact dermatitis* | British Association of Dermatologists | 2017 | `data/raw/guidelines for the management of contact dermatitis.pdf` | [DOI][4] |
| *Atopic dermatitis: Section 2, topical therapies* | American Academy of Dermatology | 2014 | `data/raw/Management and treatment of atopic dermatitis.pdf` | [DOI][5] |
| *Atopic dermatitis: Section 4, flare prevention and adjunctive approaches* | American Academy of Dermatology | 2014 | `data/raw/Prevention of disease flares atopic dermatitis.pdf` | [DOI][6] |
| *Atopic dermatitis: Section 3, phototherapy and systemic agents* | American Academy of Dermatology | 2014 | `data/raw/treatment with phototherapy atopic dermatitis.pdf` | [DOI][7] |

The manifest records the title, publisher, publication year, scope, local path, source reference, page count, layout hint, SHA-256 hash, and corpus status for every document. The pipeline indexes all seven PDFs exactly as supplied; it does not adjudicate or silently replace document versions.

## Architecture

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| Corpus integrity | `eczema_rag/config.py` | Loads configuration, resolves paths, rejects missing files, and verifies immutable SHA-256 values. |
| PDF parsing | `eczema_rag/pdf_parser.py` | Extracts native text, removes repeated margins, detects printed page labels, extracts captioned tables, records figure objects, and flags incomplete extraction. |
| Structure detection | `eczema_rag/text_utils.py` | Normalizes text, detects headings, avoids bibliography/header false positives, and maintains section hierarchy. |
| Chunking | `eczema_rag/chunker.py` | Produces configurable section-aware chunks with overlap, bounded size, stable identifiers, content hashes, page ranges, and full source metadata. |
| Embeddings | `eczema_rag/embedder.py` | Creates deterministic local TF-IDF-weighted feature-hashing vectors without external API keys. |
| Vector store | `eczema_rag/vector_store.py` | Stores vectors and metadata in SQLite, replaces collections transactionally, prevents duplicate identifiers, and performs cosine retrieval. |
| Retrieval | `eczema_rag/retriever.py` | Combines vector similarity with domain-aware reranking, filters reference-heavy chunks, supports document filters, and formats evidence citations. |
| Orchestration | `eczema_rag/pipeline.py` | Runs ingestion end to end and generates reports, manifests, parsed pages, chunks, index artifacts, and retrieval checks. |

## Installation

Python 3.11 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No external model API or credential is required for the supplied Day 1 implementation.

## Run the complete pipeline

```bash
python scripts/day1_pipeline.py
```

The command validates all source hashes, parses the seven PDFs, creates section-aware chunks, fits the local embedding model, transactionally replaces the SQLite collection, and regenerates the reports and evaluation results.

Useful options include:

```bash
python scripts/day1_pipeline.py --validate-only
python scripts/day1_pipeline.py --json-summary
python scripts/day1_pipeline.py --collection eczema_guidelines_experiment
python scripts/day1_pipeline.py --output-dir outputs/experiment
```

## Query the index

```bash
python scripts/query_index.py \
  "When is patch testing recommended for suspected allergic contact dermatitis?"
```

JSON output and document filtering are supported:

```bash
python scripts/query_index.py \
  "When should phototherapy be considered?" \
  --top-k 5 \
  --json

python scripts/query_index.py \
  "How should disease severity be assessed?" \
  --doc-id nice_cg57_atopic_eczema_under_12
```

Each hit includes the document title, issuing body, source URL or DOI, section path, PDF page range, printed-page label when detected, immutable source hash, stable chunk ID, retrieval score, and evidence text.

## Validate the generated collection

```bash
python scripts/validate_index.py
```

Validation confirms that all seven manifest entries pass their hash checks, chunk IDs and hashes are unique, per-document chunk counts match the stored collection, vector and chunk totals agree, and the stored corpus and collection metadata match the active configuration.

## Configuration

`config/pipeline.json` controls parser behavior, chunk sizes, overlap, embedding dimension, vector-store path, retrieval threshold, and logging. The default chunking parameters are 500 target words, 800 maximum words, 80 minimum words, and 80 words of overlap.

`config/questions.json` contains evaluation questions aligned with the fixed corpus, covering atopic-eczema diagnosis, clinical history, severity and quality of life, triggers, stepped treatment, pediatric referral, contact-dermatitis patch testing, phototherapy/systemic therapy, and flare prevention.

## Generated outputs

| Artifact | Purpose |
| --- | --- |
| `outputs/structured_pages.jsonl` | Page-level text, sections, tables, figures, page labels, warnings, and provenance. |
| `outputs/chunks.jsonl` | Retrieval chunks with stable IDs, hashes, section hierarchy, pages, source metadata, and content types. |
| `outputs/embedding_model.json` | Serializable state for deterministic local query embeddings. |
| `outputs/vector_store.sqlite3` | Retrieval-ready vectors, metadata, collection state, and corpus manifest. |
| `outputs/index_manifest.json` | Collection, embedding, source, and per-document chunk-count metadata. |
| `outputs/ingestion_report.md` | Human-readable ingestion metrics and limitations. |
| `outputs/index_validation.json` | Machine-readable integrity-check results. |
| `outputs/retrieval_results.md` | Evidence results for every configured evaluation question. |
| `outputs/source_credibility.md` | Document-level publisher, reference, rights note, and immutable hash provenance. |

## Known limitations

The parser performs native text extraction and detects potential OCR needs, but it does not run OCR. It records image objects without interpreting their clinical meaning. Captioned tables are extracted heuristically; complex multi-column or spanning tables may remain in linearized page text and are explicitly flagged when a caption is detected but a reliable table structure cannot be recovered. Journal-style two-column PDFs can retain some reading-order artifacts, so production use should include human review and a stronger layout model.

The built-in embedding backend is deterministic and suitable for a dependency-light prototype. The embedding and vector-store interfaces are deliberately separated so that a production deployment can replace them with a clinical embedding model and managed vector database without changing the corpus, parser, chunk metadata, or citation contract.

## References

[1]: https://www.nice.org.uk/guidance/cg57 "NICE CG57: Atopic eczema in under 12s"
[2]: https://doi.org/10.1016/j.jaip.2015.02.009 "Contact Dermatitis: A Practice Parameter-Update 2015"
[3]: https://doi.org/10.1016/j.jaad.2013.10.010 "Atopic dermatitis Section 1: Diagnosis and assessment"
[4]: https://doi.org/10.1111/bjd.15239 "BAD guidelines for the management of contact dermatitis 2017"
[5]: https://doi.org/10.1016/j.jaad.2014.03.023 "Atopic dermatitis Section 2: Topical therapies"
[6]: https://doi.org/10.1016/j.jaad.2014.08.038 "Atopic dermatitis Section 4: Flare prevention and adjunctive approaches"
[7]: https://doi.org/10.1016/j.jaad.2014.03.030 "Atopic dermatitis Section 3: Phototherapy and systemic agents"
