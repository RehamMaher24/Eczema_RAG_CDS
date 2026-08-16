# Eczema Day 1 Project

Beginner project for the Day 1 outcomes: prepare a small evidence index from two official guideline PDFs.

## Files

- `data/raw/`: the two guideline PDFs.
- `config/resources.json`: PDF list and credibility notes.
- `config/questions.json`: 8 clinical test questions.
- `scripts/day1_pipeline.py`: extracts text, chunks it, builds a simple index, and runs retrieval checks.
- `outputs/`: generated Day 1 results.

## Run

Install requirements:

```powershell
python -m pip install -r requirements.txt
```

Run the Day 1 pipeline:

```powershell
python scripts/day1_pipeline.py
```

## Outputs

- `outputs/source_credibility.md`
- `outputs/sample_pages.md`
- `outputs/chunks.jsonl`
- `outputs/embeddings.json`
- `outputs/retrieval_results.md`

Stop here for Day 1 after the retrieval results show reasonable evidence chunks.
