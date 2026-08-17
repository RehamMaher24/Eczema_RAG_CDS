#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eczema_rag.config import PipelineConfig, load_resources, resolve_path
from eczema_rag.vector_store import SQLiteVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the fixed corpus and generated vector collection."
    )
    parser.add_argument("--config", default="config/pipeline.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = PipelineConfig.load(ROOT, resolve_path(ROOT, args.config))
    resources, metadata = load_resources(ROOT, config.resources_file)
    chunks_path = config.output_directory / "chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    chunk_ids = [row["chunk_id"] for row in chunks]
    chunk_hashes = [row["chunk_hash"] for row in chunks]
    expected_counts = Counter(row["doc_id"] for row in chunks)

    with SQLiteVectorStore(
        Path(config.vector_store["path"]), int(config.embedding["dimension"])
    ) as store:
        vector_count = store.collection_count(config.collection_name)
        stored_counts = store.document_counts(config.collection_name)
        stored_manifest = store.get_corpus_manifest(config.collection_name)

    checks = {
        "resource_count_is_seven": len(resources) == 7,
        "chunk_ids_unique": len(chunk_ids) == len(set(chunk_ids)),
        "chunk_hashes_unique": len(chunk_hashes) == len(set(chunk_hashes)),
        "chunk_vector_counts_match": len(chunks) == vector_count,
        "per_document_counts_match": dict(expected_counts) == stored_counts,
        "collection_name_matches": (
            stored_manifest.get("collection_name") == config.collection_name
        ),
        "corpus_name_matches": (
            stored_manifest.get("corpus_name") == metadata.get("corpus_name")
        ),
    }
    report = {
        "valid": all(checks.values()),
        "checks": checks,
        "resource_count": len(resources),
        "chunk_count": len(chunks),
        "vector_count": vector_count,
        "document_counts": stored_counts,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
