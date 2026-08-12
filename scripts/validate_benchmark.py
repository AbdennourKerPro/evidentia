"""Validate benchmark structure and verify that gold phrases exist in chunks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from app.evaluation import load_benchmark, phrase_group_is_present


DEFAULT_BENCHMARK = Path("/data/evaluation/rag_benchmark.jsonl")
DEFAULT_PROCESSED_ROOT = Path("/data/processed")


def parse_args() -> argparse.Namespace:
    """Read paths that differ between Docker and optional host execution."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    return parser.parse_args()


def load_document_chunks(processed_root: Path, document_id: str) -> list[dict]:
    """Load the persisted Docling chunks for one benchmark document."""

    chunks_path = processed_root / document_id / "chunks.json"
    with chunks_path.open(encoding="utf-8") as chunks_file:
        return json.load(chunks_file)


def main() -> None:
    """Fail fast when a target phrase no longer exists in the processed corpus."""

    args = parse_args()
    cases = load_benchmark(args.benchmark)
    chunks_by_document: dict[str, list[dict]] = {}
    errors: list[str] = []
    verified_targets = 0

    for case in cases:
        for target_position, target in enumerate(case.evidence_targets, start=1):
            chunks = chunks_by_document.setdefault(
                target.document_id,
                load_document_chunks(args.processed_root, target.document_id),
            )
            if any(
                phrase_group_is_present(str(chunk["text"]), target.match_phrases)
                for chunk in chunks
            ):
                verified_targets += 1
            else:
                errors.append(
                    f"{case.id} target {target_position}: no phrase found in "
                    f"{target.document_id}"
                )

    categories = Counter(case.category for case in cases)
    languages = Counter(case.language for case in cases)
    review_statuses = Counter(case.review_status for case in cases)

    print(f"Benchmark: {args.benchmark}")
    print(f"Cases: {len(cases)} | categories: {dict(categories)}")
    print(f"Languages: {dict(languages)}")
    print(f"Evidence targets found: {verified_targets}")
    print(f"Human review: {dict(review_statuses)}")

    if errors:
        print("\nErrors:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("Validation passed. Gold phrases exist; semantic human review is still required.")


if __name__ == "__main__":
    main()
