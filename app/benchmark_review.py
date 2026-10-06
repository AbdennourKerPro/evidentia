"""Human-review workflow for the versioned RAG benchmark."""

from __future__ import annotations

import json
import os
import shutil
import threading
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from app.evaluation import EvaluationCase, load_benchmark, phrase_group_is_present


BENCHMARK_PATH = Path(
    os.getenv("BENCHMARK_PATH", "/data/evaluation/rag_benchmark.jsonl")
)
PROCESSED_ROOT = Path(os.getenv("PROCESSED_ROOT", "/data/processed"))
_WRITE_LOCK = threading.Lock()


class ReviewUpdate(BaseModel):
    """Fields a reviewer may correct without editing benchmark structure."""

    question: str = Field(min_length=3, max_length=1000)
    reference_answer: str = Field(max_length=5000)
    required_answer_terms: list[list[str]] = Field(default_factory=list)
    notes: str = Field(default="", max_length=3000)
    review_status: Literal["needs_human_review", "verified"]


class EvidenceExcerpt(BaseModel):
    """A concrete processed chunk containing one gold evidence phrase."""

    target_index: int
    document_id: str
    match_phrases: list[str]
    # JSON numbers are unsafe above 2^53 in browsers; preserve Qdrant IDs as text.
    chunk_id: str
    page: int
    section: str
    text: str


class ReviewCaseResponse(BaseModel):
    """One benchmark entry enriched with the chunk shown to the reviewer."""

    case: EvaluationCase
    evidence_excerpts: list[EvidenceExcerpt]


class ReviewSummary(BaseModel):
    """Progress counters displayed in the review interface."""

    total: int
    verified: int
    needs_human_review: int


class ReviewCollectionResponse(BaseModel):
    """The complete review queue and its current progress."""

    summary: ReviewSummary
    cases: list[ReviewCaseResponse]


def list_review_cases() -> ReviewCollectionResponse:
    """Load the benchmark and resolve a readable excerpt for every gold target."""

    cases = load_benchmark(BENCHMARK_PATH)
    enriched_cases = [
        ReviewCaseResponse(
            case=case,
            evidence_excerpts=_find_evidence_excerpts(case),
        )
        for case in cases
    ]
    verified = sum(case.review_status == "verified" for case in cases)
    return ReviewCollectionResponse(
        summary=ReviewSummary(
            total=len(cases),
            verified=verified,
            needs_human_review=len(cases) - verified,
        ),
        cases=enriched_cases,
    )


def update_review_case(case_id: str, update: ReviewUpdate) -> ReviewCaseResponse:
    """Validate and atomically persist one human review decision."""

    with _WRITE_LOCK:
        cases = load_benchmark(BENCHMARK_PATH)
        case_index = next(
            (index for index, case in enumerate(cases) if case.id == case_id),
            None,
        )
        if case_index is None:
            raise KeyError(case_id)

        existing = cases[case_index]
        normalized_terms = _normalize_term_groups(update.required_answer_terms)
        updated = existing.model_copy(
            update={
                "question": update.question.strip(),
                "reference_answer": update.reference_answer.strip(),
                "required_answer_terms": normalized_terms,
                "notes": update.notes.strip(),
                "review_status": update.review_status,
            }
        )
        # model_copy does not re-run validators, so validate the complete object.
        updated = EvaluationCase.model_validate(updated.model_dump())
        evidence_excerpts = _find_evidence_excerpts(updated)
        if (
            update.review_status == "verified"
            and not updated.should_abstain
            and len(evidence_excerpts) != len(updated.evidence_targets)
        ):
            raise ValueError(
                "This case cannot be verified because at least one gold evidence "
                "target is missing from the processed chunks."
            )
        cases[case_index] = updated
        _write_benchmark_atomically(cases)

    return ReviewCaseResponse(
        case=updated,
        evidence_excerpts=evidence_excerpts,
    )


def _normalize_term_groups(groups: list[list[str]]) -> list[list[str]]:
    """Trim UI input and remove empty or duplicate alternatives."""

    normalized: list[list[str]] = []
    for group in groups:
        unique_terms = list(
            dict.fromkeys(term.strip() for term in group if term.strip())
        )
        if unique_terms:
            normalized.append(unique_terms)
    return normalized


def _write_benchmark_atomically(cases: list[EvaluationCase]) -> None:
    """Keep a last-known-good backup, then replace the JSONL in one operation."""

    BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = BENCHMARK_PATH.with_suffix(".tmp")
    backup_path = BENCHMARK_PATH.with_suffix(".backup.jsonl")
    serialized = "\n".join(
        case.model_dump_json() for case in cases
    ) + "\n"

    temporary_path.write_text(serialized, encoding="utf-8")
    # Re-read the temporary file before replacing the canonical truth set.
    load_benchmark(temporary_path)
    shutil.copy2(BENCHMARK_PATH, backup_path)
    temporary_path.replace(BENCHMARK_PATH)


def _find_evidence_excerpts(case: EvaluationCase) -> list[EvidenceExcerpt]:
    """Resolve the first matching chunk for each stable text target."""

    excerpts: list[EvidenceExcerpt] = []
    for target_index, target in enumerate(case.evidence_targets):
        chunks = _load_document_chunks(target.document_id)
        matching_chunk = next(
            (
                chunk
                for chunk in chunks
                if phrase_group_is_present(str(chunk["text"]), target.match_phrases)
            ),
            None,
        )
        if matching_chunk is None:
            continue
        excerpts.append(
            EvidenceExcerpt(
                target_index=target_index,
                document_id=target.document_id,
                match_phrases=target.match_phrases,
                chunk_id=str(matching_chunk["id"]),
                page=int(matching_chunk["page"]),
                section=str(matching_chunk["section"]),
                text=str(matching_chunk["text"]),
            )
        )
    return excerpts


@lru_cache
def _load_document_chunks(document_id: str) -> list[dict[str, object]]:
    """Load each processed article once during the API process lifetime."""

    chunks_path = PROCESSED_ROOT / document_id / "chunks.json"
    with chunks_path.open(encoding="utf-8") as chunks_file:
        chunks = json.load(chunks_file)
    if not isinstance(chunks, list):
        raise ValueError(f"Expected a list of chunks in {chunks_path}")
    return chunks
