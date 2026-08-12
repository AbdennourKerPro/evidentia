"""Shared benchmark loading and deterministic evaluation helpers."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas import SearchResult


class EvidenceTarget(BaseModel):
    """A stable description of one passage that retrieval should find."""

    document_id: str = Field(min_length=1)
    match_phrases: list[str] = Field(min_length=1)


class EvaluationCase(BaseModel):
    """One versioned RAG evaluation example."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]+$")
    category: Literal["factoid", "method", "comparison", "abstention"]
    language: Literal["en", "fr"]
    question: str = Field(min_length=3)
    scope_document_ids: list[str] | None = None
    expected_document_ids: list[str] = Field(default_factory=list)
    evidence_targets: list[EvidenceTarget] = Field(default_factory=list)
    reference_answer: str = ""
    required_answer_terms: list[list[str]] = Field(default_factory=list)
    should_abstain: bool = False
    review_status: Literal["needs_human_review", "verified"] = (
        "needs_human_review"
    )
    notes: str = ""

    @model_validator(mode="after")
    def validate_ground_truth(self) -> "EvaluationCase":
        """Reject internally inconsistent ground-truth examples."""

        if self.should_abstain:
            if self.expected_document_ids or self.evidence_targets:
                raise ValueError("An abstention case cannot define expected evidence.")
            return self

        if not self.expected_document_ids:
            raise ValueError("An answerable case needs expected_document_ids.")
        if not self.evidence_targets:
            raise ValueError("An answerable case needs evidence_targets.")
        if not self.reference_answer:
            raise ValueError("An answerable case needs a reference_answer.")
        if not self.required_answer_terms:
            raise ValueError("An answerable case needs required_answer_terms.")

        expected = set(self.expected_document_ids)
        target_documents = {target.document_id for target in self.evidence_targets}
        if not target_documents.issubset(expected):
            raise ValueError("Every evidence target must belong to an expected document.")
        if self.scope_document_ids is not None and not expected.issubset(
            set(self.scope_document_ids)
        ):
            raise ValueError("The search scope must contain every expected document.")

        return self


def load_benchmark(path: Path) -> list[EvaluationCase]:
    """Parse a JSON Lines benchmark and reject duplicate identifiers."""

    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8") as benchmark_file:
        for line_number, line in enumerate(benchmark_file, start=1):
            if not line.strip():
                continue
            try:
                case = EvaluationCase.model_validate_json(line)
            except Exception as error:
                raise ValueError(
                    f"Invalid benchmark entry at {path}:{line_number}: {error}"
                ) from error
            if case.id in seen_ids:
                raise ValueError(f"Duplicate benchmark id: {case.id}")
            seen_ids.add(case.id)
            cases.append(case)

    if not cases:
        raise ValueError(f"The benchmark is empty: {path}")
    return cases


def select_cases(
    cases: Sequence[EvaluationCase],
    *,
    case_ids: Sequence[str] | None,
    max_cases: int | None,
) -> list[EvaluationCase]:
    """Apply optional command-line filters while preserving benchmark order."""

    selected = list(cases)
    if case_ids:
        requested = set(case_ids)
        selected = [case for case in selected if case.id in requested]
        missing = requested - {case.id for case in selected}
        if missing:
            raise ValueError(f"Unknown benchmark ids: {', '.join(sorted(missing))}")
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("No benchmark case remains after filtering.")
    return selected


def normalize_text(text: str) -> str:
    """Normalize case, accents, and punctuation for robust phrase matching."""

    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", without_accents.casefold()).strip()


def phrase_group_is_present(text: str, alternatives: Sequence[str]) -> bool:
    """Return true when at least one accepted wording occurs in the text."""

    normalized_text = normalize_text(text)
    return any(normalize_text(phrase) in normalized_text for phrase in alternatives)


def result_matches_target(result: SearchResult, target: EvidenceTarget) -> bool:
    """Match one retrieved chunk to a document-specific evidence target."""

    return result.document_id == target.document_id and phrase_group_is_present(
        result.text,
        target.match_phrases,
    )


def score_retrieval(
    case: EvaluationCase, results: Sequence[SearchResult]
) -> dict[str, object]:
    """Compute document and evidence retrieval metrics for one answerable case."""

    expected_documents = set(case.expected_document_ids)
    retrieved_documents = {result.document_id for result in results}
    document_recall = len(expected_documents & retrieved_documents) / len(
        expected_documents
    )

    matched_targets: set[int] = set()
    first_relevant_rank: int | None = None
    dcg = 0.0
    for rank, result in enumerate(results, start=1):
        newly_matched = {
            position
            for position, target in enumerate(case.evidence_targets)
            if position not in matched_targets and result_matches_target(result, target)
        }
        if newly_matched:
            if first_relevant_rank is None:
                first_relevant_rank = rank
            dcg += 1.0 / math.log2(rank + 1)
            matched_targets.update(newly_matched)

    evidence_recall = len(matched_targets) / len(case.evidence_targets)
    ideal_hits = min(len(case.evidence_targets), len(results))
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1)
    )

    return {
        "document_recall_at_k": document_recall,
        "evidence_recall_at_k": evidence_recall,
        "evidence_hit_at_k": float(bool(matched_targets)),
        "reciprocal_rank": 0.0
        if first_relevant_rank is None
        else 1.0 / first_relevant_rank,
        "ndcg_at_k": 0.0 if ideal_dcg == 0 else dcg / ideal_dcg,
        "matched_evidence_targets": sorted(matched_targets),
        "missing_evidence_targets": [
            position
            for position in range(len(case.evidence_targets))
            if position not in matched_targets
        ],
    }


def answer_fact_coverage(
    answer: str, required_answer_terms: Sequence[Sequence[str]]
) -> tuple[float, list[int]]:
    """Measure coverage of human-readable facts using accepted term variants."""

    matched = [
        position
        for position, alternatives in enumerate(required_answer_terms)
        if phrase_group_is_present(answer, alternatives)
    ]
    if not required_answer_terms:
        return 1.0, matched
    return len(matched) / len(required_answer_terms), matched


def mean(values: Iterable[float]) -> float:
    """Return an arithmetic mean, or zero for an empty collection."""

    materialized = list(values)
    return 0.0 if not materialized else sum(materialized) / len(materialized)


def write_json_report(path: Path, report: dict[str, object]) -> None:
    """Write a readable UTF-8 JSON report to a persistent report directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
