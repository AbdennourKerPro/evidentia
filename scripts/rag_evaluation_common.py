"""Metrics shared by the baseline and LangGraph end-to-end evaluators."""

from __future__ import annotations

from app.evaluation import (
    EvaluationCase,
    answer_fact_coverage,
    mean,
    phrase_group_is_present,
)
from app.schemas import AskResponse


def score_answer(
    case: EvaluationCase,
    response: AskResponse,
    latency_seconds: float,
) -> dict[str, object]:
    """Compare one response with abstention, facts, and document provenance."""

    abstention_correct = response.abstained == case.should_abstain
    report: dict[str, object] = {
        "id": case.id,
        "question": case.question,
        "category": case.category,
        "language": case.language,
        "expected_abstention": case.should_abstain,
        "abstention_correct": abstention_correct,
        "latency_seconds": latency_seconds,
        "response": response.model_dump(),
    }

    if case.should_abstain:
        report.update(
            {
                "fact_coverage": None,
                "citation_document_precision": None,
                "citation_document_recall": None,
                "retrieved_document_recall": None,
                "passed": abstention_correct,
            }
        )
        return report

    fact_coverage, matched_fact_groups = answer_fact_coverage(
        response.answer,
        case.required_answer_terms,
    )
    expected_documents = set(case.expected_document_ids)
    cited_documents = {citation.document_id for citation in response.citations}
    retrieved_documents = {evidence.document_id for evidence in response.evidence}
    citation_precision = (
        len(cited_documents & expected_documents) / len(cited_documents)
        if cited_documents
        else 0.0
    )
    citation_recall = len(cited_documents & expected_documents) / len(
        expected_documents
    )
    retrieved_recall = len(retrieved_documents & expected_documents) / len(
        expected_documents
    )
    matched_evidence_targets = [
        position
        for position, target in enumerate(case.evidence_targets)
        if any(
            evidence.document_id == target.document_id
            and phrase_group_is_present(evidence.text, target.match_phrases)
            for evidence in response.evidence
        )
    ]
    selected_evidence_recall = len(matched_evidence_targets) / len(
        case.evidence_targets
    )
    passed = (
        abstention_correct
        and not response.abstained
        and fact_coverage >= 0.5
        and citation_precision == 1.0
        and citation_recall == 1.0
    )
    report.update(
        {
            "reference_answer": case.reference_answer,
            "fact_coverage": fact_coverage,
            "matched_fact_groups": matched_fact_groups,
            "total_fact_groups": len(case.required_answer_terms),
            "citation_document_precision": citation_precision,
            "citation_document_recall": citation_recall,
            "retrieved_document_recall": retrieved_recall,
            "selected_evidence_recall": selected_evidence_recall,
            "matched_evidence_targets": matched_evidence_targets,
            "passed": passed,
        }
    )
    return report


def aggregate_case_reports(
    case_reports: list[dict[str, object]],
) -> dict[str, int | float]:
    """Compute the same aggregate metrics for every RAG pipeline."""

    answerable = [report for report in case_reports if not report["expected_abstention"]]
    return {
        "evaluated_cases": len(case_reports),
        "answerable_cases": len(answerable),
        "abstention_cases": len(case_reports) - len(answerable),
        "end_to_end_pass_rate": mean(
            float(bool(report["passed"])) for report in case_reports
        ),
        "abstention_accuracy": mean(
            float(bool(report["abstention_correct"])) for report in case_reports
        ),
        "mean_fact_coverage": mean(
            float(report["fact_coverage"]) for report in answerable
        ),
        "mean_citation_document_precision": mean(
            float(report["citation_document_precision"]) for report in answerable
        ),
        "mean_citation_document_recall": mean(
            float(report["citation_document_recall"]) for report in answerable
        ),
        "mean_retrieved_document_recall": mean(
            float(report["retrieved_document_recall"]) for report in answerable
        ),
        "mean_selected_evidence_recall": mean(
            float(report["selected_evidence_recall"]) for report in answerable
        ),
        "mean_latency_seconds": mean(
            float(report["latency_seconds"]) for report in case_reports
        ),
    }
