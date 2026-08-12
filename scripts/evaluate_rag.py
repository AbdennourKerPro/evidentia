"""Evaluate complete Evidentia answers with deterministic, auditable metrics."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.evaluation import (
    EvaluationCase,
    answer_fact_coverage,
    load_benchmark,
    mean,
    select_cases,
    write_json_report,
)
from app.llm_gateway import LLM_MODEL_ID
from app.qdrant_gateway import ARXIV_COLLECTION_NAME
from app.rag_service import answer_question
from app.schemas import AskResponse


DEFAULT_BENCHMARK = Path("/data/evaluation/rag_benchmark.jsonl")
DEFAULT_OUTPUT = Path("/reports/rag-evaluation.json")


def parse_args() -> argparse.Namespace:
    """Allow a quick smoke case before the slower complete CPU evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def score_answer(
    case: EvaluationCase,
    response: AskResponse,
    latency_seconds: float,
) -> dict[str, object]:
    """Compare one RAG response with abstention, facts, and document provenance."""

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
            "passed": passed,
        }
    )
    return report


def main() -> None:
    """Run the local LLM sequentially and save every answer for human auditing."""

    args = parse_args()
    cases = select_cases(
        load_benchmark(args.benchmark),
        case_ids=args.case_ids,
        max_cases=args.max_cases,
    )
    case_reports: list[dict[str, object]] = []

    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] Generating {case.id}...")
        started_at = perf_counter()
        response = answer_question(
            question=case.question,
            limit=args.limit,
            document_ids=case.scope_document_ids,
        )
        case_reports.append(
            score_answer(case, response, perf_counter() - started_at)
        )

    answerable = [report for report in case_reports if not report["expected_abstention"]]
    aggregate = {
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
        "mean_latency_seconds": mean(
            float(report["latency_seconds"]) for report in case_reports
        ),
    }
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": str(args.benchmark),
        "collection": ARXIV_COLLECTION_NAME,
        "llm_model": LLM_MODEL_ID,
        "top_k": args.limit,
        "metric_limitations": (
            "Fact coverage uses explicit term groups. It is auditable but does not "
            "replace semantic human review or a calibrated judge model."
        ),
        "aggregate": aggregate,
        "cases": case_reports,
    }
    write_json_report(args.output, report)
    print(f"\nReport written to {args.output}")
    print(
        f"Pass rate: {aggregate['end_to_end_pass_rate']:.3f} | "
        f"Fact coverage: {aggregate['mean_fact_coverage']:.3f} | "
        f"Abstention accuracy: {aggregate['abstention_accuracy']:.3f}"
    )


if __name__ == "__main__":
    main()
