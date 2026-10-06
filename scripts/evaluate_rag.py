"""Evaluate complete Evidentia answers with deterministic, auditable metrics."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.evaluation import (
    load_benchmark,
    select_cases,
    write_json_report,
)
from app.llm_gateway import capture_llm_calls, get_llm_configuration, summarize_llm_calls
from app.qdrant_gateway import ARXIV_COLLECTION_NAME
from app.rag_service import answer_question
from scripts.rag_evaluation_common import aggregate_case_reports, score_answer


DEFAULT_BENCHMARK = Path("/data/evaluation/rag_benchmark.jsonl")
DEFAULT_OUTPUT = Path("/reports/rag-evaluation-gpt6-luna.json")


def parse_args() -> argparse.Namespace:
    """Allow a quick smoke case before the complete billable API evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> None:
    """Run API calls sequentially and save every answer and its token usage."""

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
        with capture_llm_calls() as calls:
            response = answer_question(
                question=case.question,
                limit=args.limit,
                document_ids=case.scope_document_ids,
            )
        response.llm = summarize_llm_calls(calls)
        case_reports.append(
            score_answer(case, response, perf_counter() - started_at)
        )

    aggregate = aggregate_case_reports(case_reports)
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": str(args.benchmark),
        "collection": ARXIV_COLLECTION_NAME,
        "llm_model": get_llm_configuration()["model_id"],
        "llm_configuration": get_llm_configuration(),
        "pipeline": "baseline_dense_response_contract_openai_v1",
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
