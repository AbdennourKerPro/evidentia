"""Evaluate the LangGraph RAG pipeline on the same benchmark as the baseline."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.agentic_rag import answer_agentic_question
from app.evaluation import load_benchmark, select_cases, write_json_report
from app.llm_gateway import capture_llm_calls, get_llm_configuration, summarize_llm_calls
from app.qdrant_gateway import ARXIV_COLLECTION_NAME
from app.retrieval_policy import get_retrieval_profile
from scripts.rag_evaluation_common import aggregate_case_reports, score_answer


DEFAULT_BENCHMARK = Path("/data/evaluation/rag_benchmark.jsonl")
DEFAULT_OUTPUT = Path("/reports/agentic-rag-evaluation-v9.json")
PIPELINE_ID = "langgraph_candidate_coverage_v9_openai_v3"


def source_fingerprint() -> dict[str, str]:
    """Identify the production code, independent of the working directory."""

    root = Path(__file__).resolve().parents[1]
    return {
        path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "app").glob("*.py"))
    }


def parse_args() -> argparse.Namespace:
    """Select benchmark cases while keeping top-k comparable with the baseline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed cases from an existing compatible output report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run cases sequentially to keep API costs and checkpoints easy to audit."""

    args = parse_args()
    cases = select_cases(
        load_benchmark(args.benchmark),
        case_ids=args.case_ids,
        max_cases=args.max_cases,
    )
    reports_by_id = (
        _load_resumable_cases(args.output, top_k=args.limit, benchmark=args.benchmark)
        if args.resume else {}
    )

    for position, case in enumerate(cases, start=1):
        if case.id in reports_by_id:
            print(
                f"[{position}/{len(cases)}] Reusing checkpoint for {case.id}.",
                flush=True,
            )
            continue
        print(
            f"[{position}/{len(cases)}] Generating {case.id} with LangGraph...",
            flush=True,
        )
        started_at = perf_counter()
        with capture_llm_calls() as calls:
            response = answer_agentic_question(
                question=case.question,
                limit=args.limit,
                document_ids=case.scope_document_ids,
            )
        response.llm = summarize_llm_calls(calls)
        reports_by_id[case.id] = score_answer(
            case,
            response,
            perf_counter() - started_at,
        )
        ordered_reports = [
            reports_by_id[selected_case.id]
            for selected_case in cases
            if selected_case.id in reports_by_id
        ]
        write_json_report(
            args.output,
            _build_report(
                args=args,
                case_reports=ordered_reports,
                requested_cases=len(cases),
            ),
        )
        print(f"Checkpoint written after {case.id}.", flush=True)

    case_reports = [reports_by_id[case.id] for case in cases]
    report = _build_report(
        args=args,
        case_reports=case_reports,
        requested_cases=len(cases),
    )
    write_json_report(args.output, report)
    aggregate = report["aggregate"]
    print(f"\nReport written to {args.output}")
    print(
        f"Pass rate: {aggregate['end_to_end_pass_rate']:.3f} | "
        f"Fact coverage: {aggregate['mean_fact_coverage']:.3f} | "
        f"Abstention accuracy: {aggregate['abstention_accuracy']:.3f}"
    )


def _build_report(
    *,
    args: argparse.Namespace,
    case_reports: list[dict[str, object]],
    requested_cases: int,
) -> dict[str, object]:
    """Build a valid partial or final report after every completed case."""

    aggregate = aggregate_case_reports(case_reports)
    aggregate["response_contract_valid_rate"] = sum(
        float(report["response"]["execution"]["response_contract_valid"])
        for report in case_reports
    ) / len(case_reports)
    aggregate["fact_contract_valid_rate"] = sum(
        float(report["response"]["execution"]["fact_contract_valid"])
        for report in case_reports
    ) / len(case_reports)
    aggregate["correction_rate"] = sum(
        float(report["response"]["execution"]["correction_attempted"])
        for report in case_reports
    ) / len(case_reports)
    executions = [report["response"]["execution"] for report in case_reports]
    executions_with_facts = [
        execution for execution in executions if execution["fact_plan"]
    ]
    aggregate["planned_fact_cases"] = len(executions_with_facts)
    aggregate["fact_planning_fallback_rate"] = sum(
        float(execution["fact_planning_fallback"])
        for execution in executions
    ) / len(executions)
    aggregate["mean_planned_fact_coverage"] = (
        sum(
            execution["covered_planned_facts"] / len(execution["fact_plan"])
            for execution in executions_with_facts
        )
        / len(executions_with_facts)
        if executions_with_facts
        else 1.0
    )
    strategy_counts = {"global": 0, "per_source": 0}
    for case_report in case_reports:
        response = case_report["response"]
        strategy = response["execution"]["strategy"]
        strategy_counts[strategy] += 1

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "complete": len(case_reports) == requested_cases,
        "completed_cases": len(case_reports),
        "requested_cases": requested_cases,
        "benchmark": str(args.benchmark),
        "benchmark_sha256": sha256(args.benchmark.read_bytes()).hexdigest(),
        "source_sha256": source_fingerprint(),
        "collection": ARXIV_COLLECTION_NAME,
        "llm_model": get_llm_configuration()["model_id"],
        "llm_configuration": get_llm_configuration(),
        "pipeline": PIPELINE_ID,
        "retrieval_profile": get_retrieval_profile(),
        "top_k": args.limit,
        "context_policy": {
            "initial_limit": args.limit,
            "max_verified_additions": 2 if get_retrieval_profile() in {"candidate_recovery", "coverage_answer_plan"} else 0,
            "original_context_preserved": True,
        },
        "strategy_counts": strategy_counts,
        "metric_limitations": (
            "Fact coverage uses explicit term groups. Compare this report with the "
            "baseline report generated from the same verified benchmark and top-k."
        ),
        "aggregate": aggregate,
        "cases": case_reports,
    }


def _load_resumable_cases(
    output: Path, *, top_k: int | None = None, benchmark: Path | None = None
) -> dict[str, dict[str, object]]:
    """Load only reports produced by this exact pipeline implementation."""

    if not output.is_file():
        return {}
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("pipeline") != PIPELINE_ID:
        raise ValueError(
            "Cannot resume an evaluation created by another pipeline: "
            f"{payload.get('pipeline')!r}."
        )
    if payload.get("llm_configuration") != get_llm_configuration():
        raise ValueError("Cannot resume an evaluation with a different LLM configuration.")
    if payload.get("retrieval_profile") != get_retrieval_profile():
        raise ValueError("Cannot resume an evaluation with a different retrieval profile.")
    if "source_sha256" in payload and payload["source_sha256"] != source_fingerprint():
        raise ValueError("Cannot resume an evaluation after the production source changed.")
    if top_k is not None and payload.get("top_k") != top_k:
        raise ValueError("Cannot resume an evaluation with a different top-k.")
    if benchmark is not None:
        current_hash = sha256(benchmark.read_bytes()).hexdigest()
        if payload.get("benchmark_sha256") != current_hash:
            raise ValueError("Cannot resume an evaluation after the benchmark changed.")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("The resume report has no valid cases list.")
    return {
        str(case_report["id"]): case_report
        for case_report in raw_cases
        if isinstance(case_report, dict) and "id" in case_report
    }


if __name__ == "__main__":
    main()
