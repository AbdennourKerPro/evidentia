"""Evaluate one cumulative retrieval change, with repeated targets and controls.

Benchmark labels are read only for scoring, never passed to the RAG workflow.
Each profile gets a separate report; existing reports are never overwritten.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter

from app.agentic_rag import answer_agentic_question
from app.evaluation import (
    EvaluationCase,
    load_benchmark,
    result_matches_target,
    select_cases,
    write_json_report,
)
from app.llm_gateway import capture_llm_calls, get_llm_configuration, summarize_llm_calls
from app.qdrant_gateway import ARXIV_COLLECTION_NAME, list_document_chunks
from app.retrieval_policy import RETRIEVAL_PROFILES
from app.schemas import AgenticAskResponse, SearchResult
from scripts.rag_evaluation_common import aggregate_case_reports, score_answer


def parse_args() -> argparse.Namespace:
    """Require an explicit target list so this never starts all 30 cases by default."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=RETRIEVAL_PROFILES)
    parser.add_argument(
        "--benchmark", type=Path,
        default=Path("/data/evaluation/rag_benchmark.jsonl"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True, dest="case_ids")
    parser.add_argument("--control-case-id", action="append", default=[], dest="control_ids")
    parser.add_argument("--repeats", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6))
    return parser.parse_args()


def corpus_fingerprint(corpus: dict[str, list[SearchResult]]) -> str:
    """Hash stable chunk content, not dense/BM25 relevance scores."""

    serialized = [
        chunk.model_dump_json(exclude={"score"})
        for document_id in sorted(corpus)
        for chunk in sorted(corpus[document_id], key=lambda chunk: chunk.chunk_id)
    ]
    return sha256("\n".join(serialized).encode()).hexdigest()


def target_rank_diagnostics(
    case: EvaluationCase, response: AgenticAskResponse,
    corpus: dict[str, list[SearchResult]],
) -> dict[str, object]:
    """Label observed ranks only after the application has answered."""

    relevant_ids = {
        chunk.chunk_id
        for document_id in case.expected_document_ids
        for chunk in corpus[document_id]
        if any(result_matches_target(chunk, target) for target in case.evidence_targets)
    }
    ranks = []
    for audit in response.execution.retrieval_audit:
        stage_ranks = {}
        for stage in (
            "primary_dense_ids", "primary_lexical_ids", "candidate_ids", "ranked_ids"
        ):
            stage_ranks[stage] = next(
                (
                    rank for rank, chunk_id in enumerate(getattr(audit, stage), start=1)
                    if chunk_id in relevant_ids
                ), None,
            )
        ranks.append({
            "document_id": audit.document_id, "phase": audit.phase, **stage_ranks,
        })
    return {"relevant_chunk_ids": sorted(relevant_ids), "target_ranks": ranks}


def main() -> None:
    """Run targets repeatedly, controls once, and checkpoint after every response."""

    args = parse_args()
    if args.output.exists():
        raise ValueError(f"Report already exists; choose a new output: {args.output}")
    benchmark = load_benchmark(args.benchmark)
    targets = select_cases(benchmark, case_ids=args.case_ids, max_cases=None)
    controls = (
        select_cases(benchmark, case_ids=args.control_ids, max_cases=None)
        if args.control_ids else []
    )
    trials = [
        (case, repeat, "target")
        for repeat in range(1, args.repeats + 1) for case in targets
    ]
    trials += [(case, 1, "control") for case in controls]
    # Snapshot the evaluation corpus independently of runtime retrieval decisions.
    document_ids = sorted({
        document_id for case, _, _ in trials
        for document_id in case.expected_document_ids
    })
    corpus = {
        doc: list_document_chunks(ARXIV_COLLECTION_NAME, doc) for doc in document_ids
    }
    source_hashes = {
        path.as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("app").glob("*.py"))
    }
    reports = []
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "pipeline": "retrieval_ablation_v1",
        "profile": args.profile,
        "llm_configuration": get_llm_configuration(),
        "benchmark_sha256": sha256(args.benchmark.read_bytes()).hexdigest(),
        "corpus_sha256": corpus_fingerprint(corpus),
        "source_sha256": source_hashes,
        "top_k": args.limit,
        "repeats": args.repeats,
        "requested_runs": len(trials),
        "complete": False,
        "limitations": (
            "Small diagnostic panel, not a full benchmark. Repeated generations "
            "are nondeterministic; retrieval ranks identify the direct selection "
            "effect. First-trial latency includes loading local models."
        ),
        "cases": reports,
    }
    for position, (case, repeat, role) in enumerate(trials, start=1):
        print(
            f"[{position}/{len(trials)}] {args.profile}: {case.id}, "
            f"trial={repeat}, role={role}", flush=True,
        )
        started = perf_counter()
        with capture_llm_calls() as calls:
            response = answer_agentic_question(
                question=case.question, limit=args.limit,
                document_ids=case.scope_document_ids, retrieval_profile=args.profile,
            )
        response.llm = summarize_llm_calls(calls)
        scored = score_answer(case, response, perf_counter() - started)
        scored.update({"trial": repeat, "role": role})
        scored.update(target_rank_diagnostics(case, response, corpus))
        reports.append(scored)
        report["completed_runs"] = len(reports)
        report["complete"] = len(reports) == len(trials)
        report["aggregate"] = aggregate_case_reports(reports)
        report["target_aggregate"] = aggregate_case_reports([
            case_report for case_report in reports if case_report["role"] == "target"
        ])
        write_json_report(args.output, report)
        print(
            f"passed={scored['passed']}; evidence={scored.get('selected_evidence_recall')}; "
            f"facts={scored.get('fact_coverage')}; calls={len(calls)}", flush=True,
        )
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
