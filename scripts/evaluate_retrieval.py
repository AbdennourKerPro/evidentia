"""Evaluate Qdrant retrieval against the versioned Evidentia benchmark."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.embeddings import EMBEDDING_MODEL_ID, embed_query
from app.evaluation import (
    load_benchmark,
    mean,
    score_retrieval,
    select_cases,
    write_json_report,
)
from app.qdrant_gateway import ARXIV_COLLECTION_NAME, search_chunks


DEFAULT_BENCHMARK = Path("/data/evaluation/rag_benchmark.jsonl")
DEFAULT_OUTPUT = Path("/reports/retrieval-evaluation.json")


def parse_args() -> argparse.Namespace:
    """Expose small and full evaluation modes through explicit options."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args()


def main() -> None:
    """Retrieve top-k chunks, calculate metrics, and persist detailed failures."""

    args = parse_args()
    cases = select_cases(
        load_benchmark(args.benchmark),
        case_ids=args.case_ids,
        max_cases=args.max_cases,
    )
    evaluated_cases = [case for case in cases if not case.should_abstain]
    case_reports: list[dict[str, object]] = []

    for position, case in enumerate(evaluated_cases, start=1):
        started_at = perf_counter()
        results = search_chunks(
            embed_query(case.question),
            limit=args.limit,
            collection_name=ARXIV_COLLECTION_NAME,
            document_ids=case.scope_document_ids,
        )
        metrics = score_retrieval(case, results)
        latency_seconds = perf_counter() - started_at
        case_reports.append(
            {
                "id": case.id,
                "question": case.question,
                "category": case.category,
                "language": case.language,
                "expected_document_ids": case.expected_document_ids,
                "metrics": metrics,
                "latency_seconds": latency_seconds,
                "retrieved": [result.model_dump() for result in results],
            }
        )
        print(
            f"[{position}/{len(evaluated_cases)}] {case.id}: "
            f"evidence_recall@{args.limit}="
            f"{metrics['evidence_recall_at_k']:.2f}"
        )

    aggregate = {
        "evaluated_cases": len(case_reports),
        "excluded_abstention_cases": len(cases) - len(evaluated_cases),
        "document_recall_at_k": mean(
            float(report["metrics"]["document_recall_at_k"])
            for report in case_reports
        ),
        "evidence_recall_at_k": mean(
            float(report["metrics"]["evidence_recall_at_k"])
            for report in case_reports
        ),
        "evidence_hit_at_k": mean(
            float(report["metrics"]["evidence_hit_at_k"])
            for report in case_reports
        ),
        "mrr": mean(
            float(report["metrics"]["reciprocal_rank"])
            for report in case_reports
        ),
        "ndcg_at_k": mean(
            float(report["metrics"]["ndcg_at_k"]) for report in case_reports
        ),
        "mean_latency_seconds": mean(
            float(report["latency_seconds"]) for report in case_reports
        ),
    }
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": str(args.benchmark),
        "collection": ARXIV_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_ID,
        "top_k": args.limit,
        "aggregate": aggregate,
        "cases": case_reports,
    }
    write_json_report(args.output, report)
    print(f"\nReport written to {args.output}")
    print(
        f"Document recall@{args.limit}: {aggregate['document_recall_at_k']:.3f} | "
        f"Evidence recall@{args.limit}: {aggregate['evidence_recall_at_k']:.3f} | "
        f"MRR: {aggregate['mrr']:.3f}"
    )


if __name__ == "__main__":
    main()
