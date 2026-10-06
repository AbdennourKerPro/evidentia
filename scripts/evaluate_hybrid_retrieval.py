"""Compare v1, dense, hybrid-RRF, and cross-encoder chunk selection."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.embeddings import embed_queries
from app.evaluation import (
    load_benchmark,
    mean,
    score_retrieval,
    select_cases,
    write_json_report,
)
from app.hybrid_retrieval import (
    CROSS_ENCODER_MODEL_ID,
    rerank_candidate_groups,
    retrieve_hybrid_candidates,
    retrieve_multifacet_candidates,
    round_robin_select,
)
from app.query_expansion import (
    build_query_variants,
    detect_retrieval_intents,
    rerank_for_intents,
)
from app.schemas import SearchResult


DEFAULT_BENCHMARK = Path("/data/evaluation/rag_benchmark.jsonl")
DEFAULT_PLANS_REPORT = Path("/reports/agentic-rag-comparisons-verified.json")
DEFAULT_OUTPUT = Path("/reports/hybrid-retrieval-comparisons.json")


def parse_args() -> argparse.Namespace:
    """Configure a cheap comparison-only retrieval experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--plans-report", type=Path, default=DEFAULT_PLANS_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--dense-limit", type=int, default=20)
    parser.add_argument("--lexical-limit", type=int, default=20)
    parser.add_argument("--fusion-limit", type=int, default=30)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--skip-cross-encoder", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Replay saved plans so only the chunk-selection strategy changes."""

    args = parse_args()
    cases = [
        case
        for case in select_cases(
            load_benchmark(args.benchmark),
            case_ids=args.case_ids,
            max_cases=None,
        )
        if case.category == "comparison" and not case.should_abstain
    ]
    if not cases:
        raise ValueError("No answerable comparison case was selected.")

    saved_cases = _load_saved_cases(args.plans_report)
    case_reports: list[dict[str, object]] = []
    strategy_names = [
        "agentic_v1",
        "dense_balanced",
        "hybrid_rrf",
        "multifacet_rrf",
        "intent_multifacet_rrf",
    ]
    if not args.skip_cross_encoder:
        strategy_names.append("hybrid_cross_encoder")

    for position, case in enumerate(cases, start=1):
        saved_case = saved_cases.get(case.id)
        if saved_case is None:
            raise ValueError(f"Missing saved query plan for benchmark case: {case.id}")

        plans = saved_case["response"]["execution"]["planned_queries"]
        document_order = [str(plan["document_id"]) for plan in plans]
        planned_queries = [str(plan["query"]) for plan in plans]
        query_vectors = embed_queries(planned_queries)
        query_variant_groups = [
            build_query_variants(
                question=case.question,
                primary_query=query,
            )
            for query in planned_queries
        ]
        flattened_variants = [
            variant for variants in query_variant_groups for variant in variants
        ]
        flattened_variant_vectors = embed_queries(flattened_variants)
        variant_vectors_by_document: dict[str, list[list[float]]] = {}
        vector_offset = 0
        for document_id, variants in zip(
            document_order,
            query_variant_groups,
            strict=True,
        ):
            next_offset = vector_offset + len(variants)
            variant_vectors_by_document[document_id] = flattened_variant_vectors[
                vector_offset:next_offset
            ]
            vector_offset = next_offset

        retrieval_started = perf_counter()
        candidates_by_document = {
            document_id: retrieve_hybrid_candidates(
                query=query,
                query_vector=query_vector,
                document_id=document_id,
                dense_limit=args.dense_limit,
                lexical_limit=args.lexical_limit,
                fusion_limit=args.fusion_limit,
            )
            for document_id, query, query_vector in zip(
                document_order,
                planned_queries,
                query_vectors,
                strict=True,
            )
        }
        retrieval_seconds = perf_counter() - retrieval_started

        multifacet_started = perf_counter()
        multifacet_by_document = {
            document_id: retrieve_multifacet_candidates(
                queries=variants,
                query_vectors=variant_vectors_by_document[document_id],
                document_id=document_id,
                dense_limit=args.dense_limit,
                lexical_limit=args.lexical_limit,
                fusion_limit=args.fusion_limit,
            )
            for document_id, variants in zip(
                document_order,
                query_variant_groups,
                strict=True,
            )
        }
        multifacet_seconds = perf_counter() - multifacet_started
        retrieval_intents = detect_retrieval_intents(case.question)

        selections: dict[str, list[SearchResult]] = {
            "agentic_v1": _saved_evidence_as_results(saved_case),
            "dense_balanced": round_robin_select(
                {
                    document_id: candidates.dense
                    for document_id, candidates in candidates_by_document.items()
                },
                document_order=document_order,
                limit=args.limit,
            ),
            "hybrid_rrf": round_robin_select(
                {
                    document_id: candidates.fused
                    for document_id, candidates in candidates_by_document.items()
                },
                document_order=document_order,
                limit=args.limit,
            ),
            "multifacet_rrf": round_robin_select(
                {
                    document_id: candidates.fused
                    for document_id, candidates in multifacet_by_document.items()
                },
                document_order=document_order,
                limit=args.limit,
            ),
            "intent_multifacet_rrf": round_robin_select(
                {
                    document_id: rerank_for_intents(
                        candidates.fused,
                        retrieval_intents,
                    )
                    for document_id, candidates in multifacet_by_document.items()
                },
                document_order=document_order,
                limit=args.limit,
            ),
        }

        cross_encoder_seconds: float | None = None
        if not args.skip_cross_encoder:
            reranking_started = perf_counter()
            cross_encoder_groups = rerank_candidate_groups(
                [
                    (query, candidates_by_document[document_id].fused)
                    for document_id, query in zip(
                        document_order, planned_queries, strict=True
                    )
                ]
            )
            cross_encoder_seconds = perf_counter() - reranking_started
            selections["hybrid_cross_encoder"] = round_robin_select(
                cross_encoder_groups,
                document_order=document_order,
                limit=args.limit,
            )

        strategies = {
            strategy: {
                "metrics": score_retrieval(case, results),
                "selected": [result.model_dump() for result in results],
            }
            for strategy, results in selections.items()
        }
        case_reports.append(
            {
                "id": case.id,
                "question": case.question,
                "document_order": document_order,
                "planned_queries": planned_queries,
                "query_variant_groups": query_variant_groups,
                "retrieval_intents": list(retrieval_intents),
                "hybrid_retrieval_seconds": retrieval_seconds,
                "multifacet_retrieval_seconds": multifacet_seconds,
                "cross_encoder_seconds": cross_encoder_seconds,
                "candidate_counts": {
                    document_id: {
                        "dense": len(candidates.dense),
                        "lexical": len(candidates.lexical),
                        "fused": len(candidates.fused),
                    }
                    for document_id, candidates in candidates_by_document.items()
                },
                "multifacet_candidate_counts": {
                    document_id: {
                        "facets": len(query_variant_groups[position]),
                        "dense_rankings": len(candidates.dense_rankings),
                        "lexical_rankings": len(candidates.lexical_rankings),
                        "fused": len(candidates.fused),
                    }
                    for position, (document_id, candidates) in enumerate(
                        multifacet_by_document.items()
                    )
                },
                "strategies": strategies,
            }
        )
        print(
            f"[{position}/{len(cases)}] {case.id}: "
            + " | ".join(
                f"{strategy}="
                f"{strategies[strategy]['metrics']['evidence_recall_at_k']:.2f}"
                for strategy in strategy_names
            )
        )

    aggregate = {
        strategy: _aggregate_strategy(case_reports, strategy)
        for strategy in strategy_names
    }
    report: dict[str, object] = {
        "created_at": datetime.now(UTC).isoformat(),
        "benchmark": str(args.benchmark),
        "plans_report": str(args.plans_report),
        "top_k": args.limit,
        "dense_limit": args.dense_limit,
        "lexical_limit": args.lexical_limit,
        "fusion_limit": args.fusion_limit,
        "fusion": "reciprocal_rank_fusion",
        "cross_encoder_model": None
        if args.skip_cross_encoder
        else CROSS_ENCODER_MODEL_ID,
        "aggregate": aggregate,
        "cases": case_reports,
    }
    write_json_report(args.output, report)
    print(f"\nReport written to {args.output}")
    for strategy, metrics in aggregate.items():
        print(
            f"{strategy}: evidence_recall@{args.limit}="
            f"{metrics['evidence_recall_at_k']:.3f} | "
            f"MRR={metrics['mrr']:.3f}"
        )


def _load_saved_cases(path: Path) -> dict[str, dict[str, object]]:
    """Load plans and v1 evidence from the completed end-to-end report."""

    report = json.loads(path.read_text(encoding="utf-8"))
    return {str(case["id"]): case for case in report["cases"]}


def _saved_evidence_as_results(saved_case: dict[str, object]) -> list[SearchResult]:
    """Adapt v1 evidence to SearchResult for identical retrieval scoring."""

    evidence = saved_case["response"]["evidence"]
    return [
        SearchResult(
            chunk_id=-(position + 1),
            score=float(item["score"]),
            document_id=str(item["document_id"]),
            title=str(item["title"]),
            page=int(item["page"]),
            section=str(item["section"]),
            language="unknown",
            text=str(item["text"]),
        )
        for position, item in enumerate(evidence)
    ]


def _aggregate_strategy(
    case_reports: list[dict[str, object]], strategy: str
) -> dict[str, float]:
    """Aggregate the same deterministic metrics for each selection strategy."""

    metrics = [report["strategies"][strategy]["metrics"] for report in case_reports]
    return {
        "document_recall_at_k": mean(
            float(item["document_recall_at_k"]) for item in metrics
        ),
        "evidence_recall_at_k": mean(
            float(item["evidence_recall_at_k"]) for item in metrics
        ),
        "evidence_hit_at_k": mean(
            float(item["evidence_hit_at_k"]) for item in metrics
        ),
        "mrr": mean(float(item["reciprocal_rank"]) for item in metrics),
        "ndcg_at_k": mean(float(item["ndcg_at_k"]) for item in metrics),
    }


if __name__ == "__main__":
    main()
