"""Hybrid dense/BM25 retrieval, RRF fusion, and multilingual reranking."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from app.qdrant_gateway import (
    ARXIV_COLLECTION_NAME,
    list_document_chunks,
    search_chunks,
)
from app.schemas import SearchResult
from app.settings import get_reranker_model_path


CROSS_ENCODER_MODEL_ID = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
CROSS_ENCODER_MODEL_REVISION = "e1f4a37b5ad572015a1340a48133713afb98e7fe"
CROSS_ENCODER_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
)

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60

_STOP_WORDS = {
    "about",
    "and",
    "are",
    "avec",
    "comment",
    "compare",
    "dans",
    "des",
    "does",
    "from",
    "how",
    "les",
    "pour",
    "quel",
    "quelle",
    "that",
    "the",
    "their",
    "this",
    "used",
    "what",
    "which",
    "with",
}


class RerankerModelUnavailableError(RuntimeError):
    """Raised when the pinned cross-encoder files are absent from the volume."""


@dataclass(frozen=True)
class HybridCandidates:
    """The rankings produced before and after reciprocal-rank fusion."""

    dense: list[SearchResult]
    lexical: list[SearchResult]
    fused: list[SearchResult]


@dataclass(frozen=True)
class MultiFacetCandidates:
    """Per-facet rankings and their final weighted RRF fusion."""

    dense_rankings: tuple[list[SearchResult], ...]
    lexical_rankings: tuple[list[SearchResult], ...]
    fused: list[SearchResult]


@dataclass(frozen=True)
class BM25Index:
    """Small in-memory lexical index for one scientific article."""

    chunks: tuple[SearchResult, ...]
    tokenized_chunks: tuple[tuple[str, ...], ...]
    document_frequencies: dict[str, int]
    average_length: float


def is_reranker_downloaded() -> bool:
    """Check the minimal local files required by SentenceTransformers."""

    model_path = get_reranker_model_path()
    return model_path.is_dir() and all(
        (model_path / filename).is_file()
        for filename in CROSS_ENCODER_REQUIRED_FILES
    )


@lru_cache
def get_cross_encoder() -> CrossEncoder:
    """Load the multilingual reranker once and keep network access disabled."""

    if not is_reranker_downloaded():
        raise RerankerModelUnavailableError(
            "The cross-encoder is not downloaded. Run "
            "python -m scripts.download_reranker_model first."
        )

    from sentence_transformers import CrossEncoder

    return CrossEncoder(
        str(get_reranker_model_path()),
        device="cpu",
        max_length=384,
        local_files_only=True,
    )


@lru_cache(maxsize=20)
def get_bm25_index(document_id: str) -> BM25Index:
    """Build and cache a BM25 index over all Qdrant chunks of one article."""

    chunks = tuple(list_document_chunks(ARXIV_COLLECTION_NAME, document_id))
    tokenized_chunks = tuple(tuple(_tokenize(chunk.text)) for chunk in chunks)
    document_frequencies: Counter[str] = Counter()
    for tokens in tokenized_chunks:
        document_frequencies.update(set(tokens))

    average_length = (
        sum(len(tokens) for tokens in tokenized_chunks) / len(tokenized_chunks)
        if tokenized_chunks
        else 0.0
    )
    return BM25Index(
        chunks=chunks,
        tokenized_chunks=tokenized_chunks,
        document_frequencies=dict(document_frequencies),
        average_length=average_length,
    )


def clear_bm25_cache() -> None:
    """Invalidate lexical indexes after the Qdrant corpus has changed."""

    get_bm25_index.cache_clear()


def search_bm25(*, query: str, document_id: str, limit: int) -> list[SearchResult]:
    """Rank every chunk of one article with the standard BM25 formula."""

    index = get_bm25_index(document_id)
    query_terms = _tokenize(query)
    if not index.chunks or not query_terms or index.average_length == 0:
        return []

    ranked: list[tuple[float, SearchResult]] = []
    corpus_size = len(index.chunks)
    for chunk, tokens in zip(
        index.chunks, index.tokenized_chunks, strict=True
    ):
        frequencies = Counter(tokens)
        score = 0.0
        for term in set(query_terms):
            term_frequency = frequencies.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = index.document_frequencies.get(term, 0)
            inverse_document_frequency = math.log(
                1
                + (corpus_size - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            length_normalization = BM25_K1 * (
                1
                - BM25_B
                + BM25_B * len(tokens) / index.average_length
            )
            score += inverse_document_frequency * (
                term_frequency * (BM25_K1 + 1)
            ) / (term_frequency + length_normalization)

        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        chunk.model_copy(update={"score": float(score)})
        for score, chunk in ranked[:limit]
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]],
    *,
    limit: int,
    rank_constant: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[SearchResult]:
    """Fuse incomparable dense and BM25 scores using ranking positions only."""

    ranking_weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(ranking_weights) != len(rankings):
        raise ValueError("RRF weights must match the number of rankings.")

    scores: dict[int, float] = {}
    candidates: dict[int, SearchResult] = {}
    for ranking, weight in zip(rankings, ranking_weights, strict=True):
        for rank, result in enumerate(ranking, start=1):
            candidates[result.chunk_id] = result
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + weight / (
                rank_constant + rank
            )

    ordered_ids = sorted(scores, key=scores.get, reverse=True)
    return [
        candidates[chunk_id].model_copy(update={"score": scores[chunk_id]})
        for chunk_id in ordered_ids[:limit]
    ]


def retrieve_hybrid_candidates(
    *,
    query: str,
    query_vector: list[float],
    document_id: str,
    dense_limit: int = 20,
    lexical_limit: int = 20,
    fusion_limit: int = 30,
) -> HybridCandidates:
    """Combine semantic Qdrant retrieval with article-wide lexical retrieval."""

    dense = search_chunks(
        query_vector,
        limit=dense_limit,
        collection_name=ARXIV_COLLECTION_NAME,
        document_ids=[document_id],
    )
    lexical = search_bm25(
        query=query,
        document_id=document_id,
        limit=lexical_limit,
    )
    fused = reciprocal_rank_fusion(
        [dense, lexical],
        limit=fusion_limit,
    )
    return HybridCandidates(dense=dense, lexical=lexical, fused=fused)


def retrieve_multifacet_candidates(
    *,
    queries: Sequence[str],
    query_vectors: Sequence[list[float]],
    document_id: str,
    dense_limit: int = 20,
    lexical_limit: int = 20,
    fusion_limit: int = 30,
    expansion_weight: float = 2.0,
    preserve_primary: bool = False,
) -> MultiFacetCandidates:
    """Fuse dense and BM25 rankings from one primary and several facet queries."""

    if not queries or len(queries) != len(query_vectors):
        raise ValueError("Queries and query vectors must be non-empty and aligned.")

    dense_rankings: list[list[SearchResult]] = []
    lexical_rankings: list[list[SearchResult]] = []
    rankings: list[list[SearchResult]] = []
    weights: list[float] = []
    for position, (query, query_vector) in enumerate(
        zip(queries, query_vectors, strict=True)
    ):
        dense = search_chunks(
            query_vector,
            limit=dense_limit,
            collection_name=ARXIV_COLLECTION_NAME,
            document_ids=[document_id],
        )
        lexical = search_bm25(
            query=query,
            document_id=document_id,
            limit=lexical_limit,
        )
        # The original question keeps half the total budget, regardless of how
        # many expansions were generated. Baseline weights remain reproducible.
        query_weight = 1.0 if position == 0 else (
            1.0 / (len(queries) - 1) if preserve_primary else expansion_weight
        )
        dense_rankings.append(dense)
        lexical_rankings.append(lexical)
        rankings.extend([dense, lexical])
        weights.extend([query_weight, query_weight])

    fused = reciprocal_rank_fusion(
        rankings,
        limit=(
            sum(len(ranking) for ranking in rankings)
            if preserve_primary else fusion_limit
        ),
        weights=weights,
    )
    if preserve_primary:
        fused = preserve_primary_pool(
            fused, [dense_rankings[0], lexical_rankings[0]], limit=fusion_limit,
        )
    return MultiFacetCandidates(
        dense_rankings=tuple(dense_rankings),
        lexical_rankings=tuple(lexical_rankings),
        fused=fused,
    )


def preserve_primary_pool(
    fused: Sequence[SearchResult],
    primary_rankings: Sequence[Sequence[SearchResult]],
    *, limit: int, reserve_per_ranking: int = 5,
) -> list[SearchResult]:
    """Protect original-query candidates before truncation, not final answers.

    Keep the top few dense/BM25 hits even when expansion votes dominate RRF.
    All retained candidates are still reranked against the complete question.
    """

    if limit <= 0:
        return []
    # Interleave the primary dense/lexical hits before reserving their slots.
    protected_ids: list[int] = []
    for rank in range(reserve_per_ranking):
        for ranking in primary_rankings:
            if rank < len(ranking) and ranking[rank].chunk_id not in protected_ids:
                protected_ids.append(ranking[rank].chunk_id)
    protected_ids = protected_ids[:limit]
    retained_ids = set(protected_ids)
    for result in fused:
        if len(retained_ids) >= limit:
            break
        retained_ids.add(result.chunk_id)
    return [result for result in fused if result.chunk_id in retained_ids][:limit]


def rerank_candidate_groups(
    groups: Sequence[tuple[str, Sequence[SearchResult]]],
    *,
    batch_size: int = 8,
) -> dict[str, list[SearchResult]]:
    """Score all source-query/chunk pairs jointly in one cross-encoder call."""

    pairs: list[tuple[str, str]] = []
    positions: list[tuple[str, SearchResult]] = []
    for query, results in groups:
        for result in results:
            pairs.append((query, _format_cross_encoder_document(result)))
            positions.append((result.document_id, result))

    if not pairs:
        return {}

    predicted_scores = get_cross_encoder().predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    ranked_by_document: dict[str, list[SearchResult]] = {}
    for (document_id, result), score in zip(
        positions, predicted_scores, strict=True
    ):
        ranked_by_document.setdefault(document_id, []).append(
            result.model_copy(update={"score": float(score)})
        )

    for results in ranked_by_document.values():
        results.sort(key=lambda result: result.score, reverse=True)
    return ranked_by_document


def round_robin_select(
    ranked_by_document: dict[str, Sequence[SearchResult]],
    *,
    document_order: Sequence[str],
    limit: int,
) -> list[SearchResult]:
    """Select evidence fairly across sources after each source has been ranked."""

    ordered_lists = [
        ranked_by_document.get(document_id, [])
        for document_id in document_order
    ]
    selected: list[SearchResult] = []
    seen_chunk_ids: set[int] = set()
    rank = 0

    while len(selected) < limit and any(
        rank < len(results) for results in ordered_lists
    ):
        for results in ordered_lists:
            if rank >= len(results):
                continue
            result = results[rank]
            if result.chunk_id not in seen_chunk_ids:
                selected.append(result)
                seen_chunk_ids.add(result.chunk_id)
            if len(selected) == limit:
                break
        rank += 1

    return selected


def _format_cross_encoder_document(result: SearchResult) -> str:
    """Expose section and title metadata to the pairwise relevance model."""

    return (
        f"Title: {result.title}\n"
        f"Section: {result.section}\n"
        f"Text: {result.text}"
    )


def _tokenize(text: str) -> list[str]:
    """Normalize scientific text into transparent English-oriented BM25 terms."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    words = re.findall(r"[a-z0-9]+", without_accents)
    return [
        _light_stem(word)
        for word in words
        if len(word) >= 2 and word not in _STOP_WORDS
    ]


def _light_stem(word: str) -> str:
    """Merge frequent English inflections without a heavyweight NLP dependency."""

    if word.startswith("train"):
        return "train"
    if len(word) > 6 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 5 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s"):
        return word[:-1]
    return word
