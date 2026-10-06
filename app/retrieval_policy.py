"""Explicit retrieval profiles for cumulative, reproducible ablation studies."""

import os
from typing import Literal, cast


RetrievalProfile = Literal[
    "intent_baseline", "question_rerank", "original_query", "coverage_retry",
    "coverage_diagnostic", "candidate_recovery", "coverage_answer_plan",
]
RETRIEVAL_PROFILES = (
    "intent_baseline", "question_rerank", "original_query", "coverage_retry",
    "coverage_diagnostic", "candidate_recovery", "coverage_answer_plan",
)
DEFAULT_RETRIEVAL_PROFILE: RetrievalProfile = "candidate_recovery"


def validate_retrieval_profile(profile: str) -> RetrievalProfile:
    """Reject unknown experiments rather than silently using another policy."""

    if profile not in RETRIEVAL_PROFILES:
        raise ValueError(f"Unknown retrieval profile: {profile!r}")
    return cast(RetrievalProfile, profile)


def get_retrieval_profile() -> RetrievalProfile:
    """Use the deployed coverage recovery unless explicitly overridden."""

    return validate_retrieval_profile(
        os.getenv("RAG_RETRIEVAL_PROFILE", DEFAULT_RETRIEVAL_PROFILE)
    )


def preserves_original_query(profile: RetrievalProfile) -> bool:
    """The final profile includes the original-query change from step two."""

    return profile in {"original_query", "coverage_retry"}
