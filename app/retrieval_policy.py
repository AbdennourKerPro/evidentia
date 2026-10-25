"""Explicit retrieval profiles for cumulative, reproducible ablation studies."""

import os
from typing import Literal, cast


RetrievalProfile = Literal[
    "intent_baseline", "question_rerank", "original_query", "coverage_retry",
    "coverage_diagnostic", "candidate_recovery", "coverage_answer_plan",
    "candidate_recovery_v2",
]
RETRIEVAL_PROFILES = (
    "intent_baseline", "question_rerank", "original_query", "coverage_retry",
    "coverage_diagnostic", "candidate_recovery", "coverage_answer_plan",
    "candidate_recovery_v2",
)
DEFAULT_RETRIEVAL_PROFILE: RetrievalProfile = "candidate_recovery_v2"


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


def keeps_primary_hits(profile: RetrievalProfile) -> bool:
    """Recovery profiles keep every original-query hit past the RRF cut."""

    return profile in {"candidate_recovery", "coverage_answer_plan", "candidate_recovery_v2"}


def recovers_candidates(profile: RetrievalProfile) -> bool:
    """Profiles that may add verified passages after the question-coverage check."""

    return profile in {"candidate_recovery", "coverage_answer_plan", "candidate_recovery_v2"}


def uses_recovery_v2(profile: RetrievalProfile) -> bool:
    """Second-generation recovery fixes; earlier profiles stay exact ablations."""

    return profile == "candidate_recovery_v2"
