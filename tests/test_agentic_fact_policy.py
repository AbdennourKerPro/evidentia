"""Tests for the risk-based use of strict fact plans in LangGraph."""

from __future__ import annotations

import unittest

from app.agentic_rag import _plan_answer
from app.schemas import SearchResult


def _architecture_result() -> SearchResult:
    return SearchResult(
        chunk_id=1,
        score=0.5,
        document_id="blip2-2023",
        title="BLIP-2",
        page=1,
        section="Architecture",
        language="en",
        text=(
            "Learnable query vectors connect a frozen image encoder to a frozen LLM."
        ),
    )


class AgenticFactPlanningPolicyTests(unittest.TestCase):
    def test_keeps_strict_architecture_plan_for_one_article(self) -> None:
        state = _plan_answer(
            {
                "question": "Why is the Q-Former a bridge between frozen models?",
                "strategy": "global",
                "selected_results": [_architecture_result()],
            }
        )
        self.assertGreater(len(state["fact_plan"]), 0)

    def test_skips_strict_architecture_plan_for_a_comparison(self) -> None:
        state = _plan_answer(
            {
                "question": "Compare the Q-Former bridge with another projection.",
                "strategy": "per_source",
                "selected_results": [_architecture_result()],
            }
        )
        self.assertEqual(state["fact_plan"], [])


if __name__ == "__main__":
    unittest.main()
