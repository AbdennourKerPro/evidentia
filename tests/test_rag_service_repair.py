"""Tests for deterministic, evidence-only repairs after one LLM correction."""

from __future__ import annotations

import unittest

from app.rag_service import (
    _strip_internal_fact_identifiers,
    repair_corrected_answer_from_evidence,
)
from app.schemas import PlannedFact, SearchResult


def _result(*, text: str, language: str = "en") -> SearchResult:
    return SearchResult(
        chunk_id=1,
        score=0.5,
        document_id="paper",
        title="Paper",
        page=1,
        section="Results",
        language=language,
        text=text,
    )


class CorrectedAnswerRepairTests(unittest.TestCase):
    def test_strips_internal_fact_checklist_identifiers(self) -> None:
        cleaned = _strip_internal_fact_identifiers(
            "The class names are encoded as text [F1]."
        )
        self.assertEqual(cleaned, "The class names are encoded as text.")

    def test_appends_only_a_missing_evidence_derived_fact(self) -> None:
        fact = PlannedFact(
            id="F1",
            claim="paper: Le module utilise des requêtes apprenables.",
            anchor="requêtes apprenables",
            references=["S1"],
        )
        repaired, kind = repair_corrected_answer_from_evidence(
            question="Pourquoi ce module est-il utile ?",
            generated_answer=(
                "Le module est utile car il filtre les informations pertinentes "
                "pour le modèle de langage [S1]."
            ),
            results=[_result(text="Learnable queries filter information.")],
            planned_facts=[fact],
        )
        self.assertEqual(kind, "fact_plan")
        self.assertIn("requêtes apprenables [S1]", repaired)

    def test_recovers_an_english_numeric_comparison_from_english_evidence(self) -> None:
        repaired, kind = repair_corrected_answer_from_evidence(
            question="What efficiency trade-off does the method report?",
            generated_answer="Le modèle est plus efficace [S1].",
            results=[
                _result(
                    text=(
                        "The method outperforms a larger baseline by 8.7% on a "
                        "benchmark, despite having 54x fewer trainable parameters."
                    )
                )
            ],
            retrieval_intents=["quantitative_comparison"],
        )
        self.assertEqual(kind, "numeric_evidence")
        self.assertIn("8.7%", repaired)
        self.assertIn("54x fewer trainable parameters [S1].", repaired)


if __name__ == "__main__":
    unittest.main()
