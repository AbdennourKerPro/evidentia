"""Unit tests for language, citation, and multi-source validation."""

from __future__ import annotations

import unittest

from app.response_contract import (
    INSUFFICIENT_EVIDENCE_MARKER,
    detect_question_language,
    validate_answer,
)
from app.schemas import SearchResult


def _result(position: int, document_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=position,
        score=0.5,
        document_id=document_id,
        title=f"Article {document_id}",
        page=position,
        section="Method",
        language="en",
        text=f"Evidence for {document_id}.",
    )


class QuestionLanguageTests(unittest.TestCase):
    def test_detects_french_question(self) -> None:
        self.assertEqual(
            detect_question_language(
                "Comment BLIP-2 et LLaVA relient-ils la vision au langage ?"
            ),
            "fr",
        )

    def test_detects_english_question(self) -> None:
        self.assertEqual(
            detect_question_language("How do CLIP and SAM use prompts?"),
            "en",
        )

    def test_detects_chinese_question(self) -> None:
        self.assertEqual(detect_question_language("这些模型如何使用图像？"), "zh")


class AnswerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [_result(1, "paper-a"), _result(2, "paper-b")]

    def test_accepts_valid_english_multisource_answer(self) -> None:
        validation = validate_answer(
            question="How do paper A and paper B differ?",
            generated_answer=(
                "Paper A uses the first mechanism [S1]. Paper B uses a different "
                "mechanism [S2]. Therefore, they differ in how the components "
                "are connected."
            ),
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(
            {citation.document_id for citation in validation.citations},
            {"paper-a", "paper-b"},
        )

    def test_accepts_valid_french_multisource_answer(self) -> None:
        validation = validate_answer(
            question="Comment les articles A et B diffèrent-ils ?",
            generated_answer=(
                "L'article A utilise le premier mécanisme [S1]. L'article B "
                "utilise une autre connexion [S2]. Ainsi, les deux méthodes "
                "sont différentes dans leur architecture."
            ),
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertTrue(validation.is_valid)

    def test_rejects_an_uncited_required_article(self) -> None:
        validation = validate_answer(
            question="How do paper A and paper B differ?",
            generated_answer=(
                "Paper A uses the first mechanism [S1]. The two papers are "
                "different in how they connect the components."
            ),
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertFalse(validation.is_valid)
        self.assertIn(
            "missing_citation_documents:paper-b",
            validation.issues,
        )

    def test_rejects_a_wrong_script(self) -> None:
        validation = validate_answer(
            question="Comment les articles A et B diffèrent-ils ?",
            generated_answer="这两个模型使用不同的方法进行训练 [S1] [S2]。",
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertFalse(validation.is_valid)
        self.assertIn("language_mismatch:expected=fr", validation.issues)

    def test_rejects_an_out_of_range_reference(self) -> None:
        validation = validate_answer(
            question="What is the difference between the two papers?",
            generated_answer=(
                "The first method is different from the second method [S3]."
            ),
            results=self.results,
        )

        self.assertFalse(validation.is_valid)
        self.assertIn("invalid_or_missing_citations", validation.issues)

    def test_requests_review_of_comparative_abstention(self) -> None:
        validation = validate_answer(
            question="How do paper A and paper B differ?",
            generated_answer=INSUFFICIENT_EVIDENCE_MARKER,
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.issues, ("premature_abstention",))

    def test_rejects_an_answer_mixed_with_the_abstention_marker(self) -> None:
        validation = validate_answer(
            question="How do paper A and paper B differ?",
            generated_answer=(
                "Paper A and paper B use different methods [S1] [S2]. "
                + INSUFFICIENT_EVIDENCE_MARKER
            ),
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertFalse(validation.is_valid)
        self.assertIn("mixed_answer_and_abstention_marker", validation.issues)

    def test_rejects_a_truncated_citation(self) -> None:
        validation = validate_answer(
            question="How do paper A and paper B differ?",
            generated_answer=(
                "Paper A and paper B use different methods [S1] [S2"
            ),
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertFalse(validation.is_valid)
        self.assertIn("truncated_or_unbalanced_answer", validation.issues)

    def test_accepts_grouped_citation_syntax(self) -> None:
        validation = validate_answer(
            question="How do paper A and paper B differ?",
            generated_answer=(
                "The first paper uses one method and the second paper uses "
                "another method. They are different [S1, S2]."
            ),
            results=self.results,
            required_document_ids=["paper-a", "paper-b"],
        )

        self.assertTrue(validation.is_valid)
        self.assertEqual(
            {citation.document_id for citation in validation.citations},
            {"paper-a", "paper-b"},
        )

    def test_accepts_plain_abstention_without_a_source_contract(self) -> None:
        validation = validate_answer(
            question="What is not present in the corpus?",
            generated_answer=INSUFFICIENT_EVIDENCE_MARKER,
            results=self.results,
        )

        self.assertTrue(validation.is_valid)
        self.assertTrue(validation.abstained)


if __name__ == "__main__":
    unittest.main()
