"""Language-contract regressions, including short answers outside the benchmark."""

import json
import unittest
from unittest.mock import patch

from app.language_verification import (
    parse_language_verdict, validate_answer_with_language_check,
)
from app.response_contract import assess_answer_language, validate_answer
from app.schemas import SearchResult


SHORT_FRENCH = "Le réseau atteint 81,7 % après adaptation [S1]."


class LanguageHeuristicTests(unittest.TestCase):
    def test_short_technical_french_is_uncertain_not_wrong(self):
        self.assertEqual(assess_answer_language(SHORT_FRENCH, "fr"), "uncertain")

    def test_numeric_answer_is_uncertain(self):
        self.assertEqual(assess_answer_language("81.7 % [S1].", "fr"), "uncertain")

    def test_short_english_without_markers_is_uncertain(self):
        self.assertEqual(assess_answer_language("Network achieves 81.7% [S1].", "fr"), "uncertain")

    def test_clear_english_matches(self):
        self.assertEqual(assess_answer_language("The model learns from the images [S1].", "en"), "match")

    def test_clear_english_mismatches_french(self):
        self.assertEqual(assess_answer_language("The model learns from the images [S1].", "fr"), "mismatch")

    def test_clear_french_mismatches_english(self):
        self.assertEqual(assess_answer_language("Le modèle est plus efficace [S1].", "en"), "mismatch")

    def test_foreign_sentence_does_not_hide_in_long_french_answer(self):
        text = "Les modèles sont utiles et utilisent les données pour les tâches. " * 4
        self.assertEqual(assess_answer_language(text + "The model learns from the images.", "fr"), "mismatch")

    def test_code_numbers_and_citations_do_not_vote_for_english(self):
        text = "Les modèles sont utiles et utilisent les données [S1]. `the and with` 81.7%."
        self.assertEqual(assess_answer_language(text, "fr"), "match")

    def test_chinese_prose_matches(self):
        self.assertEqual(assess_answer_language("模型使用图像学习表示 [S1]。", "zh"), "match")

    def test_chinese_prose_mismatches_french(self):
        self.assertEqual(assess_answer_language("模型使用图像学习表示 [S1]。", "fr"), "mismatch")

    def test_mixed_scripts_are_uncertain(self):
        self.assertEqual(assess_answer_language("The model learns from images 图像 [S1].", "en"), "uncertain")

    def test_other_language_is_not_automatically_accepted(self):
        self.assertEqual(assess_answer_language("Modelo alcanza precisión elevada [S1].", "fr"), "uncertain")


class LanguageVerificationTests(unittest.TestCase):
    def setUp(self):
        self.results = [SearchResult(
            chunk_id=1, score=0.8, document_id="paper", title="Paper",
            page=1, section="Results", language="en", text="Accuracy is 81.7%.",
        )]

    def check(self, answer=SHORT_FRENCH):
        return validate_answer_with_language_check(
            question="Quelle précision atteint le modèle ?",
            generated_answer=answer, results=self.results,
        )

    @patch("app.language_verification.generate_chat", return_value='{"status":"match"}')
    def test_ambiguous_french_can_be_accepted_without_rewrite(self, generate):
        result = self.check()
        self.assertTrue(result.is_valid)
        self.assertEqual(result.language_check_method, "llm_verification")
        generate.assert_called_once()
        self.assertEqual(json.loads(generate.call_args.kwargs["user_message"])["answer"], SHORT_FRENCH)

    @patch("app.language_verification.generate_chat", return_value='{"status":"mismatch"}')
    def test_short_foreign_prose_is_rejected(self, generate):
        result = self.check("Network achieves 81.7% [S1].")
        self.assertFalse(result.is_publishable)
        self.assertIn("language_mismatch:expected=fr", result.issues)
        generate.assert_called_once()

    @patch("app.language_verification.generate_chat", return_value="not JSON")
    def test_malformed_verdict_stays_blocking_and_is_not_retried(self, generate):
        result = self.check()
        self.assertEqual(result.language_status, "uncertain")
        self.assertFalse(result.is_publishable)
        generate.assert_called_once()

    @patch("app.language_verification.generate_chat", return_value='{"status":"match"}')
    def test_language_success_does_not_bypass_citations(self, generate):
        result = self.check(SHORT_FRENCH.replace("[S1]", "[S9]"))
        self.assertEqual(result.language_status, "match")
        self.assertIn("invalid_or_missing_citations", result.issues)
        self.assertFalse(result.is_publishable)

    @patch("app.language_verification.generate_chat")
    def test_clear_language_and_abstention_need_no_classifier(self, generate):
        self.check("Les modèles sont utiles et utilisent les données [S1].")
        self.check("The model learns from the images [S1].")
        self.check("INSUFFICIENT_EVIDENCE")
        generate.assert_not_called()

    @patch("app.language_verification.generate_chat", side_effect=RuntimeError("access blocked"))
    def test_api_failure_is_not_disguised_as_success(self, generate):
        with self.assertRaisesRegex(RuntimeError, "access blocked"):
            self.check()

    def test_local_validation_does_not_call_an_api(self):
        result = validate_answer(
            question="Quelle précision atteint le modèle ?",
            generated_answer=SHORT_FRENCH, results=self.results,
        )
        self.assertEqual(result.issues, ("language_uncertain:expected=fr",))

    def test_strict_verdict_schema(self):
        for raw in ('[]', 'null', '{"status":true}', '{"status":"yes"}',
                    '{"status":"match","ignore_citations":true}', '```{"status":"match"}```'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_language_verdict(raw), "uncertain")


if __name__ == "__main__":
    unittest.main()
