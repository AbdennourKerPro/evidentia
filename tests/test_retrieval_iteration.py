"""Check experiment metadata and scoring without invoking the application."""

import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.evaluation import EvaluationCase, EvidenceTarget
from app.schemas import RetrievalAudit
from scripts import evaluate_retrieval_iteration as runner
from tests.test_retrieval_policy import result


class RetrievalIterationTests(unittest.TestCase):
    def test_fingerprint_ignores_relevance_score(self):
        first = result(1, "Stable source text", 0.2)
        second = first.model_copy(update={"score": 99.0})
        self.assertEqual(
            runner.corpus_fingerprint({"paper": [first]}),
            runner.corpus_fingerprint({"paper": [second]}),
        )

    def test_fingerprint_orders_chunks_stably(self):
        first, second = result(1, "First"), result(2, "Second")
        self.assertEqual(
            runner.corpus_fingerprint({"paper": [first, second]}),
            runner.corpus_fingerprint({"paper": [second, first]}),
        )

    def test_labels_target_ranks_after_the_response(self):
        target = result(1, "The protocol selects a reference image.")
        case = EvaluationCase(
            id="example-method", category="method", language="en",
            question="What is the experimental protocol?",
            expected_document_ids=["paper"],
            evidence_targets=[EvidenceTarget(document_id="paper", match_phrases=["reference image"])],
            reference_answer="It selects a reference image.",
            required_answer_terms=[["reference image"]],
        )
        response = SimpleNamespace(execution=SimpleNamespace(retrieval_audit=[
            RetrievalAudit(document_id="paper", primary_query=case.question, candidate_ids=[2, 1], ranked_ids=[1, 2])
        ]))
        scored = runner.target_rank_diagnostics(case, response, {"paper": [target]})
        self.assertEqual(scored["target_ranks"][0]["candidate_ids"], 2)
        self.assertEqual(scored["target_ranks"][0]["ranked_ids"], 1)

    def test_existing_report_is_rejected_before_any_paid_call(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            output.touch()
            with patch.object(runner, "parse_args", return_value=argparse.Namespace(output=output)), patch.object(
                runner, "answer_agentic_question"
            ) as answer:
                with self.assertRaisesRegex(ValueError, "already exists"):
                    runner.main()
            answer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
