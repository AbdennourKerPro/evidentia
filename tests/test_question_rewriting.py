import json
import os
import unittest
from unittest.mock import patch

from app import conversation_service, paper_registry, question_rewriting
from app.llm_gateway import LlmGatewayError
from app.schemas import AgenticAskResponse, DiscoveryAudit
from tests.dynamic_helpers import RegistryTestCase

CID = "11111111-2222-3333-4444-555555555555"


class RewriteTests(unittest.TestCase):
    def test_clean_rewrite(self):
        self.assertEqual(question_rewriting.clean_rewrite('  "Explain the terms of the SimCLR loss"\n'), "Explain the terms of the SimCLR loss")
        self.assertEqual(question_rewriting.clean_rewrite("First line?\nsecond line"), "First line?")
        self.assertIsNone(question_rewriting.clean_rewrite("  "))
        self.assertIsNone(question_rewriting.clean_rewrite("x" * 501))

    def test_first_question_never_calls_the_llm(self):
        with patch.object(question_rewriting, "generate_chat", side_effect=AssertionError):
            self.assertEqual(question_rewriting.rewrite_follow_up("What is SimCLR?", [], []), "What is SimCLR?")

    def test_follow_up_uses_previous_questions_and_titles(self):
        with patch.object(question_rewriting, "generate_chat", return_value="Explain the terms of SimCLR's NT-Xent loss.") as chat:
            result = question_rewriting.rewrite_follow_up(
                "Explain the different terms.", ["What is the training objective in SimCLR?", "Give me the formula"],
                ["A Simple Framework for Contrastive Learning"])
        self.assertEqual(result, "Explain the terms of SimCLR's NT-Xent loss.")
        message = chat.call_args.kwargs["user_message"]
        self.assertIn("SimCLR", message)
        self.assertIn("A Simple Framework", message)
        self.assertIn("CURRENT QUESTION:\nExplain the different terms.", message)

    def test_failures_and_garbage_fall_back_to_the_original(self):
        with patch.object(question_rewriting, "generate_chat", side_effect=LlmGatewayError("down", code="x")):
            self.assertEqual(question_rewriting.rewrite_follow_up("And the terms?", ["q one"], []), "And the terms?")
        with patch.object(question_rewriting, "generate_chat", return_value=""):
            self.assertEqual(question_rewriting.rewrite_follow_up("And the terms?", ["q one"], []), "And the terms?")


class ConversationRewriteTests(RegistryTestCase):
    def response(self, question):
        return AgenticAskResponse.model_validate({
            "question": question, "answer": "ok [S1]", "citations": [], "evidence": [], "retrieved_chunks": 0,
            "abstained": False, "execution": {"strategy": "global", "expected_language": "en", "detected_document_ids": [],
            "planned_queries": [], "candidate_chunks": 0, "selected_chunks": 0, "correction_attempted": False,
            "response_contract_valid": True, "steps": []}})

    def run_turn(self, question, **kwargs):
        events = []
        paper = self.make_paper("2002.05709", "v3", title="A Simple Framework for Contrastive Learning")
        paper_registry.attach_paper(CID, paper.document_id, added_by="agent")
        with patch.object(conversation_service, "rewrite_follow_up", **kwargs) as rewrite, \
             patch.object(conversation_service, "run_discovery", return_value=DiscoveryAudit()) as discovery, \
             patch.object(conversation_service, "answer_agentic_question", side_effect=lambda **k: self.response(k["question"])) as answer, \
             patch.object(conversation_service, "summarize_llm_calls", return_value=None):
            response = conversation_service.answer_for_conversation(
                conversation_id=CID, question=question, limit=5, pipeline="agentic", discovery="auto",
                document_ids=None, emit=lambda e, d: events.append((e, d)))
        return response, events, rewrite, discovery, answer

    def test_follow_up_is_rewritten_for_discovery_and_answering(self):
        paper_registry.record_turn(CID, "What is the training objective in SimCLR?")
        response, events, rewrite, discovery, answer = self.run_turn(
            "Explain the different terms.", return_value="Explain the terms of SimCLR's loss.")
        self.assertEqual(rewrite.call_args.args[1], ["What is the training objective in SimCLR?"])
        self.assertEqual(discovery.call_args.args[1], "Explain the terms of SimCLR's loss.")
        self.assertEqual(discovery.call_args.kwargs["recorded_question"], "Explain the different terms.")
        self.assertEqual(answer.call_args.kwargs["question"], "Explain the terms of SimCLR's loss.")
        self.assertTrue(answer.call_args.kwargs["check_formulas"])
        self.assertIn(("question.rewritten", {"question": "Explain the terms of SimCLR's loss."}), events)
        self.assertEqual(response.execution.rewritten_question, "Explain the terms of SimCLR's loss.")
        # The history keeps what the user actually typed.
        self.assertEqual(paper_registry.recent_questions(CID, 2)[-1], "Explain the different terms.")

    def test_first_question_and_unchanged_question_report_no_rewrite(self):
        response, events, rewrite, *_ = self.run_turn("What is SimCLR?", return_value="unused")
        self.assertEqual(rewrite.call_count, 0)
        self.assertIsNone(response.execution.rewritten_question)
        paper_registry.record_turn(CID, "earlier question here")
        response, events, rewrite, *_ = self.run_turn("Fully standalone question?", side_effect=lambda q, p, t: q)
        self.assertEqual(rewrite.call_count, 1)
        self.assertIsNone(response.execution.rewritten_question)
        self.assertNotIn("question.rewritten", [e for e, _ in events])

    def test_discovery_prompt_does_not_repeat_the_original_question_as_history(self):
        from app.source_discovery import build_user_message

        paper_registry.record_turn(CID, "first question here")
        paper_registry.record_turn(CID, "Explain the different terms.")
        text = build_user_message(CID, "Explain the terms of SimCLR's loss.", "Explain the different terms.")
        self.assertIn("- first question here", text)
        self.assertNotIn("- Explain the different terms.", text)


if __name__ == "__main__":
    unittest.main()
