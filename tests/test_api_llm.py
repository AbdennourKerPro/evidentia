"""HTTP and real LangGraph integration with mocked retrieval and no external I/O."""

from __future__ import annotations

import json
import os
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import SearchResult
from tests.openai_fixtures import FAKE_API_KEY, TEST_ENV, mock_client, response_body


def evidence(document_id: str, chunk_id: int) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, score=0.9, document_id=document_id, title=document_id,
        page=2, section="Method", language="en",
        text="This method learns representations from its input data.",
    )


class ApiLlmTests(unittest.TestCase):
    def setUp(self) -> None:
        # Existing call-count tests deliberately exercise the v8 profile.
        # Deployment defaults must not silently change their mocked sequence.
        self.environment = patch.dict(os.environ, {**TEST_ENV, "RAG_RETRIEVAL_PROFILE": "question_rerank"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.http = TestClient(app)
        self.addCleanup(self.http.close)
        self.requests: list[dict] = []

    def install_answers(self, answers: list[str]) -> None:
        def handler(request):
            self.requests.append(json.loads(request.content))
            if not answers:
                raise AssertionError("Unexpected extra LLM request.")
            return httpx.Response(200, json=response_body(answers.pop(0)))

        client = mock_client(handler)
        self.addCleanup(client.close)
        client_patch = patch("app.llm_gateway._get_client", return_value=client)
        client_patch.start()
        self.addCleanup(client_patch.stop)

    def test_status_is_not_a_remote_access_check(self) -> None:
        with patch("app.llm_gateway.OpenAI") as constructor:
            response = self.http.get("/llm/status")
        constructor.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["configured"])
        self.assertFalse(response.json()["access_verified"])
        self.assertEqual(response.json()["model_id"], "gpt-6-luna")
        self.assertNotIn(FAKE_API_KEY, response.text)
        self.assertNotIn("downloaded", response.json())

    def test_ui_exposes_model_and_usage_elements(self) -> None:
        response = self.http.get("/ui/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="llm-status"', response.text)
        self.assertIn('class="llm-usage answer-state"', response.text)
        self.assertNotIn("Qwen local", response.text)

    def test_ui_has_conversation_navigation_and_modal_settings(self) -> None:
        response = self.http.get("/ui/")
        self.assertIn('id="conversation-list"', response.text)
        self.assertIn('<dialog id="settings-dialog"', response.text)
        self.assertIn('id="settings-button"', response.text)
        self.assertNotIn('class="topbar"', response.text)
        self.assertIn('<script type="module"', response.text)
        self.assertGreater(response.text.index('id="pipeline-mode"'), response.text.index('<dialog'))
        self.assertGreater(response.text.index('href="/review/"'), response.text.index('<dialog'))
        for path in ("/ui/markdown.js", "/ui/conversations.js"):
            asset = self.http.get(path)
            self.assertEqual(asset.status_code, 200)
            self.assertIn("javascript", asset.headers["content-type"])

    def test_ui_serves_name_independent_vector_logo(self) -> None:
        response = self.http.get("/ui/")
        self.assertIn('class="brand-logo"', response.text)
        self.assertIn('aria-label="Accueil — Articles scientifiques"', response.text)
        self.assertNotIn("Evidentia", response.text)
        self.assertNotIn('class="brand sidebar-label"', response.text)
        logo = self.http.get("/ui/logo.svg")
        self.assertEqual(logo.status_code, 200)
        self.assertIn("image/svg+xml", logo.headers["content-type"])
        root = ET.fromstring(logo.text)
        self.assertEqual(root.attrib["viewBox"], "0 0 100 100")
        tags = [element.tag.rsplit("}", 1)[-1] for element in root.iter()]
        self.assertEqual(tags.count("path"), 2)
        self.assertFalse(set(tags) & {"image", "script", "foreignObject", "rect", "circle"})

    def test_status_reports_missing_key_without_breaking_health(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            status_response = self.http.get("/llm/status")
            health_response = self.http.get("/health")
        self.assertFalse(status_response.json()["configured"])
        self.assertEqual(health_response.status_code, 200)

    def test_invalid_configuration_has_sanitized_http_error(self) -> None:
        with patch.dict(os.environ, {"OPENAI_MAX_OUTPUT_TOKENS": FAKE_API_KEY}):
            response = self.http.get("/llm/status")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "invalid_llm_configuration")
        self.assertNotIn(FAKE_API_KEY, response.text)

    def test_baseline_returns_citations_full_chunks_and_usage(self) -> None:
        self.install_answers(["The method learns representations from its input data [S1]."])
        with patch("app.rag_service.embed_query", return_value=[0.1]), patch(
            "app.rag_service.search_chunks", return_value=[evidence("clip-2021", 1)]
        ):
            response = self.http.post("/ask", json={"question": "What does the method do?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertFalse(payload["abstained"])
        self.assertEqual(payload["citations"][0]["reference"], "S1")
        self.assertIn("input data", payload["evidence"][0]["text"])
        self.assertEqual(payload["llm"]["call_count"], 1)
        self.assertEqual(payload["llm"]["total_tokens"], 46)

    def test_no_evidence_does_not_call_api(self) -> None:
        with patch("app.rag_service.embed_query", return_value=[0.1]), patch(
            "app.rag_service.search_chunks", return_value=[]
        ), patch("app.llm_gateway._get_client") as client_factory:
            response = self.http.post("/ask", json={"question": "What does the method do?"})
        client_factory.assert_not_called()
        self.assertTrue(response.json()["abstained"])
        self.assertEqual(response.json()["llm"]["call_count"], 0)

    def test_baseline_short_french_answer_has_one_language_check(self) -> None:
        self.install_answers([
            "Le réseau atteint 81,7 % après adaptation [S1].",
            '{"status":"match"}',
        ])
        with patch("app.rag_service.embed_query", return_value=[0.1]), patch(
            "app.rag_service.search_chunks", return_value=[evidence("clip-2021", 1)]
        ):
            response = self.http.post("/ask", json={"question": "Quelle précision atteint le modèle ?"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["abstained"])
        self.assertEqual(response.json()["llm"]["call_count"], 2)

    def test_missing_key_on_ask_is_not_a_500_error(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch(
            "app.rag_service.embed_query", return_value=[0.1]
        ), patch("app.rag_service.search_chunks", return_value=[evidence("clip-2021", 1)]):
            response = self.http.post("/ask", json={"question": "What does the method do?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["code"], "missing_api_key")

    def test_langgraph_collects_planning_and_generation_calls(self) -> None:
        self.install_answers([
            json.dumps({
                "clip-2021": "CLIP representations from input data",
                "dinov2-2023": "DINOv2 representations from input data",
            }),
            "The first method learns representations from its input data [S1]. "
            "The second method also learns representations from its input data [S2].",
        ])
        def retrieve(*, document_id, **kwargs):
            result = evidence(document_id, 1 if document_id == "clip-2021" else 2)
            return SimpleNamespace(fused=[result], dense_rankings=[[result]], lexical_rankings=[[result]])

        with patch("app.agentic_rag.embed_queries", side_effect=lambda queries: [[0.1] for _ in queries]), patch(
            "app.agentic_rag.retrieve_multifacet_candidates", side_effect=retrieve
        ):
            response = self.http.post("/agentic/ask", json={
                "question": "Compare CLIP and DINOv2.",
                "document_ids": ["clip-2021", "dinov2-2023"],
                "limit": 2,
            })
        payload = response.json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertFalse(payload["abstained"], payload)
        self.assertEqual(payload["execution"]["strategy"], "per_source")
        self.assertEqual(payload["llm"]["call_count"], 2)
        self.assertEqual(payload["llm"]["total_tokens"], 92)
        self.assertTrue(all(request["model"] == "gpt-6-luna" for request in self.requests))
        self.assertEqual(len(payload["citations"]), 2)

    def test_langgraph_correction_uses_same_api_and_is_counted(self) -> None:
        self.install_answers([
            "La méthode apprend des représentations à partir des données [S1].",
            "The method learns representations from the input data [S1].",
        ])
        with patch("app.agentic_rag.embed_query", return_value=[0.1]), patch(
            "app.agentic_rag.search_chunks", return_value=[evidence("clip-2021", 1)]
        ):
            response = self.http.post("/agentic/ask", json={"question": "What does the method do?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertFalse(payload["abstained"], payload)
        self.assertTrue(payload["execution"]["correction_attempted"])
        self.assertEqual(payload["llm"]["call_count"], 2)
        self.assertEqual(len(self.requests), 2)

    def test_candidate_recovery_http_includes_coverage_calls(self) -> None:
        aspect = "How the method learns representations"
        self.install_answers([
            json.dumps({"requirements": [{
                "aspect": aspect, "document_id": "clip-2021",
                "search_query": "representation learning mechanism",
            }]}),
            json.dumps({"requirements": [{
                "aspect": aspect, "document_id": "clip-2021",
                "reference": "S1", "quote": "learns representations from its input data",
                "assessment": "The quote directly describes the mechanism.",
                "search_query": None,
            }]}),
            "The method learns representations from its input data [S1].",
        ])
        with patch.dict(os.environ, {"RAG_RETRIEVAL_PROFILE": "candidate_recovery"}), patch(
            "app.agentic_rag.embed_query", return_value=[0.1]
        ), patch("app.agentic_rag.search_chunks", return_value=[evidence("clip-2021", 1)]):
            response = self.http.post("/agentic/ask", json={"question": "What does the method do?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertEqual(payload["execution"]["retrieval_profile"], "candidate_recovery")
        self.assertTrue(payload["execution"]["coverage_checks"][0]["sufficient"])
        self.assertEqual(payload["llm"]["call_count"], 3)
        self.assertEqual(payload["citations"][0]["reference"], "S1")
        self.assertIn("input data", payload["evidence"][0]["text"])

    def test_langgraph_uncertain_language_is_checked_without_rewrite(self) -> None:
        self.install_answers([
            "What does this model do?",
            "Le réseau atteint 81,7 % après adaptation [S1].",
            '{"status":"match"}',
        ])
        with patch("app.agentic_rag.embed_query", return_value=[0.1]), patch(
            "app.agentic_rag.search_chunks", return_value=[evidence("clip-2021", 1)]
        ):
            response = self.http.post("/agentic/ask", json={"question": "Que fait ce modèle ?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertFalse(payload["abstained"], payload)
        self.assertFalse(payload["execution"]["correction_attempted"])
        self.assertEqual(payload["execution"]["language_status"], "match")
        self.assertEqual(payload["execution"]["language_check_method"], "llm_verification")
        self.assertEqual(payload["llm"]["call_count"], 3)

    def test_langgraph_uncertain_verdict_cannot_loop_or_publish(self) -> None:
        self.install_answers([
            "What does this model do?",
            "Le réseau atteint 81,7 % après adaptation [S1].",
            '{"status":"uncertain"}',
            "Le réseau atteint 81,7 % après adaptation [S1].",
            '{"status":"uncertain"}',
        ])
        with patch("app.agentic_rag.embed_query", return_value=[0.1]), patch(
            "app.agentic_rag.search_chunks", return_value=[evidence("clip-2021", 1)]
        ):
            response = self.http.post("/agentic/ask", json={"question": "Que fait ce modèle ?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertTrue(payload["abstained"], payload)
        self.assertTrue(payload["execution"]["correction_attempted"])
        self.assertIn("language_uncertain:expected=fr", payload["execution"]["contract_issues"])
        self.assertEqual(payload["llm"]["call_count"], 5)


if __name__ == "__main__":
    unittest.main()
