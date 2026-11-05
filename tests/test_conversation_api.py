import json
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import conversation_service, paper_ingestion, paper_registry
from app.main import app
from app.schemas import AgenticAskResponse, DiscoveryAudit
from tests.dynamic_helpers import RegistryTestCase

CID = "11111111-2222-3333-4444-555555555555"
OTHER = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
MAMBA = "arxiv-2312.00752v2"


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = [l for l in block.splitlines() if not l.startswith(":")]
        if not lines:
            continue
        name = lines[0].removeprefix("event: ")
        events.append((name, json.loads(lines[1].removeprefix("data: "))))
    return events


def fake_answer(question="q", **overrides):
    return AgenticAskResponse.model_validate({
        "question": question, "answer": "Mamba is a selective SSM [S1].", "citations": [], "evidence": [],
        "retrieved_chunks": 0, "abstained": False,
        "execution": {"strategy": "global", "expected_language": "en", "detected_document_ids": [],
                      "planned_queries": [], "candidate_chunks": 0, "selected_chunks": 0,
                      "correction_attempted": False, "response_contract_valid": True, "steps": []},
        **overrides,
    })


class ConversationApiTests(RegistryTestCase):
    def setUp(self):
        super().setUp()
        env = patch.dict(os.environ, {"OPENAI_API_KEY": "unit-test-key", "OPENAI_REASONING_EFFORT": "none"})
        env.start()
        self.addCleanup(env.stop)
        self.http = TestClient(app)
        self.addCleanup(self.http.close)

    def ask(self, body, cid=CID):
        with self.http.stream("POST", f"/conversations/{cid}/ask", json=body) as response:
            return response.status_code, response.headers.get("content-type"), parse_sse("".join(response.iter_text()))

    def test_invalid_uuid_is_rejected_everywhere(self):
        for call in (lambda: self.http.put("/conversations/not-a-uuid"),
                     lambda: self.http.get("/conversations/x/papers"),
                     lambda: self.http.delete("/conversations/x"),
                     lambda: self.http.post("/conversations/x/ask", json={"question": "abc def"})):
            self.assertEqual(call().status_code, 422)

    def test_sse_event_order_and_discovery_attached_to_the_response(self):
        def fake_discovery(conversation_id, question, emit, **kwargs):
            emit("discovery.started", {})
            emit("discovery.tool_call", {"name": "search_arxiv", "arguments": {}})
            return DiscoveryAudit(focus_document_ids=[], stopped_reason="finished", model_calls=1)

        with patch.object(conversation_service, "run_discovery", side_effect=fake_discovery), \
             patch.object(conversation_service, "answer_agentic_question", return_value=fake_answer()) as answer:
            status, content_type, events = self.ask({"question": "What is Mamba?"})
        self.assertEqual(status, 200)
        self.assertTrue(content_type.startswith("text/event-stream"))
        names = [name for name, _ in events]
        self.assertEqual(names, ["discovery.started", "discovery.tool_call", "answer.started", "answer", "done"])
        payload = dict(events)["answer"]
        self.assertEqual(payload["execution"]["discovery"]["stopped_reason"], "finished")
        self.assertEqual(payload["llm"]["call_count"], 0)
        self.assertEqual(paper_registry.recent_questions(CID), ["What is Mamba?"])
        self.assertEqual(answer.call_args.kwargs["corpus_document_ids"], [])

    def test_answer_only_sees_this_conversations_ready_papers(self):
        mine = self.make_paper()
        theirs = self.make_paper("2401.00001", "v1")
        building = self.make_paper("2401.00002", "v1", status="indexing")
        paper_registry.attach_paper(CID, mine.document_id, added_by="agent")
        paper_registry.attach_paper(CID, building.document_id, added_by="agent")
        paper_registry.attach_paper(OTHER, theirs.document_id, added_by="agent")
        with patch.object(conversation_service, "answer_agentic_question", return_value=fake_answer()) as answer:
            self.ask({"question": "tell me more", "discovery": "off",
                      "document_ids": [theirs.document_id, mine.document_id]})
        kwargs = answer.call_args.kwargs
        self.assertEqual(kwargs["corpus_document_ids"], [mine.document_id])
        self.assertEqual(kwargs["document_ids"], [mine.document_id])  # foreign id dropped
        self.assertEqual(list(kwargs["catalog"]), [mine.document_id])

    def test_discovery_off_never_calls_the_agent_and_empty_conversation_abstains(self):
        with patch.object(conversation_service, "run_discovery", side_effect=AssertionError("no discovery")):
            _, _, events = self.ask({"question": "What is Mamba?", "discovery": "off"})
        payload = dict(events)["answer"]
        self.assertTrue(payload["abstained"])
        self.assertNotIn("discovery.started", [n for n, _ in events])

    def test_errors_become_sse_error_events(self):
        from app.llm_gateway import LlmGatewayError

        with patch.object(conversation_service, "run_discovery",
                          side_effect=LlmGatewayError("quota", code="upstream_rate_limit")):
            _, _, events = self.ask({"question": "What is Mamba?"})
        self.assertEqual([n for n, _ in events], ["error", "done"])
        self.assertEqual(events[0][1]["code"], "upstream_rate_limit")
        with patch.object(conversation_service, "run_discovery", side_effect=RuntimeError("secret detail")):
            _, _, events = self.ask({"question": "What is Mamba?"})
        self.assertNotIn("secret", json.dumps(events))

    def test_papers_listing_detach_and_gc(self):
        paper = self.make_paper()
        paper_registry.attach_paper(CID, paper.document_id, added_by="agent")
        paper_registry.attach_paper(OTHER, paper.document_id, added_by="agent")
        listing = self.http.get(f"/conversations/{CID}/papers").json()
        self.assertEqual([p["document_id"] for p in listing["papers"]], [paper.document_id])
        with patch.object(paper_ingestion, "delete_document_chunks") as delete, \
             patch.object(paper_ingestion, "clear_bm25_cache"):
            self.assertEqual(self.http.delete(f"/conversations/{CID}/papers/{paper.document_id}").status_code, 200)
            delete.assert_not_called()  # still used by the other conversation
            self.assertEqual(self.http.delete(f"/conversations/{CID}/papers/{paper.document_id}").status_code, 404)
            self.assertEqual(self.http.delete(f"/conversations/{OTHER}/papers/{paper.document_id}").status_code, 200)
            delete.assert_called_once()
        self.assertIsNone(paper_registry.get_paper(paper.document_id))

    def test_deleting_a_conversation_purges_only_unshared_unpinned_papers(self):
        shared = self.make_paper("2401.00001", "v1")
        solo = self.make_paper("2401.00002", "v1")
        pinned = self.make_paper("2304.02643", "v1", pinned=True)
        for p in (shared, solo, pinned):
            paper_registry.attach_paper(CID, p.document_id, added_by="agent")
        paper_registry.attach_paper(OTHER, shared.document_id, added_by="agent")
        paper_registry.record_turn(CID, "hello there")
        with patch.object(paper_ingestion, "delete_document_chunks"), patch.object(paper_ingestion, "clear_bm25_cache"):
            result = self.http.delete(f"/conversations/{CID}").json()
        self.assertEqual(result["purged_documents"], [solo.document_id])
        self.assertFalse(paper_registry.conversation_exists(CID))
        self.assertIsNone(paper_registry.get_paper(solo.document_id))
        self.assertIsNotNone(paper_registry.get_paper(shared.document_id))
        self.assertIsNotNone(paper_registry.get_paper(pinned.document_id))

    def test_manual_add_uses_the_same_ingestion(self):
        paper = self.make_paper()
        with patch.object(paper_ingestion, "ingest_arxiv_paper", return_value=paper) as ingest:
            response = self.http.post(f"/conversations/{CID}/papers", json={"arxiv_id": "2312.00752v2"})
        self.assertEqual(response.status_code, 200)
        ingest.assert_called_once_with("2312.00752", "v2")
        self.assertEqual(paper_registry.list_conversation_papers(CID)[0].added_by, "user")
        with patch.object(paper_ingestion, "ingest_arxiv_paper",
                          side_effect=paper_ingestion.IngestionError("not found")):
            self.assertEqual(self.http.post(f"/conversations/{CID}/papers", json={"arxiv_id": "2312.00753"}).status_code, 422)
        self.assertEqual(self.http.post(f"/conversations/{CID}/papers", json={"arxiv_id": "hep-th/9901001"}).status_code, 422)


class ClassicEndpointsStayScopedTests(unittest.TestCase):
    """The classic API must never surface papers added by conversations."""

    def test_arxiv_search_and_documents_are_limited_to_historical_papers(self):
        from app.schemas import IndexedDocument

        http = TestClient(app)
        self.addCleanup(http.close)
        docs = [IndexedDocument(document_id="clip-2021", title="CLIP", indexed_chunks=3),
                IndexedDocument(document_id=MAMBA, title="Mamba", indexed_chunks=3)]
        with patch("app.main.list_indexed_documents", return_value=docs):
            ids = [d["document_id"] for d in http.get("/arxiv/documents").json()["documents"]]
        self.assertEqual(ids, ["clip-2021"])
        with patch("app.main.embed_query", return_value=[0.1]), patch("app.main.search_chunks", return_value=[]) as search:
            http.get("/arxiv/search", params={"query": "anything"})
        self.assertIn("clip-2021", search.call_args.kwargs["document_ids"])
        self.assertNotIn(MAMBA, search.call_args.kwargs["document_ids"])


if __name__ == "__main__":
    unittest.main()
