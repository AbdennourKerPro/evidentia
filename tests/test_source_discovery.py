import json
import os
import unittest
from unittest.mock import patch

import httpx

from app import arxiv_client, paper_ingestion, paper_registry, source_discovery
from app.llm_gateway import _get_client
from tests.dynamic_helpers import RegistryTestCase
from tests.openai_fixtures import TEST_ENV, function_call_body, mock_client

CID = "11111111-2222-3333-4444-555555555555"
META = arxiv_client.ArxivPaperMeta(arxiv_id="2312.00752", latest_version="v2",
                                   title="Mamba: Linear-Time Sequence Modeling", abstract="SSM")


class SourceDiscoveryTests(RegistryTestCase):
    def setUp(self):
        super().setUp()
        env = patch.dict(os.environ, {**TEST_ENV, "MAX_NEW_PAPERS_PER_TURN": "2", "MAX_PAPERS_PER_CONVERSATION": "3"})
        env.start()
        self.addCleanup(env.stop)
        _get_client.cache_clear()
        self.addCleanup(_get_client.cache_clear)
        self.requests = []

    def script(self, calls):
        bodies = [function_call_body(n, a, f"c{i}") for i, (n, a) in enumerate(calls)]

        def handler(request):
            self.requests.append(json.loads(request.content))
            return httpx.Response(200, json=bodies.pop(0))

        client = mock_client(handler)
        self.addCleanup(client.close)
        p = patch("app.llm_gateway._get_client", return_value=client)
        p.start()
        self.addCleanup(p.stop)

    def fake_ingest(self, arxiv_id, version=None, *, short_name=None, on_progress=None):
        if on_progress:
            on_progress("downloading", {})
        return paper_registry.upsert_paper(
            document_id=f"arxiv-{arxiv_id}{version or 'v1'}", arxiv_id=arxiv_id, version=version or "v1",
            title=f"Paper {arxiv_id}", short_name=short_name, status="ready", chunk_count=7)

    def test_search_add_finish_flow(self):
        self.script([
            ("search_arxiv", {"query": "mamba selective state space", "max_results": 3, "category": None, "sort_by": "relevance"}),
            ("add_paper_to_conversation", {"arxiv_id": "2312.00752", "short_name": "Mamba", "reason": "asked about Mamba"}),
            ("finish_source_selection", {"focus_document_ids": ["arxiv-2312.00752v1", "ghost"], "rationale": "r"}),
        ])
        events = []
        with patch.object(arxiv_client, "search", return_value=[META]) as search, \
             patch.object(paper_ingestion, "ingest_arxiv_paper", side_effect=self.fake_ingest):
            paper_registry.record_turn(CID, "What is Mamba?")
            audit = source_discovery.run_discovery(CID, "What is Mamba?", lambda k, p: events.append(k))
        search.assert_called_once()
        self.assertEqual([c.name for c in audit.tool_calls],
                         ["search_arxiv", "add_paper_to_conversation", "finish_source_selection"])
        self.assertTrue(all(c.ok for c in audit.tool_calls))
        self.assertEqual(audit.added_document_ids, ["arxiv-2312.00752v1"])
        self.assertEqual(audit.focus_document_ids, ["arxiv-2312.00752v1"])  # unknown id filtered
        self.assertEqual(audit.stopped_reason, "finished")
        self.assertIn("paper.added", events)
        self.assertIn("ingestion.progress", events)
        attached = paper_registry.list_conversation_papers(CID)
        self.assertEqual([(p.document_id, p.added_by) for p in attached], [("arxiv-2312.00752v1", "agent")])

    def test_prompt_lists_attached_papers_and_previous_questions(self):
        paper = self.make_paper("2304.02643", "v1", title="Segment Anything", pinned=True)
        paper_registry.attach_paper(CID, paper.document_id, added_by="seed")
        paper_registry.record_turn(CID, "first question")
        paper_registry.record_turn(CID, "second question")
        self.script([("finish_source_selection", {"focus_document_ids": [paper.document_id], "rationale": "r"})])
        source_discovery.run_discovery(CID, "second question")
        user_text = self.requests[0]["input"][0]["content"]
        self.assertIn("arxiv-2304.02643v1", user_text)
        self.assertIn("- first question", user_text)
        self.assertEqual(user_text.count("second question"), 1)  # not repeated as history
        self.assertEqual({t["name"] for t in self.requests[0]["tools"]},
                         {"search_arxiv", "lookup_arxiv_ids", "add_paper_to_conversation", "finish_source_selection"})
        self.assertTrue(all(t["strict"] and t["parameters"]["additionalProperties"] is False
                            for t in self.requests[0]["tools"]))

    def test_download_budget_per_turn(self):
        adds = [("add_paper_to_conversation", {"arxiv_id": f"2401.0000{i}", "short_name": None, "reason": "r"})
                for i in range(1, 4)]
        self.script([*adds, ("finish_source_selection", {"focus_document_ids": [], "rationale": "r"})])
        with patch.object(paper_ingestion, "ingest_arxiv_paper", side_effect=self.fake_ingest) as ingest:
            audit = source_discovery.run_discovery(CID, "q about three papers")
        self.assertEqual(ingest.call_count, 2)
        self.assertEqual([c.error for c in audit.tool_calls][:3], [None, None, "turn_budget_exceeded"])

    def test_cached_papers_do_not_use_the_download_budget(self):
        for i in range(1, 3):
            self.make_paper(f"2401.0000{i}", "v1")
        adds = [("add_paper_to_conversation", {"arxiv_id": f"2401.0000{i}", "short_name": None, "reason": "r"})
                for i in (1, 2)]
        self.script([*adds, ("finish_source_selection", {"focus_document_ids": [], "rationale": "r"})])
        with patch.object(paper_ingestion, "ingest_arxiv_paper", side_effect=self.fake_ingest):
            audit = source_discovery.run_discovery(CID, "q")
        self.assertTrue(all(c.ok for c in audit.tool_calls))
        self.assertEqual(len(audit.added_document_ids), 2)

    def test_conversation_full(self):
        for i in range(1, 4):
            p = self.make_paper(f"2402.0000{i}", "v1")
            paper_registry.attach_paper(CID, p.document_id, added_by="agent")
        self.script([
            ("add_paper_to_conversation", {"arxiv_id": "2401.00009", "short_name": None, "reason": "r"}),
            ("finish_source_selection", {"focus_document_ids": [], "rationale": "r"}),
        ])
        with patch.object(paper_ingestion, "ingest_arxiv_paper", side_effect=self.fake_ingest) as ingest:
            audit = source_discovery.run_discovery(CID, "q")
        ingest.assert_not_called()
        self.assertEqual(audit.tool_calls[0].error, "conversation_full")

    def test_ingestion_failure_is_reported_not_attached(self):
        self.script([
            ("add_paper_to_conversation", {"arxiv_id": "2401.00001", "short_name": None, "reason": "r"}),
            ("finish_source_selection", {"focus_document_ids": [], "rationale": "r"}),
        ])
        with patch.object(paper_ingestion, "ingest_arxiv_paper",
                          side_effect=paper_ingestion.IngestionError("PDF too large")):
            audit = source_discovery.run_discovery(CID, "q")
        self.assertEqual(audit.tool_calls[0].error, "ingestion_failed")
        self.assertEqual(paper_registry.list_conversation_papers(CID), [])

    def test_search_budget_and_errors_and_in_conversation_flags(self):
        paper = self.make_paper("2312.00752", "v2")
        paper_registry.attach_paper(CID, paper.document_id, added_by="agent")
        search_args = {"query": "state space model", "max_results": 2, "category": None, "sort_by": "relevance"}
        self.script([("search_arxiv", search_args)] * 4
                    + [("finish_source_selection", {"focus_document_ids": [], "rationale": "r"})])
        with patch.object(arxiv_client, "search", return_value=[META]):
            audit = source_discovery.run_discovery(CID, "q")
        errors = [c.error for c in audit.tool_calls]
        self.assertEqual(errors[:3], [None, None, None])
        self.assertEqual(errors[3], "search_budget_exceeded")
        session = source_discovery._Session(CID, lambda *_: None)
        with patch.object(arxiv_client, "search", return_value=[META]):
            result = session.search_arxiv(search_args)
        self.assertTrue(result["results"][0]["in_conversation"])
        self.assertTrue(result["results"][0]["cached"])
        with patch.object(arxiv_client, "search", side_effect=arxiv_client.ArxivError("down")):
            self.assertEqual(source_discovery._Session(CID, lambda *_: None).search_arxiv(search_args)["error"],
                             "search_failed")

    def test_unknown_tool(self):
        self.assertEqual(source_discovery._Session(CID, lambda *_: None).execute("rm_rf", {})["error"], "unknown_tool")


if __name__ == "__main__":
    unittest.main()
