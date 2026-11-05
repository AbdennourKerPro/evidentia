import json
import unittest
from pathlib import Path

from app import paper_registry as registry
from app.agentic_rag import DOCUMENT_CATALOG
from tests.dynamic_helpers import RegistryTestCase

CONVERSATION = "11111111-2222-3333-4444-555555555555"
OTHER = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


class PaperRegistryTests(RegistryTestCase):
    def test_uuid_validation(self):
        self.assertTrue(registry.is_valid_conversation_id(CONVERSATION))
        self.assertFalse(registry.is_valid_conversation_id("../../etc/passwd"))
        self.assertFalse(registry.is_valid_conversation_id("abc"))

    def test_attach_list_and_detach(self):
        paper = self.make_paper()
        self.assertTrue(registry.attach_paper(CONVERSATION, paper.document_id, added_by="agent", reason="why"))
        self.assertFalse(registry.attach_paper(CONVERSATION, paper.document_id, added_by="agent"))
        listed = registry.list_conversation_papers(CONVERSATION)
        self.assertEqual([p.document_id for p in listed], [paper.document_id])
        self.assertEqual(listed[0].added_by, "agent")
        self.assertEqual(registry.list_conversation_papers(OTHER), [])
        self.assertTrue(registry.detach_paper(CONVERSATION, paper.document_id))
        self.assertEqual(registry.list_conversation_papers(CONVERSATION), [])

    def test_only_ready_filter(self):
        ready = self.make_paper()
        pending = self.make_paper("2401.00001", "v1", status="indexing")
        for p in (ready, pending):
            registry.attach_paper(CONVERSATION, p.document_id, added_by="agent")
        self.assertEqual(len(registry.list_conversation_papers(CONVERSATION)), 2)
        self.assertEqual(
            [p.document_id for p in registry.list_conversation_papers(CONVERSATION, only_ready=True)],
            [ready.document_id],
        )

    def test_orphans_ignore_pinned_attached_and_in_progress(self):
        used = self.make_paper("2401.00001", "v1")
        unused = self.make_paper("2401.00002", "v1")
        pinned = self.make_paper("2401.00003", "v1", pinned=True)
        running = self.make_paper("2401.00004", "v1", status="converting")
        failed = self.make_paper("2401.00005", "v1", status="failed")
        registry.attach_paper(CONVERSATION, used.document_id, added_by="agent")
        orphans = set(registry.orphan_documents(0))
        self.assertEqual(orphans, {unused.document_id, failed.document_id})
        self.assertNotIn(pinned.document_id, orphans)
        self.assertNotIn(running.document_id, orphans)
        # A just-updated paper is protected by the minimum age.
        self.assertEqual(registry.orphan_documents(3600), [])

    def test_delete_conversation_keeps_papers_and_frees_orphans(self):
        shared = self.make_paper("2401.00001", "v1")
        registry.attach_paper(CONVERSATION, shared.document_id, added_by="agent")
        registry.attach_paper(OTHER, shared.document_id, added_by="agent")
        registry.record_turn(CONVERSATION, "q1")
        registry.delete_conversation(CONVERSATION)
        self.assertFalse(registry.conversation_exists(CONVERSATION))
        self.assertEqual(registry.orphan_documents(0), [])  # still used by OTHER
        registry.delete_conversation(OTHER)
        self.assertEqual(registry.orphan_documents(0), [shared.document_id])

    def test_recent_questions_keep_last_three_in_order(self):
        for question in ["a q", "b q", "c q", "d q"]:
            registry.record_turn(CONVERSATION, question)
        self.assertEqual(registry.recent_questions(CONVERSATION, 3), ["b q", "c q", "d q"])

    def test_find_paper_without_version_prefers_ready(self):
        self.make_paper("2304.02643", "v4", status="failed")
        ready = self.make_paper("2304.02643", "v1", pinned=True)
        self.assertEqual(registry.find_paper("2304.02643").document_id, ready.document_id)
        self.assertIsNone(registry.find_paper("2304.02643", "v9"))

    def test_seed_is_idempotent_and_keeps_exact_aliases(self):
        sources = Path("data/sources/arxiv_papers.json")
        if not sources.is_file():
            self.skipTest("catalog not mounted")
        for _ in range(2):
            registry.seed_legacy_papers(catalog=DOCUMENT_CATALOG, sources_path=sources)
        pinned = {p.document_id: p for p in registry.list_pinned_papers()}
        self.assertEqual(set(pinned), set(DOCUMENT_CATALOG))
        for document_id, metadata in DOCUMENT_CATALOG.items():
            self.assertEqual(tuple(pinned[document_id].aliases), tuple(metadata["aliases"]))
            self.assertEqual(pinned[document_id].title, metadata["title"])


if __name__ == "__main__":
    unittest.main()
