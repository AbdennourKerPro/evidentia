import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app import arxiv_client, paper_ingestion, paper_registry
from app.schemas import EvidenceChunk
from tests.dynamic_helpers import RegistryTestCase

META = arxiv_client.ArxivPaperMeta(
    arxiv_id="2312.00752", latest_version="v2", title="Mamba: Linear-Time Sequence Modeling", abstract="x"
)


def fake_download(arxiv_id, version, destination, *, max_megabytes, on_bytes=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"%PDF-1.5")
    return "0" * 64


def fake_convert(*, pdf_path, output_dir, source, max_pages):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "document.md").write_text("md")
    (output_dir / "conversion-manifest.json").write_text(json.dumps({"formula_enrichment": True}))
    return object()


def fake_chunk(*, document_id, title, document, output_dir):
    return [EvidenceChunk(id=i, document_id=document_id, title=title, page=1, section="S",
                          language="en", text=f"chunk {i}") for i in (1, 2, 3)]


class PaperIngestionTests(RegistryTestCase):
    def patches(self, **overrides):
        base = dict(
            lookup=patch.object(arxiv_client, "lookup", return_value=[META]),
            download=patch.object(arxiv_client, "download_pdf", side_effect=fake_download),
            convert=patch("app.paper_ingestion.document_ingestion.convert_pdf_to_artifacts", side_effect=fake_convert),
            chunk=patch("app.paper_ingestion.document_ingestion.chunk_document", side_effect=fake_chunk),
            embed=patch("app.paper_ingestion.embed_passages", side_effect=lambda t: [[0.1, 0.2]] * len(t)),
            index=patch("app.paper_ingestion.index_chunks", return_value=3),
            delete=patch("app.paper_ingestion.delete_document_chunks"),
            bm25=patch("app.paper_ingestion.clear_bm25_cache"),
        )
        base.update(overrides)
        mocks = {}
        for name, p in base.items():
            mocks[name] = p.start()
            self.addCleanup(p.stop)
        return mocks

    def test_full_ingestion_reports_stages_and_registers(self):
        mocks = self.patches()
        stages = []
        paper = paper_ingestion.ingest_arxiv_paper(
            "2312.00752", None, short_name="Mamba", on_progress=lambda s, p: stages.append(s)
        )
        self.assertEqual(stages, ["downloading", "converting", "indexing", "ready"])
        self.assertEqual(paper.document_id, "arxiv-2312.00752v2")
        self.assertEqual((paper.status, paper.chunk_count), ("ready", 3))
        self.assertIn(r"\bmamba\b", paper.aliases)
        mocks["index"].assert_called_once()

    def test_cache_hit_skips_everything(self):
        mocks = self.patches()
        paper_ingestion.ingest_arxiv_paper("2312.00752")
        for m in mocks.values():
            m.reset_mock()
        again = paper_ingestion.ingest_arxiv_paper("2312.00752")
        self.assertEqual(again.status, "ready")
        for name in ("lookup", "download", "convert", "index"):
            mocks[name].assert_not_called()

    def test_paper_converted_without_formula_decoding_is_redone(self):
        mocks = self.patches()
        old = self.make_paper("2312.00752", "v2")  # ready, but no manifest: converted before decoding existed
        paper_registry.attach_paper("11111111-2222-3333-4444-555555555555", old.document_id, added_by="agent")
        paper = paper_ingestion.ingest_arxiv_paper("2312.00752")
        mocks["convert"].assert_called_once()
        mocks["delete"].assert_called_once()  # stale chunks removed before re-indexing
        self.assertEqual(paper.chunk_count, 3)
        self.assertEqual(  # the conversation keeps the paper
            [p.document_id for p in paper_registry.list_conversation_papers("11111111-2222-3333-4444-555555555555")],
            [old.document_id])
        mocks["convert"].reset_mock()
        paper_ingestion.ingest_arxiv_paper("2312.00752")
        mocks["convert"].assert_not_called()  # now up to date

    def test_pinned_papers_are_never_redone(self):
        mocks = self.patches()
        pinned = self.make_paper("2304.02643", "v1", pinned=True)
        self.assertEqual(paper_ingestion.ingest_arxiv_paper("2304.02643", "v1").document_id, pinned.document_id)
        mocks["convert"].assert_not_called()

    def test_failure_marks_failed_and_cleans_artifacts(self):
        mocks = self.patches(
            convert=patch("app.paper_ingestion.document_ingestion.convert_pdf_to_artifacts",
                          side_effect=RuntimeError("boom with secrets")),
        )
        with self.assertRaises(paper_ingestion.IngestionError) as caught:
            paper_ingestion.ingest_arxiv_paper("2312.00752")
        self.assertNotIn("secrets", str(caught.exception))  # no raw exception text
        paper = paper_registry.get_paper("arxiv-2312.00752v2")
        self.assertEqual(paper.status, "failed")
        self.assertFalse((Path(self.tmp.name) / "cache" / "raw" / "arxiv-2312.00752v2.pdf").exists())
        mocks["delete"].assert_called_once()

    def test_unknown_paper(self):
        self.patches(lookup=patch.object(arxiv_client, "lookup", return_value=[]))
        with self.assertRaises(paper_ingestion.IngestionError):
            paper_ingestion.ingest_arxiv_paper("2312.00752")

    def test_purge_respects_pinned_and_in_progress(self):
        self.patches()
        pinned = self.make_paper("2401.00001", "v1", pinned=True)
        running = self.make_paper("2401.00002", "v1", status="indexing")
        ready = self.make_paper("2401.00003", "v1")
        self.assertFalse(paper_ingestion.purge_document(pinned.document_id))
        self.assertFalse(paper_ingestion.purge_document(running.document_id))
        self.assertTrue(paper_ingestion.purge_document(ready.document_id))
        self.assertIsNone(paper_registry.get_paper(ready.document_id))
        self.assertIsNotNone(paper_registry.get_paper(pinned.document_id))

    def test_aliases_refuse_short_names(self):
        self.assertEqual(paper_ingestion.build_aliases("Foo", "AB"), [])
        self.assertEqual(paper_ingestion.build_aliases("Segment Anything", "SAM"), [r"\bsam\b"])
        self.assertEqual(
            paper_ingestion.build_aliases("DINOv2: Learning Robust Features", None), [r"\bdinov2\b"]
        )


if __name__ == "__main__":
    unittest.main()
