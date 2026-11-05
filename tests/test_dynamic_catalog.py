"""The graph must stay identical for the classic corpus and strictly scoped for others."""

import unittest
from pathlib import Path
from unittest.mock import patch

from app import agentic_rag, conversation_service, paper_registry
from app.agentic_rag import DOCUMENT_CATALOG, _analyze_question, _retrieve_global, _search_scope
from app.schemas import SearchResult
from tests.dynamic_helpers import RegistryTestCase

QUESTIONS = [
    "How many GPUs are used to train the ViT-B and ViT-L models of SAM?",
    "Compare the training paradigms of CLIP, DINOv2, and BLIP-2.",
    "What makes Segment Anything promptable?",
    "How does LLaVA use instruction data?",
    "Quelle est la différence entre CLIP et DINO v2 ?",
    "What is the capital of France?",
    "Compare these two approaches.",
]
MAMBA = "arxiv-2312.00752v2"
S4 = "arxiv-2111.00396v3"


def state(question, **extra):
    base = {"question": question, "catalog": dict(DOCUMENT_CATALOG), "selected_document_ids": None}
    base.update(extra)
    return base


def hit(document_id, chunk_id=1):
    return SearchResult(chunk_id=chunk_id, score=0.5, document_id=document_id, title=document_id,
                        page=1, section="Method", language="en", text="text")


class CatalogParityTests(RegistryTestCase):
    def test_registry_catalog_routes_exactly_like_the_constant(self):
        sources = Path("data/sources/arxiv_papers.json")
        paper_registry.seed_legacy_papers(catalog=DOCUMENT_CATALOG, sources_path=sources)
        pinned = {p.document_id: p for p in paper_registry.list_pinned_papers()}
        # Detection order follows catalogue order; a conversation orders it by attachment.
        catalog = {d: {"title": pinned[d].title, "aliases": tuple(pinned[d].aliases)} for d in DOCUMENT_CATALOG}
        for question in QUESTIONS:
            for selected in (None, ["clip-2021", "sam-2023"]):
                old = _analyze_question(state(question, selected_document_ids=selected))
                new = _analyze_question(state(question, selected_document_ids=selected, catalog=catalog))
                self.assertEqual(old["strategy"], new["strategy"], question)
                self.assertEqual(old["detected_document_ids"], new["detected_document_ids"], question)

    def test_partial_state_defaults_to_the_classic_corpus(self):
        result = _analyze_question({"question": "How does SAM work?"})
        self.assertEqual(result["detected_document_ids"], ["sam-2023"])


class DynamicRoutingTests(unittest.TestCase):
    catalog = {
        MAMBA: {"title": "Mamba: Linear-Time Sequence Modeling", "aliases": (r"\bmamba\b",)},
        S4: {"title": "Efficiently Modeling Long Sequences with Structured State Spaces", "aliases": (r"\bs4\b",)},
    }

    def test_alias_detection_and_comparison_routing(self):
        result = _analyze_question(state("Compare Mamba and S4", catalog=self.catalog))
        self.assertEqual(result["strategy"], "per_source")
        self.assertEqual(result["detected_document_ids"], [MAMBA, S4])
        self.assertEqual(_analyze_question(state("How does Mamba work?", catalog=self.catalog))["strategy"], "global")

    def test_classic_names_are_not_detected_in_a_dynamic_catalog(self):
        result = _analyze_question(state("What is CLIP?", catalog=self.catalog))
        self.assertEqual(result["detected_document_ids"], [])

    def test_focus_documents_are_added_and_restricted_to_the_catalog(self):
        result = _analyze_question(state("Compare them", catalog=self.catalog,
                                         focus_document_ids=[MAMBA, S4, "not-in-catalog"]))
        self.assertEqual(result["detected_document_ids"], [MAMBA, S4])
        self.assertEqual(result["strategy"], "per_source")
        # focus never overrides a user selection
        narrowed = _analyze_question(state("Compare them", catalog=self.catalog,
                                           focus_document_ids=[MAMBA, S4], selected_document_ids=[MAMBA]))
        self.assertEqual(narrowed["detected_document_ids"], [MAMBA])

    def test_search_scope_is_selection_else_corpus_never_unbounded(self):
        self.assertEqual(_search_scope({"corpus_document_ids": [MAMBA, S4]}), [MAMBA, S4])
        self.assertEqual(_search_scope({"corpus_document_ids": [MAMBA, S4], "selected_document_ids": [S4]}), [S4])
        self.assertEqual(_search_scope({}), list(DOCUMENT_CATALOG))

    def test_global_retrieval_always_passes_a_document_filter(self):
        seen = []

        def fake_search(vector, *, limit, collection_name, document_ids=None):
            seen.append(document_ids)
            return [hit(MAMBA)]

        with patch.object(agentic_rag, "search_chunks", side_effect=fake_search), \
             patch.object(agentic_rag, "embed_query", return_value=[0.1]), \
             patch.object(agentic_rag, "english_search_query", return_value=None):
            _retrieve_global(state("Which model is faster?", catalog=self.catalog, limit=3,
                                   retrieval_profile="candidate_recovery_v2",
                                   corpus_document_ids=[MAMBA, S4], detected_document_ids=[]))
        self.assertTrue(seen)
        self.assertTrue(all(ids == [MAMBA, S4] for ids in seen), seen)

    def test_single_paper_conversation_uses_focused_hybrid_retrieval(self):
        from app.hybrid_retrieval import MultiFacetCandidates

        candidates = MultiFacetCandidates(dense_rankings=([hit(MAMBA)],), lexical_rankings=([hit(MAMBA)],), fused=[hit(MAMBA)])
        with patch.object(agentic_rag, "retrieve_multifacet_candidates", return_value=candidates) as hybrid, \
             patch.object(agentic_rag, "embed_queries", return_value=[[0.1]]), \
             patch.object(agentic_rag, "english_search_query", return_value=None), \
             patch.object(agentic_rag, "rerank_candidate_groups", return_value={MAMBA: [hit(MAMBA)]}):
            result = _retrieve_global(state("How is it trained?", catalog=self.catalog, limit=3,
                                            retrieval_profile="candidate_recovery_v2",
                                            corpus_document_ids=[MAMBA], detected_document_ids=[]))
        self.assertEqual(hybrid.call_args.kwargs["document_id"], MAMBA)
        self.assertEqual(result["retrieval_batches"][0]["document_id"], MAMBA)


class EmptyCorpusTests(RegistryTestCase):
    def test_no_paper_means_abstention_without_any_llm_or_vector_call(self):
        with patch.object(agentic_rag, "search_chunks", side_effect=AssertionError("no search")), \
             patch.object(agentic_rag, "generate_chat", side_effect=AssertionError("no llm")):
            response = agentic_rag.answer_agentic_question(
                question="What is Mamba?", limit=3, corpus_document_ids=[], catalog={})
        self.assertTrue(response.abstained)
        self.assertEqual(response.retrieved_chunks, 0)
        self.assertEqual(response.execution.steps, [])

    def test_french_question_gets_french_abstention(self):
        response = agentic_rag.answer_agentic_question(
            question="Qu'est-ce que Mamba et comment fonctionne-t-il ?", limit=3, corpus_document_ids=[], catalog={})
        self.assertEqual(response.execution.expected_language, "fr")


if __name__ == "__main__":
    unittest.main()
