"""candidate_recovery_v2 invariants with generic fixtures, no benchmark or network."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app.agentic_rag import (
    _answer_guidelines, _correct_answer, _recover_candidates, _retrieve_global,
    _route_after_coverage, _route_after_selection,
)
from app.candidate_recovery import (
    CANDIDATES_PER_GAP, CANDIDATES_PER_GAP_V2, MAX_ADDITIONS,
    assess_question_coverage, plan_articles, recover_candidate_evidence,
)
from app.hybrid_retrieval import (
    MultiFacetCandidates, drop_reference_sections, is_reference_section, retrieve_multifacet_candidates,
)
from app.main import app
from app.rag_service import ABSTENTION_CORRECTION_NOTE, SOURCE_FIDELITY_GUIDELINES, generate_answer_draft
from app.retrieval_policy import (
    RETRIEVAL_PROFILES, keeps_primary_hits, recovers_candidates, uses_recovery_v2,
    validate_retrieval_profile,
)
from app.schemas import CoverageRequirement, EvidenceCoverage
from tests.openai_fixtures import TEST_ENV, mock_client, response_body
from tests.test_retrieval_policy import result


def from_article(chunk_id: int, document_id: str, text: str, section: str = "Method"):
    return result(chunk_id, text).model_copy(update={"document_id": document_id, "section": section})


class RecoveryV2ProfileTests(unittest.TestCase):
    def test_profile_is_valid_and_isolated(self):
        self.assertEqual(validate_retrieval_profile("candidate_recovery_v2"), "candidate_recovery_v2")
        self.assertTrue(keeps_primary_hits("candidate_recovery_v2"))
        self.assertTrue(recovers_candidates("candidate_recovery_v2"))
        self.assertEqual([profile for profile in RETRIEVAL_PROFILES if uses_recovery_v2(profile)], ["candidate_recovery_v2"])

    def test_profile_routes_through_check_and_recovery(self):
        coverage = EvidenceCoverage(sufficient=False, requirements=[CoverageRequirement(aspect="Gap aspect")])
        state = {"retrieval_profile": "candidate_recovery_v2", "coverage_checks": [coverage]}
        self.assertEqual(_route_after_selection(state), "check")
        self.assertEqual(_route_after_coverage(state), "recover")

    def test_answer_guidelines_are_profile_gated(self):
        self.assertEqual(_answer_guidelines({"retrieval_profile": "candidate_recovery"}), "")
        self.assertEqual(_answer_guidelines({"retrieval_profile": "candidate_recovery_v2"}), SOURCE_FIDELITY_GUIDELINES)

    def test_answer_guidelines_extend_the_system_message(self):
        with patch("app.rag_service.generate_chat", return_value="Supported [S1].") as generate:
            generate_answer_draft(question="What?", results=[result(1, "Evidence text")], answer_guidelines="EXTRA RULE")
            generate_answer_draft(question="What?", results=[result(1, "Evidence text")])
        self.assertTrue(generate.call_args_list[0].kwargs["system_message"].endswith("\nEXTRA RULE"))
        self.assertNotIn("EXTRA RULE", generate.call_args_list[1].kwargs["system_message"])


class PlanArticlesTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            from_article(1, "paper", "The model is trained on curated images."),
            from_article(2, "other", "[12] An unrelated citation.", section="References"),
        ]
        self.plan = json.dumps({"requirements": [
            {"aspect": "Training data curation", "document_id": "paper", "search_query": "data curation"},
        ]})
        self.verdict = json.dumps({"requirements": [{
            "aspect": "Training data curation", "document_id": "paper", "reference": "S1",
            "quote": "trained on curated images", "search_query": None,
        }]})

    def test_detected_articles_present_in_results_are_planned(self):
        self.assertEqual(plan_articles(self.results), ["paper", "other"])
        self.assertEqual(plan_articles(self.results, ["paper"]), ["paper"])
        self.assertEqual(plan_articles(self.results, ["missing"]), ["missing"])

    def test_incidental_article_does_not_invalidate_the_plan(self):
        with patch("app.candidate_recovery.generate_chat", side_effect=[self.plan, self.verdict]) as generate:
            coverage = assess_question_coverage(question="How is data curated?", results=self.results, document_ids=["paper"])
        self.assertEqual(json.loads(generate.call_args_list[0].kwargs["user_message"])["articles"], ["paper"])
        self.assertTrue(coverage.sufficient)

    def test_without_detection_every_retrieved_article_is_required(self):
        with patch("app.candidate_recovery.generate_chat", return_value=self.plan) as generate:
            coverage = assess_question_coverage(question="How is data curated?", results=self.results)
        self.assertEqual(coverage.diagnostic, "invalid_question_requirements")
        generate.assert_called_once()

    def test_detected_article_absent_from_results_is_reported_unsupported(self):
        plan = self.plan.replace('"paper"', '"missing"')
        verdict = json.dumps({"requirements": [{
            "aspect": "Training data curation", "document_id": "missing", "reference": None,
            "quote": None, "search_query": None,
        }]})
        with patch("app.candidate_recovery.generate_chat", side_effect=[plan, verdict]):
            coverage = assess_question_coverage(question="How?", results=self.results[:1], document_ids=["missing"])
        self.assertFalse(coverage.sufficient)
        self.assertEqual(coverage.requirements[0].search_query, "data curation")


class ReferenceSectionTests(unittest.TestCase):
    def test_reference_headings_are_detected(self):
        for section in ("References", "references", "7 References", "6.1. Bibliography", " REFERENCES "):
            self.assertTrue(is_reference_section(section), section)
        for section in ("Method", "Reference implementation", "Related work", ""):
            self.assertFalse(is_reference_section(section), section)

    def test_reference_chunks_are_dropped_before_fusion(self):
        dense = [from_article(1, "paper", "[3] Citation", section="References"), result(2, "Method text")]
        lexical = [from_article(3, "paper", "[4] Citation", section="8 Bibliography"), result(4, "Result text")]
        rankings = iter([dense, lexical, dense, lexical])
        with patch("app.hybrid_retrieval.search_chunks", side_effect=lambda *a, **k: next(rankings)), patch(
            "app.hybrid_retrieval.search_bm25", side_effect=lambda **k: next(rankings)
        ):
            kept = retrieve_multifacet_candidates(
                queries=["q"], query_vectors=[[0.1]], document_id="paper", exclude_references=True,
            )
            unfiltered = retrieve_multifacet_candidates(queries=["q"], query_vectors=[[0.1]], document_id="paper")
        self.assertEqual({hit.chunk_id for hit in kept.fused}, {2, 4})
        self.assertEqual({hit.chunk_id for hit in unfiltered.fused}, {1, 2, 3, 4})
        self.assertEqual(drop_reference_sections(dense), [dense[1]])


class RecoveryV2Tests(unittest.TestCase):
    def setUp(self):
        self.initial = [result(1, "The model reports a test accuracy of 71 percent.")]
        self.pool = [result(2, "The corpus is described in the appendix.")]
        self.found = [result(3, "The corpus contains nine million samples.")]
        self.gap = CoverageRequirement(aspect="Training corpus size", document_id="paper", search_query="training corpus size")
        self.coverage = EvidenceCoverage(sufficient=False, requirements=[self.gap])
        self.searches = []

    def verdict(self, reference=None, quote=None):
        return json.dumps({"requirements": [{
            "aspect": self.gap.aspect, "document_id": "paper", "reference": reference, "quote": quote, "search_query": None,
        }]})

    def fallback(self, gap):
        self.searches.append(gap)
        return [*self.initial, *self.pool, *self.found]

    def recover(self, verdicts, **changes):
        parameters = dict(
            selected=self.initial, candidates=self.pool, coverage=self.coverage, fallback_search=self.fallback,
            candidates_per_gap=CANDIDATES_PER_GAP_V2, search_unresolved_gaps=True, added_support_only=True,
        )
        parameters.update(changes)
        with patch("app.candidate_recovery.rerank_candidate_groups", side_effect=lambda groups: {"paper": list(groups[0][1])}), patch(
            "app.candidate_recovery.generate_chat", side_effect=verdicts
        ) as generate:
            recovered = recover_candidate_evidence(**parameters)
        return recovered, generate

    def test_unresolved_gap_gets_one_fallback_and_second_verification(self):
        recovered, generate = self.recover([self.verdict(), self.verdict("S2", "nine million samples")])
        self.assertEqual(self.searches, [self.gap])
        self.assertEqual(generate.call_count, 2)
        second_window = json.loads(generate.call_args_list[1].kwargs["user_message"])
        # Initial and already-proposed chunks are excluded from the fallback proposals.
        self.assertEqual([item["text"] for item in second_window["passages"]], [self.initial[0].text, self.found[0].text])
        self.assertEqual([item.chunk_id for item in recovered.results], [1, 3])
        self.assertEqual(recovered.coverage.requirements[0].reference, "S2")
        self.assertTrue(recovered.coverage.sufficient)
        self.assertTrue(recovered.audit.fallback_used)
        self.assertEqual(recovered.audit.considered_chunk_ids, [2, 3])

    def test_resolved_gap_does_not_search(self):
        recovered, generate = self.recover([self.verdict("S2", "described in the appendix")])
        self.assertEqual(self.searches, [])
        self.assertFalse(recovered.audit.fallback_used)
        self.assertEqual(recovered.audit.added_chunk_ids, [2])
        generate.assert_called_once()

    def test_gap_searched_for_an_empty_pool_is_not_searched_again(self):
        recovered, generate = self.recover([self.verdict()], candidates=[])
        self.assertEqual(self.searches, [self.gap])
        generate.assert_called_once()
        self.assertEqual(recovered.results, self.initial)

    def test_initial_passage_cannot_close_a_gap(self):
        recovered, generate = self.recover([self.verdict("S1", "test accuracy of 71 percent"), self.verdict()])
        self.assertEqual(self.searches, [self.gap])
        self.assertEqual(generate.call_count, 2)
        self.assertIsNone(recovered.coverage.requirements[0].reference)
        self.assertFalse(recovered.coverage.sufficient)
        self.assertEqual(recovered.results, self.initial)

    def test_previous_profiles_still_accept_an_initial_passage(self):
        recovered, generate = self.recover(
            [self.verdict("S1", "test accuracy of 71 percent")],
            candidates_per_gap=CANDIDATES_PER_GAP, search_unresolved_gaps=False, added_support_only=False,
        )
        self.assertEqual(self.searches, [])
        self.assertEqual(recovered.coverage.requirements[0].reference, "S1")
        generate.assert_called_once()

    def test_additions_stay_bounded_across_both_verifications(self):
        gaps = [self.gap.model_copy(update={"aspect": f"Quantity {i}"}) for i in range(2)]
        pool = [result(10, "Quantity zero is nine million samples.")]
        found = [result(20 + i, f"Quantity one is {i} million samples.") for i in range(3)]
        first = json.dumps({"requirements": [
            {"aspect": "Quantity 0", "document_id": "paper", "reference": "S2", "quote": "nine million samples", "search_query": None},
            {"aspect": "Quantity 1", "document_id": "paper", "reference": None, "quote": None, "search_query": None},
        ]})
        second = json.dumps({"requirements": [
            {"aspect": "Quantity 1", "document_id": "paper", "reference": "S2", "quote": "0 million samples", "search_query": None},
        ]})
        with patch("app.candidate_recovery.rerank_candidate_groups", side_effect=lambda groups: {"paper": list(groups[0][1])}), patch(
            "app.candidate_recovery.generate_chat", side_effect=[first, second]
        ):
            recovered = recover_candidate_evidence(
                selected=self.initial, candidates=pool, coverage=EvidenceCoverage(sufficient=False, requirements=gaps),
                fallback_search=lambda gap: found, candidates_per_gap=CANDIDATES_PER_GAP_V2,
                search_unresolved_gaps=True, added_support_only=True,
            )
        self.assertEqual(recovered.audit.added_chunk_ids, [10, 20])
        self.assertLessEqual(len(recovered.audit.added_chunk_ids), MAX_ADDITIONS)
        self.assertEqual(recovered.results[0], self.initial[0])
        self.assertTrue(recovered.coverage.sufficient)

    def test_candidates_per_gap_is_wider(self):
        pool = [result(10 + i, f"Candidate {i}") for i in range(10)]
        _, generate = self.recover([self.verdict(), self.verdict()], candidates=pool)
        first_window = json.loads(generate.call_args_list[0].kwargs["user_message"])["passages"]
        self.assertEqual(len(first_window), 1 + CANDIDATES_PER_GAP_V2)
        self.assertEqual((CANDIDATES_PER_GAP, CANDIDATES_PER_GAP_V2), (3, 8))

    def test_agentic_node_passes_profile_options(self):
        state = {
            "retrieval_profile": "candidate_recovery_v2", "selected_results": self.initial,
            "candidate_pool": self.pool, "coverage_checks": [self.coverage],
        }
        captured = {}

        def fake_recovery(**kwargs):
            captured.update(kwargs)
            return recover_candidate_evidence(selected=kwargs["selected"], candidates=[], coverage=EvidenceCoverage(sufficient=None))

        with patch("app.agentic_rag.recover_candidate_evidence", side_effect=fake_recovery):
            update = _recover_candidates(state)
            _recover_candidates({**state, "retrieval_profile": "candidate_recovery"})
            old_options = dict(captured)
            _recover_candidates(state)
        self.assertEqual(captured["candidates_per_gap"], CANDIDATES_PER_GAP_V2)
        self.assertTrue(captured["search_unresolved_gaps"] and captured["added_support_only"])
        self.assertEqual(old_options["candidates_per_gap"], CANDIDATES_PER_GAP)
        self.assertFalse(old_options["search_unresolved_gaps"] or old_options["added_support_only"])
        self.assertIn("fallback_used=False", update["trace"][0]["detail"])

    def run_fallback(self, gap, profile="candidate_recovery_v2", **patches):
        state = {
            "retrieval_profile": profile, "selected_results": self.initial,
            "candidate_pool": [], "coverage_checks": [self.coverage],
        }
        captured = {}

        def fake_recovery(**kwargs):
            captured["found"] = kwargs["fallback_search"](gap)
            return recover_candidate_evidence(selected=kwargs["selected"], candidates=[], coverage=EvidenceCoverage(sufficient=None))

        with patch("app.agentic_rag.recover_candidate_evidence", side_effect=fake_recovery), patch(
            "app.agentic_rag.embed_query", return_value=[0.1]
        ), patch("app.agentic_rag.search_chunks", return_value=patches.get("dense", [])) as dense, patch(
            "app.agentic_rag.retrieve_multifacet_candidates",
            return_value=MultiFacetCandidates(dense_rankings=([],), lexical_rankings=([],), fused=patches.get("fused", [])),
        ) as hybrid:
            _recover_candidates(state)
        return captured["found"], dense, hybrid

    def test_agentic_fallback_drops_reference_chunks_without_article(self):
        hits = [from_article(5, "paper", "[1] Citation.", section="References"), *self.found]
        found, _, hybrid = self.run_fallback(self.gap.model_copy(update={"document_id": None}), dense=hits)
        self.assertEqual(found, self.found)
        hybrid.assert_not_called()

    def test_agentic_fallback_is_hybrid_for_one_article(self):
        found, dense, hybrid = self.run_fallback(self.gap, fused=self.found)
        self.assertEqual(found, self.found)
        dense.assert_not_called()
        self.assertEqual(hybrid.call_args.kwargs["queries"], ["training corpus size"])
        self.assertEqual(hybrid.call_args.kwargs["document_id"], "paper")
        self.assertTrue(hybrid.call_args.kwargs["exclude_references"])

    def test_previous_profile_fallback_stays_dense(self):
        found, _, hybrid = self.run_fallback(self.gap, profile="candidate_recovery", dense=self.found)
        self.assertEqual(found, self.found)
        hybrid.assert_not_called()


class RecoveryV2AnswerTests(unittest.TestCase):
    def test_guidelines_keep_abstention_for_an_absent_subject(self):
        self.assertIn("exactly as written", SOURCE_FIDELITY_GUIDELINES)
        self.assertIn("never replaces abstention", SOURCE_FIDELITY_GUIDELINES)
        self.assertIn("INSUFFICIENT_EVIDENCE", SOURCE_FIDELITY_GUIDELINES)

    def test_correction_note_is_only_added_to_corrections(self):
        evidence = [result(1, "Evidence text")]
        with patch("app.rag_service.generate_chat", return_value="Supported [S1].") as generate:
            generate_answer_draft(question="What?", results=evidence, correction_note="NOTE")
            generate_answer_draft(
                question="What?", results=evidence, previous_answer="INSUFFICIENT_EVIDENCE",
                correction_issues=["premature_abstention"], correction_note="NOTE",
            )
        self.assertNotIn("NOTE", generate.call_args_list[0].kwargs["user_message"])
        self.assertTrue(generate.call_args_list[1].kwargs["user_message"].endswith("NOTE"))

    def correct(self, profile):
        state = {
            "question": "What?", "retrieval_profile": profile, "strategy": "global",
            "selected_results": [result(1, "Evidence text")], "generated_answer": "INSUFFICIENT_EVIDENCE",
            "answer_validation": SimpleNamespace(issues=("premature_abstention",)), "fact_plan": [],
        }
        with patch("app.agentic_rag.generate_answer_draft", return_value="INSUFFICIENT_EVIDENCE") as draft, patch(
            "app.agentic_rag.repair_corrected_answer_from_evidence", return_value=("INSUFFICIENT_EVIDENCE", None)
        ):
            _correct_answer(state)
        return draft.call_args.kwargs

    def test_v2_correction_allows_abstention(self):
        self.assertEqual(self.correct("candidate_recovery_v2")["correction_note"], ABSTENTION_CORRECTION_NOTE)
        self.assertEqual(self.correct("candidate_recovery")["correction_note"], "")


class RecoveryV2RetrievalTests(unittest.TestCase):
    def retrieve(self, profile):
        state = {"question": "Which encoders does the paper explore?", "retrieval_profile": profile,
                 "detected_document_ids": ["paper"], "limit": 5}
        hits = [result(1, "Encoder text")]
        with patch("app.agentic_rag.retrieve_multifacet_candidates", return_value=MultiFacetCandidates(
            dense_rankings=(hits,), lexical_rankings=(hits,), fused=hits,
        )) as hybrid, patch("app.agentic_rag.embed_queries", return_value=[[0.1]]), patch(
            "app.agentic_rag.embed_query", return_value=[0.1]
        ), patch("app.agentic_rag.search_chunks", return_value=hits) as dense, patch(
            "app.agentic_rag.rerank_candidate_groups", return_value={"paper": hits}
        ):
            update = _retrieve_global(state)
        return update, hybrid, dense

    def test_detected_article_uses_hybrid_retrieval_with_one_variant(self):
        update, hybrid, dense = self.retrieve("candidate_recovery_v2")
        self.assertEqual(len(hybrid.call_args.kwargs["queries"]), 1)
        self.assertTrue(hybrid.call_args.kwargs["exclude_references"])
        dense.assert_not_called()
        self.assertEqual(update["candidate_pool"][0].chunk_id, 1)

    def test_previous_profile_keeps_dense_route_for_one_variant(self):
        _, hybrid, dense = self.retrieve("candidate_recovery")
        hybrid.assert_not_called()
        dense.assert_called()


class RecoveryV2HttpTests(unittest.TestCase):
    def test_v2_http_filters_references_and_adds_answer_guidelines(self):
        requests = []
        aspect = "How the method learns representations"
        answers = [
            json.dumps({"requirements": [{
                "aspect": aspect, "document_id": "clip-2021", "search_query": "representation learning mechanism",
            }]}),
            json.dumps({"requirements": [{
                "aspect": aspect, "document_id": "clip-2021", "reference": "S1",
                "quote": "learns representations from its input data",
                "assessment": "The quote directly describes the mechanism.", "search_query": None,
            }]}),
            "The method learns representations from its input data [S1].",
        ]

        def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json=response_body(answers.pop(0)))

        client = mock_client(handler)
        self.addCleanup(client.close)
        hits = [
            from_article(1, "clip-2021", "This method learns representations from its input data."),
            from_article(2, "llava-2023", "[7] A cited paper about representations.", section="References"),
        ]
        with patch.dict(os.environ, {**TEST_ENV, "RAG_RETRIEVAL_PROFILE": "candidate_recovery_v2"}), patch(
            "app.llm_gateway._get_client", return_value=client
        ), patch("app.agentic_rag.embed_query", return_value=[0.1]), patch(
            "app.agentic_rag.search_chunks", return_value=hits
        ), TestClient(app) as http:
            response = http.post("/agentic/ask", json={"question": "What does the method do?"})
        payload = response.json()
        self.assertEqual(response.status_code, 200, payload)
        self.assertEqual(payload["execution"]["retrieval_profile"], "candidate_recovery_v2")
        self.assertEqual([item["document_id"] for item in payload["evidence"]], ["clip-2021"])
        self.assertEqual(json.loads(requests[0]["input"][0]["content"])["articles"], ["clip-2021"])
        self.assertIn(SOURCE_FIDELITY_GUIDELINES, requests[2]["instructions"])
        self.assertEqual(len(requests), 3)


if __name__ == "__main__":
    unittest.main()
