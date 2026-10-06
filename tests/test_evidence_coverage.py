"""Validate the coverage judge and bounded recovery without any API calls."""

import json
import unittest
from unittest.mock import patch

from app.agentic_rag import (
    _retrieve_missing_evidence, _route_after_coverage,
    answer_agentic_question,
)
from app.evidence_coverage import assess_evidence_coverage, missing_search_queries, parse_coverage_assessment
from app.schemas import EvidenceCoverage, CoverageRequirement
from tests.test_retrieval_policy import result


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.evidence = result(1, "The method selects a reference image during testing.")

    def parse(self, **overrides):
        item = {"aspect": "Experimental selection", "reference": "S1", "quote": "selects a reference image", "search_query": None}
        item.update(overrides)
        return parse_coverage_assessment(json.dumps({"requirements": [item]}), [self.evidence])

    def test_accepts_exact_quote_and_reference(self):
        self.assertTrue(self.parse().sufficient)

    def test_rejects_invented_quote(self):
        self.assertFalse(self.parse(quote="Uses a hidden validation dataset").sufficient)

    def test_rejects_unknown_reference(self):
        self.assertFalse(self.parse(reference="S99").sufficient)

    def test_empty_or_malformed_plan_is_unknown(self):
        for raw in ("not JSON", '{"requirements": []}', "null"):
            self.assertIsNone(parse_coverage_assessment(raw, [self.evidence]).sufficient)

    def test_limits_complementary_queries(self):
        assessment = EvidenceCoverage(sufficient=False, requirements=[
            CoverageRequirement(aspect=f"Aspect {i}", search_query=f"search {i}") for i in range(4)
        ])
        self.assertEqual(len(missing_search_queries(assessment, "Original?")), 2)

    def test_unknown_plan_falls_back_to_original_question(self):
        assessment = EvidenceCoverage(sufficient=None)
        self.assertEqual(missing_search_queries(assessment, "Original?"), ["Original?"])

    def test_no_evidence_does_not_call_api(self):
        with patch("app.evidence_coverage.generate_chat") as generate:
            self.assertFalse(assess_evidence_coverage(question="What?", results=[]).sufficient)
        generate.assert_not_called()

    def test_retry_is_bounded_to_one_attempt(self):
        state = {"detected_document_ids": ["paper"], "coverage_checks": [EvidenceCoverage(sufficient=False)]}
        self.assertEqual(_route_after_coverage(state), "retry")
        state["supplementary_search_attempted"] = True
        self.assertEqual(_route_after_coverage(state), "plan")

    def test_sufficient_evidence_skips_retry(self):
        self.assertEqual(_route_after_coverage({"coverage_checks": [EvidenceCoverage(sufficient=True)]}), "plan")

    def test_retry_preserves_source_scope_and_top_k(self):
        from app.hybrid_retrieval import MultiFacetCandidates
        candidates = MultiFacetCandidates(dense_rankings=([self.evidence],), lexical_rankings=([self.evidence],), fused=[self.evidence])
        state = {
            "question": "How is the example method tested?", "limit": 1, "strategy": "global",
            "detected_document_ids": ["paper"], "selected_document_ids": ["paper"],
            "selected_results": [], "coverage_checks": [EvidenceCoverage(sufficient=False)],
        }
        with patch("app.agentic_rag.embed_queries", return_value=[[0.1]]), patch(
            "app.agentic_rag.retrieve_multifacet_candidates", return_value=candidates
        ) as retrieve, patch("app.agentic_rag.rerank_candidate_groups", return_value={"paper": [self.evidence]}):
            updated = _retrieve_missing_evidence(state)
        self.assertEqual(retrieve.call_args.kwargs["document_id"], "paper")
        self.assertTrue(updated["supplementary_search_attempted"])
        self.assertEqual(len(updated["selected_results"]), 1)

    def test_graph_rechecks_after_recovery_without_looping(self):
        with patch("app.agentic_rag.embed_query", return_value=[0.1]), patch(
            "app.agentic_rag.search_chunks", return_value=[self.evidence]
        ), patch("app.agentic_rag.assess_evidence_coverage", return_value=EvidenceCoverage(sufficient=False)) as check, patch(
            "app.agentic_rag._coverage_document_ids", return_value=["paper"]
        ), patch("app.agentic_rag.embed_queries", return_value=[[0.1]]), patch(
            "app.agentic_rag.retrieve_multifacet_candidates"
        ) as retrieve, patch("app.agentic_rag.rerank_candidate_groups", return_value={"paper": [self.evidence]}), patch(
            "app.agentic_rag.generate_answer_draft", return_value="The method selects a reference image during testing [S1]."
        ):
            from app.hybrid_retrieval import MultiFacetCandidates
            retrieve.return_value = MultiFacetCandidates(dense_rankings=([self.evidence],), lexical_rankings=([self.evidence],), fused=[self.evidence])
            response = answer_agentic_question(question="How is the method tested?", limit=1, retrieval_profile="coverage_retry")
        self.assertEqual(check.call_count, 2)
        self.assertEqual(retrieve.call_count, 1)
        self.assertTrue(response.execution.supplementary_search_attempted)
        self.assertEqual(len(response.execution.coverage_checks), 2)


if __name__ == "__main__":
    unittest.main()
