"""Candidate recovery invariants with generic fixtures, no benchmark or network."""

import json
import unittest
from unittest.mock import patch

from app.agentic_rag import _route_after_coverage, _generation_coverage_plan
from app.candidate_recovery import (
    _parse_scoped_assessment, recover_candidate_evidence, format_coverage_plan,
    assess_question_coverage, parse_question_requirements, verify_fixed_requirements,
    select_recovery_gaps,
)
from app.schemas import CoverageRequirement, EvidenceCoverage
from tests.test_retrieval_policy import result


class CandidateRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.initial = [result(1, "The model reports a test accuracy of 71 percent.")]
        self.pool = [result(2, "The corpus contains nine million samples.")]
        self.gap = CoverageRequirement(aspect="Training corpus size", document_id="paper", search_query="training corpus size")
        self.coverage = EvidenceCoverage(sufficient=False, requirements=[self.gap])

    def verdict(self, reference="S2", quote="nine million samples", **changes):
        requirement = {"aspect": self.gap.aspect, "document_id": "paper", "reference": reference, "quote": quote, "search_query": None}
        requirement.update(changes)
        return json.dumps({"requirements": [requirement]})

    def run_recovery(self, raw=None, **changes):
        parameters = dict(selected=self.initial, candidates=self.pool, coverage=self.coverage)
        parameters.update(changes)
        with patch("app.candidate_recovery.rerank_candidate_groups", return_value={"paper": self.pool}) as rerank, patch(
            "app.candidate_recovery.generate_chat", return_value=raw or self.verdict()
        ) as generate:
            updated = recover_candidate_evidence(**parameters)
        return updated, rerank, generate

    def test_preserves_original_and_adds_verified_candidate(self):
        updated, rerank, generate = self.run_recovery()
        self.assertEqual([item.chunk_id for item in updated.results], [1, 2])
        self.assertEqual(updated.results[0], self.initial[0])
        self.assertEqual(updated.coverage.requirements[0].reference, "S2")
        self.assertTrue(updated.coverage.sufficient)
        self.assertEqual(updated.audit.added_chunk_ids, [2])
        self.assertEqual(rerank.call_args.args[0][0][0], "training corpus size")
        generate.assert_called_once()

    def test_invented_quote_cannot_add_a_candidate(self):
        updated, _, _ = self.run_recovery(self.verdict(quote="two billion training samples"))
        self.assertEqual(updated.results, self.initial)
        self.assertFalse(updated.coverage.sufficient)

    def test_unknown_reference_cannot_add_a_candidate(self):
        updated, _, _ = self.run_recovery(self.verdict(reference="S99"))
        self.assertEqual(updated.audit.added_chunk_ids, [])

    def test_changed_requirement_cannot_weaken_question(self):
        updated, _, _ = self.run_recovery(self.verdict(aspect="Anything numerical"))
        self.assertEqual(updated.results, self.initial)
        self.assertEqual(updated.audit.diagnostic, "invalid_or_changed_requirements")

    def test_malformed_verification_preserves_context(self):
        updated, _, generate = self.run_recovery("not JSON")
        self.assertEqual(updated.results, self.initial)
        generate.assert_called_once()

    def test_unknown_decomposition_never_triggers_an_api_guess(self):
        updated, rerank, generate = self.run_recovery(coverage=EvidenceCoverage(sufficient=None))
        self.assertEqual(updated.results, self.initial)
        rerank.assert_not_called()
        generate.assert_not_called()

    def test_complete_selection_needs_no_recovery(self):
        supported = self.gap.model_copy(update={"reference": "S1", "quote": "reports a test accuracy"})
        _, rerank, generate = self.run_recovery(coverage=EvidenceCoverage(sufficient=True, requirements=[supported]))
        rerank.assert_not_called()
        generate.assert_not_called()

    def test_empty_pool_needs_no_verification(self):
        updated, _, generate = self.run_recovery(candidates=[])
        self.assertEqual(updated.audit.diagnostic, "no_additional_candidates")
        generate.assert_not_called()

    def test_empty_pool_uses_fallback_search_for_the_gap(self):
        calls = []

        def fallback(gap):
            calls.append(gap)
            return [*self.initial, *self.pool]

        updated, rerank, generate = self.run_recovery(candidates=[], fallback_search=fallback)
        self.assertEqual(calls, [self.gap])
        self.assertEqual(rerank.call_args.args[0][0][1], self.pool)
        self.assertEqual(updated.audit.added_chunk_ids, [2])
        self.assertTrue(updated.coverage.sufficient)
        generate.assert_called_once()

    def test_fallback_is_not_used_when_pool_has_candidates(self):
        calls = []
        self.run_recovery(fallback_search=lambda gap: calls.append(gap) or [])
        self.assertEqual(calls, [])

    def test_empty_fallback_keeps_context_without_verification(self):
        updated, _, generate = self.run_recovery(candidates=[], fallback_search=lambda gap: list(self.initial))
        self.assertEqual(updated.audit.diagnostic, "no_additional_candidates")
        self.assertEqual(updated.results, self.initial)
        generate.assert_not_called()

    def test_wrong_article_is_excluded_before_ranking(self):
        wrong = self.pool[0].model_copy(update={"document_id": "other"})
        _, rerank, generate = self.run_recovery(candidates=[wrong])
        rerank.assert_not_called()
        generate.assert_not_called()

    def test_same_candidate_is_not_added_twice(self):
        updated, _, _ = self.run_recovery(candidates=[self.pool[0], self.pool[0], self.initial[0]])
        self.assertEqual([item.chunk_id for item in updated.results], [1, 2])

    def test_remaps_proposal_reference_to_final_reference(self):
        second = result(3, "The corpus contains nine million samples.")
        updated, _, _ = self.run_recovery(self.verdict(reference="S3"), candidates=[*self.pool, second])
        # Ranked proposals are mocked here, so explicitly expose both.
        with patch("app.candidate_recovery.rerank_candidate_groups", return_value={"paper": [*self.pool, second]}), patch(
            "app.candidate_recovery.generate_chat", return_value=self.verdict(reference="S3")
        ):
            updated = recover_candidate_evidence(selected=self.initial, candidates=[*self.pool, second], coverage=self.coverage)
        self.assertEqual([item.chunk_id for item in updated.results], [1, 3])
        self.assertEqual(updated.coverage.requirements[0].reference, "S2")

    def test_limits_added_text_and_keeps_gap_unresolved(self):
        with patch("app.candidate_recovery.MAX_ADDED_CHARACTERS", 5):
            updated, _, _ = self.run_recovery()
        self.assertEqual(updated.results, self.initial)
        self.assertFalse(updated.coverage.sufficient)

    def test_scope_is_verified_against_source_not_only_quote(self):
        wrong = self.pool[0].model_copy(update={"document_id": "other"})
        parsed = _parse_scoped_assessment(self.verdict(), [self.initial[0], wrong])
        self.assertFalse(parsed.sufficient)
        self.assertIsNone(parsed.requirements[0].reference)

    def test_invalid_document_scope_is_unknown(self):
        parsed = _parse_scoped_assessment(self.verdict(document_id="invented"), [*self.initial, *self.pool])
        self.assertIsNone(parsed.sufficient)

    def test_diagnostic_profile_cannot_recover(self):
        self.assertEqual(_route_after_coverage({"retrieval_profile": "coverage_diagnostic", "coverage_checks": [self.coverage]}), "plan")

    def test_candidate_profile_routes_to_bounded_recovery(self):
        self.assertEqual(_route_after_coverage({"retrieval_profile": "candidate_recovery", "coverage_checks": [self.coverage]}), "recover")

    def test_unsupported_requirement_is_not_in_generation_plan(self):
        self.assertEqual(format_coverage_plan(self.coverage), "")

    def test_generation_plan_is_an_independent_profile(self):
        supported = self.gap.model_copy(update={"reference": "S1", "quote": "reports a test accuracy"})
        coverage = EvidenceCoverage(sufficient=True, requirements=[supported])
        self.assertEqual(_generation_coverage_plan({"retrieval_profile": "candidate_recovery", "coverage_checks": [coverage]}), "")
        self.assertIn("Training corpus size", _generation_coverage_plan({"retrieval_profile": "coverage_answer_plan", "coverage_checks": [coverage]}))

    def test_requirement_planner_never_sees_retrieved_text(self):
        plan = json.dumps({"requirements": [{"aspect": self.gap.aspect, "document_id": "paper", "search_query": "training corpus size"}]})
        with patch("app.candidate_recovery.generate_chat", side_effect=[plan, self.verdict(reference=None, quote=None)]) as generate:
            coverage = assess_question_coverage(question="What is the corpus size?", results=self.initial)
        planner_input = json.loads(generate.call_args_list[0].kwargs["user_message"])
        self.assertEqual(set(planner_input), {"question", "articles"})
        self.assertNotIn(self.initial[0].text, generate.call_args_list[0].kwargs["user_message"])
        self.assertFalse(coverage.sufficient)
        self.assertEqual(coverage.requirements[0].search_query, "training corpus size")
        self.assertEqual(generate.call_count, 2)

    def test_malformed_plan_skips_verification(self):
        with patch("app.candidate_recovery.generate_chat", return_value="invalid") as generate:
            coverage = assess_question_coverage(question="What?", results=self.initial)
        self.assertIsNone(coverage.sufficient)
        generate.assert_called_once()

    def test_empty_context_skips_both_calls(self):
        with patch("app.candidate_recovery.generate_chat") as generate:
            coverage = assess_question_coverage(question="What?", results=[])
        self.assertFalse(coverage.sufficient)
        generate.assert_not_called()

    def test_duplicate_or_invented_requirements_are_rejected(self):
        valid = {"aspect": self.gap.aspect, "document_id": "paper", "search_query": "training corpus size"}
        for items in ([valid, valid], [{**valid, "document_id": "invented"}], [{**valid, "search_query": None}], [valid] * 5):
            self.assertIsNone(parse_question_requirements(json.dumps({"requirements": items}), self.initial))

    def test_verifier_cannot_omit_a_requirement(self):
        second = self.gap.model_copy(update={"aspect": "Evaluation accuracy"})
        with patch("app.candidate_recovery.generate_chat", return_value=self.verdict(reference="S1", quote="test accuracy of 71 percent")):
            checked = verify_fixed_requirements([self.gap, second], self.initial)
        self.assertIsNone(checked.sufficient)

    def test_maximum_two_additions_and_gaps(self):
        gaps = [self.gap.model_copy(update={"aspect": f"Requested quantity {i}"}) for i in range(4)]
        candidates = [result(i + 2, f"The corpus contains quantity {i} with nine million samples.") for i in range(6)]
        raw = json.dumps({"requirements": [
            {"aspect": gap.aspect, "document_id": "paper", "reference": f"S{i + 2}", "quote": "nine million samples", "search_query": None}
            for i, gap in enumerate(gaps[:2])
        ]})
        with patch("app.candidate_recovery.rerank_candidate_groups", return_value={"paper": candidates}), patch(
            "app.candidate_recovery.generate_chat", return_value=raw
        ):
            updated = recover_candidate_evidence(selected=self.initial, candidates=candidates, coverage=EvidenceCoverage(sufficient=False, requirements=gaps))
        self.assertEqual(len(updated.audit.missing_aspects), 2)
        self.assertEqual(len(updated.audit.added_chunk_ids), 2)
        self.assertEqual(updated.results[0], self.initial[0])
        self.assertFalse(updated.coverage.sufficient)

    def test_multi_article_plan_cannot_omit_an_article_or_merge_scopes(self):
        other = self.initial[0].model_copy(update={"chunk_id": 9, "document_id": "other"})
        valid = {"aspect": "Input interface", "document_id": "paper", "search_query": "input interface"}
        for items in ([valid], [valid, {**valid, "document_id": None}], [valid, {**valid, "document_id": "other"}]):
            parsed = parse_question_requirements(json.dumps({"requirements": items}), [self.initial[0], other])
            if len(items) == 2 and items[1]["document_id"] == "other":
                self.assertEqual(len(parsed), 2)
            else:
                self.assertIsNone(parsed)

    def test_gap_budget_is_distributed_across_articles(self):
        first = self.gap.model_copy(update={"aspect": "First source structure"})
        second = self.gap.model_copy(update={"aspect": "First source operation"})
        other = self.gap.model_copy(update={"aspect": "Second source structure", "document_id": "other"})
        self.assertEqual(select_recovery_gaps([first, second, other]), [first, other])

    def test_single_source_retains_question_order(self):
        gaps = [self.gap.model_copy(update={"aspect": f"Aspect {i}"}) for i in range(3)]
        self.assertEqual(select_recovery_gaps(gaps), gaps[:2])

    def test_supported_requirements_do_not_consume_gap_budget(self):
        supported = self.gap.model_copy(update={"reference": "S1"})
        other = self.gap.model_copy(update={"document_id": "other"})
        self.assertEqual(select_recovery_gaps([supported, self.gap, other]), [self.gap, other])

    def test_gap_selection_remains_bounded_with_three_articles(self):
        gaps = [self.gap.model_copy(update={"document_id": f"article-{i}"}) for i in range(3)]
        self.assertEqual(select_recovery_gaps(gaps), gaps[:2])


if __name__ == "__main__":
    unittest.main()
