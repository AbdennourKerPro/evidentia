"""No-network checks for isolated retrieval changes."""

import unittest
from unittest.mock import patch

from app.agentic_rag import _rank_candidates
from app.hybrid_retrieval import preserve_primary_pool, retrieve_multifacet_candidates
from app.query_expansion import build_query_variants
from app.retrieval_policy import validate_retrieval_profile
from app.schemas import SearchResult


def result(chunk_id: int, text: str, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id, score=score, document_id="paper", title="Example",
        page=1, section="Method", language="en", text=text,
    )


class RetrievalPolicyTests(unittest.TestCase):
    def test_question_reranker_uses_entire_question(self):
        candidates = [result(1, "Architecture"), result(2, "Experimental protocol")]
        with patch("app.agentic_rag.rerank_candidate_groups", return_value={"paper": [candidates[1], candidates[0]]}) as rerank:
            _rank_candidates(question="How is this evaluated?", candidates=candidates, intents=[], profile="question_rerank")
        self.assertEqual(rerank.call_args.args[0], [("How is this evaluated?", candidates)])

    def test_baseline_does_not_load_cross_encoder(self):
        with patch("app.agentic_rag.rerank_candidate_groups") as rerank:
            _rank_candidates(question="What?", candidates=[], intents=[], profile="intent_baseline")
        rerank.assert_not_called()

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_retrieval_profile("typo")

    def test_original_query_precedes_planned_query(self):
        variants = build_query_variants(
            question="What is the experimental protocol?", primary_query="Paper mechanism",
            preserve_original=True,
        )
        self.assertEqual(variants[:2], ["What is the experimental protocol?", "Paper mechanism"])

    def test_original_candidate_survives_expansion_truncation(self):
        fused = [result(i, "Expansion hit") for i in range(1, 12)]
        selected = preserve_primary_pool(fused, [[fused[-1]]], limit=3)
        self.assertEqual(len(selected), 3)
        self.assertIn(11, [hit.chunk_id for hit in selected])
        self.assertEqual(len({hit.chunk_id for hit in selected}), 3)

    def test_expansion_weights_share_one_budget(self):
        with patch("app.hybrid_retrieval.search_chunks", return_value=[]), patch(
            "app.hybrid_retrieval.search_bm25", return_value=[]
        ), patch("app.hybrid_retrieval.reciprocal_rank_fusion", return_value=[]) as fusion:
            retrieve_multifacet_candidates(
                queries=["original", "facet one", "facet two"],
                query_vectors=[[0.1], [0.2], [0.3]], document_id="paper",
                preserve_primary=True,
            )
        self.assertEqual(fusion.call_args.kwargs["weights"], [1.0, 1.0, 0.5, 0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
