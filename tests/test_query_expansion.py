"""Tests for deterministic retrieval-intent routing and query expansion."""

from __future__ import annotations

import unittest

from app.query_expansion import (
    build_query_variants,
    detect_retrieval_intents,
    rerank_for_intents,
)
from app.schemas import SearchResult


def _result(*, chunk_id: int, section: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        document_id="paper",
        title="Paper",
        page=1,
        section=section,
        language="en",
        text=text,
    )


class QueryExpansionTests(unittest.TestCase):
    def test_detects_numeric_scale_in_english(self) -> None:
        self.assertIn(
            "numeric_scale",
            detect_retrieval_intents(
                "Compare the reported training-data scale of three models."
            ),
        )

    def test_detects_numeric_scale_in_french(self) -> None:
        self.assertIn(
            "numeric_scale",
            detect_retrieval_intents("Sur combien de paires CLIP est-il entraîné ?"),
        )

    def test_detects_prompt_mechanism(self) -> None:
        self.assertIn(
            "prompt_mechanism",
            detect_retrieval_intents("How do prompts play different roles?"),
        )

    def test_detects_architecture_and_capabilities(self) -> None:
        intents = detect_retrieval_intents(
            "Which zero-shot capability emerges from the Q-Former bridge?"
        )
        self.assertEqual(intents, ("architecture_bridge", "capabilities"))

    def test_detects_a_training_objective_comparison(self) -> None:
        intents = detect_retrieval_intents(
            "Compare the supervision signals and pre-training objectives."
        )
        self.assertIn("training_objective", intents)

    def test_detects_a_general_zero_shot_inference_mechanism(self) -> None:
        intents = detect_retrieval_intents(
            "How does a model turn zero-shot matching into image classification?"
        )
        self.assertIn("inference_mechanism", intents)

    def test_detects_a_general_quantitative_efficiency_comparison(self) -> None:
        self.assertIn(
            "quantitative_comparison",
            detect_retrieval_intents(
                "What efficiency trade-off does the method report?"
            ),
        )

    def test_generic_question_keeps_one_query(self) -> None:
        variants = build_query_variants(
            question="What limitation do the authors discuss?",
            primary_query="Paper limitations",
        )
        self.assertEqual(variants, ["Paper limitations"])

    def test_expansion_does_not_contain_benchmark_answers(self) -> None:
        variants = build_query_variants(
            question="Compare the reported training-data scale.",
            primary_query="Training data scale",
        )
        expanded = " ".join(variants)
        for forbidden_answer in ("400 million", "LVD-142M", "SA-1B", "11M"):
            self.assertNotIn(forbidden_answer, expanded)

    def test_numeric_reranking_prefers_dataset_description_over_ablation(self) -> None:
        ablation = _result(
            chunk_id=1,
            section="Ablations",
            text="We subsample 11M images to 1M and use approximately 100M masks.",
            score=0.9,
        )
        dataset = _result(
            chunk_id=2,
            section="Introduction",
            text=(
                "The SA-1B dataset contains 11M licensed images and 1.1B masks."
            ),
            score=0.5,
        )
        ranked = rerank_for_intents(
            [ablation, dataset],
            ["numeric_scale"],
        )
        self.assertEqual(ranked[0].chunk_id, dataset.chunk_id)

    def test_prompt_reranking_prefers_exact_input_types(self) -> None:
        transfer = _result(
            chunk_id=1,
            section="Zero-Shot Experiments",
            text="We evaluate several promptable segmentation tasks.",
            score=0.9,
        )
        mechanism = _result(
            chunk_id=2,
            section="Model Architecture",
            text=(
                "The prompt encoder supports sparse points and boxes and dense masks."
            ),
            score=0.5,
        )
        ranked = rerank_for_intents(
            [transfer, mechanism],
            ["prompt_mechanism"],
        )
        self.assertEqual(ranked[0].chunk_id, mechanism.chunk_id)

    def test_inference_reranking_prefers_the_decision_rule(self) -> None:
        overview = _result(
            chunk_id=1,
            section="Introduction",
            text="The model transfers to many zero-shot tasks.",
            score=0.9,
        )
        mechanism = _result(
            chunk_id=2,
            section="Using the model for zero-shot transfer",
            text=(
                "Candidate class names are embedded as text and cosine similarity "
                "scores are scaled by a temperature before softmax."
            ),
            score=0.5,
        )
        ranked = rerank_for_intents(
            [overview, mechanism],
            ["inference_mechanism"],
        )
        self.assertEqual(ranked[0].chunk_id, mechanism.chunk_id)


if __name__ == "__main__":
    unittest.main()
