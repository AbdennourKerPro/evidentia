"""Tests for deterministic evidence fact planning and completeness checks."""

from __future__ import annotations

import unittest

from app.fact_planning import (
    format_answer_coverage_requirements,
    format_fact_requirements,
    plan_evidence_facts,
)
from app.response_contract import INSUFFICIENT_EVIDENCE_MARKER, validate_answer
from app.schemas import PlannedFact, SearchResult


def _result(position: int, text: str, document_id: str = "paper") -> SearchResult:
    return SearchResult(
        chunk_id=position,
        score=0.5,
        document_id=document_id,
        title=f"Article {document_id}",
        page=position,
        section="Method",
        language="en",
        text=text,
    )


class DeterministicFactPlanTests(unittest.TestCase):
    def test_extracts_frozen_feature_results_in_french(self) -> None:
        planning = plan_evidence_facts(
            question=(
                "Que montre le modèle sur les caractéristiques visuelles gelées "
                "sans fine-tuning ?"
            ),
            results=[
                _result(
                    1,
                    "Visual features close the gap with (weakly) supervised "
                    "alternatives without the need for finetuning and are "
                    "compatible with linear layers.",
                )
            ],
            retrieval_intents=["architecture_bridge"],
        )
        anchors = {fact.anchor for fact in planning.facts}
        self.assertFalse(planning.used_fallback)
        self.assertIn("sans fine-tuning", anchors)
        self.assertIn("faiblement supervisées", anchors)
        self.assertIn("couches linéaires", anchors)
        self.assertTrue(all(fact.references == ["S1"] for fact in planning.facts))

    def test_extracts_bridge_components_and_their_exact_references(self) -> None:
        planning = plan_evidence_facts(
            question="Pourquoi le Q-Former relie-t-il la vision à un LLM gelé ?",
            results=[
                _result(
                    1,
                    "Learnable query vectors extract the most relevant visual "
                    "features from the frozen image encoder.",
                ),
                _result(
                    2,
                    "The module acts as an information bottleneck before a frozen LLM.",
                ),
            ],
            retrieval_intents=["architecture_bridge"],
        )
        facts_by_anchor = {fact.anchor: fact for fact in planning.facts}
        self.assertEqual(facts_by_anchor["requêtes apprenables"].references, ["S1"])
        self.assertTrue(
            facts_by_anchor["requêtes apprenables"].claim.startswith("paper:")
        )
        self.assertEqual(facts_by_anchor["LLM gelé"].references, ["S2"])
        self.assertIn("caractéristiques visuelles pertinentes", facts_by_anchor)

    def test_projection_is_required_only_when_the_question_requests_a_bridge(self) -> None:
        evidence = [_result(1, "We use a linear projection between the models.")]
        unrelated = plan_evidence_facts(
            question="Quels composants restent gelés ?",
            results=evidence,
            retrieval_intents=["architecture_bridge"],
        )
        related = plan_evidence_facts(
            question="Comment relier les représentations par projection ?",
            results=evidence,
            retrieval_intents=["architecture_bridge"],
        )
        self.assertEqual(unrelated.facts, ())
        self.assertEqual(related.facts[0].anchor, "projection linéaire")

    def test_does_not_apply_architecture_ontology_to_a_generic_question(self) -> None:
        planning = plan_evidence_facts(
            question="What limitation is discussed?",
            results=[_result(1, "The system uses a frozen image encoder.")],
            retrieval_intents=[],
        )
        self.assertEqual(planning.facts, ())
        self.assertFalse(planning.used_fallback)

    def test_extracts_contrastive_and_teacher_student_supervision(self) -> None:
        planning = plan_evidence_facts(
            question="Compare the supervision and pre-training objectives.",
            results=[
                _result(
                    1,
                    "Given image-text pairs, we optimize a contrastive objective.",
                    document_id="clip",
                ),
                _result(
                    2,
                    "Discriminative Self-supervised Pre-training uses a student "
                    "and a teacher network.",
                    document_id="dino",
                ),
            ],
            retrieval_intents=["training_objective"],
        )
        anchors = {fact.anchor: fact.references for fact in planning.facts}
        self.assertEqual(anchors["image-text pairs"], ["S1"])
        self.assertEqual(anchors["contrastive objective"], ["S1"])
        self.assertEqual(anchors["self-supervised"], ["S2"])
        self.assertEqual(anchors["teacher and student"], ["S2"])

    def test_extracts_an_inference_decision_rule_only_from_explicit_evidence(self) -> None:
        planning = plan_evidence_facts(
            question="How is zero-shot matching used for classification?",
            results=[
                _result(
                    1,
                    "The names of all classes are encoded as possible texts. "
                    "We compute cosine similarity, scale it by a temperature "
                    "parameter, and normalize the scores with softmax.",
                )
            ],
            retrieval_intents=["inference_mechanism"],
        )
        self.assertEqual(
            {fact.anchor for fact in planning.facts},
            {"class names", "cosine similarity", "temperature", "softmax"},
        )

    def test_does_not_accept_an_inaccurate_french_bottleneck_synonym(self) -> None:
        planning = plan_evidence_facts(
            question="Pourquoi le module est-il un goulot d'étranglement ?",
            results=[_result(1, "The module acts as an information bottleneck.")],
            retrieval_intents=["architecture_bridge"],
        )
        fact = planning.facts[0]
        self.assertNotIn("entrave d'information", fact.alternatives)

    def test_formats_auditable_generation_requirements(self) -> None:
        rendered = format_fact_requirements(
            [
                PlannedFact(
                    id="F1",
                    claim="The module uses learnable queries.",
                    anchor="learnable queries",
                    references=["S1"],
                )
            ]
        )
        self.assertIn("F1", rendered)
        self.assertIn('Required exact anchor: "learnable queries"', rendered)
        self.assertIn("[S1]", rendered)

    def test_architecture_guide_requests_exact_model_and_bridge_names(self) -> None:
        rendered = format_answer_coverage_requirements(["architecture_bridge"])
        self.assertIn("visual encoder and language model", rendered)
        self.assertIn("linear projection", rendered)


class FactCompletenessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.results = [
            _result(1, "Evidence one."),
            _result(2, "Evidence two."),
        ]
        self.facts = [
            PlannedFact(
                id="F1",
                claim="The bridge uses trainable queries.",
                anchor="trainable queries",
                references=["S1"],
            ),
            PlannedFact(
                id="F2",
                claim="The language model stays frozen.",
                anchor="language model stays frozen",
                references=["S2"],
            ),
        ]

    def test_accepts_every_anchor_with_a_local_allowed_citation(self) -> None:
        validation = validate_answer(
            question="How does the bridge connect the models?",
            generated_answer=(
                "The bridge uses trainable queries [S1]. The language model "
                "stays frozen [S2]."
            ),
            results=self.results,
            planned_facts=self.facts,
        )
        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.covered_fact_ids, ("F1", "F2"))
        self.assertEqual(validation.cited_fact_ids, ("F1", "F2"))

    def test_rejects_a_missing_anchor(self) -> None:
        validation = validate_answer(
            question="How does the bridge connect the models?",
            generated_answer=(
                "The bridge uses trainable queries [S1]. The second component "
                "does not change [S2]."
            ),
            results=self.results,
            planned_facts=self.facts,
        )
        self.assertIn("missing_planned_fact_anchors:F2", validation.issues)
        self.assertTrue(validation.is_publishable)
        self.assertFalse(validation.fact_contract_valid)

    def test_rejects_a_citation_detached_from_its_fact(self) -> None:
        validation = validate_answer(
            question="How does the bridge connect the models?",
            generated_answer=(
                "The bridge uses trainable queries. The evidence supports this "
                "description [S1]. The language model stays frozen [S2]."
            ),
            results=self.results,
            planned_facts=self.facts,
        )
        self.assertIn("missing_planned_fact_citations:F1", validation.issues)
        self.assertTrue(validation.is_publishable)

    def test_accepts_a_controlled_anchor_equivalent(self) -> None:
        fact = PlannedFact(
            id="F1",
            claim="The module uses learnable queries.",
            anchor="learnable queries",
            alternatives=["learnable query vectors"],
            references=["S1"],
        )
        validation = validate_answer(
            question="How does the bridge work?",
            generated_answer="It uses learnable query vectors [S1].",
            results=self.results,
            planned_facts=[fact],
        )
        self.assertTrue(validation.is_valid)

    def test_rejects_abstention_when_a_fact_plan_exists(self) -> None:
        validation = validate_answer(
            question="How does the bridge connect the models?",
            generated_answer=INSUFFICIENT_EVIDENCE_MARKER,
            results=self.results,
            planned_facts=self.facts,
        )
        self.assertEqual(validation.issues, ("premature_abstention",))


if __name__ == "__main__":
    unittest.main()
