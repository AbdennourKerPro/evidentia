import unittest

from app import agentic_rag
from app.formula_fidelity import extract_equations, support_ratio, unsupported_equations
from app.response_contract import describe_validation_issues
from app.schemas import SearchResult

NTXENT = (r"\ell_{i,j} = -\log \frac{\exp(\operatorname{sim}(z_i,z_j)/\tau)}"
          r"{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\operatorname{sim}(z_i,z_k)/\tau)}")
# The real chunk indexed for SimCLR: Docling could not decode the equation.
UNDECODED = ("We treat the other 2(N - 1) augmented examples as negatives. Let sim(u, v) = u⊤v/‖u‖‖v‖ denote "
             "the dot product between ℓ2 normalized u and v. Then the loss function for a positive pair of "
             "examples (i, j) is defined as\n<!-- formula-not-decoded -->\nwhere ✶[k̸=i] ∈ {0, 1} is an "
             "indicator function and τ denotes a temperature parameter.")
DECODED = UNDECODED.replace("<!-- formula-not-decoded -->", (
    r"\ell_{i,j} = - \log \frac{ \exp( \mathrm{sim}( z_i, z_j ) / \tau ) }"
    r"{ \sum_{k=1}^{2N} \mathbb{1}_{[k \ne i]} \exp( \mathrm{sim}( z_i, z_k ) / \tau ) }"))


def passage(text):
    return SearchResult(chunk_id=1, score=1, document_id="d", title="t", page=2, section="s", language="en", text=text)


class FormulaFidelityTests(unittest.TestCase):
    def test_only_equation_like_math_is_checked(self):
        answer = (r"Inline \(\tau\) and \(2(N-1)\) and costs $5 and $6, but \[" + NTXENT + r"\] and $L = \sum_{i=1}^{N} \log p(x_i \mid x_{<i}) + \lambda \|\theta\|^2$.")
        found = extract_equations(answer)
        self.assertEqual(len(found), 2)
        self.assertTrue(found[0].startswith(r"\ell_{i,j}"))

    def test_compound_equations_are_checked_part_by_part(self):
        compound = NTXENT.replace("\\operatorname{sim}(z_i,z_j)", "s_{i,j}") + r", \qquad s_{i,j} = \frac{z_i^\top z_j}{\|z_i\|\|z_j\|}"
        found = extract_equations(r"\[" + compound + r"\]")
        self.assertEqual(len(found), 1)  # the short definition of s_{i,j} is not checked
        self.assertNotIn("qquad", found[0])

    def test_memory_formula_is_unsupported_when_the_passage_has_no_decoded_equation(self):
        answer = r"The loss is \[" + NTXENT + r"\] [S1]."
        self.assertEqual(unsupported_equations(answer, [passage(UNDECODED)]), [NTXENT])
        self.assertLess(support_ratio(NTXENT, UNDECODED), 0.5)

    def test_decoded_equation_supports_a_rewritten_formula(self):
        self.assertGreaterEqual(support_ratio(NTXENT, DECODED), 0.9)
        self.assertEqual(unsupported_equations(r"\[" + NTXENT + r"\]", [passage("unrelated"), passage(DECODED)]), [])

    def test_equation_written_in_prose_is_supported(self):
        answer = r"Similarity is \(\operatorname{sim}(u,v)=\frac{u^\top v}{\|u\|\|v\|}\)."
        self.assertEqual(unsupported_equations(answer, [passage(UNDECODED)]), [])

    def test_no_math_or_no_equations_means_nothing_to_report(self):
        self.assertEqual(unsupported_equations("Plain answer [S1].", [passage("x")]), [])
        self.assertEqual(unsupported_equations(r"Only \(\tau\) here.", []), [])

    def test_correction_prompt_explains_the_issue(self):
        text = " ".join(describe_validation_issues(["unsupported_formula"]))
        self.assertIn("not available in the retrieved passages", text)


class GraphHookTests(unittest.TestCase):
    def state(self, check):
        return {
            "question": "Give me the formula of the loss.",
            "generated_answer": r"The loss is \[" + NTXENT + r"\] [S1].",
            "selected_results": [passage(UNDECODED)],
            "strategy": "global", "fact_plan": [], "check_formulas": check,
        }

    def test_flag_adds_a_blocking_issue_only_in_agent_mode(self):
        flagged = agentic_rag._validate_answer(self.state(True))["answer_validation"]
        self.assertIn("unsupported_formula", flagged.issues)
        self.assertFalse(flagged.is_publishable)  # still present after the correction -> abstention
        plain = agentic_rag._validate_answer(self.state(False))["answer_validation"]
        self.assertNotIn("unsupported_formula", plain.issues)


if __name__ == "__main__":
    unittest.main()


class PartialAnswerTests(unittest.TestCase):
    """Agent mode answers what the passages support instead of abstaining on a missing detail."""

    def state(self, partial, profile="candidate_recovery_v2"):
        return {"retrieval_profile": profile, "partial_answers": partial}

    def test_guidelines_are_added_only_in_agent_mode(self):
        from app.rag_service import PARTIAL_ANSWER_GUIDELINES, SOURCE_FIDELITY_GUIDELINES

        self.assertEqual(agentic_rag._answer_guidelines(self.state(False)), SOURCE_FIDELITY_GUIDELINES)
        agent = agentic_rag._answer_guidelines(self.state(True))
        self.assertTrue(agent.startswith(SOURCE_FIDELITY_GUIDELINES))
        self.assertTrue(agent.endswith(PARTIAL_ANSWER_GUIDELINES))
        # Older profiles stay byte-identical unless the agent flag is set.
        self.assertEqual(agentic_rag._answer_guidelines(self.state(False, "question_rerank")), "")
        self.assertEqual(agentic_rag._answer_guidelines(self.state(True, "question_rerank")), PARTIAL_ANSWER_GUIDELINES)

    def test_partial_rule_forbids_outside_knowledge_and_keeps_the_marker_for_irrelevant_evidence(self):
        from app.rag_service import PARTIAL_ANSWER_GUIDELINES, PARTIAL_CORRECTION_NOTE

        self.assertIn("Never fill the gap with outside knowledge", PARTIAL_ANSWER_GUIDELINES)
        self.assertIn("say nothing relevant", PARTIAL_ANSWER_GUIDELINES)
        self.assertIn("return only INSUFFICIENT_EVIDENCE", PARTIAL_CORRECTION_NOTE)

    def test_prompt_reaches_the_generation_call(self):
        from unittest.mock import patch

        from app import rag_service
        from app.rag_service import PARTIAL_ANSWER_GUIDELINES

        with patch.object(rag_service, "generate_chat", return_value="ok [S1]") as chat:
            rag_service.generate_answer_draft(
                question="Give me the LaTeX formula", results=[passage("text")],
                answer_guidelines=agentic_rag._answer_guidelines(self.state(True)))
        self.assertIn(PARTIAL_ANSWER_GUIDELINES, chat.call_args.kwargs["system_message"])
