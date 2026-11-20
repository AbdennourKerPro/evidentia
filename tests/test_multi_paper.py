"""Several papers in one conversation: balanced evidence, comparison routing, answer guidelines."""

import unittest

from app import agentic_rag
from app.agentic_rag import _analyze_question, _answer_guidelines
from app.hybrid_retrieval import ensure_paper_representation
from app.rag_service import MULTI_PAPER_GUIDELINES, PARTIAL_ANSWER_GUIDELINES
from app.schemas import SearchResult

A, B, C = "arxiv-1111.00001v1", "arxiv-2222.00002v1", "arxiv-3333.00003v1"
CATALOG = {A: {"title": "Mamba", "aliases": (r"\bmamba\b",)}, B: {"title": "Attention", "aliases": (r"\btransformer\b",)},
           C: {"title": "Other", "aliases": (r"\bother\b",)}}


def hit(chunk_id: int, document_id: str) -> SearchResult:
    return SearchResult(chunk_id=chunk_id, score=1 / chunk_id, document_id=document_id, title=document_id,
                        page=1, section="s", language="en", text=f"t{chunk_id}")


class EnsurePaperRepresentationTests(unittest.TestCase):
    def test_a_relevant_absent_paper_replaces_the_weakest_passage_of_a_repeated_one(self):
        ranked = [hit(1, A), hit(2, A), hit(3, A), hit(4, B), hit(5, A)]
        selected = ensure_paper_representation(ranked, ranked[:3], window=5)
        self.assertEqual([r.chunk_id for r in selected], [1, 2, 4])

    def test_a_paper_outside_the_window_is_not_forced_in(self):
        ranked = [hit(1, A), hit(2, A), hit(3, A), hit(4, A), hit(5, B)]
        self.assertEqual(ensure_paper_representation(ranked, ranked[:3], window=4), ranked[:3])

    def test_it_never_removes_the_only_passage_of_a_paper_or_changes_the_size(self):
        ranked = [hit(1, A), hit(2, B), hit(3, C)]
        self.assertEqual(ensure_paper_representation(ranked, ranked[:2], window=3), ranked[:2])
        many = [hit(1, A), hit(2, A), hit(3, B), hit(4, C)]
        result = ensure_paper_representation(many, many[:2], window=4)
        self.assertEqual(len(result), 2)
        self.assertEqual({r.document_id for r in result}, {A, B})


class AnalyzeQuestionTests(unittest.TestCase):
    def state(self, question, **extra):
        return {"question": question, "catalog": CATALOG, "selected_document_ids": None,
                "corpus_document_ids": [A, B], "focus_document_ids": [], **extra}

    def test_comparison_without_names_uses_every_conversation_paper_in_agent_mode(self):
        update = _analyze_question(self.state("Compare the two approaches.", multi_paper=True))
        self.assertEqual((update["strategy"], update["detected_document_ids"]), ("per_source", [A, B]))

    def test_other_modes_keep_their_behaviour(self):
        update = _analyze_question(self.state("Compare the two approaches."))
        self.assertEqual((update["strategy"], update["detected_document_ids"]), ("global", []))
        plain = _analyze_question(self.state("How does it work?", multi_paper=True))
        self.assertEqual(plain["strategy"], "global")

    def test_named_papers_and_agent_focus_still_decide(self):
        named = _analyze_question(self.state("Compare Mamba and the transformer.", multi_paper=True, corpus_document_ids=[A, B, C]))
        self.assertEqual(named["detected_document_ids"], [A, B])
        focus = _analyze_question(self.state("How do they differ?", multi_paper=True, focus_document_ids=[A, B]))
        self.assertEqual(focus["detected_document_ids"], [A, B])

    def test_the_corpus_fallback_is_capped(self):
        many = {f"arxiv-{i}": {"title": str(i), "aliases": ()} for i in range(6)}
        update = _analyze_question(self.state("Compare them.", multi_paper=True, catalog=many, corpus_document_ids=list(many)))
        self.assertEqual(update["strategy"], "global")


class GuidelineTests(unittest.TestCase):
    base = {"retrieval_profile": "candidate_recovery_v2", "partial_answers": True}

    def test_multi_paper_guidance_needs_the_flag_and_two_papers(self):
        self.assertNotIn(MULTI_PAPER_GUIDELINES, _answer_guidelines({**self.base}))
        self.assertNotIn(MULTI_PAPER_GUIDELINES, _answer_guidelines({**self.base, "multi_paper": True, "corpus_document_ids": [A]}))
        text = _answer_guidelines({**self.base, "multi_paper": True, "corpus_document_ids": [A, B]})
        self.assertIn(PARTIAL_ANSWER_GUIDELINES, text)
        self.assertTrue(text.endswith(MULTI_PAPER_GUIDELINES))

    def test_uncovered_subjects_are_stated_to_the_writer(self):
        text = _answer_guidelines({**self.base, "multi_paper": True, "corpus_document_ids": [A],
                                   "uncovered_subjects": ["Transformer", "BERT"]})
        self.assertIn("No attached paper was found for: Transformer; BERT.", text)
        self.assertNotIn("No attached paper was found", _answer_guidelines({**self.base, "uncovered_subjects": ["X"]}))


if __name__ == "__main__":
    unittest.main()
