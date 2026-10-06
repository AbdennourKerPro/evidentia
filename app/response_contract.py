"""Deterministic language and citation contract for generated RAG answers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal, Sequence

from app.schemas import Citation, PlannedFact, SearchResult


ResponseLanguage = Literal["en", "fr", "zh"]
LanguageStatus = Literal["match", "mismatch", "uncertain"]

INSUFFICIENT_EVIDENCE_MARKER = "INSUFFICIENT_EVIDENCE"
_CITATION_GROUP_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_VALID_CITATION_GROUP_PATTERN = re.compile(
    r"S\d+(?:\s*[,;]\s*S\d+)*",
    flags=re.IGNORECASE,
)
_REFERENCE_POSITION_PATTERN = re.compile(r"S(\d+)", flags=re.IGNORECASE)
_WORD_PATTERN = re.compile(r"[a-zà-öø-ÿ]+", flags=re.IGNORECASE)
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")

_FRENCH_MARKERS = {
    "afin",
    "ainsi",
    "avec",
    "ce",
    "ces",
    "cette",
    "comme",
    "comment",
    "dans",
    "des",
    "donc",
    "elle",
    "elles",
    "entre",
    "est",
    "et",
    "leur",
    "leurs",
    "les",
    "mais",
    "par",
    "pour",
    "que",
    "quel",
    "quelle",
    "qui",
    "selon",
    "sont",
    "une",
    "utilise",
    "utilisent",
}

_ENGLISH_MARKERS = {
    "according",
    "and",
    "are",
    "as",
    "between",
    "but",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "that",
    "the",
    "their",
    "these",
    "they",
    "this",
    "to",
    "uses",
    "what",
    "which",
    "with",
}

LANGUAGE_LABELS: dict[ResponseLanguage, str] = {
    "en": "English",
    "fr": "French",
    "zh": "Chinese",
}

LOCALIZED_ABSTENTIONS: dict[ResponseLanguage, str] = {
    "en": "I cannot answer from the retrieved evidence alone.",
    "fr": "Je ne peux pas répondre à partir des seules preuves retrouvées.",
    "zh": "我无法仅根据检索到的证据回答。",
}


@dataclass(frozen=True)
class AnswerValidation:
    """Result of deterministic checks applied to one generated draft."""

    expected_language: ResponseLanguage
    citations: tuple[Citation, ...]
    issues: tuple[str, ...]
    abstained: bool
    planned_fact_ids: tuple[str, ...] = ()
    covered_fact_ids: tuple[str, ...] = ()
    cited_fact_ids: tuple[str, ...] = ()
    language_status: LanguageStatus | None = None
    language_check_method: str = "heuristic"

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def is_publishable(self) -> bool:
        """Allow advisory completeness issues when no blocking defect remains."""

        return all(_is_advisory_issue(issue) for issue in self.issues)

    @property
    def fact_contract_valid(self) -> bool:
        return not any(_is_advisory_issue(issue) for issue in self.issues)


def detect_question_language(question: str) -> ResponseLanguage:
    """Infer the response language for the project's English/French/Chinese UI."""

    if _script_ratio(question, _CJK_PATTERN) >= 0.2:
        return "zh"

    tokens = _tokens(question)
    french_score = sum(token in _FRENCH_MARKERS for token in tokens)
    english_score = sum(token in _ENGLISH_MARKERS for token in tokens)
    has_french_diacritic = any(
        character in "àâçéèêëîïôùûüÿœ"
        for character in question.casefold()
    )
    if french_score > english_score or has_french_diacritic:
        return "fr"
    return "en"


def validate_answer(
    *,
    question: str,
    generated_answer: str,
    results: Sequence[SearchResult],
    required_document_ids: Sequence[str] = (),
    planned_facts: Sequence[PlannedFact] = (),
) -> AnswerValidation:
    """Check language, citations, source coverage, and planned fact coverage."""

    expected_language = detect_question_language(question)
    required_documents = tuple(dict.fromkeys(required_document_ids))
    fact_plan = tuple(planned_facts)
    planned_fact_ids = tuple(fact.id for fact in fact_plan)
    issues: list[str] = []

    if generated_answer.strip() == INSUFFICIENT_EVIDENCE_MARKER:
        available_documents = {result.document_id for result in results}
        if fact_plan or (
            required_documents and set(required_documents) <= available_documents
        ):
            issues.append("premature_abstention")
        return AnswerValidation(
            expected_language=expected_language,
            citations=(),
            issues=tuple(issues),
            abstained=True,
            planned_fact_ids=planned_fact_ids,
        )

    if INSUFFICIENT_EVIDENCE_MARKER in generated_answer:
        issues.append("mixed_answer_and_abstention_marker")

    if not _appears_complete(generated_answer):
        issues.append("truncated_or_unbalanced_answer")
    language_status = assess_answer_language(generated_answer, expected_language)
    if language_status == "mismatch":
        issues.append(f"language_mismatch:expected={expected_language}")
    elif language_status == "uncertain":
        issues.append(f"language_uncertain:expected={expected_language}")

    citations = resolve_citations(generated_answer, results)
    if citations is None:
        issues.append("invalid_or_missing_citations")
        resolved_citations: tuple[Citation, ...] = ()
    else:
        resolved_citations = tuple(citations)
        cited_documents = {citation.document_id for citation in citations}
        missing_documents = [
            document_id
            for document_id in required_documents
            if document_id not in cited_documents
        ]
        if missing_documents:
            issues.append(
                "missing_citation_documents:" + ",".join(missing_documents)
            )

    covered_fact_ids, cited_fact_ids = _validate_planned_facts(
        generated_answer,
        fact_plan,
    )
    missing_anchors = [
        fact_id for fact_id in planned_fact_ids if fact_id not in covered_fact_ids
    ]
    if missing_anchors:
        issues.append("missing_planned_fact_anchors:" + ",".join(missing_anchors))
    missing_fact_citations = [
        fact_id
        for fact_id in covered_fact_ids
        if fact_id not in cited_fact_ids
    ]
    if missing_fact_citations:
        issues.append(
            "missing_planned_fact_citations:" + ",".join(missing_fact_citations)
        )

    return AnswerValidation(
        expected_language=expected_language,
        citations=resolved_citations,
        issues=tuple(issues),
        abstained=False,
        planned_fact_ids=planned_fact_ids,
        covered_fact_ids=tuple(covered_fact_ids),
        cited_fact_ids=tuple(cited_fact_ids),
        language_status=language_status,
    )


def resolve_citations(
    generated_answer: str, results: Sequence[SearchResult]
) -> list[Citation] | None:
    """Resolve only server-assigned references present in the evidence window."""

    positions: list[int] = []
    for group in _CITATION_GROUP_PATTERN.findall(generated_answer):
        stripped_group = group.strip()
        if not _VALID_CITATION_GROUP_PATTERN.fullmatch(stripped_group):
            continue
        positions.extend(
            int(match)
            for match in _REFERENCE_POSITION_PATTERN.findall(stripped_group)
        )
    if not positions or any(
        position < 1 or position > len(results) for position in positions
    ):
        return None

    unique_positions = list(dict.fromkeys(positions))
    return [
        Citation(
            reference=f"S{position}",
            document_id=results[position - 1].document_id,
            title=results[position - 1].title,
            page=results[position - 1].page,
            section=results[position - 1].section,
            score=results[position - 1].score,
        )
        for position in unique_positions
    ]


def describe_validation_issues(issues: Sequence[str]) -> list[str]:
    """Translate machine-readable issue codes into corrective instructions."""

    descriptions: list[str] = []
    for issue in issues:
        if issue == "premature_abstention":
            descriptions.append(
                "The draft abstained even though evidence from every required "
                "article is available. Re-check the passages before abstaining."
            )
        elif issue == "mixed_answer_and_abstention_marker":
            descriptions.append(
                "Do not mix an answer with INSUFFICIENT_EVIDENCE. Either provide "
                "a fully supported cited answer or return only that marker."
            )
        elif issue == "truncated_or_unbalanced_answer":
            descriptions.append(
                "The draft is truncated or has unbalanced brackets. Finish every "
                "sentence and every citation before returning the answer."
            )
        elif issue.startswith(("language_mismatch:expected=", "language_uncertain:expected=")):
            language = issue.rsplit("=", maxsplit=1)[-1]
            descriptions.append(
                f"The complete answer must be written in {LANGUAGE_LABELS[language]}."
            )
        elif issue == "invalid_or_missing_citations":
            descriptions.append(
                "Use at least one valid server reference such as [S1], and use "
                "only references shown in EVIDENCES."
            )
        elif issue.startswith("missing_citation_documents:"):
            document_ids = issue.split(":", maxsplit=1)[1]
            descriptions.append(
                "Add a supported claim and citation for each missing article: "
                f"{document_ids}."
            )
        elif issue.startswith("missing_planned_fact_anchors:"):
            fact_ids = issue.split(":", maxsplit=1)[1]
            descriptions.append(
                "Explicitly cover these FACT CHECKLIST items and copy each "
                f"required exact anchor: {fact_ids}."
            )
        elif issue.startswith("missing_planned_fact_citations:"):
            fact_ids = issue.split(":", maxsplit=1)[1]
            descriptions.append(
                "For each listed FACT CHECKLIST item, place one of its allowed "
                f"references in the same sentence as its exact anchor: {fact_ids}."
            )
        else:
            descriptions.append(issue)
    return descriptions


def _validate_planned_facts(
    answer: str,
    facts: Sequence[PlannedFact],
) -> tuple[list[str], list[str]]:
    """Check exact semantic anchors and their local source references."""

    normalized_answer = _normalize_phrase(answer)
    covered: list[str] = []
    cited: list[str] = []
    for fact in facts:
        normalized_anchors = [
            _normalize_phrase(anchor)
            for anchor in (fact.anchor, *fact.alternatives)
            if _normalize_phrase(anchor)
        ]
        if not any(anchor in normalized_answer for anchor in normalized_anchors):
            continue
        covered.append(fact.id)
        if _fact_has_local_citation(answer, fact):
            cited.append(fact.id)
    return covered, cited


def _fact_has_local_citation(answer: str, fact: PlannedFact) -> bool:
    """Require one allowed source in the sentence or bullet containing the anchor."""

    normalized_anchors = [
        _normalize_phrase(anchor)
        for anchor in (fact.anchor, *fact.alternatives)
        if _normalize_phrase(anchor)
    ]
    allowed_references = {reference.upper() for reference in fact.references}
    units = re.split(r"(?<=[.!?])\s+|\n+", answer)
    for unit in units:
        normalized_unit = _normalize_phrase(unit)
        if not any(anchor in normalized_unit for anchor in normalized_anchors):
            continue
        unit_references = {
            f"S{position}"
            for group in _CITATION_GROUP_PATTERN.findall(unit)
            if _VALID_CITATION_GROUP_PATTERN.fullmatch(group.strip())
            for position in _REFERENCE_POSITION_PATTERN.findall(group)
        }
        if unit_references & allowed_references:
            return True
    return False


def _normalize_phrase(text: str) -> str:
    """Normalize case, accents, and punctuation for robust anchor matching."""

    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    characters = [
        character if character.isalnum() else " "
        for character in without_accents.casefold()
    ]
    return re.sub(r"\s+", " ", "".join(characters)).strip()


def _is_advisory_issue(issue: str) -> bool:
    return issue.startswith(
        ("missing_planned_fact_anchors:", "missing_planned_fact_citations:")
    )


def language_prose(text: str) -> str:
    """Ignore non-prose noise, without deleting names or foreign sentences."""

    text = re.sub(r"```[\s\S]*?```|`[^`\n]*`", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[\s*S\d+(?:\s*[,;]\s*S\d+)*\s*\]", " ", text, flags=re.I)
    # Formulae and numeric values do not establish a natural language.
    text = re.sub(r"\\\([\s\S]*?\\\)|\\\[[\s\S]*?\\\]|\$[^$\n]*\$", " ", text)
    return re.sub(r"\d+(?:[.,]\d+)*\s*%?", " ", text)


def assess_answer_language(text: str, expected: ResponseLanguage) -> LanguageStatus:
    """Separate evidence of another language from insufficient lexical evidence.

    These conservative heuristics are not a general language detector. Ambiguous
    short, technical, mixed, or unsupported-language prose needs verification.
    No paper name, benchmark identifier, or numerical result is special-cased.
    """

    prose = language_prose(text)
    cjk_ratio = _script_ratio(prose, _CJK_PATTERN)
    if cjk_ratio >= 0.5:
        return "match" if expected == "zh" else "mismatch"
    if cjk_ratio > 0:
        return "uncertain"

    def scores(unit: str) -> tuple[int, int]:
        tokens = _tokens(unit)
        return (
            sum(token in (_FRENCH_MARKERS | {"le", "la"}) for token in tokens),
            sum(token in _ENGLISH_MARKERS for token in tokens),
        )

    french, english = scores(prose)
    if expected == "zh":
        return "mismatch" if max(french, english) >= 2 else "uncertain"
    wanted, other = (french, english) if expected == "fr" else (english, french)
    if other >= 2 and other >= 2 * max(wanted, 1):
        return "mismatch"
    # A whole foreign sentence must not disappear in a longer correct answer.
    for unit in re.split(r"(?<=[.!?])\s+|\n+", prose):
        fr, en = scores(unit)
        local_wanted, local_other = (fr, en) if expected == "fr" else (en, fr)
        if local_other >= 2 and local_other >= 2 * max(local_wanted, 1):
            return "mismatch"
    if wanted >= 2 and wanted >= 2 * max(other, 1):
        return "match"
    return "uncertain"


def _appears_complete(text: str) -> bool:
    """Catch common max-token truncation without making a semantic judgment."""

    stripped = text.rstrip()
    if not stripped:
        return False
    if stripped.count("[") != stripped.count("]"):
        return False
    return stripped[-1] in ".!?…。！？)]}\"'»”"


def _tokens(text: str) -> list[str]:
    """Return lowercase lexical tokens while preserving French characters."""

    return [token.casefold() for token in _WORD_PATTERN.findall(text)]


def _script_ratio(text: str, pattern: re.Pattern[str]) -> float:
    """Measure one script against all Unicode alphabetic characters."""

    alphabetic_count = sum(character.isalpha() for character in text)
    if alphabetic_count == 0:
        return 0.0
    return len(pattern.findall(unicodedata.normalize("NFC", text))) / alphabetic_count
