"""Deterministic evidence-derived fact planning for answer completeness."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from app.response_contract import ResponseLanguage, detect_question_language
from app.schemas import PlannedFact, SearchResult


MAX_PLANNED_FACTS = 8


@dataclass(frozen=True)
class FactPlanningResult:
    """Extracted fact plan plus whether the supported planner was unavailable."""

    facts: tuple[PlannedFact, ...]
    used_fallback: bool


@dataclass(frozen=True)
class EvidenceConceptRule:
    """A reusable evidence pattern with localized claim and anchor text."""

    key: str
    evidence_pattern: re.Pattern[str]
    localized_text: dict[ResponseLanguage, tuple[str, str]]
    localized_alternatives: dict[ResponseLanguage, tuple[str, ...]] | None = None
    question_pattern: re.Pattern[str] | None = None


_ARCHITECTURE_CONCEPT_RULES = (
    EvidenceConceptRule(
        key="no_finetuning",
        evidence_pattern=re.compile(
            r"without (?:the need for )?fine[- ]?tuning|without finetuning"
        ),
        localized_text={
            "en": (
                "The features are used without fine-tuning.",
                "without fine-tuning",
            ),
            "fr": (
                "Les caractéristiques sont utilisées sans fine-tuning.",
                "sans fine-tuning",
            ),
            "zh": ("The features are used without fine-tuning.", "without fine-tuning"),
        },
        localized_alternatives={
            "en": ("without finetuning",),
            "fr": ("sans finetuning", "sans nécessiter de fine-tuning"),
            "zh": ("without finetuning",),
        },
    ),
    EvidenceConceptRule(
        key="weak_supervision",
        evidence_pattern=re.compile(r"weakly\)? supervised"),
        localized_text={
            "en": (
                "The result is compared with weakly supervised alternatives.",
                "weakly supervised alternatives",
            ),
            "fr": (
                "Le résultat est comparé à des alternatives faiblement supervisées.",
                "faiblement supervisées",
            ),
            "zh": (
                "The result is compared with weakly supervised alternatives.",
                "weakly supervised alternatives",
            ),
        },
        localized_alternatives={
            "en": ("weakly-supervised alternatives",),
            "fr": ("alternatives supervisées faibles", "supervision faible"),
            "zh": ("weakly-supervised alternatives",),
        },
    ),
    EvidenceConceptRule(
        key="linear_readout",
        evidence_pattern=re.compile(
            r"linear (?:layer|layers|classifier|classifiers|probe|probes)"
        ),
        localized_text={
            "en": (
                "The information is accessible with linear layers or classifiers.",
                "linear layers",
            ),
            "fr": (
                "L'information est accessible avec des couches linéaires.",
                "couches linéaires",
            ),
            "zh": (
                "The information is accessible with linear layers.",
                "linear layers",
            ),
        },
        localized_alternatives={
            "en": ("linear classifiers", "linear probe"),
            "fr": ("classifieurs linéaires", "couche linéaire"),
            "zh": ("linear classifiers",),
        },
    ),
    EvidenceConceptRule(
        key="learnable_queries",
        evidence_pattern=re.compile(r"learnable quer"),
        localized_text={
            "en": (
                "The module uses learnable queries.",
                "learnable queries",
            ),
            "fr": (
                "Le module utilise des requêtes apprenables.",
                "requêtes apprenables",
            ),
            "zh": ("The module uses learnable queries.", "learnable queries"),
        },
        localized_alternatives={
            "en": ("learnable query vectors",),
            "fr": (
                "vecteurs de requête apprenants",
                "requêtes apprenantes",
                "learnable queries",
            ),
            "zh": ("learnable query vectors",),
        },
    ),
    EvidenceConceptRule(
        key="relevant_visual_features",
        evidence_pattern=re.compile(
            r"(?:most )?relevant visual (?:feature|features)|"
            r"visual (?:feature|features|representation|representations).{0,40}most relevant|"
            r"most useful visual feature"
        ),
        localized_text={
            "en": (
                "The module extracts the most relevant visual features.",
                "relevant visual features",
            ),
            "fr": (
                "Le module extrait les caractéristiques visuelles pertinentes.",
                "caractéristiques visuelles pertinentes",
            ),
            "zh": (
                "The module extracts the most relevant visual features.",
                "relevant visual features",
            ),
        },
        localized_alternatives={
            "en": ("visual features most relevant",),
            "fr": (
                "représentations visuelles pertinentes",
                "informations visuelles pertinentes",
            ),
            "zh": ("visual features most relevant",),
        },
    ),
    EvidenceConceptRule(
        key="frozen_image_encoder",
        evidence_pattern=re.compile(r"frozen (?:image|visual) encoder"),
        localized_text={
            "en": (
                "The system keeps a frozen image encoder.",
                "frozen image encoder",
            ),
            "fr": (
                "Le système conserve un encodeur d'images gelé.",
                "encodeur d'images gelé",
            ),
            "zh": ("The system keeps a frozen image encoder.", "frozen image encoder"),
        },
        localized_alternatives={
            "en": ("image encoder remains frozen",),
            "fr": (
                "encodeur d'image gelé",
                "encodeur d'image figé",
            ),
            "zh": ("image encoder remains frozen",),
        },
    ),
    EvidenceConceptRule(
        key="frozen_language_model",
        evidence_pattern=re.compile(r"frozen (?:llm|language model)"),
        localized_text={
            "en": (
                "The system keeps a frozen language model.",
                "frozen language model",
            ),
            "fr": (
                "Le système conserve un LLM gelé.",
                "LLM gelé",
            ),
            "zh": ("The system keeps a frozen language model.", "frozen language model"),
        },
        localized_alternatives={
            "en": ("frozen LLM", "language model remains frozen"),
            "fr": ("LLM figé", "modèle de langage gelé"),
            "zh": ("frozen LLM",),
        },
    ),
    EvidenceConceptRule(
        key="information_bottleneck",
        evidence_pattern=re.compile(r"information bottleneck"),
        localized_text={
            "en": (
                "The module acts as an information bottleneck.",
                "information bottleneck",
            ),
            "fr": (
                "Le module agit comme un goulot d'étranglement d'information.",
                "goulot d'étranglement d'information",
            ),
            "zh": ("The module acts as an information bottleneck.", "information bottleneck"),
        },
        localized_alternatives={
            "en": ("bottleneck",),
            "fr": ("goulot d'étranglement",),
            "zh": ("bottleneck",),
        },
    ),
    EvidenceConceptRule(
        key="linear_projection",
        evidence_pattern=re.compile(
            r"linear projection|projection matrix|linearly project"
        ),
        localized_text={
            "en": (
                "A linear projection connects the representation spaces.",
                "linear projection",
            ),
            "fr": (
                "Une projection linéaire relie les espaces de représentation.",
                "projection linéaire",
            ),
            "zh": ("A linear projection connects the spaces.", "linear projection"),
        },
        localized_alternatives={
            "en": ("projection matrix", "projected linearly"),
            "fr": ("matrice de projection", "couche de projection"),
            "zh": ("projection matrix",),
        },
        question_pattern=re.compile(
            r"projection|connect|bridge|align|relier|relie|pont"
        ),
    ),
)

_TRAINING_CONCEPT_RULES = (
    EvidenceConceptRule(
        key="paired_image_text",
        evidence_pattern=re.compile(
            r"image[-, ]+text (?:pair|pairs|pairing|pairings)|"
            r"\(image, text\) (?:pair|pairs|pairing|pairings)"
        ),
        localized_text={
            "en": ("The supervision uses image-text pairs.", "image-text pairs"),
            "fr": (
                "La supervision utilise des paires image-texte.",
                "paires image-texte",
            ),
            "zh": ("The supervision uses image-text pairs.", "image-text pairs"),
        },
        localized_alternatives={
            "en": ("image and text pairs", "paired images and text"),
            "fr": ("paires d'images et de textes",),
            "zh": ("image and text pairs",),
        },
    ),
    EvidenceConceptRule(
        key="contrastive_objective",
        evidence_pattern=re.compile(
            r"contrastive objective|contrastive language-image|"
            r"symmetric cross entropy loss|maximize the cosine similarity"
        ),
        localized_text={
            "en": ("Training uses a contrastive objective.", "contrastive objective"),
            "fr": (
                "L'entraînement utilise un objectif contrastif.",
                "objectif contrastif",
            ),
            "zh": ("Training uses a contrastive objective.", "contrastive objective"),
        },
        localized_alternatives={
            "en": ("contrastive loss", "contrastive learning"),
            "fr": ("perte contrastive", "apprentissage contrastif"),
            "zh": ("contrastive loss",),
        },
    ),
    EvidenceConceptRule(
        key="self_supervision",
        evidence_pattern=re.compile(
            r"discriminative self-supervised pre-training|"
            r"pretrained.{0,40}(?:with )?no supervision|self-supervised objective"
        ),
        localized_text={
            "en": ("Training is self-supervised.", "self-supervised"),
            "fr": ("L'entraînement est auto-supervisé.", "auto-supervisé"),
            "zh": ("Training is self-supervised.", "self-supervised"),
        },
        localized_alternatives={
            "en": ("without supervision", "no supervision"),
            "fr": ("sans supervision",),
            "zh": ("without supervision",),
        },
    ),
    EvidenceConceptRule(
        key="teacher_student",
        evidence_pattern=re.compile(
            r"student and a teacher|student.{0,80}teacher|teacher.{0,80}student"
        ),
        localized_text={
            "en": ("The objective uses teacher and student networks.", "teacher and student"),
            "fr": (
                "L'objectif utilise des réseaux enseignant et étudiant.",
                "enseignant et étudiant",
            ),
            "zh": ("The objective uses teacher and student networks.", "teacher and student"),
        },
        localized_alternatives={
            "en": ("teacher-student", "student and teacher"),
            "fr": ("teacher-student", "teacher et student"),
            "zh": ("teacher-student",),
        },
    ),
)

_INFERENCE_CONCEPT_RULES = (
    EvidenceConceptRule(
        key="candidate_class_text",
        evidence_pattern=re.compile(
            r"(?:names?|descriptions?) of (?:all |the )?(?:target )?(?:dataset'?s )?"
            r"classes|candidate classes|possible texts"
        ),
        localized_text={
            "en": (
                "Candidate class names are encoded as text.",
                "class names",
            ),
            "fr": (
                "Les noms de classes candidats sont encodés comme texte.",
                "noms de classes",
            ),
            "zh": ("Candidate class names are encoded as text.", "class names"),
        },
        localized_alternatives={
            "en": ("candidate classes", "possible texts", "class descriptions"),
            "fr": ("classes candidates", "descriptions de classes"),
            "zh": ("candidate classes", "possible texts"),
        },
    ),
    EvidenceConceptRule(
        key="cosine_similarity",
        evidence_pattern=re.compile(r"cosine similarity"),
        localized_text={
            "en": (
                "Image and text embeddings are compared with cosine similarity.",
                "cosine similarity",
            ),
            "fr": (
                "Les embeddings image et texte sont comparés par similarité cosinus.",
                "similarité cosinus",
            ),
            "zh": ("Embeddings use cosine similarity.", "cosine similarity"),
        },
        localized_alternatives={
            "en": ("cosine-similarity",),
            "fr": ("cosine similarity",),
            "zh": ("cosine-similarity",),
        },
    ),
    EvidenceConceptRule(
        key="temperature_scaling",
        evidence_pattern=re.compile(r"temperature (?:parameter|scaling)|scaled by (?:a )?temperature"),
        localized_text={
            "en": (
                "A temperature parameter scales the similarity scores.",
                "temperature",
            ),
            "fr": (
                "Un paramètre de température met à l'échelle les scores.",
                "température",
            ),
            "zh": ("A temperature parameter scales the scores.", "temperature"),
        },
        localized_alternatives={
            "en": ("temperature parameter", "temperature scaling"),
            "fr": ("paramètre de température",),
            "zh": ("temperature parameter",),
        },
    ),
    EvidenceConceptRule(
        key="softmax_decision",
        evidence_pattern=re.compile(r"softmax|probability distribution"),
        localized_text={
            "en": (
                "Softmax converts the scores into a probability distribution.",
                "softmax",
            ),
            "fr": (
                "Softmax transforme les scores en distribution de probabilités.",
                "softmax",
            ),
            "zh": ("Softmax forms a probability distribution.", "softmax"),
        },
        localized_alternatives={
            "en": ("probability distribution",),
            "fr": ("distribution de probabilités",),
            "zh": ("probability distribution",),
        },
    ),
)

_INTENT_PLANNING_REQUIREMENTS: dict[str, str] = {
    "numeric_scale": (
        "For each requested entity ask: What is the dataset name? What object is "
        "counted? What exact value and unit are reported?"
    ),
    "quantitative_comparison": (
        "State the compared methods, the exact task or benchmark, the reported "
        "performance difference, and the resource or parameter trade-off only "
        "when all details are supported by EVIDENCES."
    ),
    "prompt_mechanism": (
        "For each model ask: What is the prompt's role? Which concrete input "
        "types or templates are used? What operation or output follows?"
    ),
    "architecture_bridge": (
        "Answer each supported coverage question: What stays frozen or is used "
        "without fine-tuning? What is trainable? What visual information flows? "
        "What is the functional role? What comparison is reported? What simple "
        "readout demonstrates the result? In a comparison, explicitly name the "
        "visual encoder and language model shown in each article and state the "
        "exact bridge type; write 'linear projection' or its natural translation "
        "when the evidence describes a linear projector."
    ),
    "inference_mechanism": (
        "Explain the supported inference pipeline in order: candidate class text, "
        "representation or score, normalization, and decision rule. Name cosine "
        "similarity, temperature, or softmax only when EVIDENCES state them."
    ),
    "capabilities": (
        "Ask separately: Which task families are evaluated? Which emerging "
        "capabilities are reported? What condition makes them zero-shot?"
    ),
    "training_objective": (
        "For each model identify the supervision data and the exact objective or "
        "loss. State whether training is contrastive, predictive, self-supervised, "
        "or teacher-student, and name image-level or patch-level targets when shown."
    ),
}


def plan_evidence_facts(
    *,
    question: str,
    results: Sequence[SearchResult],
    retrieval_intents: Sequence[str] = (),
) -> FactPlanningResult:
    """Build an auditable checklist only where deterministic rules are supported."""

    if not results:
        return FactPlanningResult(facts=(), used_fallback=False)

    language = detect_question_language(question)
    normalized_question = _normalize(question)
    facts: list[PlannedFact] = []
    active_rules: list[EvidenceConceptRule] = []
    if "architecture_bridge" in retrieval_intents:
        active_rules.extend(_ARCHITECTURE_CONCEPT_RULES)
    if "inference_mechanism" in retrieval_intents:
        active_rules.extend(_INFERENCE_CONCEPT_RULES)
    if "training_objective" in retrieval_intents:
        active_rules.extend(_TRAINING_CONCEPT_RULES)
    if not active_rules:
        return FactPlanningResult(facts=(), used_fallback=False)

    for rule in active_rules:
        if rule.question_pattern and not rule.question_pattern.search(normalized_question):
            continue
        references_by_document: dict[str, list[str]] = {}
        for position, result in enumerate(results, start=1):
            if not rule.evidence_pattern.search(_normalize(result.text)):
                continue
            references_by_document.setdefault(result.document_id, []).append(
                f"S{position}"
            )
        claim, anchor = rule.localized_text[language]
        alternatives = (
            rule.localized_alternatives.get(language, ())
            if rule.localized_alternatives
            else ()
        )
        for document_id, references in references_by_document.items():
            facts.append(
                PlannedFact(
                    id=f"F{len(facts) + 1}",
                    claim=f"{document_id}: {claim}",
                    anchor=anchor,
                    alternatives=list(alternatives),
                    references=references,
                )
            )
            if len(facts) == MAX_PLANNED_FACTS:
                break
        if len(facts) == MAX_PLANNED_FACTS:
            break

    return FactPlanningResult(
        facts=tuple(facts),
        used_fallback=not facts,
    )


def format_fact_requirements(facts: Sequence[PlannedFact]) -> str:
    """Render the validated plan as explicit instructions for answer generation."""

    if not facts:
        return ""
    lines = []
    for fact in facts:
        references = ", ".join(f"[{reference}]" for reference in fact.references)
        lines.extend(
            [
                f"- {fact.id}: {fact.claim}",
                f'  Required exact anchor: "{fact.anchor}"',
                "  Accepted equivalents: "
                + (", ".join(f'"{item}"' for item in fact.alternatives) or "none"),
                f"  Allowed supporting references: {references}",
            ]
        )
    return (
        "FACT CHECKLIST CONTRACT:\n"
        "Explicitly cover every fact below. In the same sentence, copy its "
        "required anchor exactly and cite at least one allowed reference. Do not "
        "show fact IDs or describe the checklist in the answer. Every claim starts "
        "with its document ID: never attribute that fact to another article.\n"
        + "\n".join(lines)
    )


def format_answer_coverage_requirements(retrieval_intents: Sequence[str]) -> str:
    """Give generation an evidence-conditional guide for every detected intent."""

    requirements = [
        _INTENT_PLANNING_REQUIREMENTS[intent]
        for intent in retrieval_intents
        if intent in _INTENT_PLANNING_REQUIREMENTS
    ]
    if not requirements:
        return ""
    return (
        "EVIDENCE COVERAGE GUIDE:\n- "
        + "\n- ".join(requirements)
        + "\nAddress every coverage question supported by EVIDENCES."
    )


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()
