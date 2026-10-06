"""Transparent intent detection and query expansion for difficult retrievals."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal, Sequence

from app.schemas import SearchResult


RetrievalIntent = Literal[
    "numeric_scale",
    "quantitative_comparison",
    "prompt_mechanism",
    "architecture_bridge",
    "inference_mechanism",
    "capabilities",
    "training_objective",
]

_INTENT_PATTERNS: dict[RetrievalIntent, re.Pattern[str]] = {
    "numeric_scale": re.compile(
        r"\b(scale|size|count|number|how many|training data|dataset|corpus|"
        r"taille|nombre|combien|donnees|jeu de donnees|million|billion|milliard)\b"
    ),
    "quantitative_comparison": re.compile(
        r"\b(efficiency|efficient|outperform|fewer|more|"
        r"ratio|percentage|percent|performance gain|gain|efficacit[eé]|"
        r"surpasse|moins|plus|rapport)\b"
    ),
    "prompt_mechanism": re.compile(
        r"\b(prompt|prompts|prompting|template|templates|invite|invites)\b"
    ),
    "architecture_bridge": re.compile(
        r"\b(q[- ]?former|bottleneck|bridge|projection|frozen|connect|alignment|"
        r"goulot|pont|projection|gele|gelees|relie|relier|alignement)\b"
    ),
    "inference_mechanism": re.compile(
        r"\b(?:zero[- ]?shot).{0,80}\b(?:classif(?:y|ier|ication)?|"
        r"predict(?:ion)?|decision|infer(?:ence)?|classif(?:ier|ication)|"
        r"pr[eé]dire|d[eé]cision)\b|\b(?:classif(?:y|ier|ication)?|"
        r"predict(?:ion)?|infer(?:ence)?|classif(?:ier|ication)|"
        r"pr[eé]dire).{0,80}\b(?:zero[- ]?shot)\b"
    ),
    "capabilities": re.compile(
        r"\b(capability|capabilities|task|tasks|benchmark|highlighted|emerging|"
        r"capacite|capacites|tache|taches|evaluation)\b"
    ),
}

# Keep the suffixes outside the alternatives so terms such as "pre-training"
# and "supervision" are recognized as complete words.
_INTENT_PATTERNS["training_objective"] = re.compile(
    r"\b(?:supervision|supervised|self-supervised|unsupervised|"
    r"pre[- ]?train(?:ing|ed)?|objective|loss|teacher|student|contrastive|"
    r"supervise|auto-supervise|preentrainement|objectif|perte)\b"
)

_FACET_QUERIES: dict[RetrievalIntent, str] = {
    "numeric_scale": (
        "Exact reported dataset scale and quantitative details: dataset name, "
        "number of images, image-text pairs, masks, annotations, samples, "
        "million, billion."
    ),
    "quantitative_comparison": (
        "Exact experimental comparison and efficiency trade-off: compared "
        "methods, benchmark or task, reported performance difference, and "
        "parameter, memory, compute, or data ratio."
    ),
    "prompt_mechanism": (
        "Exact prompt mechanism and prompt input types: class text templates, "
        "context ensembling, points, boxes, text, masks, sparse prompts, dense "
        "prompts, prompt encoder."
    ),
    "architecture_bridge": (
        "Exact architecture of the trainable bridge: learnable queries, "
        "information bottleneck, relevant visual features, frozen image encoder, "
        "frozen language model, projection."
    ),
    "inference_mechanism": (
        "Exact inference and decision mechanism: candidate labels or text "
        "descriptions, image and text representations, similarity or scoring, "
        "normalization or temperature, probabilities, and final predicted class."
    ),
    "capabilities": (
        "Exact reported evaluation tasks and emerging zero-shot capabilities: "
        "task names, instruction following, visual reasoning, conversation."
    ),
    "training_objective": (
        "Exact pre-training supervision and objective: paired or unpaired data, "
        "labels or text, contrastive or predictive loss, teacher-student targets, "
        "global and local views, masked patches, image-level and patch-level loss."
    ),
}


def detect_retrieval_intents(question: str) -> tuple[RetrievalIntent, ...]:
    """Return only explicit, auditable intents found in the user question."""

    normalized = _normalize(question)
    return tuple(
        intent
        for intent, pattern in _INTENT_PATTERNS.items()
        if pattern.search(normalized)
    )


def build_query_variants(
    *, question: str, primary_query: str, preserve_original: bool = False,
) -> list[str]:
    """Add one English retrieval facet for each detected difficult intent."""

    variants = [primary_query.strip()]
    if preserve_original:
        variants.insert(0, question.strip())
    for intent in detect_retrieval_intents(question):
        variants.append(f"{primary_query.strip()} {_FACET_QUERIES[intent]}")
    return list(dict.fromkeys(variant for variant in variants if variant))


def rerank_for_intents(
    results: Sequence[SearchResult],
    intents: Sequence[RetrievalIntent],
) -> list[SearchResult]:
    """Stably prioritize sections and terms implied by explicit query intents."""

    intent_set = set(intents)
    if not intent_set:
        return list(results)
    return sorted(
        results,
        key=lambda result: (_intent_score(result, intent_set), result.score),
        reverse=True,
    )


def _intent_score(
    result: SearchResult,
    intents: set[RetrievalIntent],
) -> int:
    """Compute an auditable lexical score without benchmark-specific answers."""

    text = _normalize(f"{result.section} {result.text}")
    section = _normalize(result.section)
    score = 0

    if "numeric_scale" in intents:
        magnitude_count = len(
            re.findall(
                r"\b\d+(?:[.,]\d+)?\s*(?:m|b|million|billion)\b",
                text,
            )
        )
        score += min(magnitude_count, 3) * 2
        score += 2 * int(bool(re.search(r"\b[a-z]+-\d+[mb]\b", text)))
        score += sum(
            term in text
            for term in ("dataset", "corpus", "images", "masks", "pairs")
        )
        score += int(
            any(
                term in text
                for term in ("contains", "consists", "includes", "assemble")
            )
        )
        score += 3 * int(
            any(
                term in section
                for term in (
                    "introduction",
                    "abstract",
                    "data processing",
                    "pretraining data",
                    "dataset",
                    "composition",
                )
            )
        )
        score -= 3 * int(
            any(
                term in section
                for term in ("ablation", "cost", "impact", "fairness", "recipe")
            )
        )

    if "quantitative_comparison" in intents:
        score += 2 * len(re.findall(r"\b\d+(?:[.,]\d+)?(?:\s*%|x\b)", text))
        score += sum(
            term in text
            for term in (
                "outperform",
                "compared",
                "fewer",
                "more trainable",
                "parameters",
                "compute-efficient",
                "efficiency",
                "accuracy",
                "benchmark",
            )
        )
        score += 3 * int(
            any(
                term in section
                for term in ("result", "experiment", "evaluation", "abstract", "introduction")
            )
        )
        score -= 2 * int(any(term in section for term in ("related work", "references")))

    if "prompt_mechanism" in intents:
        score += sum(
            term in text
            for term in (
                "prompt encoder",
                "sparse",
                "dense",
                "points",
                "boxes",
                "masks",
                "templates",
                "ensembl",
                "class",
            )
        )
        score += 3 * int(
            any(
                term in section
                for term in ("model", "method", "prompt engineering", "architecture")
            )
        )

    if "architecture_bridge" in intents:
        score += sum(
            term in text
            for term in (
                "q-former",
                "learnable quer",
                "information bottleneck",
                "visual features",
                "frozen image encoder",
                "frozen language model",
                "projection",
            )
        )

    if "inference_mechanism" in intents:
        score += sum(
            term in text
            for term in (
                "zero-shot classification",
                "candidate class",
                "class names",
                "text descriptions",
                "cosine similarity",
                "temperature",
                "softmax",
                "probability distribution",
                "most probable",
            )
        )
        score += 4 * int(
            any(
                term in section
                for term in (
                    "zero-shot transfer",
                    "zero-shot classification",
                    "inference",
                    "method",
                    "approach",
                )
            )
        )
        score -= 2 * int(any(term in section for term in ("limitations", "related work", "references")))

    if "capabilities" in intents:
        score += sum(
            term in text
            for term in (
                "visual question answering",
                "image captioning",
                "image-text retrieval",
                "zero-shot",
                "instruction",
                "visual reasoning",
                "conversation",
            )
        )
        score += 2 * int(
            any(term in section for term in ("abstract", "introduction", "experiment"))
        )

    if "training_objective" in intents:
        score += sum(
            term in text
            for term in (
                "pre-training objective",
                "contrastive objective",
                "cross-entropy loss",
                "image-text",
                "image, text",
                "teacher",
                "student",
                "self-supervised",
                "image-level objective",
                "patch-level objective",
                "masked patch",
                "global crop",
            )
        )
        score += 4 * int(
            any(
                term in section
                for term in (
                    "method",
                    "pre-training",
                    "pretraining",
                    "objective",
                    "approach",
                )
            )
        )
        score -= 3 * int(
            any(term in section for term in ("related work", "references"))
        )

    return score


def _normalize(text: str) -> str:
    """Remove accents and normalize spacing before deterministic routing."""

    decomposed = unicodedata.normalize("NFKD", text.casefold())
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()
