"""Detect equations that an answer states but that no retrieved passage contains.

A model can reproduce a famous equation from memory and attach a citation to it. The
passages are the only allowed source, so such an equation must be removed or the answer
must abstain. The check is a heuristic limited to equation-like math: it compares the
normalised equation with each passage and accepts it when enough of it appears there.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher

from app.schemas import SearchResult


MIN_EQUATION_CHARACTERS = 20
MIN_BLOCK_CHARACTERS = 5
SUPPORT_THRESHOLD = 0.5
MAX_REPORTED = 3

_MATH = re.compile(
    r"\$\$(?P<a>.+?)\$\$|\\\[(?P<b>.+?)\\\]|\\\((?P<c>.+?)\\\)"
    r"|(?<![\\$\d])\$(?!\s)(?P<d>[^$\n]{1,300}?)(?<![\s\\])\$(?!\d)",
    re.DOTALL,
)
_EQUATION_HINT = re.compile(r"=|\\(?:frac|dfrac|sum|prod|int|log|exp|min|max|argmin|argmax)\b")
_MARKUP = re.compile(
    r"\\(?:left|right|big|Big|bigg|Bigg|mathrm|mathbf|mathbb|mathcal|mathit|operatorname|text|"
    r"displaystyle|textstyle|quad|qquad|limits|nolimits)\b|\\[,;:! ]"
)
_UNICODE = {
    "τ": "tau", "ℓ": "ell", "∑": "sum", "∏": "prod", "∫": "int", "θ": "theta", "α": "alpha",
    "β": "beta", "γ": "gamma", "λ": "lambda", "μ": "mu", "σ": "sigma", "φ": "phi", "ϕ": "phi",
    "ε": "epsilon", "δ": "delta", "π": "pi", "ω": "omega", "η": "eta", "−": "-", "⊤": "top",
    "✶": "1", "𝟙": "1", "≤": "leq", "≥": "geq", "‖": "|",
}


_SEGMENT_BREAK = re.compile(r"\\q?quad\b|\\\\|;\s*(?=\\)|,\s*\\q?quad")


def extract_equations(answer: str) -> list[str]:
    """Equation-like parts of the answer's math.

    A span such as ``loss = ..., \\qquad s = ...`` is split into its equations: short
    definitions (a lone similarity, a symbol) are not checked, because PDF text flattens
    their indices and exponents beyond recognition.
    """

    equations = []
    for match in _MATH.finditer(answer):
        tex = next(group for group in match.groups() if group is not None).strip()
        for segment in _SEGMENT_BREAK.split(tex):
            segment = segment.strip(" ,;.\n")
            if len(squash(segment)) >= MIN_EQUATION_CHARACTERS and _EQUATION_HINT.search(segment):
                equations.append(segment)
    return equations


def squash(text: str) -> str:
    """Lowercase alphanumerics only, with layout macros and notation variants removed."""

    text = "".join(_UNICODE.get(char, char) for char in text)
    text = unicodedata.normalize("NFKD", text)
    text = _MARKUP.sub("", text)
    # Layout and negation have no counterpart in flattened PDF text ("k̸=i" loses its
    # slash, fractions lose \frac): ignore them on both sides.
    text = re.sub(r"\\(?:d|t)?frac\b|\\(?:neq?|not)\b|≠", "", text)
    text = re.sub(r"\\([A-Za-z]+)", r"\1", text)
    return re.sub(r"[^a-z0-9]", "", text.lower())


def support_ratio(equation: str, passage: str) -> float:
    """Share of the equation found in the passage in runs of at least five characters."""

    needle, haystack = squash(equation), squash(passage)
    if not needle or not haystack:
        return 0.0
    blocks = SequenceMatcher(None, needle, haystack, autojunk=False).get_matching_blocks()
    return sum(block.size for block in blocks if block.size >= MIN_BLOCK_CHARACTERS) / len(needle)


def unsupported_equations(answer: str, results: Sequence[SearchResult]) -> list[str]:
    """Equations of ``answer`` that no single passage supports well enough."""

    return [
        equation
        for equation in extract_equations(answer)
        if all(support_ratio(equation, result.text) < SUPPORT_THRESHOLD for result in results)
    ][:MAX_REPORTED]
