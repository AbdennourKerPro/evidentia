"""Small text helpers shared by question routing and paper alias creation."""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """Lowercase text and remove accents so French routing stays predictable."""

    decomposed = unicodedata.normalize("NFKD", text.lower())
    without_accents = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", without_accents).strip()
