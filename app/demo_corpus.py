"""A small synthetic corpus used to validate retrieval before PDF ingestion."""

from app.schemas import EvidenceChunk


DEMO_CHUNKS: tuple[EvidenceChunk, ...] = (
    EvidenceChunk(
        id=1,
        document_id="heat-notes-2024",
        title="Carnet d'observation : revêtements réfléchissants",
        page=2,
        section="Résultat principal",
        language="fr",
        text=(
            "Dans les rues testées, les surfaces à albédo élevé ont réduit la "
            "température mesurée en journée. L'effet est plus marqué lorsque "
            "l'ombre des arbres reste faible."
        ),
    ),
    EvidenceChunk(
        id=2,
        document_id="heat-notes-2024",
        title="Carnet d'observation : revêtements réfléchissants",
        page=4,
        section="Limites",
        language="fr",
        text=(
            "Les observations ont été réalisées pendant une seule période "
            "estivale. Elles ne permettent pas d'attribuer à elles seules une "
            "causalité au matériau de voirie."
        ),
    ),
    EvidenceChunk(
        id=3,
        document_id="sensor-protocol-2025",
        title="Field protocol for temperature sensors",
        page=1,
        section="Calibration",
        language="en",
        text=(
            "Each temperature sensor was compared with a reference instrument "
            "before deployment and after retrieval. Measurements with a drift "
            "above the predefined threshold were excluded from the analysis."
        ),
    ),
    EvidenceChunk(
        id=4,
        document_id="canopy-study-2025",
        title="Étude exploratoire de la couverture arborée",
        page=3,
        section="Méthode",
        language="fr",
        text=(
            "La couverture arborée a été estimée à partir d'images aériennes. "
            "Les zones ont été regroupées par densité de canopée avant la "
            "comparaison des températures de surface."
        ),
    ),
    EvidenceChunk(
        id=5,
        document_id="rain-garden-review-2025",
        title="Rain gardens and runoff retention",
        page=5,
        section="Findings",
        language="en",
        text=(
            "The reviewed sites reported slower runoff after intense rainfall "
            "when rain gardens included a permeable soil layer and regular "
            "maintenance of the inlet."
        ),
    ),
    EvidenceChunk(
        id=6,
        document_id="evidence-methods-2025",
        title="Guide de lecture des résultats scientifiques",
        page=2,
        section="Interprétation",
        language="fr",
        text=(
            "Une réponse fondée sur des sources doit distinguer une observation, "
            "une corrélation et une conclusion causale. Elle doit conserver le "
            "lien vers le document et la page qui soutiennent chaque affirmation."
        ),
    ),
)
