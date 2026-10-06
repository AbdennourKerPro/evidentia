# Résumé — retrieval intentionnel multi-facettes v4

## Changement

Le pipeline combine maintenant expansion par facettes, recherches dense et
BM25 par variante, fusion RRF pondérée et reranking déterministe selon
l'intention explicite de la question.

## Validation

- Tests déterministes : 21/21.
- Deux échecs comparatifs ciblés : 2/2 réussites, couverture factuelle 1,00.
- Ensemble des comparaisons : 5/5 réussites, couverture factuelle moyenne 0,88.
- Audit de quatre anciens échecs hors comparaison : 2/4 réussites.

Les deux échecs persistants (`dinov2-frozen-features-fr` et
`blip2-bottleneck-fr`) disposent des bonnes preuves et citations. Ils relèvent
de la complétude de la réponse générée, pas du retrieval.

## Rapports

- `intent-multifacet-retrieval-two-failures.json`
- `agentic-rag-intent-multifacet-two-failures.json`
- `agentic-rag-intent-multifacet-regression-seven.json`

La prochaine amélioration recommandée est un plan de faits extrait des preuves,
suivi d'une validation de complétude factuelle avant finalisation.

