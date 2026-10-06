# Validation ciblée — pipeline v5

Date : 15 août 2026

Pipeline : `langgraph_fact_plan_intent_multifacet_response_contract_v5`

## État

- 38/38 tests unitaires réussis dans Docker ;
- 2/2 échecs factuels persistants corrigés ;
- 5/5 comparaisons de garde réussies ;
- campagne finale de 30 cas préparée mais non exécutée à la demande de
  l'utilisateur.

## Résultats ciblés

| Cas | Réussite | Couverture factuelle | Rapport |
|---|---:|---:|---|
| `dinov2-frozen-features-fr` | oui | 0,50 | `agentic-rag-fact-plan-v5-dinov2-v5.json` |
| `blip2-bottleneck-fr` | oui | 0,50 | `agentic-rag-fact-plan-v5-blip2-publishable.json` |
| `compare-clip-dinov2-supervision-en` | oui | 1,00 | `agentic-rag-v5-clip-dinov2-attribution-guard.json` |
| `compare-blip2-llava-bridge-fr` | oui | 0,60 | `agentic-rag-v5-blip2-llava-guard-v2.json` |
| `compare-clip-sam-prompts-en` | oui | 0,60 | `agentic-rag-v5-comparisons-remaining-guard.json` |
| `compare-training-data-scale-en` | oui | 1,00 | `agentic-rag-v5-comparisons-remaining-guard.json` |
| `compare-frozen-components-en` | oui | 0,80 | `agentic-rag-v5-comparisons-remaining-guard.json` |

La réponse CLIP/DINOv2 a été relue : les objectifs ne sont plus transférés
d'un article à l'autre. Le cas BLIP-2/LLaVA cite des preuves provenant des deux
articles et distingue le Q-Former du projecteur linéaire de LLaVA.

## Relance différée des 30 cas

```powershell
docker compose run --rm api python -m scripts.evaluate_agentic_rag `
  --output /reports/agentic-rag-v5-final-30.json `
  --resume
```

Le rapport est sauvegardé après chaque cas. La même commande reprend les cas
manquants après une interruption.

## Interprétation prudente

Les résultats ci-dessus valident les régressions connues. Ils ne constituent
pas le résultat final sur l'ensemble du benchmark. Le taux final, la ventilation
par catégorie et les taux de contrats devront être lus dans le rapport de 30
cas après son exécution.
