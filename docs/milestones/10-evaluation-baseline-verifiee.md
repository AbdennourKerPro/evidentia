# Milestone 10 — Baseline d'évaluation vérifiée

## Objectif

Mesurer le RAG simple sur une vérité terrain relue avant d'introduire un
reranker, LangGraph, du SFT ou un composant multimodal. Cette baseline devient
le point de comparaison des prochaines architectures.

## Préconditions contrôlées

- les 30 cas portent le statut `verified` ;
- les 31 cibles de preuve existent toujours dans les chunks Docling ;
- le benchmark complet passe la validation Pydantic ;
- le `top_k` est fixé à 5 pour le retrieval et la génération ;
- les cinq articles sont interrogés dans la collection
  `arxiv_chunks_e5_small`.

## Deux évaluations complémentaires

### Retrieval

La commande suivante n'appelle pas Qwen :

```powershell
docker compose run --rm api python -m scripts.evaluate_retrieval `
  --limit 5 `
  --output /reports/retrieval-evaluation-verified.json
```

Elle obtient 94,7 % de rappel documentaire, 76,0 % de rappel des preuves et un
MRR de 0,580 sur les 25 cas répondables.

### End-to-end

La campagne complète utilise le même pipeline que `POST /ask` :

```powershell
docker compose run --rm api python -u -m scripts.evaluate_rag `
  --limit 5 `
  --output /reports/rag-evaluation-verified.json
```

Le modèle et l'embedding sont chargés une fois dans un conteneur temporaire,
puis les 30 cas sont exécutés séquentiellement. La campagne dure 24 minutes et
6 secondes sur le CPU de la machine.

## Résultats

| Mesure | Baseline |
|---|---:|
| Réussite end-to-end | 70,0 % |
| Réussite sur les 25 cas répondables | 64,0 % |
| Couverture factuelle | 59,0 % |
| Décision réponse/abstention | 80,0 % |
| Abstention correcte hors corpus | 100 % |
| Citations documentaires, précision/rappel | 76,0 % / 76,0 % |
| Latence moyenne end-to-end | 47,84 s |

## Lecture architecturale

Le RAG simple est fort sur les questions de méthode mono-article (11/12) et
conservateur hors corpus (5/5 abstentions correctes). Sa faiblesse structurante
est la comparaison : 0/5.

Qdrant classe les chunks indépendamment. Rien ne contraint le top-5 à contenir
une preuve de chaque article nécessaire. Un article sémantiquement dominant
occupe donc souvent quatre ou cinq places, même si son document concurrent est
explicitement demandé. Cette erreur se situe avant le LLM : un prompt ne peut
pas reconstruire une preuve absente de son contexte.

Le français révèle une seconde faiblesse. Deux des six cas répondables passent,
contre quatorze sur dix-neuf en anglais. Les causes combinent retrieval
cross-lingue insuffisant, réponse dans une langue incorrecte et couverture
incomplète.

## Comportement du garde-fou

Le validateur de citations transforme une sortie sans références valides en
abstention. Il évite ainsi une réponse non traçable sur `blip2-capabilities-en`
et `compare-frozen-components-en`. Cette sécurité améliore la fiabilité, mais
elle masque actuellement la sortie brute du modèle dans le rapport final. Une
future instrumentation devra conserver cette sortie uniquement dans le rapport
d'évaluation pour diagnostiquer précisément l'erreur de format.

## Prochaine hypothèse à tester

La prochaine variante doit améliorer le retrieval avant de modifier Qwen :

1. identifier les documents ou concepts demandés ;
2. décomposer une comparaison en une sous-question par source ;
3. récupérer un quota de chunks par source ;
4. fusionner les candidats ;
5. appliquer un reranker ;
6. générer avec des preuves équilibrées.

Cette orchestration constitue un premier cas naturel pour LangGraph. La même
campagne permettra de vérifier si le graphe améliore les comparaisons sans
dégrader les questions simples, les abstentions ni la latence.

Le rapport humain détaillé est enregistré dans
`reports/rag-evaluation-verified-summary.md` et les données complètes restent
dans les deux rapports JSON versionnés.
