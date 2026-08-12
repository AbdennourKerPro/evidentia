# Milestone 08 — Benchmark et évaluation du RAG

## Objectif

Établir une baseline mesurable avant d'ajouter un graphe agentique, du
fine-tuning ou un VLM. Sans cette baseline, une modification peut sembler
meilleure sur quelques exemples tout en dégradant le comportement général.

## Nature du benchmark

Le fichier `data/evaluation/rag_benchmark.jsonl` contient 30 cas
semi-synthétiques :

| Type | Rôle |
|---|---|
| Questions factuelles | Vérifier la récupération d'un résultat ou d'un nombre précis. |
| Questions de méthode | Vérifier l'explication d'une architecture ou d'un entraînement. |
| Comparaisons | Exiger des preuves provenant de deux ou trois articles. |
| Abstentions | Vérifier que le système refuse une question absente du corpus. |

Une question synthétique n'est pas automatiquement une vérité terrain. Le
script peut prouver que le fragment cité existe dans les chunks, mais seul un
humain peut confirmer que la question, la réponse de référence et la preuve ont
le même sens. C'est pourquoi chaque entrée commence avec
`review_status: needs_human_review`.

## Vérité terrain robuste au re-chunking

Un identifiant de point Qdrant dépend du découpage du PDF. Si la taille ou la
méthode de chunking change, un benchmark fondé uniquement sur ces identifiants
devient caduc. Chaque `evidence_target` associe donc :

- un `document_id` attendu ;
- une ou plusieurs formulations textuelles acceptées.

Le validateur recherche ces fragments dans les chunks persistés. Après un
re-chunking, le texte peut changer de point tout en restant une preuve valide.

## Évaluation du retrieval

`scripts/evaluate_retrieval.py` encode chaque question avec E5, interroge
Qdrant et calcule :

- **document recall@k** : proportion des articles attendus présents dans le
  top-k ;
- **evidence recall@k** : proportion des passages de référence couverts ;
- **hit@k** : présence d'au moins une preuve correcte ;
- **MRR** : inverse du rang de la première preuve correcte ;
- **nDCG@k** : récompense les preuves correctes placées tôt dans le classement ;
- latence moyenne de l'embedding et de la recherche.

Les cas d'abstention sont exclus du retrieval : une base vectorielle renvoie
toujours les voisins les plus proches, même lorsqu'ils ne répondent pas à la
question. L'abstention se juge au niveau end-to-end.

## Évaluation end-to-end

`scripts/evaluate_rag.py` appelle le même pipeline que `POST /ask`. Il mesure :

- exactitude de l'abstention ;
- couverture de groupes de faits attendus ;
- précision et rappel des documents cités ;
- rappel des documents présents dans le contexte ;
- taux de réussite selon un seuil déterministe ;
- latence complète de recherche et de génération.

La couverture des faits repose sur des expressions alternatives explicites.
Elle est simple, reproductible et auditable, mais elle ne mesure pas toutes les
paraphrases et ne garantit pas seule la fidélité sémantique. Les rapports
conservent donc la réponse complète, les citations et tous les chunks pour une
revue humaine. Un juge sémantique calibré pourra être ajouté plus tard et devra
être comparé à cette revue humaine.

## Fichiers et responsabilités

| Fichier | Responsabilité |
|---|---|
| `app/evaluation.py` | Schémas du benchmark, normalisation et fonctions de métriques communes. |
| `scripts/validate_benchmark.py` | Validation structurelle et existence des preuves dans les chunks. |
| `scripts/evaluate_retrieval.py` | Évaluation rapide de l'embedding et de Qdrant, sans charger le LLM. |
| `scripts/evaluate_rag.py` | Évaluation plus lente du pipeline complet avec Qwen/OpenVINO. |
| `reports/*.json` | Résultats détaillés persistés sur la machine hôte. |

## Exécution dans Docker

Validation du benchmark :

```powershell
docker compose run --rm api python -m scripts.validate_benchmark
```

Retrieval complet avec cinq chunks :

```powershell
docker compose run --rm api python -m scripts.evaluate_retrieval --limit 5
```

Test end-to-end d'un seul cas avant une campagne longue :

```powershell
docker compose run --rm api python -m scripts.evaluate_rag `
  --case-id clip-training-objective-en
```

Campagne end-to-end complète :

```powershell
docker compose run --rm api python -m scripts.evaluate_rag
```

Les dossiers montés par Compose ont deux politiques distinctes : le benchmark
est en lecture seule dans `/data/evaluation`, alors que `/reports` est
inscriptible. Un script d'évaluation ne peut donc pas modifier silencieusement
la vérité terrain, mais ses rapports survivent à la suppression du conteneur
temporaire.

## Interprétation

Le retrieval doit être corrigé avant le prompt ou le LLM lorsqu'une preuve
n'arrive jamais dans le top-k. Si les bonnes preuves sont récupérées mais que la
réponse omet un fait ou cite le mauvais article, le problème se situe plutôt
dans l'orchestration, le prompt, le modèle ou la validation des citations.

Cette séparation fournira la baseline commune aux prochaines variantes : RAG
simple, RAG avec reranking, graphe LangGraph, modèle SFT puis composants
multimodaux.

## Validation et première baseline

Le validateur confirme :

- 30 cas uniques : 12 méthodes, 8 faits, 5 comparaisons et 5 abstentions ;
- 22 questions anglaises et 8 françaises ;
- 31 cibles de preuve présentes dans les chunks persistés ;
- 30 cas encore marqués pour revue sémantique humaine.

La première campagne de retrieval sur les 25 cas répondables avec `top_k=5`
obtient :

| Mesure | Baseline |
|---|---:|
| Document recall@5 | 0,947 |
| Evidence recall@5 | 0,760 |
| MRR | 0,580 |
| Latence moyenne embedding + Qdrant | 0,419 s |

Les principales erreurs portent sur les comparaisons multi-articles : une seule
requête vectorielle favorise souvent plusieurs chunks d'un même article au lieu
de diversifier les sources. Cette observation fournit une future cible concrète
pour la décomposition de requête, le reranking ou un graphe agentique.

Un smoke test end-to-end sur `clip-training-objective-en` passe avec une
couverture des faits de 1,0, une précision documentaire des citations de 1,0 et
une latence voisine de 100 secondes, chargement initial de Qwen inclus. La
campagne RAG complète reste volontairement séparée car elle effectue 30
générations CPU séquentielles.
