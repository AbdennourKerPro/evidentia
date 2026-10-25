# v9 — Compte rendu final de couverture et récupération

30 septembre 2026. GPT-6-Luna, reasoning low. Benchmark et scoring inchangés.
Rapport : `agentic-rag-v9-candidate-coverage-final-30-20260930.json`.

## Résultats

| Mesure | v8 | v9 finale |
|---|---:|---:|
| Cas réussis | 28/30 | 29/30 |
| Couverture moyenne des groupes de termes | 82,3 % | 89,7 % |
| Rappel des passages cibles sélectionnés | 81,3 % | 89,3 % |
| Comparaisons réussies | 4/5 | 5/5 |
| Abstentions attendues réussies | 5/5 | 5/5 |
| Contrats de réponse valides | 29/30 | 29/30 |
| Appels LLM | 45 | 118 |
| Tokens cumulés | 80 976 | 207 623 |
| Latence moyenne observée | 6,10 s | 13,52 s |

Les temps sont des observations de campagnes distinctes, pas un benchmark de
performance à conditions strictement identiques. Les chiffres de tokens ne sont
pas des montants facturés ; ils incluent les tokens rapportés par l'API, y compris
ceux dont la tarification peut différer en présence de cache.

## Gains et perte, par rapport à v8

- `clip-data-scale-fr` : échec -> réussite, couverture 0,5, preuve cible présente.
- `compare-clip-sam-prompts-en` : échec -> réussite, couverture 1,0, deux preuves
  cibles présentes ; le passage des prompts SAM est ajouté depuis la réserve.
- `dinov2-curation-en` : réussite -> échec, couverture 0,25.
- Les 27 autres décisions réussite/échec sont inchangées. En particulier,
  `compare-blip2-llava-bridge-fr` réussit avec une couverture de 0,8.

Le bilan est donc deux gains et une perte au score, pas une non-régression stricte.
Les neuf répétitions ciblées de la variante finale passent, ainsi que quatre
témoins, mais ce sous-ensemble ne remplace pas la campagne complète.

## Lecture du cas DINOv2

La réponse décrit la sélection d'images visuellement similaires à des datasets
curatés dans une source web brute, la déduplication, les filtres et le retrieval
par exemples/clusters, sans utiliser le texte ou les métadonnées.

Le score lexical ne reconnaît notamment pas « visually similar » comme
« visual similarity », « raw web-image source » comme « uncurated », ni
« using images rather than text or metadata » comme « without metadata ».
Il s'agit d'un **faux négatif probable après lecture humaine**, pas d'une
réussite automatiquement reclassée. Le JSON reste à 29/30 et les termes du
benchmark n'ont pas été modifiés pour faire passer cette réponse. Une revue
humaine explicite ou une évaluation sémantique indépendante reste nécessaire.

## Ce que la version fait réellement

Décomposition de la question sans voir les chunks, vérification d'extraits exacts,
reranking des candidats selon les aspects manquants, budget partagé entre articles,
et ajout conditionnel de deux passages maximum. Les cinq passages initiaux restent
dans le même ordre et gardent leurs références. Aucune nouvelle requête Qdrant,
aucun téléchargement ni aucune règle spécifique aux identifiants de cas.

La récupération est tentée dans 15 cas et ajoute au moins un passage dans six cas.
Les contrats existants de langue, citations et faits restent actifs. Le seul
contrat de réponse invalide est l'abstention `abstain-clip-electricity-en`
(`premature_abstention`, non publiable), également observée en v8 : son résultat
d'abstention est correct pour le benchmark, ce qui ne signifie pas que tous les
contrats sont validés. Le score de passage ne mesure pas la fidélité sémantique
de chaque affirmation citée ni l'exhaustivité d'une réponse au seuil de 50 %.

## Décision et prochaine validation

Le profil `candidate_recovery` est disponible et évalué, mais `question_rerank`
reste le profil par défaut. La perte DINOv2 au score et le surcoût des appels sont
explicitement conservés. Le profil ajoutant un plan au prompt de génération a
été rejeté après son ablation défavorable. Les rapports des neuf variantes restent
disponibles dans `reports/` ; les principes et scripts sont expliqués dans
`docs/milestones/21-candidate-coverage-recovery.md`.

La prochaine validation utile est une revue de DINOv2 et un petit jeu de questions
inédites, non utilisé pour ajuster ces prompts. Le développement répété sur les
mêmes 30 cas ne constitue pas une démonstration de généralisation.

## Mise à jour de déploiement après revue du cas DINOv2

À la demande de l'utilisateur, le profil `candidate_recovery` est désormais
activé par défaut pour l'API et le mode LangGraph de l'UI. Le score automatique
original demeure **29/30**. Après revue qualitative de la réponse DINOv2 et de ses
preuves S1/S3 dans la conversation, ce cas est considéré correct sur le fond,
soit **30/30 après adjudication de ce seul cas**. La revue a été faite par
l'assistant, pas par un panel humain indépendant ; le déploiement est autorisé
par l'utilisateur. Ni les critères ni le rapport JSON n'ont été réécrits.

Les modifications depuis l'évaluation concernent uniquement le profil par défaut,
sa configuration Compose et les tests qui doivent isoler explicitement les
anciens enchaînements d'appels. Les prompts et la récupération sont inchangés.
