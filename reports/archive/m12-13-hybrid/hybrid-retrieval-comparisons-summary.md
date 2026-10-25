# Évaluation du retrieval hybride sur les cinq comparaisons

## Question expérimentale

Cette expérience vérifie si une recherche hybride sélectionne de meilleurs
passages que le retrieval LangGraph v1, avant de modifier le LLM. Les plans de
recherche déjà sauvegardés sont rejoués à l'identique : seules la création des
candidats et leur classement changent.

Le benchmark est `data/evaluation/rag_benchmark.jsonl`. Les cinq questions de
comparaison ont été relues manuellement et possèdent onze chunks cibles au
total. La fenêtre finale reste limitée à cinq chunks.

## Variantes comparées

1. `agentic_v1` : résultat sauvegardé du dense Qdrant suivi du petit reranking
   lexical historique ;
2. `dense_balanced` : top-20 dense par source, puis sélection équilibrée ;
3. `hybrid_rrf` : top-20 dense et top-20 BM25 par source, fusion RRF en top-30,
   puis sélection équilibrée ;
4. `hybrid_cross_encoder` : mêmes candidats hybrides, puis reclassement avec
   `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` sur CPU.

## Résultats retrieval-only

| Stratégie | Rappel exact macro @5 | Chunks cibles | Au moins une cible | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| LangGraph v1 | 0,400 | 4/11 | 0,600 | 0,350 | 0,306 |
| Dense équilibré | 0,300 | 3/11 | 0,400 | 0,150 | 0,178 |
| **Dense + BM25 + RRF** | **0,567** | **6/11** | 0,800 | 0,550 | **0,477** |
| Hybride + cross-encoder | 0,467 | 5/11 | **1,000** | **0,633** | 0,416 |

Le rappel macro est la moyenne des rappels de chaque question. Le compteur
`chunks cibles` est un rappel micro sur les onze passages. Ces deux mesures ne
pondèrent donc pas les questions de la même façon.

RRF apporte le meilleur compromis pour la génération : six chunks cibles
contre quatre auparavant. Le cross-encoder place plus souvent une première
preuve très haut, mais il élimine une preuve utile supplémentaire de la petite
fenêtre finale. Il est donc conservé comme variante expérimentale, sans être
activé dans le graphe v2.

## Temps de calcul du retrieval

- dense + BM25 + RRF : 0,133 s en moyenne par question, embeddings exclus ;
- cross-encoder : 16,269 s supplémentaires en moyenne sur CPU ;
- modèle de reranking : environ 471 Mo, téléchargé une fois dans le volume
  Docker `model_cache`.

Le cross-encoder est appliqué uniquement à la shortlist fusionnée, jamais aux
938 chunks complets. Malgré cela, son coût n'est pas justifié par son rappel
inférieur sur ce petit benchmark.

## Vérification bout en bout

La variante RRF a ensuite remplacé le reranking v1 dans LangGraph. Qwen a de
nouveau planifié les recherches et généré les réponses ; cette seconde
campagne ne rejoue donc pas les plans sauvegardés.

| Mesure | LangGraph v1 | LangGraph hybride v2 |
|---|---:|---:|
| Réussite end-to-end | 2/5 (0,400) | 2/5 (0,400) |
| Couverture factuelle | 0,320 | **0,480** |
| Précision documentaire des citations | 0,600 | **0,800** |
| Rappel documentaire des citations | 0,600 | **0,700** |
| Documents attendus dans les chunks | 1,000 | 1,000 |
| Bonne décision réponse/abstention | 0,600 | **0,800** |
| Latence moyenne | 84,05 s | 104,62 s |

La couverture factuelle gagne 0,16 point, soit 50 % relativement à v1. Le taux
de passage ne progresse pas : deux anciens échecs passent, mais deux anciennes
réussites régressent. L'une conserve 0,8 de couverture factuelle mais ne cite
plus les deux documents attendus. La question sur l'échelle des données reste
une abstention incorrecte.

Décomposition moyenne de la v2 :

- planification : 22,74 s ;
- une branche hybride : 116,7 ms ;
- génération : 81,71 s.

Le surcoût observé par rapport à v1 ne vient pas principalement de BM25/RRF ;
les durées variables des générations Qwen dominent la mesure.

## Résultats par question

| Cas | Passage v1 → v2 | Couverture v1 → v2 | Observation |
|---|---:|---:|---|
| CLIP–DINOv2, supervision | oui → non | 0,60 → 0,40 | Réponse correcte dans l'ensemble mais détails cibles incomplets. |
| BLIP-2–LLaVA, pont | non → oui | 0,00 → 0,60 | Les preuves sur Q-Former et projection deviennent accessibles. |
| CLIP–SAM, prompts | non → oui | 0,20 → 0,60 | Les rôles distincts des prompts sont mieux couverts. |
| Échelle des données | non → non | 0,00 → 0,00 | Le modèle s'abstient malgré une amélioration retrieval-only partielle. |
| Composants gelés | oui → non | 0,80 → 0,80 | Couverture stable, mais citations limitées à un seul article. |

Le cas français révèle en outre un problème indépendant : Qwen produit du
chinois affiché avec un encodage incorrect au lieu du français. La métrique
factuelle reconnaît les termes techniques, mais la réponse n'est pas
acceptable pour un utilisateur. Ce défaut devra être traité au niveau du
contrôle de langue et de la génération.

## Fichiers de preuve

- retrieval-only : `reports/hybrid-retrieval-comparisons.json` ;
- retrieval-only sans cross-encoder :
  `reports/hybrid-retrieval-comparisons-no-cross.json` ;
- end-to-end v1 : `reports/agentic-rag-comparisons-verified.json` ;
- end-to-end hybride v2 :
  `reports/agentic-rag-hybrid-comparisons-verified.json`.

## Décision

Activer BM25 + RRF pour les comparaisons multi-articles. Garder le retrieval
dense simple pour les questions globales afin de ne pas complexifier toute la
baseline. Conserver le cross-encoder et son script d'évaluation, mais ne pas le
charger dans le serveur courant.

La prochaine amélioration doit cibler la stabilité de la réponse : respect de
la langue demandée, obligation de couvrir et citer chaque source comparée, et
traitement spécifique des questions numériques. Ensuite, l'évaluation devra
être étendue aux 25 cas répondables avant toute conclusion générale.
