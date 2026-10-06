# Milestone 15 — Retrieval multi-facettes guidé par l'intention

## Objectif

Corriger les deux dernières comparaisons en échec sans mémoriser les réponses du
benchmark dans le code :

- rôle des prompts dans CLIP et SAM ;
- échelle des données de CLIP, DINOv2 et SAM.

La difficulté ne venait plus de la couverture des articles. Le retrieval
équilibré ramenait bien un nombre de chunks par source, mais pas toujours le
passage qui répondait à la facette exacte de la question. Un passage sur une
ablation chiffrée pouvait, par exemple, précéder la description canonique du
dataset.

## Architecture v4

```text
question
  -> détection déterministe des intentions
  -> plan de recherche par article
  -> requête principale + requêtes de facettes
  -> dense et BM25 pour chaque variante
  -> fusion RRF pondérée
  -> reranking explicable selon l'intention
  -> sélection équilibrée entre les articles
  -> génération Qwen 2.5 7B
  -> contrat langue/citations/couverture des sources
```

Le pipeline déclaré par l'évaluateur est
`langgraph_intent_multifacet_rrf_response_contract_v4`.

## 1. Détection de l'intention

`app/query_expansion.py` détecte actuellement quatre familles à partir de la
question :

- `numeric_scale` : taille, nombre, dataset, corpus, million, milliard ;
- `prompt_mechanism` : prompt, template, prompting ;
- `architecture_bridge` : Q-Former, pont, projection, composant gelé ;
- `capabilities` : tâches, capacités, benchmark.

La détection est déterministe, bilingue anglais/français et inspectable. Elle ne
fait pas appel au LLM. Une question générique ne déclenche aucune expansion.

## 2. Expansion multi-facettes

`build_query_variants()` conserve la requête planifiée comme première variante,
puis ajoute une formulation orientée vers chaque intention détectée. Pour
`numeric_scale`, elle recherche par exemple les types d'unités et d'objets
attendus — images, paires, masques, millions, milliards — sans contenir les
valeurs propres à CLIP, DINOv2 ou SAM.

Ce dernier point évite une fuite du benchmark : ni `400 million`, ni
`LVD-142M`, ni `SA-1B`, ni `11M` ne sont codés dans les requêtes.

## 3. Dense + BM25 + RRF sur chaque facette

`app/hybrid_retrieval.py` expose `retrieve_multifacet_candidates()` :

1. une recherche dense compare les embeddings pour chaque variante ;
2. BM25 recherche les correspondances lexicales pour chaque variante ;
3. RRF fusionne toutes les listes à partir des rangs, sans rendre comparables
   artificiellement les scores dense et BM25 ;
4. les variantes spécialisées reçoivent un poids supérieur à la requête
   générale.

La sortie conserve les listes intermédiaires pour l'observabilité et les tests.

## 4. Reranking guidé par l'intention

`rerank_for_intents()` réordonne la petite liste fusionnée avec des règles
lisibles. Pour une question de taille de dataset, une introduction ou une
section `Data Processing` contenant un nom de corpus, des unités et une phrase
de composition est favorisée ; une section d'ablation ou de coût est pénalisée.

Pour une question sur les prompts, les passages contenant les types d'entrées
précis (`points`, `boxes`, `masks`, `sparse`, `dense`, `templates`) et les
sections d'architecture ou de méthode sont favorisés.

Ce reranking ne remplace pas la pertinence sémantique : il ne s'applique qu'aux
candidats déjà trouvés par dense/BM25 et seulement lorsqu'une intention explicite
est détectée.

## 5. Intégration LangGraph

`app/agentic_rag.py` transporte maintenant dans l'état et les tâches parallèles :

- `query_variants` ;
- leurs `query_vectors` ;
- `retrieval_intents`.

Le chemin comparatif applique cette logique indépendamment dans chaque branche
d'article avant la sélection round-robin. Le chemin global l'applique aussi
lorsqu'un seul article est explicitement nommé ou sélectionné. L'API expose les
variantes et intentions dans `execution.planned_queries`, et la trace indique le
nombre de facettes et les intentions utilisées.

## Tests déterministes

`tests/test_query_expansion.py` couvre :

- la détection anglaise et française ;
- les quatre intentions ;
- l'absence d'expansion sur une question générique ;
- l'absence des réponses du benchmark dans les facettes ;
- la préférence pour une description de dataset plutôt qu'une ablation ;
- la préférence pour les types de prompts exacts plutôt qu'une expérience
  seulement apparentée.

Avec les tests du contrat de réponse : **21/21 tests réussissent** dans l'image
Docker.

## Résultats retrieval sur les deux échecs ciblés

Avec `top_k=5` :

| Stratégie | Rappel moyen des preuves cibles |
|---|---:|
| agentic v1 | 0,000 |
| dense équilibré | 0,000 |
| hybride RRF | 0,417 |
| multi-facettes RRF | 0,417 |
| intention + multi-facettes RRF | **0,833** |

Par cas :

- prompts CLIP/SAM : rappel **1,00** ;
- échelle des datasets : rappel littéral **0,67**.

Le second cas contient néanmoins les trois faits dans les cinq chunks retenus.
Le passage DINOv2 sélectionné écrit `LVD-142M (142M images)` alors que la cible
littérale du benchmark recherche une autre phrase du même article. Le test
bout-en-bout permet de vérifier si cette preuve équivalente suffit réellement.

## Résultats bout en bout

Les deux comparaisons auparavant en échec réussissent maintenant :

- **2/2 réussites** ;
- couverture factuelle moyenne : **1,00** ;
- précision et rappel des articles cités : **1,00** ;
- contrat de réponse valide : **1,00** ;
- aucune correction supplémentaire déclenchée.

En combinant cette campagne avec la régression des trois autres comparaisons :

- **5/5 comparaisons réussies** ;
- couverture factuelle moyenne : **0,88** ;
- précision et rappel des articles cités : **1,00** ;
- latence moyenne : environ **128,3 s** par question sur CPU/OpenVINO.

Rapports sources :

- `reports/intent-multifacet-retrieval-two-failures.json` ;
- `reports/agentic-rag-intent-multifacet-two-failures.json` ;
- `reports/agentic-rag-intent-multifacet-regression-seven.json`.

## Audit des autres catégories

Les comparaisons étaient le problème principal, mais pas l'unique problème. Les
quatre anciens échecs hors comparaison ont été rejoués :

| Cas | Catégorie | Résultat v4 | Couverture factuelle |
|---|---|---:|---:|
| `clip-data-scale-fr` | factoid | réussite | 1,00 |
| `dinov2-frozen-features-fr` | factoid | échec | 0,25 |
| `blip2-bottleneck-fr` | method | échec | 0,25 |
| `blip2-capabilities-en` | factoid | réussite | 1,00 |

Pour les deux échecs persistants, le bon article est retrouvé et cité, les cinq
chunks contiennent les éléments utiles, la langue est correcte et le contrat est
valide. L'échec se situe donc après le retrieval : Qwen paraphrase la preuve sans
énoncer explicitement assez de faits atomiques attendus.

Le contrat v3 vérifie la langue, la syntaxe des citations et la couverture des
articles, mais pas encore la complétude sémantique des affirmations. La prochaine
itération recommandée est un **plan de réponse fondé sur les preuves** : extraire
avant rédaction une courte checklist de faits par sous-question, générer depuis
cette checklist, puis vérifier que chaque fait retenu apparaît avec une citation.
Cette étape doit rester indépendante des réponses de référence du benchmark.

## Limites et prochaine validation

La campagne ciblée prouve la correction des régressions connues, pas la qualité
universelle du système. Avant de figer une version, il faudra rejouer les 30 cas
du benchmark. Les deux cas de complétude factuelle doivent être traités avant
cette campagne complète afin d'éviter de payer une nouvelle fois le coût de
génération sans nouvelle hypothèse technique.

