# Milestone 24 — Récupération de couverture v2

Date : 24 octobre 2026. Branche : `fix/coverage-recovery-v2`.

## 1. Pourquoi cette étape

Après le milestone 21 et le profil `keep_primary_hits`, il restait un échec sur
le benchmark anglais (30 cas) et trois sur le benchmark heldout (40 cas). Les
traces montraient quatre défauts génériques de la récupération, pas des faits
manquants dans le corpus :

1. Le plan d'exigences exigeait une exigence pour **chaque** article présent dans
   les résultats. Un chunk `References` de LLaVA, remonté pour une question
   DINOv2, rendait le plan invalide (`invalid_question_requirements`).
2. Les chunks bibliographiques occupent des places dans la réserve sans jamais
   pouvoir prouver une réponse.
3. La recherche de secours n'était appelée que si une lacune n'avait **aucun**
   candidat ; une réserve non vide mais inutile bloquait toute autre recherche.
4. Le vérificateur de récupération pouvait citer un passage initial que le premier
   contrôle venait de juger insuffisant : la lacune « disparaissait » sans preuve
   nouvelle.

## 2. Correctifs

Tous les correctifs sont derrière un nouveau profil, `candidate_recovery_v2`.
Les profils existants restent des ablations reproductibles à l'identique.

| Correctif | Mécanisme |
|---|---|
| Articles du plan | Si des articles sont détectés dans la question, le plan ne porte que sur ceux présents dans les résultats (ou sur les détectés s'il n'y en a aucun). Sinon, le comportement reste le même. |
| Bibliographie | Les chunks dont la section est `References`/`Bibliography` (éventuellement numérotée) sont filtrés après la recherche dense/BM25, avant la fusion RRF et le reranking. Aucune réindexation. |
| Recherche de secours | Après la vérification de la réserve, chaque lacune toujours sans preuve déclenche **une** recherche avec sa requête anglaise (hors chunks déjà proposés), puis une seconde vérification limitée à ces lacunes. Le budget reste de 2 ajouts et 16 000 caractères au total. Aucun passage initial n'est retiré. L'audit expose `fallback_used`. |
| Cohérence du vérificateur | Pendant la récupération, une lacune ne peut être fermée que par un passage **ajouté**. Une référence vers un passage initial vaut absence de preuve. |
| Réserve par lacune | 8 candidats proposés par lacune (au lieu de 3). |
| Prompt de réponse | Recopier noms de modèles, jeux de données, identifiants et valeurs exactement, avec leur provenance (« X de Y ») ; ne jamais affirmer qu'un article ne dit pas quelque chose, mais dire que les passages fournis ne le couvrent pas. |

Aucune règle propre à un article ni aucun fait du benchmark n'est encodé.

Compose fixe aussi `OPENAI_REASONING_EFFORT: none`, indépendamment de `.env`.

## 3. Résultats

Une passe par benchmark, `gpt-6-luna`, `reasoning_effort=none`, top-k 5.

| Benchmark | Référence | v2 | Abstention | Couverture des faits | Latence moyenne |
|---|---:|---:|---:|---:|---:|
| english30 | 29/30 (`keep-primary-hits`, 09/10) | **30/30** | 0,967 → 1,000 | 0,917 → 0,931 | 13,5 s → 18,8 s |
| heldout | 37/40 (`main-358445c`, 09/10) | **38/40** | 1,000 → 0,975 | 0,903 → 0,887 | 10,7 s → 20,3 s |

Rapports : `reports/agentic-rag-english30-recovery-v2-20261024.json` et
`reports/agentic-rag-heldout-draft-recovery-v2-20261024.json`.

Effets mesurés des mécanismes :

- plans invalides au premier contrôle : 3 → 0 (english30), 2 → 0 (heldout) ;
- recherche de secours utilisée : 12 cas (english30), 17 cas (heldout) ;
- passages ajoutés : 3 → 8 (english30), 11 → 12 (heldout) ;
- appels LLM moyens : 3,8 → 4,3 et 4,1 → 4,6 ; tokens d'entrée : +58 % et +49 %.

Cas cibles :

| Cas | Avant | v2 | Explication |
|---|---|---|---|
| `english20-blip2-peak-learning-rate` | abstention | réussi | Lacune détectée, passage du learning rate ajouté depuis la réserve élargie. |
| `heldout-dinov2-curated-data-fr` | échoué (1/3 faits) | réussi (2/3) | Passage sur la déduplication et la recherche d'images proches ajouté. |
| `heldout-dinov2-text-guided-argument-fr` | échoué | réussi | Avant, un chunk `References` occupait S4 et le plan était invalide (aucune récupération), d'où « DINOv2 n'avance pas d'argument ». Le filtre bibliographique libère la place : le passage d'introduction sur les légendes entre dans le top 5 et la couverture est suffisante d'emblée. |
| `heldout-blip2-encoders-fr` | échoué | échoué | Voir limites. |

## 4. Échecs et régression

**`heldout-abstain-midjourney-fr` (régression).** « Midjourney utilise-t-il le
même objectif d'entraînement que CLIP ? » doit produire une abstention. Retrieval,
couverture et premier brouillon sont identiques à l'ancien run : le brouillon
s'abstient. Le validateur signale `premature_abstention` (un fait CLIP est planifié),
d'où une correction. Avant, la correction s'abstenait encore. Avec la nouvelle
consigne « dire que les passages ne couvrent pas », elle publie une réponse partielle :
l'objectif de CLIP, puis « le passage ne décrit pas Midjourney ». La consigne
crée une troisième voie entre réponse et abstention. Cause probable : correctif f
combiné à la pression de correction. Une seule passe : la variance du LLM n'est pas exclue.

**`heldout-blip2-encoders-fr` (échec persistant).** La preuve est dans le chunk
`3.4. Model Pre-training` (« ViT-L/14 from CLIP … ViT-G/14 from EVA-CLIP »).
Elle n'apparaît ni dans les 5 passages initiaux, ni parmi les 14 candidats
proposés (réserve + recherche de secours). La question a une seule variante de
requête, donc elle passe par la route globale non focalisée : recherche dense
seulement, sans BM25, alors que l'article est détecté. La recherche de secours est
elle aussi dense. Le texte du chunk est dominé par le pré-entraînement, ce qui le
fait manquer. La réponse cite « ViT-L et ViT-G » sans `/14` ni EVA-CLIP.

## 5. Décision et limites

- Le profil par défaut reste `candidate_recovery` : la régression sur une
  abstention bloque le passage de `candidate_recovery_v2` en défaut.
- Pistes génériques : restreindre la consigne f aux réponses par ailleurs
  appuyées (garder `INSUFFICIENT_EVIDENCE` si la question principale n'est pas
  couverte) ; utiliser la recherche hybride dense+BM25 pour un article détecté,
  même avec une seule variante, et pour la recherche de secours.
- Coût : la seconde vérification et la fenêtre plus large doublent environ la
  durée du nœud de récupération (médiane ≈ 11 s contre 3–5 s). Les cas sans
  récupération sont aussi plus lents qu'au 09/10, ce qui indique une variance de
  latence de l'API.
- Une passe par benchmark seulement, sur des benchmarks en brouillon.
