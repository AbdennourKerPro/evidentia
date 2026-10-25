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

## 5. Itération v2b (25 octobre 2026)

**Règle d'évaluation.** À partir de cette date, l'évaluation porte uniquement sur
des cas 100 % anglais : English30, plus les 20 cas `language="en"` du heldout.
L'évaluateur accepte `--language en`. Le filtre est appliqué à la lecture, consigné
dans le rapport (`language_filter`) et vérifié à la reprise ; les fichiers du
benchmark ne changent pas.

**Les échecs n'étaient pas propres au français.** Une sonde ponctuelle (questions
traduites, hors benchmark) reproduisait les deux échecs en anglais : réponse
partielle sur Midjourney 2 fois sur 2, encodeurs BLIP-2 incomplets 2 fois sur 2.

**Correctifs v2b**, toujours limités à `candidate_recovery_v2` :

- La consigne f ne couvre plus que les détails secondaires d'une réponse par
  ailleurs appuyée. Quand la question porte principalement sur une entité ou un
  sujet absent des passages, la sortie reste `INSUFFICIENT_EVIDENCE`. La consigne
  de recopie exacte des identifiants est conservée.
- La correction `premature_abstention` rappelle que l'abstention reste valide si
  le sujet principal est absent.
- Un article détecté passe toujours par la fusion dense+BM25, même avec une seule
  variante de requête. La question traduite en anglais sert de variante
  supplémentaire si besoin. La recherche de secours d'une lacune liée à un article
  est elle aussi hybride.

Sonde après correctifs (3 passes) : BLIP-2 retrouve le chunk `3.4 Model
Pre-training` en S1 et cite « ViT-L/14 from CLIP and ViT-G/14 from EVA-CLIP »
(3/3) ; Midjourney s'abstient (3/3).

**Évaluation** : 3 passes v2b, `reasoning_effort=none`, top-k 5. Références :
`candidate_recovery`, `keep-primary-hits` du 09/10 pour English30, et une passe
du 25/10 pour heldout-en.

| Benchmark | Référence | v2b run 1 / 2 / 3 | Moyenne v2b | Abstention | Couverture des faits | Latence | Appels LLM |
|---|---:|---:|---:|---:|---:|---:|---:|
| English30 | 29/30 | 29 / 30 / 30 | 29,7/30 | 0,967 → 1,000 | 0,917 → 0,931 | 13,5 s → 19,3 s | 3,77 → 4,28 |
| heldout-en | 20/20 | 20 / 20 / 20 | 20/20 | 1,000 → 1,000 | 0,917 → 0,927 | 16,6 s → 21,0 s | 3,95 → 4,33 |

Rapports : `reports/agentic-rag-english30-recovery-v2b-20261025-run{1,2,3}.json`,
`reports/agentic-rag-heldout-en-recovery-v2b-20261025-run{1,2,3}.json`,
`reports/agentic-rag-heldout-en-candidate-recovery-20261025.json`.

- `english20-blip2-peak-learning-rate` : échec de la référence, réussi 3/3.
- **Régression** `english20-sam-vit-b-l-gpus` : réussi 2/3 (la référence passait
  sur sa passe unique). Dans le run 1, retrieval et couverture sont corrects : S1
  contient « To train ViT-B and ViT-L, … distributed across 128 GPUs », et le
  vérificateur cite cette phrase pour les deux exigences. La réponse affirme
  pourtant « The supplied passages do not state the number of GPUs used
  specifically to train the ViT-B and ViT-L models. [S1] ». La formule vient
  directement de la consigne f. S2, qui donne 256 GPUs pour le modèle ViT-H par
  défaut, a probablement poussé le modèle à hésiter. Cause probable : la
  consigne f donne au générateur une formule de repli qu'il emploie à tort sur un
  détail pourtant présent. La référence n'ayant qu'une passe, sa propre stabilité
  sur ce cas n'est pas mesurée.

## 6. Itération v2c (25 octobre 2026) : comparaison à variance égale

**Correctif.** La formule « les passages fournis ne couvrent pas » est retirée
de la consigne f. Restent la recopie exacte des identifiants et valeurs avec
leur provenance, la règle d'abstention quand le sujet principal est absent, et
le rappel dans la correction `premature_abstention`.

**Protocole.** 3 passes de `candidate_recovery_v2` (v2c) et 3 passes de la
référence `candidate_recovery`, sur English30 et heldout-en, avec
`reasoning_effort=none`. Les deux séries ont tourné en parallèle (une file
chacune), donc dans les mêmes conditions de charge et d'API. Les latences
absolues sont plus élevées qu'en exécution isolée.

| Benchmark | Profil | Réussis par passe | Moyenne | Abstention | Couverture des faits | Latence | Appels LLM |
|---|---|---|---:|---:|---:|---:|---:|
| English30 | référence | 29 / 29 / 29 | 0,967 | 0,967 | 0,898 | 22,3 s | 3,76 |
| English30 | v2c | 30 / 30 / 30 | **1,000** | 1,000 | 0,937 | 26,2 s | 4,20 |
| heldout-en | référence | 20 / 20 / 20 | 1,000 | 1,000 | 0,927 | 24,4 s | 3,95 |
| heldout-en | v2c | 20 / 20 / 20 | 1,000 | 1,000 | 0,892 | 28,3 s | 4,40 |

Rapports : `reports/agentic-rag-{english30|heldout-en}-{recovery-v2c|candidate-recovery}-20261025-run{1,2,3}.json`.

Cas par cas :

- **Aucune régression.** Aucun cas réussi 3/3 par la référence n'échoue avec v2c.
- `english20-sam-vit-b-l-gpus` : 3/3 pour la référence comme pour v2c. Les six
  réponses donnent « 128 GPUs [S1] » ; le refus erroné vu en v2b a disparu
  avec la formule.
- `english20-blip2-peak-learning-rate` : référence 0/3 (abstention : la lacune
  reste sans preuve), v2c 3/3 (« peak learning rate of 1e-4 »).

Couverture des faits en baisse sur heldout-en (0,927 → 0,892) ; aucun cas ne
passe sous le seuil de réussite :

- `heldout-dinov2-sam-data-scale-en` : 0,5 sur les 6 passes v2c (English30 et
  heldout), contre 1,0 sur les 3 passes English30 de la référence. Avec v2c, la
  récupération ajoute pour SAM un passage `7.6 Ablations` au lieu du passage
  d'introduction. La réponse cite alors « approximately 100M masks » au lieu de
  « more than 1 billion masks ». La réserve plus large (8 candidats par lacune)
  laisse le vérificateur choisir un passage qui couvre l'exigence formulée (une
  échelle de données SAM) sans être le chiffre de référence. Effet systématique
  à suivre.
- `heldout-sam-ambiguous-prompt-en`, `heldout-llava-gpt4-data-en` : écarts de
  formulation d'une passe à l'autre (0,67 contre 1,0), présents aussi dans la
  référence.

## 7. Décision et limites

- **Le profil par défaut de `compose.yaml` passe à `candidate_recovery_v2`** :
  aucune régression au sens défini (cas réussi 3/3 par la référence, raté au
  moins une fois par v2c) et pass rate moyen supérieur ou égal (1,000 contre
  0,967 sur English30, 1,000 contre 1,000 sur heldout-en).
- `DEFAULT_RETRIEVAL_PROFILE` dans `app/retrieval_policy.py` reste
  `candidate_recovery` hors Compose. Une variable `RAG_RETRIEVAL_PROFILE`
  définie dans `.env` reste prioritaire sur la valeur par défaut de Compose.
- Coût : environ +0,45 appel LLM par question et environ +4 s de latence moyenne
  à charge égale.
- Limites : benchmarks en brouillon, 3 passes seulement ; la baisse de
  couverture sur l'échelle de données SAM montre qu'un vérificateur qui accepte
  une preuve « suffisante » ne garantit pas le meilleur passage.
