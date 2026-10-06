# Évaluation complète v7 — question_rerank — 30 cas

## Conclusion

La campagne est complète : **27/30 cas réussis, soit 90 %**, contre
28/30 (93,3 %) pour le rapport GPT-6 Luna précédent. Le correctif SAM fonctionne,
mais le remplacement global du classement par intentions dans les branches
hybrides introduit deux régressions de sélection. Cette version n'est donc
pas une amélioration globale démontrée du produit.

L'application n'a pas été modifiée ni remise à une version précédente pendant
cette évaluation. Aucune correction de langue ou de retrieval n'a été effectuée
après observation des erreurs. Les profils optionnels `original_query` et
`coverage_retry` n'ont pas été évalués sur les 30 cas lors de cette campagne.

## Protocole

- Exécution réelle dans l'image Docker actuelle, avec les scripts inclus dans
  l'image, sans montage d'une autre version du code.
- Pipeline : `langgraph_retrieval_ablation_v7_openai_v3`.
- Profil explicitement fixé à `question_rerank` pour le processus d'évaluation.
- Modèle : `gpt-6-luna`, Responses API, reasoning `low`.
- Timeout : 60 secondes ; une relance SDK autorisée ; sortie maximale 4 096 tokens.
- Top-k : 5 ; collection `arxiv_chunks_e5_small`.
- 30 cas vérifiés humainement : 25 answerable et 5 abstentions attendues.
- Une génération de pipeline par cas, avec ses éventuelles corrections internes.
- Configuration LLM, top-k et empreinte du benchmark identiques au rapport précédent.
- Checkpoint après chaque cas ; rapport final `complete=true`, 30/30 cas terminés.
- Nouveau nom de rapport ; le rapport précédent n'a pas été écrasé.

Empreinte SHA-256 du benchmark :
`31917579b4753cf27d3eca756dbf7b8ef23917f6bfcb1bdb6a7b2414a4aab74b`.

Commande exécutée depuis le dossier de l'application, sur une seule ligne :

```powershell
docker compose run --rm --no-deps -e RAG_RETRIEVAL_PROFILE=question_rerank api python -m scripts.evaluate_agentic_rag --limit 5 --output /reports/agentic-rag-v7-question-rerank-30-20260930.json
```

## Résultats agrégés

| Indicateur | Rapport précédent | Nouvelle campagne |
| --- | ---: | ---: |
| Cas réussis | 28/30 | 27/30 |
| Réussite globale | 93,3 % | 90,0 % |
| Couverture moyenne des groupes de faits, 25 cas answerable | 86,2 % | 82,4 % |
| Rappel moyen des passages cibles sélectionnés | 84,7 % | 79,3 % |
| Rappel des articles retrouvés | 100 % | 100 % |
| Précision des articles cités | 96 % | 92 % |
| Rappel des articles cités | 96 % | 92 % |
| Comparaisons réussies | 5/5 | 4/5 |
| Abstentions attendues correctement publiées | 5/5 | 5/5 |
| Contrat de réponse valide | 93,3 % | 93,3 % |
| Taux de tentative de correction | 33,3 % | 30,0 % |
| Latence moyenne par cas | 3,75 s | 7,49 s |
| Latence médiane par cas | 2,29 s | 6,02 s |
| Appels LLM logiques | 45 | 44 |
| Tokens totaux observés | 80 456 | 80 461 |

La précision et le rappel des citations mesurent **l'identité des articles**,
pas la véracité de chaque affirmation ni l'adéquation de chaque citation.
Le rappel des articles à 100 % ne signifie donc pas que les bons passages
ont été retenus.

L'accuracy d'abstention agrégée vaut 93,3 % : elle porte sur les 30 cas, pas
seulement sur les cinq questions sans réponse. CLIP et ScienceQA sont deux
abstentions sur des questions answerable. Les cinq abstentions attendues sont
toutes correctes. Deux contrats sont marqués invalides, dont l'un concerne
une question d'abstention finalement publiée comme une abstention sûre.

Répartition des réussites :

- méthode : 12/12 ;
- factoid : 6/8 ;
- comparaison : 4/5 ;
- abstention : 5/5.

## Les trois échecs

### 1. `clip-data-scale-fr` — nouvelle régression de sélection

Question : « Sur combien de paires image-texte CLIP est-il pré-entraîné et quel
résultat zero-shot marquant obtient-il sur ImageNet ? »

Le pipeline s'abstient car les preuves retenues sont insuffisantes. Le passage
cible contenant la taille de 400 millions de paires était au rang 1 du pool
fusionné, mais passe au rang 6 après cross-encoder. Il sort donc du top-5.
Les passages sélectionnés portent surtout sur les performances et expériences
de transfert, plutôt que sur la taille du corpus demandée.

- couverture des faits : 0 % ;
- rappel de preuve cible : 0 % ;
- aucune correction interne : le marqueur d'abstention est considéré valide ;
- précédent rapport : réussite avec preuve cible présente.

Le diagnostic est une perte de couverture d'une question à deux aspects.
Les rangs ont été vérifiés à partir des audits du rapport et des chunks
convertis locaux correspondant à la phrase cible du benchmark.

### 2. `llava-scienceqa-fr` — échec de langue persistant

Question : « Quelle précision LLaVA combiné à GPT-4 atteint-il sur ScienceQA
après fine-tuning ? »

Les preuves de 92,53 % sont présentes : rappel de preuve cible à 100 %.
Le brouillon puis sa correction échouent au contrat
`language_mismatch:expected=fr`. La réponse finale est donc une abstention.
Le validateur de langue n'avait pas été corrigé dans cette version.

Le rapport ne conserve pas le texte de ces deux brouillons, seulement leur
longueur et les erreurs de validation. Il ne permet pas d'affirmer qu'ils
étaient réellement anglais. La faiblesse du détecteur sur une réponse
française courte reste un mécanisme plausible, confirmé séparément par les
tests diagnostiques, mais pas prouvé sur le texte de ces brouillons.

### 3. `compare-blip2-llava-bridge-fr` — nouvelle régression comparative

Question : « Compare le mécanisme utilisé par BLIP-2 et LLaVA pour relier un
encodeur visuel à un modèle de langage. »

Les deux articles sont retrouvés, mais leurs passages architecturaux cibles
sont écartés par le reranker :

| Article | Rang dans le pool fusionné | Rang après cross-encoder |
| --- | ---: | ---: |
| BLIP-2 | 2 | 8 |
| LLaVA | 1 | 6 |

La sélection équilibrée conserve trois chunks BLIP-2 et deux chunks LLaVA,
donc aucun de ces passages cibles n'est fourni au LLM. La réponse mentionne
Q-Former et une projection, mais pas toutes les caractéristiques attendues.
Elle précise elle-même que les extraits ne permettent pas de nommer certains
modèles. La correction de l'abstention initiale produit une réponse publiable,
mais sa couverture des faits est de 40 %, sous le seuil de réussite de 50 %.

Le rappel des deux preuves cibles vaut 0 %, contre 100 % dans le précédent
rapport. Cette régression ne peut pas être attribuée uniquement à une variante
de formulation du LLM : les preuves pertinentes sont effectivement exclues.

## Correctif SAM confirmé

`sam-ambiguity-en` passe de l'échec à la réussite, avec 100 % de couverture
des groupes de faits et 100 % de rappel de la preuve cible. La réponse explique
l'évaluation oracle : choisir la prédiction correspondant le mieux à la vérité
terrain parmi les trois masques, au lieu d'imposer le premier masque classé
par SAM. Le passage oracle est effectivement dans le contexte utilisé.

Le bilan de changement de réussite est donc : un cas corrigé, deux nouvelles
régressions, un échec persistant. Les 26 autres cas conservent leur statut.

## Tableau de chaque cas

Les colonnes « faits » et « preuve » représentent les scores actuels en
pourcentage. Un tiret indique une question d'abstention, sans score de faits.

| Cas | Avant | Maintenant | Faits | Preuve cible |
| --- | --- | --- | ---: | ---: |
| clip-training-objective-en | OK | OK | 100 | 100 |
| clip-zero-shot-classifier-en | OK | OK | 100 | 100 |
| clip-data-scale-fr | OK | Échec | 0 | 0 |
| clip-prompt-ensembling-en | OK | OK | 75 | 100 |
| dinov2-objectives-en | OK | OK | 100 | 100 |
| dinov2-curation-en | OK | OK | 100 | 100 |
| dinov2-frozen-features-fr | OK | OK | 75 | 0 |
| dinov2-distillation-en | OK | OK | 100 | 100 |
| sam-architecture-en | OK | OK | 100 | 100 |
| sam-prompt-types-fr | OK | OK | 100 | 100 |
| sam-ambiguity-en | Échec | OK | 100 | 100 |
| sam-dataset-scale-en | OK | OK | 100 | 100 |
| blip2-two-stages-en | OK | OK | 100 | 100 |
| blip2-bottleneck-fr | OK | OK | 100 | 100 |
| blip2-efficiency-en | OK | OK | 100 | 100 |
| blip2-capabilities-en | OK | OK | 100 | 100 |
| llava-synthetic-data-en | OK | OK | 75 | 100 |
| llava-architecture-en | OK | OK | 100 | 100 |
| llava-feature-alignment-en | OK | OK | 60 | 100 |
| llava-scienceqa-fr | Échec | Échec | 0 | 100 |
| compare-clip-dinov2-supervision-en | OK | OK | 100 | 0 |
| compare-blip2-llava-bridge-fr | OK | Échec | 40 | 0 |
| compare-clip-sam-prompts-en | OK | OK | 60 | 50 |
| compare-training-data-scale-en | OK | OK | 75 | 33,3 |
| compare-frozen-components-en | OK | OK | 100 | 100 |
| abstain-clip-electricity-en | OK | OK | — | — |
| abstain-sam-mars-en | OK | OK | — | — |
| abstain-blip2-radiology-en | OK | OK | — | — |
| abstain-dinov2-future-benchmark-fr | OK | OK | — | — |
| abstain-llava-local-latency-fr | OK | OK | — | — |

Certains cas passent sans retrouver la phrase cible annotée. Le score ne
garantit donc pas à lui seul que toute la réponse est soutenue par les preuves
attendues. Ces cas restent intéressants pour une revue qualitative ultérieure.

## Usage et latence

La campagne comporte 44 appels LLM logiques, 74 202 tokens d'entrée et 6 259
tokens de sortie, soit 80 461 tokens. Les éventuelles relances internes du SDK
ne sont pas comptées séparément par ce compteur. Aucun montant monétaire
n'est estimé ici.

La latence moyenne observée est de 7,49 secondes et la médiane de 6,02 secondes.
Le premier cas prend 36,04 secondes, avec chargement des modèles locaux dans
le nouveau processus. Hors ce premier cas, la moyenne est de 6,51 secondes.
Le cross-encoder apporte du calcul CPU supplémentaire ; les deux campagnes
ne constituent toutefois pas une mesure contrôlée de son coût isolé, car
elles n'ont pas été exécutées à cache et charge machine strictement identiques.

## Suite proposée, non implémentée pendant cette campagne

Le reranking par question entière corrige la recherche du protocole SAM, mais
ne préserve pas toujours toutes les facettes d'une question. La prochaine
itération devrait tester une sélection couvrant les sous-questions et les
sources, plutôt qu'un remplacement systématique de tous les classements par
un seul score global. Il faut conserver le gain SAM tout en récupérant les
passages chiffrés et architecturaux actuellement exclus.

Le validateur de langue doit être corrigé séparément. Les profils optionnels
peuvent également être évalués, sans supposer qu'ils corrigeront ces régressions.
Une seule campagne reste sensible à la variabilité de génération : répéter
les cas sensibles et ajouter des questions inédites permettra ensuite de
vérifier la généralisation.

Le protocole à paramètres constants et l'inspection des traces ont été
cadrés avec [OpenAI Docs](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
Toutes les valeurs numériques et les diagnostics ci-dessus proviennent des
rapports locaux, pas de cette documentation.

Rapport actuel : `agentic-rag-v7-question-rerank-30-20260930.json`.
Référence conservée : `agentic-rag-evaluation-gpt6-luna.json`.
