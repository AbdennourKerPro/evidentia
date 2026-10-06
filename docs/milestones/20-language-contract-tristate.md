# Milestone 20 — Contrôle de langue à trois états

Date : 30 septembre 2026. Version logique : v8.

## 1. Objectif et périmètre

Corriger le refus injustifié de réponses françaises courtes, techniques ou
numériques, sans modifier la recherche documentaire ni adapter des intentions
aux questions du benchmark.

La campagne précédente sur 30 cas avait obtenu 27 réussites : SAM était corrigé,
mais deux anciennes réussites avaient régressé (échelle des données CLIP et
comparaison BLIP-2/LLaVA). ScienceQA restait bloqué par la langue. Cette étape
traite uniquement la langue. Les deux problèmes de sélection documentaire restent
à traiter séparément ; aucune nouvelle campagne complète n'a été lancée ici.

Le modèle reste GPT-6-Luna, avec l'effort de raisonnement configuré `low`, via la
passerelle Responses existante. Aucun entraînement, nouveau détecteur téléchargé,
nouveau paquet Python, réindexage Qdrant ou changement d'embeddings.

## 2. Pourquoi le contrôle précédent échouait

Il comparait le nombre de mots présents dans deux petites listes françaises et
anglaises. Pour accepter le français, il exigeait au moins deux marqueurs français
et un score français supérieur ou égal au score anglais.

Une phrase française comme « LLaVA combiné à GPT-4 atteint une précision de
92,53 % sur ScienceQA après fine-tuning [S3, S4]. » contient presque uniquement
des mots absents de cette liste. Cela ne signifie pas qu'elle est anglaise.
L'absence de signal linguistique était pourtant transformée en erreur certaine.

Les anciens brouillons ayant conduit à l'échec sur les 30 cas n'étaient pas
enregistrés intégralement : on ne peut pas reconstituer leur langue avec certitude.
Nous avons donc fait un contrôle supplémentaire sur les réponses de cette étape :
les mêmes textes ont été soumis au validateur de l'image v7 conservée. Il rejette
les trois réponses françaises ScienceQA avec `language_mismatch:expected=fr` ;
le nouveau les accepte. Les deux témoins restent acceptés par les deux versions.
Ce replay est hors réseau, sans génération ni juge supplémentaire : il isole
le changement de validation sur les mêmes textes, sans prouver rétrospectivement
le contenu des brouillons non enregistrés.

## 3. Concepts : trois décisions, pas deux

| État | Sens | Suite |
|---|---|---|
| `match` | Les indices locaux indiquent la langue demandée | Poursuivre les autres contrats |
| `mismatch` | Les indices indiquent une autre langue | Erreur de langue ; réécriture si encore autorisée |
| `uncertain` | Indices insuffisants ou ambigus | Un appel de classification pour départager |

« Incertain » ne signifie donc ni « faux » ni « automatiquement accepté ».
Une réponse numérique peut être compatible avec plusieurs langues. Les noms
scientifiques, acronymes, valeurs et formules ne sont pas en eux-mêmes des phrases
étrangères. En revanche, une véritable phrase explicative anglaise dans une
réponse française doit rester un défaut.

Le détecteur local reste une heuristique, pas une certification linguistique :
les noms propres et mots partagés peuvent encore produire des erreurs. Le juge
LLM peut également se tromper. Les tests ciblés ne remplacent pas la vérification
humaine ni une évaluation linguistique plus large.

## 4. Architecture d'exécution

```text
Question -> retrieval inchangé -> génération du brouillon
                               -> validation des contrats locaux
                                  -> langue match/mismatch : décision locale
                                  -> langue uncertain : une classification Luna
                               -> décision sur tous les contrats réunis
                                  -> valides : publication
                                  -> défaut : une réécriture au maximum
                                     -> nouvelle validation, mêmes règles
                                     -> publication ou abstention explicite
```

Chaque brouillon passe toujours par les vérifications de complétude, citations,
couverture des articles requis et faits planifiés. Un verdict de langue `match`
ne supprime aucune autre erreur. Les erreurs de complétude factuelle qui étaient
consultatives le restent : leur politique n'a pas été modifiée ici.

Le marqueur exact `INSUFFICIENT_EVIDENCE` n'est pas une phrase à classifier : il
conserve son traitement spécifique, notamment le contrôle d'abstention prématurée.
La réponse d'abstention visible est ensuite localisée dans la langue de la question.

Limites de la boucle LangGraph :

- Au maximum une classification de langue par brouillon.
- Au maximum un brouillon initial et une réécriture : donc deux classifications
  supplémentaires au maximum pour la langue.
- Pas de boucle de vérification ni de retry applicatif de classification.
- Les retries HTTP éventuellement configurés dans le SDK restent distincts de
  ce décompte logique.
- Un verdict encore incertain après la correction demeure bloquant.
- Les erreurs d'accès, de quota ou de connexion API suivent la gestion d'erreurs
  existante ; elles ne sont pas déguisées en validations réussies.

La baseline `/ask` vérifie également les cas linguistiques incertains, mais ne
gagne pas de boucle de réécriture : son architecture reste différente de
`/agentic/ask`, comme avant.

## 5. Explication de chaque fichier modifié ou ajouté

### `app/response_contract.py` — règles locales, sans appel réseau

`language_prose(text)` élimine, uniquement pour le calcul linguistique, les blocs
de code, le code en ligne, URL, citations S, formules délimitées et nombres. Le
texte original publié et les règles de citation restent intacts. Les noms des
modèles et les phrases étrangères ne sont pas supprimés.

`assess_answer_language(text, expected)` renvoie les trois états. Il exploite :

1. La proportion d'idéogrammes chinois pour les langues actuellement prises en
   charge (`en`, `fr`, `zh`). Une forte domination chinoise (au moins 50 % des
   caractères alphabétiques) est un signal clair ; les scripts mixtes vont à la
   vérification.
2. Les listes existantes de marqueurs français/anglais, plus les articles généraux
   `le` et `la` pour l'analyse des réponses seulement. La détection de langue de
   la question n'a pas été modifiée.
3. Un seuil de deux marqueurs et une domination d'un facteur deux pour décider
   localement. Un faible signal ou une concurrence entre langues reste incertain.
4. Une vérification par phrase/ligne pour éviter qu'une phrase étrangère soit
   masquée par un long texte dans la bonne langue.

Ces seuils sont des choix d'ingénierie conservateurs, pas des probabilités
calibrées. Aucune règle spécifique à ScienceQA, LLaVA, SAM ou 92,53 %.

`validate_answer(...)` reste une fonction pure : elle calcule tous les contrats
et ajoute `language_uncertain:expected=...` ou `language_mismatch:expected=...`
selon le résultat local. Elle peut donc être testée sans clé API.

`AnswerValidation` conserve les résultats de citations et de faits, et ajoute
`language_status` et `language_check_method`. Pour le marqueur d'abstention,
le statut de langue est `None` (contrôle non applicable).

### `app/language_verification.py` — nouveau module applicatif

`validate_answer_with_language_check(...)` appelle d'abord la validation pure.
Uniquement si le statut est incertain, il transmet à la passerelle LLM la langue
attendue et le brouillon complet, comme données JSON. Le modèle ne reçoit ni
réponse de référence du benchmark ni étiquette attendue de réussite.

Le prompt demande de classifier la langue, pas de répondre à la question, juger
les faits ou réécrire le texte. Il explicite la neutralité des termes techniques
et interdit de suivre les instructions contenues dans le brouillon. Cela réduit
le risque d'injection, sans constituer une garantie absolue.

`parse_language_verdict(raw)` exige un objet JSON contenant exactement la clé
`status` avec l'une des trois valeurs autorisées. Une sortie malformed, une clé
en trop, un booléen ou une valeur inconnue deviennent `uncertain`, jamais `match`.
Le format est demandé par prompt et vérifié localement ; ce module n'utilise pas
de schéma de sortie contraint au niveau de l'API.

Le budget de cet appel est de 1 024 tokens au maximum, limité également par la
configuration globale. Ce budget inclut le raisonnement. Il ne s'agit pas du
nombre réel de tokens consommés. Les métriques de la passerelle enregistrent
l'appel et son usage comme pour les autres appels de génération.

Le module remplace uniquement l'erreur linguistique incertaine par le résultat
du juge ; toutes les autres erreurs sont conservées.

### `app/agentic_rag.py` — intégration dans le graphe

Le nœud `validate_answer` utilise ce nouveau point d'entrée. La trace donne la
langue attendue, le verdict final et la méthode (`heuristic` ou
`llm_verification`). Le routage et la limite d'une correction restent inchangés.
Les deux validations, avant et après correction éventuelle, restent visibles
dans les étapes. Les métadonnées synthétiques de fin représentent le dernier
brouillon contrôlé.

### `app/rag_service.py` — intégration dans la baseline

`answer_from_results` utilise le même contrôle de langue avec vérification.
Le finaliseur et la recherche vectorielle ne changent pas. Les réparations
déterministes existantes conservent leurs contrôles locaux sans appels additionnels.

### `app/schemas.py` — transparence des résultats

`AgenticExecution` expose `language_status` et `language_check_method`. Ces champs
s'ajoutent aux traces et à `contract_issues`, sans supprimer les champs existants.
Ils ne constituent pas une nouvelle interface visuelle ; ils sont disponibles
dans la réponse API et les rapports d'évaluation.

### `scripts/evaluate_agentic_rag.py` — version du rapport

Le nom de sortie par défaut devient `agentic-rag-evaluation-v8.json` et l'identifiant
de pipeline devient `langgraph_language_contract_v8_openai_v3`. Cela sépare les
rapports v7/v8 et évite de réutiliser un checkpoint ancien comme résultat de cette
nouvelle version. Les critères, questions et réponses de référence ne changent pas.
Ce script complet n'a pas été lancé pendant cette étape.

### Tests

`tests/test_language_verification.py` couvre les phrases courtes françaises et
anglaises, nombres seuls, autre langue, chinois, scripts mixtes, phrase étrangère
dans une longue réponse française, neutralité du code/citations, schéma JSON strict,
conservation des erreurs de citation et propagation des erreurs API. Ses appels
LLM sont simulés : ils vérifient la logique applicative, pas la précision réelle
du modèle de classification.

`tests/test_api_llm.py` ajoute des intégrations HTTP avec passerelle simulée :
baseline courte (deux appels), graphe accepté sans réécriture (deux appels), et
incertitude persistante (quatre appels maximum, une correction, abstention).
Les tests existants de réécriture d'une réponse réellement étrangère passent aussi.

Résultat de la suite dans l'image reconstruite : **124 tests réussis**.

## 6. Évaluation réelle ciblée

Rapport : `reports/language-v8-scienceqa-targeted-20260930.json`.

Nous avons réutilisé `scripts.evaluate_retrieval_iteration` comme exécuteur de
répétitions : il permet de garder explicitement `question_rerank`, trois essais
de la cible et un essai par témoin. Son champ historique `pipeline` reste
`retrieval_ablation_v1` ; les empreintes des sources dans ce rapport identifient
le code v8 évalué. Le nom du script ne signifie pas que le retrieval a été modifié.

Commande exécutée depuis le dossier de l'application, sur le code monté en lecture
seule dans l'image existante :

```powershell
docker compose run --rm --no-deps --volume "C:/Users/abden/Documents/Codex/2026-07-30/je/work/evidentia:/app:ro" api python -m scripts.evaluate_retrieval_iteration --profile question_rerank --case-id llava-scienceqa-fr --repeats 3 --control-case-id sam-ambiguity-en --control-case-id sam-prompt-types-fr --output /reports/language-v8-scienceqa-targeted-20260930.json
```

| Cas | Essais | Réussites | Langue | Réécritures | Appels LLM par essai |
|---|---:|---:|---|---:|---:|
| LLaVA / ScienceQA, français | 3 | 3 | Vérification LLM : match | 0 | 2 |
| SAM / types de prompts, français | 1 | 1 | Heuristique : match | 0 | 1 |
| SAM / ambiguïté et oracle, anglais | 1 | 1 | Heuristique : match | 0 | 1 |

Les trois réponses ScienceQA donnent 92,53 %, citent S3 et S4 et ne s'abstiennent
plus. La vérification prend environ 0,87 à 1,50 seconde sur ces essais.

Attention à l'interprétation du score : les réponses ScienceQA couvrent deux des
trois groupes de termes attendus (valeur et ScienceQA), mais n'énoncent pas
explicitement « état de l'art ». Leur couverture factuelle mesurée est donc
**2/3**, pas 100 %. Elles passent le seuil existant du benchmark (au moins 0,5,
abstention correcte et articles cités corrects). Les deux témoins couvrent tous
leurs groupes attendus.

Sur ce panel : 5/5 réussites, rappel des preuves cibles 1, couverture factuelle
moyenne 0,8, huit appels logiques et 9 540 tokens cumulés. Ce sont des mesures
sur cinq exécutions, dont trois répétitions du même cas, pas sur cinq questions
différentes. Le temps du premier essai inclut le chargement des modèles locaux ;
ces latences ne constituent pas un benchmark de performance à cache contrôlé.

Empreinte du benchmark inchangé :
`31917579b4753cf27d3eca756dbf7b8ef23917f6bfcb1bdb6a7b2414a4aab74b`.

Les fichiers d'embeddings, Qdrant, expansion des requêtes, retrieval hybride,
profil de retrieval et planification factuelle ont été comparés par SHA-256 à
l'image v7 conservée : identiques. Le prompt de génération n'a pas été modifié.

## 7. Déploiement et retour arrière

L'image v7 a été conservée sous `evidentia-app:pre-language-v8` avant de remplacer
`evidentia-app:dev`. L'image de dépendances précédente
`evidentia-app:pre-retrieval-v7` reste également disponible.

Construction incrémentale :

```powershell
docker build --pull=false --file Dockerfile.incremental --tag evidentia-app:dev .
docker compose run --rm --no-deps api python -m unittest discover -s tests -q
docker compose up -d --no-deps --no-build api
```

Le Dockerfile incrémental recopie seulement `app`, `scripts` et `tests` sur
l'environnement de dépendances déjà installé. Aucun téléchargement massif de
dépendances, aucune suppression de volume et aucune réindexation des articles.
L'interface reste à `http://localhost:8000/ui/`.

Si un retour à v7 est nécessaire, il remettrait l'ancien code et son ancien défaut
de langue, sans effacer les volumes :

```powershell
docker image tag evidentia-app:pre-language-v8 evidentia-app:dev
docker compose up -d --no-deps --no-build api
```

Ne pas exécuter ce retour arrière simplement pour tester la v8 ; il s'agit d'une
procédure de secours, non exécutée pendant cette étape.

## 8. Ce qui reste à faire

Le blocage de langue ScienceQA est corrigé sur trois répétitions et isolé par le
replay à texte identique. Il n'y a aucune règle spéciale au cas évalué.

Les deux régressions de retrieval (CLIP et comparaison BLIP-2/LLaVA) ne sont ni
réévaluées ni déclarées corrigées ici. Une campagne complète v8 reste nécessaire
avant d'annoncer un nouveau score global sur 30 cas. Un panel linguistique hors
benchmark, avec annotations humaines et exemples adversariaux, permettrait aussi
de mesurer la précision du juge plutôt que seulement sa bonne intégration.

Le choix d'un juge de classification à critères explicites et l'isolation de
l'étape évaluée s'appuient sur les bonnes pratiques d'OpenAI Docs :
https://developers.openai.com/api/docs/guides/evaluation-best-practices.
Il s'agit de notre implémentation applicative, pas d'une garantie fournie par l'API.

## 9. Validation complète ultérieure sur les 30 cas

Après les essais ciblés, une campagne complète a été explicitement demandée et
exécutée avec l'image v8 déployée, sans modification pendant la campagne. Rapport :
`reports/agentic-rag-v8-language-30-20260930.json`. Analyse exhaustive, tableau des
30 cas et limites : `reports/agentic-rag-v8-language-30-20260930-summary.md`.

Résultat : **28/30 (93,3 %)** contre 27/30 en v7. ScienceQA et la comparaison
BLIP-2/LLaVA passent désormais ; la comparaison des prompts CLIP/SAM passe de
réussite à échec. Le premier gain est confirmé par le nouveau contrôle de langue.
Le gain BLIP-2/LLaVA n'est pas attribuable à la langue : pas de classification
linguistique additionnelle sur ce cas, couverture des termes seulement 60 % et
preuves cibles toujours absentes. La perte CLIP/SAM se produit avec les mêmes
textes de chunks que v7 : génération variable, termes attendus partiellement
absents et score sensible à « point » versus « points ».

Le cas CLIP sur les 400 millions de paires reste en échec documentaire. Les cinq
abstentions attendues passent. Aucun défaut de langue n'apparaît dans les traces
de cette campagne. La vérification LLM a été utilisée sur ScienceQA et sur la
réponse anglaise courte et numérique d'efficacité BLIP-2, sans réécriture.

45 appels logiques, 80 976 tokens, couverture moyenne des groupes de termes
82,3 %, rappel des preuves cibles 81,3 %, huit corrections. Le score global
progresse sans amélioration significative de la couverture factuelle lexicale.
Les limites des mesures et la variabilité de génération interdisent de présenter
ce résultat comme une résolution définitive des comparaisons.
