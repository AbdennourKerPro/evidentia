# Milestone 19 — Retrieval : correction SAM et ablation successive

Date : 30 septembre 2026.

## 1. Résultat et périmètre

Le correctif principal est le remplacement du classement prioritaire par
intentions lexicales par un cross-encoder qui évalue la question complète face
à chaque candidat. Sur les trois répétitions du cas `sam-ambiguity-en`, le
passage oracle remonte du rang 12 au rang 1. Les trois réponses du profil
`question_rerank` couvrent tous les groupes de faits du benchmark.

Trois améliorations ont été implémentées puis évaluées successivement, sans
modifier le modèle, le corpus, les réponses de référence ou les seuils de
réussite. Le profil minimal efficace, `question_rerank`, est actif par défaut.
Les deux améliorations supplémentaires restent disponibles comme profils
expérimentaux : leur bénéfice supplémentaire sur cette erreur n'est pas établi.

Le validateur de langue n'a pas été modifié. L'erreur LLaVA/ScienceQA reste une
étape distincte. Aucune campagne complète de 30 cas n'a été lancée.

## 2. Pourquoi l'ancienne réponse était incomplète

La question demande comment SAM traite l'ambiguïté d'un point pendant
l'évaluation. Il faut distinguer deux décisions :

- en fonctionnement normal, SAM classe les masques avec une confiance basée sur
  une estimation de leur IoU ;
- dans l'évaluation oracle, on choisit parmi les trois prédictions le masque qui
  correspond le mieux à la vérité terrain, afin de mesurer la qualité sans
  imposer la résolution de l'ambiguïté par le modèle.

L'ancienne réponse décrivait surtout la première décision. La seconde est bien
présente dans le corpus, section `D.1. Zero-Shot Single Point Valid Mask Evaluation`.

Les nouvelles traces permettent de localiser précisément le problème :

| Étape de l'ancien retrieval | Rang du passage cible |
| --- | ---: |
| Recherche dense sur la requête principale | 1 |
| Recherche BM25 sur la requête principale | 2 |
| Fusion multi-facettes RRF | 1 |
| Classement prioritaire par intentions | 12 |

Il ne s'agissait donc pas d'un article manquant, ni d'un échec de l'embedding.
Le passage était retrouvé, puis rétrogradé avant la sélection des cinq chunks.

## 3. Rappel des composants techniques

Un **candidat** est un chunk retrouvé lors de la recherche élargie. Un chunk
**sélectionné** est un candidat finalement envoyé au LLM. On peut donc avoir
retrouvé une bonne preuve sans que le LLM la voie.

La recherche dense utilise les embeddings E5 et Qdrant. BM25 recherche des
correspondances lexicales dans les chunks d'un article. La fusion RRF combine
les positions dans les différentes listes ; elle ne mélange pas directement
les scores numériques de BM25 et de la similarité dense.

Un **cross-encoder** reçoit conjointement `(question, passage)` et produit un
score de classement. Il peut examiner leurs interactions au lieu de comparer
deux embeddings calculés séparément. Le modèle local existant est
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, avec la révision déjà épinglée dans
`app/hybrid_retrieval.py`. Il tourne sur CPU et ne fait pas d'appel génératif à
OpenAI. Son score n'est pas une probabilité de véracité, ni une similarité E5.

Les intentions servent toujours à produire les variantes de recherche et à
construire les plans de faits existants. Seule leur priorité lexicale dans le
classement est retirée des nouveaux profils. Les comparaisons continuent de
sélectionner les sources en round-robin, afin de conserver plusieurs articles
dans le contexte.

Le nouveau reranking est appliqué aux branches hybrides multi-facettes,
focalisées sur un article ou utilisées par source dans une comparaison. Le
fallback global à une recherche dense unique reste inchangé : cette étape
ne constitue pas une refonte de toutes les routes de découverte.

## 4. Profils cumulatifs : une variable supplémentaire par étape

### Référence : `intent_baseline`

Reproduit le retrieval précédent : variantes existantes, pondération initiale
des expansions et classement `(_intent_score, score_rrf)` prioritaire.
Le cross-encoder n'intervient pas dans ce profil.

### Étape 1 : `question_rerank`

Conserve le même pool de candidats que la référence. Le cross-encoder classe
ce pool contre la question complète. Le titre et la section du chunk restent
présents dans le texte fourni au reranker. Les cinq premiers résultats sont
sélectionnés pour un article ; les comparaisons restent équilibrées entre
articles après classement par source.

Cette étape ne change ni le prompt de génération, ni les contrats de réponse.
Elle ajoute du calcul local, mais aucun appel LLM génératif.

### Étape 2 : `original_query`

Ajoute à l'étape 1 trois éléments cohérents pour préserver la recherche
originale :

1. la question utilisateur non modifiée devient la première variante ;
2. son poids RRF vaut 1, et les autres variantes se partagent un budget total
   de 1, indépendamment de leur nombre ; chaque variante possède une liste
   dense et une liste BM25 ;
3. les cinq premiers hits de chaque liste originale sont protégés contre la
   troncature du pool de 30 candidats, sans doublons.

La protection porte sur l'entrée du reranker, pas sur le contexte final : les
chunks protégés doivent encore être classés contre la question. Cela évite
d'imposer artificiellement une preuve simplement parce qu'elle vient de la
requête originale. Une requête planifiée pour un article reste une variante
complémentaire ; elle ne remplace plus la question utilisateur.

### Étape 3 : `coverage_retry`

Ajoute à l'étape 2 un contrôle de couverture avant la génération :

```text
sélection initiale
  -> check_evidence_coverage
       -> suffisant : plan de faits puis génération
       -> manque/incertain et budget disponible : retrieve_missing_evidence
            -> nouvelle sélection, toujours au plus cinq chunks
            -> check_evidence_coverage une seconde fois
            -> plan de faits puis génération, sans nouvelle boucle de recherche
```

Luna identifie entre un et quatre aspects demandés par la question. Pour chaque
aspect couvert, il doit fournir une référence `S…` et une citation textuelle.
Le serveur vérifie que cette référence existe et que le texte cité apparaît
dans le chunk indiqué. Un JSON invalide n'est jamais considéré comme une
preuve de couverture.

Un aspect non couvert peut produire une requête complémentaire. On exécute au
plus deux requêtes de manque dans une seule tentative de récupération, limitée
aux articles détectés ou déjà concernés par la recherche, au maximum cinq.
La question originale reste également dans cette recherche complémentaire.
Les anciens chunks sélectionnés sont réunis avec les nouveaux candidats,
dédupliqués puis rerankés. Les références finales sont reconstruites ensuite,
afin de ne pas mélanger les anciennes références avec la nouvelle sélection.

La vérification locale des citations empêche les citations inventées, mais ne
prouve pas que le jugement sémantique de Luna est correct. Le contrôle est un
signal de récupération, pas un nouveau veto absolu sur la réponse. Si la preuve
reste absente après la tentative, le pipeline conserve ses règles existantes
de génération fondée sur les sources et d'abstention. L'évaluation Mars a
effectivement conservé l'abstention après deux contrôles insuffisants.

Ce mécanisme est un workflow LangGraph borné. Il n'introduit pas encore de
`tools` dans les requêtes Responses API, ni de téléchargement dynamique arXiv.
Il prépare la structure d'une future recherche pilotée par outils.

## 5. Explication des fichiers et fonctions

### `app/retrieval_policy.py`

Centralise les quatre profils et leur validation. `get_retrieval_profile()` lit
`RAG_RETRIEVAL_PROFILE`. `preserves_original_query()` indique si l'étape 2 est
incluse. Une valeur inconnue est rejetée plutôt que remplacée silencieusement.
L'évaluateur peut passer un profil explicitement sans changer le profil de l'UI.

### `app/agentic_rag.py`

`answer_agentic_question()` accepte le profil expérimental et initialise les
traces. `_rank_candidates()` choisit l'ancien classement ou le cross-encoder.
`_candidate_audit()` conserve les identités des chunks dans les listes
originales, le pool et le classement final.

Les nouveaux nœuds `_check_evidence_coverage()` et
`_retrieve_missing_evidence()` réalisent l'étape 3.
`_route_after_selection()` active ce chemin uniquement pour `coverage_retry`.
`_route_after_coverage()` vérifie le budget ; une seconde recherche est interdite
après la première tentative. `_coverage_document_ids()` borne les sources.

Les reducers `Annotated[..., operator.add]` accumulent les audits et contrôles,
y compris ceux des branches par source. Le plan de faits, la génération et les
contrats langue/citations existants restent ensuite dans le même graphe.

### `app/query_expansion.py` et `app/hybrid_retrieval.py`

`build_query_variants(..., preserve_original=True)` place la question originale
en tête. `retrieve_multifacet_candidates(..., preserve_primary=True)` adapte les
poids et conserve les candidats originaux grâce à `preserve_primary_pool()`.
Les paramètres restent désactivés pour les deux premiers profils, afin de
reproduire leur comportement séparément.

### `app/evidence_coverage.py`

`assess_evidence_coverage()` fait un appel Luna avec un budget de sortie de
1 536 tokens. `parse_coverage_assessment()` valide la forme, les références et
les extraits. `sufficient=None` signifie « jugement inexploitable », pas « vrai ».
`missing_search_queries()` déduplique et limite les requêtes ; à défaut de
requête valide, elle revient à la question originale.

Les erreurs API restent gérées par la passerelle existante : elles ne sont pas
camouflées comme une réussite du contrôle. Aucune clé n'est écrite dans les
traces, rapports ou fichiers source.

### `app/schemas.py`

Ajoute `RetrievalAudit`, `CoverageRequirement` et `EvidenceCoverage`. La réponse
agentic expose `retrieval_profile`, `retrieval_audit`, `coverage_checks` et
`supplementary_search_attempted`. Cela rend l'exécution inspectable sans changer
la structure principale réponse/citations/chunks utilisée par l'interface.

### `scripts/evaluate_retrieval_iteration.py`

Exécute un profil à la fois. `--case-id` est obligatoire, pour ne pas lancer
accidentellement tout le benchmark. Les cibles sont répétées ; les contrôles
sont exécutés une fois. `--repeats` est limité à cinq.

Le script suit les étapes suivantes :

1. refuser un chemin de rapport déjà existant, avant tout appel facturé ;
2. charger le benchmark et sélectionner uniquement les cas demandés ;
3. prendre une empreinte des chunks des articles attendus et des fichiers source ;
4. appeler le pipeline avec seulement la question, le scope, le top-k et le profil ;
5. mesurer les appels LLM et calculer les métriques après la réponse ;
6. retrouver les rangs des preuves attendues dans les audits ;
7. sauvegarder un checkpoint après chaque essai.

`corpus_fingerprint()` ignore les scores et trie les chunks pour produire une
empreinte stable. `target_rank_diagnostics()` applique les annotations du
benchmark uniquement après la réponse. Les formulations attendues ne sont
jamais injectées dans les requêtes, le reranker, le juge ou la génération.

Les rapports produits pendant cette session contiennent les empreintes des
fichiers clés au moment de chaque exécution. Le script final étend cette
empreinte à tous les modules Python du dossier `app` et distingue la phase
initiale d'une recherche complémentaire.

### `scripts/evaluate_agentic_rag.py`

L'évaluateur général conserve son usage précédent. Le pipeline passe en v7,
enregistre le profil actif et refuse une reprise avec un profil différent.
Le chemin de sortie par défaut devient
`/reports/agentic-rag-evaluation-v7.json`, afin de préserver le rapport initial
GPT-6 Luna. Cet évaluateur général n'a pas été lancé sur les 30 cas ici.

### Tests

- `test_retrieval_policy.py` : question complète transmise au reranker, baseline
  sans cross-encoder, conservation des candidats, pondération et validation ;
- `test_evidence_coverage.py` : citations exactes, JSON invalide, références
  inconnues, nombre de recherches, scope, top-k et terminaison du graphe ;
- `test_retrieval_iteration.py` : empreintes stables, annotation des rangs après
  réponse et refus de remplacer un rapport avant un appel facturé ;
- `test_evaluation_checkpoint.py` : ajoute le refus d'une reprise avec un profil
  différent, en plus des vérifications modèle, top-k et benchmark existantes.

La suite finale contient **101 tests**, tous passés dans l'image mise à jour,
sans appels LLM facturés pendant ces tests.

## 6. Protocole expérimental et résultats

Configuration commune : GPT-6 Luna, reasoning `low`, top-k 5, collection
`arxiv_chunks_e5_small`. Les quatre expériences ont les mêmes empreintes de
benchmark et de corpus attendu.

Chaque étape comporte six exécutions : trois répétitions de `sam-ambiguity-en`,
puis trois contrôles : `sam-architecture-en`, `compare-clip-sam-prompts-en` et
`abstain-sam-mars-en`. Au total : 24 exécutions, pas 24 cas indépendants.

| Profil | Passage oracle dans le contexte | Rang après reranking | Couverture moyenne des termes SAM | Réussite SAM, seuil existant | Contrôles réussis |
| --- | --- | ---: | ---: | --- | --- |
| Référence | 0/3 | 12 | 53,3 % | 2/3 | 3/3 |
| Question complète | 3/3 | 1 | 100 % | 3/3 | 3/3 |
| + Requête originale | 3/3 | 1 | 93,3 % | 3/3 | 2/3 |
| + Couverture/recherche bornée | 3/3 | 1 | 100 % | 3/3 | 3/3 |

Le critère global existant accepte une couverture de termes de 50 % : les
deux réussites partielles de la référence ne signifient donc pas que la réponse
expliquait le protocole oracle. Le passage était absent et les trois réponses
décrivaient le classement normal par confiance. Les rangs et la présence de la
preuve sont ici les indicateurs les plus directs de l'effet correctif.

L'étape 2 fournit encore le bon protocole dans ses trois réponses. Un essai
obtient 80 % de couverture lexicale bien que le protocole soit expliqué : le
score mesure les variantes textuelles annotées, pas une compréhension sémantique
parfaite. Nous n'avons pas modifié ces annotations après avoir vu les résultats.

La comparaison CLIP/SAM conserve seulement 50 % de rappel des passages cibles
dans les quatre expériences. Elle passe au seuil global sauf à l'étape 2.
Les cinq chunks sélectionnés sont identiques entre les étapes 1 et 2 ; la
variation de réussite ne démontre donc pas une régression de sélection entre
ces deux runs. Elle souligne une faiblesse résiduelle de ce contrôle et la
variabilité de la génération. Aucune règle spéciale n'a été ajoutée pour
rattraper ce cas pendant la correction SAM.

Dans l'étape 3, les trois contrôles SAM de couverture sont suffisants dès le
premier passage : **aucune récupération supplémentaire ne corrige le cas SAM**.
Une récupération a effectivement lieu sur la question Mars, suivie d'un second
jugement insuffisant et d'une abstention correcte. La borne d'une tentative est
donc vérifiée aussi par une exécution réelle, en plus des tests sans API.

### Appels et usage observés

| Profil, six exécutions | Appels LLM logiques | Tokens entrée | Tokens sortie |
| --- | ---: | ---: | ---: |
| Référence | 8 | 12 650 | 1 203 |
| Question complète | 7 | 9 675 | 957 |
| + Requête originale | 8 | 12 373 | 976 |
| + Couverture/recherche bornée | 15 | 23 096 | 2 815 |

Total observé : 38 appels logiques et 63 745 tokens. Les éventuelles tentatives
internes du SDK ne sont pas comptées comme des appels séparés par ce compteur.
Sur SAM seul, le profil minimal utilise un appel par question ; le contrôle de
couverture en utilise deux, sans changement de sélection ni gain correctif
direct observable sur cette erreur.

La première latence de chaque processus inclut le chargement des modèles
locaux. Les essais ne justifient pas une conclusion de vitesse globale, ni une
estimation de gain sur des questions inédites. Trois répétitions sont un
diagnostic, pas une validation statistique de production. Le panel réutilise des
questions du benchmark ; ce n'est pas un jeu de test indépendant.

### Rapports conservés

- `reports/retrieval-sam-00-baseline.json`
- `reports/retrieval-sam-01-question-rerank.json`
- `reports/retrieval-sam-02-original-query.json`
- `reports/retrieval-sam-03-coverage-retry.json`

Chaque rapport contient les réponses complètes, preuves sélectionnées, traces,
rangs cibles, usage LLM et configuration. Le rapport initial de 30 cas
`reports/agentic-rag-evaluation-gpt6-luna.json` est conservé.

## 7. Configuration et reproduction

Le profil de l'application se choisit dans `.env` :

```dotenv
RAG_RETRIEVAL_PROFILE=question_rerank
```

Les autres valeurs sont `intent_baseline`, `original_query`, `coverage_retry`.
L'étape 3 ajoute un contrôle LLM par question ayant des preuves, et un second
contrôle en cas de récupération. Il faut donc l'activer en connaissance de ce
surcoût. Une modification d'environnement demande la recréation de l'API :

```powershell
docker compose up -d --no-deps --no-build api
```

Pour reproduire une étape, utiliser un nom de sortie inédit. Exemple sur une
seule ligne PowerShell :

```powershell
docker compose run --rm --no-deps api python -m scripts.evaluate_retrieval_iteration --profile question_rerank --case-id sam-ambiguity-en --repeats 3 --control-case-id sam-architecture-en --control-case-id compare-clip-sam-prompts-en --control-case-id abstain-sam-mars-en --output /reports/retrieval-sam-question-rerank-new-run.json
```

Changer `--profile` et le nom de sortie permet de mesurer une autre étape.
Ce script refuse les rapports déjà existants ; il ne reprend pas une campagne
interrompue. L'évaluateur général possède toujours son mécanisme `--resume`.

Pour les tests sans appels facturés :

```powershell
docker compose run --rm --no-deps api python -m unittest discover -s tests -q
```

## 8. Mise à jour Docker sans changer les dépendances

La reconstruction ordinaire a été interrompue : le cache ne réutilisait pas la
couche de dépendances, et le build voulait télécharger de nouvelles versions
transitives. Déployer un environnement différent de celui des essais aurait
affaibli leur comparabilité et entraîné des téléchargements inutiles.

`Dockerfile.incremental` utilise l'image locale testée comme base et copie
uniquement `app`, `scripts` et `tests`. Il hérite de son Python, de ses
dépendances, variables d'image et commande de démarrage. Aucun fichier `.env`
n'est incorporé ; Compose continue de fournir la configuration au démarrage.

L'image précédente a été conservée sous `evidentia-app:pre-retrieval-v7` pour
permettre un retour arrière. Le build incrémental a réussi et les 101 tests ont
ensuite été exécutés dans l'image finale. L'API seule est recréée ; les volumes
des articles, des modèles et de Qdrant ne sont pas supprimés.

Pour une future mise à jour de code uniquement, en préservant une base locale :

```powershell
docker image tag evidentia-app:dev evidentia-app:pre-retrieval-v7
docker build --pull=false --file Dockerfile.incremental --tag evidentia-app:dev .
docker compose up -d --no-deps --no-build api
```

Attention : la première commande remplace ce tag de sauvegarde par l'image
courante. Choisir un autre tag si l'on souhaite garder plusieurs versions,
puis le fournir via `--build-arg BASE_IMAGE=nom-du-tag`. Le Dockerfile ordinaire
reste nécessaire pour une modification volontaire des dépendances.

## 9. Décision et prochaines vérifications

Le défaut retenu est `question_rerank`, car il corrige directement le mécanisme
identifié sans ajouter d'appel génératif. Les variantes plus complexes sont
conservées pour comparaison, pas déclarées meilleures sans preuve.

Il reste à traiter séparément la validation de langue, à mieux examiner le
contrôle comparatif incomplet, puis à ajouter des questions inédites sur les
protocoles expérimentaux. Une prochaine évaluation globale devra employer le
rapport v7 et ne pas écraser le rapport initial. Aucune généralisation de
performance aux 30 cas n'est revendiquée ici.

L'approche d'évaluation ciblée, les répétitions et l'inspection des traces ont
été structurées avec **OpenAI Docs**, notamment le
[guide officiel d'évaluation](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
Les mesures de cette trace viennent des rapports locaux, pas de ce guide.

## 10. Campagne complète ultérieure sur les 30 cas

Après cette ablation, une évaluation complète a été explicitement demandée
et exécutée avec le profil actif `question_rerank`, sans modification du code.
Le rapport `reports/agentic-rag-v7-question-rerank-30-20260930.json` contient
30/30 cas terminés. Son compte rendu détaillé se trouve dans
`reports/agentic-rag-v7-question-rerank-30-20260930-summary.md`.

Résultat : 27/30 réussites (90 %), contre 28/30 auparavant. SAM est corrigé,
mais deux régressions de sélection apparaissent sur `clip-data-scale-fr` et
`compare-blip2-llava-bridge-fr`. L'échec de langue de ScienceQA persiste.
Les cinq abstentions attendues sont correctes ; quatre comparaisons sur cinq
passent. La couverture moyenne des faits est de 82,4 % et celle des passages
cibles de 79,3 %.

Cette campagne montre que le gain observé sur le cas SAM ne se généralise pas
tel quel à l'ensemble du benchmark. La décision minimale de l'ablation doit
donc être réexaminée avec ces nouveaux résultats, sans supposer que le
cross-encoder sur question entière est toujours supérieur au classement
précédent. Le profil de l'application n'a pas été changé automatiquement
à la suite de l'évaluation.
