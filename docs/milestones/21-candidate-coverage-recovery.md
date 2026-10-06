# Milestone 21 — Récupération de preuves guidée par la couverture

Date : 30 septembre 2026. Version logique : v9.

## 1. Pourquoi cette étape

La v8 passait 28/30 questions. Deux cas échouaient : la taille du dataset CLIP et
la comparaison des prompts CLIP/SAM. Les preuves existaient déjà dans les candidats.
Sur CLIP, des passages avec 400 millions de paires étaient aux rangs 1 et 2 RRF,
puis 12 et 6 après reranking global. Sur SAM, le passage des types de prompts
était au rang 1 RRF, puis 7 après reranking. La sélection initiale ne le gardait pas.

Il n'était donc pas nécessaire de télécharger d'autres articles ou d'entraîner
un nouveau modèle pour récupérer ces preuves. Le problème était la couverture
de la question par une fenêtre de cinq passages, puis l'utilisation de ces passages.

Objectif : compléter uniquement les aspects manquants, sans retirer les preuves
initiales, changer les intentions lexicales, apprendre les réponses du benchmark
ou modifier son score pour faire passer les cas.

## 2. Concepts et architecture

La pertinence d'un chunk et la couverture d'une question sont deux choses
différentes : cinq passages tous pertinents pour un mécanisme peuvent omettre
un nombre, une liste de formats ou un aspect d'une comparaison.

La couverture est représentée par des exigences atomiques : chacune doit pouvoir
être appuyée par un court extrait d'un article. Les exigences sont définies sans
voir les chunks, pour ne pas être ajustées à ce que le moteur a retrouvé.

```text
retrieval v8 inchangé -> cinq passages initiaux + réserve de candidats
  -> plan des exigences, à partir de la question seule
  -> vérification des exigences dans les passages initiaux
     -> complet ou inconnu : génération habituelle
     -> manque confirmé :
        reranking des candidats contre les sous-questions manquantes
        -> vérification d'extraits exacts sur une courte liste de propositions
        -> ajout de zéro, un ou deux passages, sans remplacer les cinq premiers
        -> génération habituelle + contrats de langue/citations/faits existants
```

Cette version ne fait pas de nouveaux appels Qdrant pendant la récupération.
Elle utilise la réserve issue de la recherche initiale. La recherche complémentaire
et le fetching arXiv restent des étapes futures si la preuve est absente de la réserve.
Il s'agit de nœuds orchestrés dans LangGraph, pas de function calling natif où le
LLM exécuterait librement des outils.

## 3. Des expériences isolées, y compris les échecs

Les rapports suivants sont conservés dans `reports/`. Les premières variantes
ont été modifiées progressivement ; leurs empreintes de sources sont enregistrées
dans les rapports. Les fichiers actuels représentent la dernière variante, pas
toutes les anciennes versions de prompt.

| Rapport | Modification testée | Cibles réussies | Témoins réussis |
|---|---|---:|---:|
| `coverage-v9-01-diagnostic.json` | Diagnostic définissant encore les exigences en voyant les chunks | 0/2 | 2/2 |
| `coverage-v9-02-candidates.json` | Récupération depuis les candidats, avec ce diagnostic couplé | 3/6 | 4/5 |
| `coverage-v9-03-fixed-requirements.json` | Exigences définies avec la question seule, puis vérifiées sans les changer | 4/6 | 5/5 |
| `coverage-v9-04-scoped-requirements.json` | Exigences séparées par article, liste explicite pour les interfaces | 4/6 | 4/5 |
| `coverage-v9-05-atomic-candidates.json` | Exigences atomiques : formats d'entrée séparés du mécanisme | **6/6** | **5/5** |
| `coverage-v9-06-answer-plan.json` | Même récupération, avec les liens exigence/preuve dans le prompt de réponse | 3/6 | 5/5 |
| `coverage-v9-07-mechanism-requirements.json` | Structure/operation des connecteurs distinguées des formats d'entrée | 6/9 | 4/4 |
| `coverage-v9-08-completeness-verifier.json` | Explication complète/partielle avant de choisir la citation ; exemple générique d'interface | 8/9 | 4/4 |
| `coverage-v9-09-source-fair-budget.json` | Budget partagé entre articles et longueur de citation explicitée | **9/9** | **4/4** |

Les six essais cibles sont trois répétitions de CLIP et trois de CLIP/SAM. Les
cinq témoins sont SAM/oracle, ScienceQA, goulot BLIP-2, comparaison BLIP-2/LLaVA
et abstention SAM/Mars. Le premier diagnostic n'utilisait que deux témoins et une
répétition de chaque cible. Les six premières campagnes totalisent 59 exécutions
ciblées, avec un nombre
d'appels LLM variable selon les routes.
À partir de la septième campagne, la comparaison BLIP-2/LLaVA devient une cible
répétée trois fois, au même titre que les deux échecs initiaux ; quatre témoins
sont conservés, soit 13 exécutions par campagne.

Deux enseignements :

1. Un juge qui définit et vérifie les exigences en voyant les mêmes passages peut
   considérer des exemples SAM comme une interface complète. La séparation
   question-only/verification a supprimé cette possibilité structurelle, mais pas
   toute erreur de jugement du modèle.
2. Une exigence regroupant deux articles ne peut pas être prouvée par un seul
   extrait d'une seule source. Même au sein d'un article, mélanger liste de formats
   et mécanisme oblige parfois à plusieurs extraits. Les exigences atomiques
   évitent ces lacunes artificielles.

Le plan ajouté à la génération n'a pas amélioré les résultats ciblés : il a parfois
écarté des informations pertinentes et utilisé des formes singulières non reconnues
par le score lexical. Il reste un profil d'ablation, mais n'est pas le choix retenu.
Nous ne prétendons pas que les différences de scores ont toutes une cause unique :
chaque essai contient des appels LLM non déterministes. Les deux dernières campagnes
ciblées ont en partie tourné simultanément ; leurs latences ne sont pas comparables
comme mesures de performance à ressources constantes.

## 4. Explication du nouveau module `app/candidate_recovery.py`

### Définir les exigences

`assess_question_coverage(question, results)` fait deux appels bornés au modèle
configuré, toujours GPT-6-Luna :

- Premier appel : la question et les identifiants des articles seulement, **pas le
  contenu des passages**. Le modèle propose entre une et quatre exigences et des
  requêtes anglaises courtes. Un exemple générique d'articles A/B illustre la
  séparation entre formats d'entrée et traitement ; il ne contient aucun fait
  sur CLIP, SAM ou une valeur du benchmark.
  Pour une comparaison de connecteurs, les exigences portent plutôt sur la
  structure concrète et la transformation/sélection des représentations. Les
  formats d'entrée ne sont pas imposés à toutes les questions d'architecture.
- Deuxième appel : les exigences figées et les passages. Le modèle doit fournir
  un extrait exact et sa référence S pour chaque exigence, ou indiquer qu'il ne
  trouve pas de support.

`parse_question_requirements` contrôle les types, longueurs, doublons, articles
autorisés et nombre d'exigences. Pour plusieurs articles, chaque article doit
avoir une exigence et les exigences ne peuvent pas avoir une portée inter-articles
anonyme. Un plan invalide reste inconnu ; il ne provoque pas de recherche devinée.
Cette limite de quatre exigences peut empêcher un plan de couvrir plus de quatre
articles : la récupération revient alors au contexte initial plutôt que de le
modifier sans plan valide.

### Vérifier des exigences sans les réécrire

`verify_fixed_requirements` compare le nombre, l'ordre, le libellé exact et
l'identifiant de source des exigences retournées à celles reçues. Le vérificateur
ne peut pas raccourcir une exigence pour accepter un passage incomplet. Les requêtes
des lacunes sont reprises du plan de la question, pas de nouvelles inventions du
vérificateur.

`_parse_scoped_assessment` réutilise la validation de citations textuelles de
`app/evidence_coverage.py`, puis vérifie que l'article de la référence est celui
de l'exigence. Une citation exacte dans le mauvais article n'est pas un support.

La présence exacte d'un extrait est vérifiée localement. Son adéquation à
l'exigence reste un jugement LLM : une vraie citation peut être insuffisante ou
mal interprétée. Le code ne transforme pas cela en garantie de vérité.

Le prompt final demande une courte explication complète/partielle avant le choix
de citation et utilise un exemple fictif (clavier, gestes, voix), sans faits du
benchmark. Cette explication guide le juge, mais n'est pas une preuve formelle
ni un champ du rapport : les exigences et citations vérifiées sont conservées.
Les citations doivent avoir entre 8 et 600 caractères ; un extrait trop long
est rejeté plutôt que tronqué en une citation qui n'existe pas dans la source.

### Récupérer dans la réserve

`recover_candidate_evidence` :

1. Conserve les passages initiaux, dans leur ordre, avec leurs références stables.
2. Prend au maximum deux exigences sans preuve, en faisant un tour des articles
   avant de reprendre une seconde lacune du même article. L'ordre du plan est
   conservé au sein de chaque article. Ce partage empêche un premier article
   d'épuiser le budget des comparaisons. Avec trois articles en lacune, seuls
   les deux premiers sont traités : la limite reste explicite.
3. Exclut les chunks déjà sélectionnés et limite les candidats à l'article demandé.
4. Utilise le cross-encoder existant contre la requête de **chaque lacune**, pas
   contre la question entière. Retient jusqu'à trois propositions par lacune.
5. Vérifie une fois les exigences figées dans les propositions et le contexte
   initial. Ce n'est pas une nouvelle recherche ni un remplacement de top-k.
6. Ajoute uniquement les passages auxquels un extrait vérifié est associé.

Les références de la fenêtre temporaire du juge sont remappées vers les références
de la fenêtre finale par identifiant de chunk. Ainsi, un candidat temporairement
nommé S8 peut devenir S6 sans décaler les S1..S5 existants. Deux exigences peuvent
utiliser le même nouveau chunk sans l'ajouter deux fois.

Limites explicites :

- Deux lacunes traitées, trois propositions par lacune.
- Deux ajouts maximum, 16 000 caractères cumulés maximum pour ces ajouts.
- Les cinq passages initiaux ne sont jamais retirés ou réordonnés.
- Un seul passage dans le nœud de récupération, sans boucle.
- Un plan inconnu, une vérification malformée ou aucun support conserve le contexte.
- Une addition dépassant le budget n'est pas faite et sa lacune reste sans preuve.
- Les erreurs d'accès/quota API sont propagées par la passerelle existante.
- Le budget est un plafond de caractères ajoutés, pas une mesure exacte du total
  des tokens d'entrée. Les passages initiaux ne sont pas tronqués par ce module.

La sélection finale peut donc passer de cinq à six ou sept chunks. Le gain testé
combine une sélection ciblée et une augmentation **conditionnelle** de contexte :
il ne démontre pas à lui seul une supériorité sur une baseline prenant toujours
sept chunks. Cette dernière serait une ablation future distincte.

### Profil de génération expérimental

`format_coverage_plan` construit un bloc de données JSON liant exigences,
extraits et références vérifiés. `coverage_answer_plan` l'ajoute au prompt du
brouillon et de sa correction. Les exigences sans support ne sont pas présentées
comme faits. Ce profil a été évalué, mais ses moins bons résultats ont conduit à
ne pas l'activer comme comportement retenu.

## 5. Modifications des composants existants

### `app/agentic_rag.py`

Le state garde maintenant `candidate_pool`, avec un réducteur d'accumulation pour
les branches parallèles par source. Le retrieval global multifacette conserve ses
30 candidats, même lorsque cinq seulement sont sélectionnés. Les recherches
simples sans multifacettes gardent leur petite réserve telle quelle : elles ne
font pas secrètement une recherche supplémentaire.

Un nœud `recover_candidates` complète les preuves après le contrôle de couverture,
puis rejoint `plan_answer`. Il ne revient pas au contrôle initial et ne boucle pas.
Les anciennes routes de récupération globale sont conservées dans `coverage_retry`.
Le routage distingue les profils pour ne pas activer plusieurs modifications à la
fois. Le contrôle de langue à trois états, les citations et la limite d'une correction
de réponse de v8 restent actifs.

Les traces montrent les candidats considérés, les ajouts, le nombre de passages
préservés et le résultat de couverture. Les S-references du contexte initial
restent valides. La récupération peut modifier la réponse générée en ajoutant du
contexte : préserver les chunks n'est pas une garantie mathématique de non-régression.

### `app/schemas.py`

`CoverageRequirement` possède un `document_id` optionnel. `CandidateRecoveryAudit`
enregistre le déclenchement, les identifiants considérés/ajoutés, les lacunes et
un diagnostic. `AgenticExecution` expose cette structure et les contrôles de
couverture dans les réponses API et rapports. Les champs existants restent présents.

### `app/retrieval_policy.py`

Trois nouveaux profils indépendants des précédentes ablations :

- `coverage_diagnostic` : contrôle seul, sans modifier sélection ou génération.
- `candidate_recovery` : contrôle + récupération conditionnelle ; génération habituelle.
- `coverage_answer_plan` : même récupération + bloc de couverture dans le prompt.

Ils utilisent le classement v8 de question entière, sans activer `original_query`
ou l'ancien `coverage_retry`. Pas de nouvelle liste d'intentions ni de traitement
codé spécifiquement pour les deux identifiants de cas évalués.

### `app/rag_service.py`

Un argument optionnel `coverage_plan` est ajouté au générateur. Vide, il ne
modifie pas le prompt. Il n'est rempli que par le profil de génération expérimental.
La baseline `/ask` conserve son comportement ; la nouvelle orchestration concerne
`/agentic/ask`.

### Scripts d'évaluation

`scripts/evaluate_retrieval_iteration.py` est réutilisé sans nouveau script :
répétitions explicites des cibles, témoins une fois, scoring et diagnostic des
rangs effectués après la réponse. Les labels du benchmark ne sont pas transmis
aux nœuds de production. Son champ historique de pipeline reste `retrieval_ablation_v1` ;
les profils et empreintes de sources distinguent les campagnes.

`scripts/evaluate_agentic_rag.py` prend l'identifiant de pipeline
`langgraph_candidate_coverage_v9_openai_v3`, un nom de rapport par défaut v9 et
une description de la politique de contexte. Le top-k déclaré est la limite
initiale ; les ajouts conditionnels et le nombre réel de chunks sont aussi enregistrés.
Les checkpoints v8 ne sont pas repris comme résultats v9.

## 6. Tests sans réseau

`tests/test_candidate_recovery.py` utilise des articles fictifs, un corpus de neuf
millions d'exemples et des réponses API simulées. Il couvre : conservation des
chunks initiaux, extraits inventés, références inconnues, scope incorrect, exigences
changées ou omises, aucun candidat, plan malformé, question-only sans texte retrouvé,
doublons, remapping des références, budgets d'ajout, limites de lacunes, route
diagnostic indépendante et séparation du profil de génération.

Ces tests contrôlent les invariants de code, pas la pertinence réelle du juge LLM.
La suite complète sur les sources v9 compte **155 tests réussis** avant déploiement.
Les nouveaux tests couvrent aussi le partage du budget entre articles et le
refus de reprendre un checkpoint avec une empreinte de code différente. Les
nouveaux rapports complets enregistrent les empreintes SHA-256 des modules de
production ; les anciens rapports qui n'en possèdent pas gardent seulement les
contrôles historiques de pipeline, modèle, profil, top-k et benchmark.

## 7. Reproduire les évaluations ciblées

Depuis le dossier complet de l'application, avant d'intégrer les sources à l'image :

```powershell
docker compose run --rm --no-deps --volume "C:/Users/abden/Documents/Codex/2026-07-30/je/work/evidentia:/app:ro" api python -m scripts.evaluate_retrieval_iteration --profile candidate_recovery --case-id clip-data-scale-fr --case-id compare-clip-sam-prompts-en --repeats 3 --control-case-id sam-ambiguity-en --control-case-id llava-scienceqa-fr --control-case-id blip2-bottleneck-fr --control-case-id compare-blip2-llava-bridge-fr --control-case-id abstain-sam-mars-en --output /reports/nouveau-rapport-couverture.json
```

Changer le profil en `coverage_diagnostic` ou `coverage_answer_plan` isole la
présence du contrôle ou du bloc de génération. Toujours choisir un nouveau fichier
pour préserver les mesures précédentes. Les prompts finaux diffèrent des premières
variantes historiques : leurs anciens résultats ne doivent pas être présentés
comme exactement reproductibles avec le seul profil actuel.

## 8. Cadre méthodologique et limites

Les critères du benchmark et ses 30 cas n'ont pas été changés. Une réussite au seuil
de 50 % de groupes de termes n'atteste pas d'une réponse exhaustive. Sur les
répétitions CLIP, la couverture reste 50 % malgré la récupération des 400 millions :
le résultat cité est souvent 76,2 % sur ImageNet plutôt que la comparaison ResNet-50
attendue par les autres groupes de termes. Les réponses doivent être relues pour
vérifier quelle variante et quel résultat la source décrit.

Les mots singuliers/pluriels peuvent aussi changer le score sans changer le sens.
Le score lexical est conservé pour comparer les campagnes, mais ne remplace pas
une revue humaine. Le benchmark a servi plusieurs fois au développement : les
résultats sont in-sample, non une démonstration sur des questions inédites.

L'analyse des traces et des gains/pertes par cas est cadrée avec OpenAI Docs,
notamment [les bonnes pratiques d'évaluation](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
Les mesures ci-dessus viennent des rapports locaux, pas de ce guide. Aucune
utilisation de la plateforme Evals n'est nécessaire : les scripts locaux appellent
la passerelle Responses et calculent leur propre scoring.

## 9. Première validation complète et régression constatée

`agentic-rag-v9-candidate-coverage-30-20260930.json` contient une première campagne
complète avec le profil `candidate_recovery`, après la variante atomique et avant
la distinction structure/operation des connecteurs : **29/30**, contre 28/30 en v8.
Les deux échecs initiaux passent, mais la comparaison BLIP-2/LLaVA régresse :
couverture lexicale 0,4, contre 0,6 en v8. Les 27 autres cas restent réussis.
Ce résultat n'est donc pas une validation sans régression.

La question BLIP-2/LLaVA demande le mécanisme de liaison ; elle ne demande pas
explicitement de nommer les modèles. Une exigence de formats d'entrée y était
mal adaptée. La variante suivante distingue généralement la structure concrète
du connecteur de son operation sur les représentations. Sur trois répétitions de
chacune des trois questions fragiles, CLIP et BLIP-2/LLaVA passent 3/3 chacun,
mais CLIP/SAM échoue 3/3 : le juge accepte encore parfois un exemple de point ou
de texte comme preuve de tous les formats supportés. Les quatre témoins passent.

Coût de la première campagne complète : 112 appels LLM et 176 809 tokens cumulés,
contre 45 appels et 80 976 tokens pour v8. Latence moyenne observée : 11,65 s contre
6,10 s. Ce surcoût vient notamment des deux vérifications de couverture par
question, plus l'appel de récupération lorsqu'il est nécessaire. Le gain net ne
justifie pas à lui seul de masquer cette régression ni ce coût supplémentaire.

## 10. Validation finale et disponibilité

La variante finale ajoute un exemple générique au vérificateur pour distinguer
un exemple d'une interface complète, partage les deux lacunes entre articles,
et explicite la borne de 600 caractères des citations. Le budget ne croît pas.
La campagne ciblée finale passe **9/9 cibles et 4/4 témoins**. La campagne complète
`agentic-rag-v9-candidate-coverage-final-30-20260930.json` passe **29/30** :

- Les deux échecs initiaux passent ; les cinq comparaisons passent, y compris
  BLIP-2/LLaVA, qui avait régressé dans la première campagne complète v9.
- Les cinq abstentions attendues passent. Un contrat demeure invalide pour
  l'abstention CLIP/électricité (`premature_abstention`), comme en v8. Le cas
  est correctement abstenu mais marqué non publiable : passage au benchmark
  et validité de tous les contrats sont des notions distinctes.
- DINOv2/curation perd sa réussite au **score lexical** : le texte dit notamment
  « visually similar », « raw web-image source » et « using images rather than
  text or metadata », non reconnus par les groupes de termes attendus. Après
  lecture, c'est un faux négatif probable, pas une omission évidente du mécanisme.
  Nous ne le reclassons pas en réussite : le rapport reste à 29/30.

La couverture moyenne des groupes de termes passe de 82,3 % en v8 à 89,7 % ;
le rappel des passages cibles, de 81,3 % à 89,3 %. La récupération est tentée dans
15 cas et ajoute une preuve dans six cas. Le coût observé est de **118 appels et
207 623 tokens**, contre 45 appels et 80 976 tokens en v8. Latence moyenne observée
13,52 s ; ces campagnes ne sont pas un benchmark de performance contrôlé.

Le nouveau code est installé dans l'image, mais **`question_rerank` reste le
profil par défaut** : une non-régression stricte au score n'est pas démontrée et
le surcoût n'est pas négligeable. `candidate_recovery` est disponible en option.
La variante `coverage_answer_plan`, moins bonne, n'est pas activée. Les 155 tests
automatisés passent sur les sources et dans l'image construite, sans dépendances
modifiées ni réindexation. L'image précédente est conservée sous
`evidentia-app:pre-coverage-v9` ; aucun volume de modèles ou Qdrant n'est supprimé.

### Activer le profil depuis PowerShell, quand souhaité

Depuis le dossier du projet :

```powershell
$env:RAG_RETRIEVAL_PROFILE = "candidate_recovery"
docker compose up -d --no-deps --no-build api
Remove-Item Env:\RAG_RETRIEVAL_PROFILE
```

La variable est transmise à Compose pour recréer seulement l'API avec le nouveau
profil. `--no-build` utilise l'image déjà testée, `--no-deps` laisse Qdrant intact.
La dernière ligne retire la variable de la session PowerShell, pas du conteneur
déjà démarré. Un futur `compose up` sans variable reviendra au défaut ; pour un
choix persistant, renseigner uniquement `RAG_RETRIEVAL_PROFILE=candidate_recovery`
dans `.env`, sans recopier ni afficher les secrets existants.

Pour revenir au profil précédent, reprendre les trois lignes avec
`"question_rerank"`. Le retour de profil n'exige pas de reconstruire l'image.

### Reproduire la validation complète sans changer l'interface active

```powershell
docker compose run --rm --no-deps --env RAG_RETRIEVAL_PROFILE=candidate_recovery api python -m scripts.evaluate_agentic_rag --output /reports/nouvelle-validation-v9.json
```

Le conteneur d'évaluation temporaire appelle le modèle configuré, utilise Qdrant
existant et écrit un rapport dans le volume `reports`. Toujours choisir un nom
nouveau. `--resume` vérifie le modèle, le profil, le benchmark, le top-k et, pour
les nouveaux rapports, l'empreinte des modules Python de production.

La prochaine validation utile est une revue humaine explicite de DINOv2 et des
questions inédites réservées à l'évaluation. Il ne faut pas injecter les mots
attendus dans les prompts uniquement pour transformer un faux négatif en score
positif. Les résultats actuels restent in-sample.

## 11. Activation pour l'interface, demandée après la revue du cas DINOv2

Le 30 septembre 2026, l'utilisateur demande de déployer le profil amélioré après
la revue qualitative de la réponse DINOv2 dans la conversation. La réponse a été
comparée aux passages S1 et S3 : sélection d'images visuellement similaires à des
datasets curatés depuis une source web brute, déduplication, filtres et sélection
par exemples/clusters, sans texte ou métadonnées. Les informations attendues sont
présentes malgré les synonymes non reconnus par le score lexical.

Le bilan adopté est **29/30 automatiquement, 30/30 après validation qualitative
de ce cas**. Le rapport automatique original n'est pas modifié. Cette revue a
été réalisée par l'assistant dans la conversation, et le déploiement demandé par
l'utilisateur ; ce n'est pas une nouvelle campagne de 30 jugements humains
indépendants, ni une garantie sur des questions inédites.

`candidate_recovery` devient maintenant le défaut dans `app/retrieval_policy.py`
et le fallback persistant de `compose.yaml`. Aucun override de ce profil n'était
présent dans `.env`. La logique de récupération et les prompts sont inchangés
depuis la campagne complète ; seule la configuration de profil diffère dans les
modules de production. Aucune nouvelle évaluation complète n'est nécessaire pour
ce changement de configuration : la campagne précédente utilisait déjà ce profil
explicitement. L'image est reconstruite sans mise à jour de dépendances, puis
l'API seule est recréée ; Qdrant, articles et modèles restent intacts.

L'UI sélectionne déjà **LangGraph** par défaut et envoie les questions à
`POST /agentic/ask`. Cet endpoint lit le profil de l'API : il utilise donc désormais
la récupération améliorée, sans besoin de modifier le JavaScript. Le mode
**Baseline** continue volontairement à appeler `/ask`, sans cette récupération.
Les références et les textes complets des passages ajoutés utilisent le même
format que ceux des passages initiaux.

Les tests HTTP de comptage d'appels v8 sélectionnent dorénavant explicitement
`question_rerank`, plutôt que dépendre du défaut de déploiement. Un nouveau test
HTTP couvre le profil `candidate_recovery` avec décomposition, vérification et
génération simulées, et vérifie profil, citations, texte des preuves et trois
appels comptabilisés. La suite passe à **156 tests**. Ce sont des réponses API
simulées, non un changement des critères du benchmark.

Pour revenir au profil précédent, conserver un override explicite
`RAG_RETRIEVAL_PROFILE=question_rerank` lors de la recréation de l'API ; retirer
l'override revient maintenant à `candidate_recovery`, pas à v8. L'image de retour
`evidentia-app:pre-coverage-v9` reste disponible. Le surcoût documenté en section 10
reste applicable ; l'activation ne le supprime pas.

Vérification effective dans l'UI : question de comparaison CLIP/SAM, avec ces deux
articles sélectionnés et le mode LangGraph. La trace montre
`ranking=candidate_recovery`, puis `recover_candidates` avec six propositions,
un ajout et cinq passages préservés. La réponse est affichée, les six citations
sont présentes et le bouton S6 ouvre le texte entier du passage SAM ajouté,
notamment l'énumération des prompts sparse et dense. Les contrats finaux sont
validés après une correction de citations des faits planifiés. Les endpoints
`/health` et `/ready` sont vérifiés après la finalisation du déploiement.
