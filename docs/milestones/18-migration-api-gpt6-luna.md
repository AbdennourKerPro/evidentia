# Milestone 18 — Migration du LLM vers GPT‑6 Luna via API

Date : 30 septembre 2026.

## 1. Objectif et périmètre

Remplacer l'inférence locale Qwen/OpenVINO par GPT‑6 Luna via la Responses API.
Cette étape change uniquement le moteur de génération et ses paramètres
directement liés. Elle ne met pas encore en place la découverte arXiv, l'ingestion
à la demande ou la boucle de tool calling.

Tous les appels génératifs partagent le même gateway et la même configuration :
planification des recherches par article, rédaction et correction éventuelle.
Il n'y a pas de repli silencieux vers Qwen ou vers un autre modèle.

Les embeddings E5, BM25, la fusion RRF, le classement par intentions, le
cross-encoder lorsqu'il est utilisé, Qdrant et les contrats de réponse restent
locaux. Un modèle d'embedding ou un cross-encoder n'est pas un appel au LLM
génératif.

Les PDF, chunks, collections Qdrant, anciens rapports et anciens poids Qwen
n'ont pas été supprimés. Il n'est pas nécessaire de réindexer les articles.
Les résultats historiques obtenus avec Qwen doivent rester identifiés comme tels.

## 2. Rappel : API et inférence locale

Avant, le conteneur lisait les poids Qwen, chargeait le tokenizer et exécutait
OpenVINO sur le PC. Maintenant, il transmet les instructions, la question et
les passages nécessaires au service OpenAI, puis récupère une réponse.

L'interface, FastAPI et la recherche de preuves restent exécutés sur le PC.
La génération nécessite une connexion Internet et une clé API. Les questions
et passages transmis quittent donc la machine : ne pas y inclure de documents
confidentiels sans vérifier les règles de traitement des données applicables.

La clé reste côté serveur. Elle n'est jamais envoyée au navigateur, ni incluse
dans les réponses HTTP, les rapports d'évaluation ou les journaux du gateway.
Elle existe toutefois dans l'environnement du conteneur ; une personne ayant
accès à Docker peut inspecter cet environnement.

Les appels utilisent `store=False`. Cela ne constitue pas une garantie de
Zero Data Retention ni une suppression de tous les journaux du fournisseur.

## 3. Architecture conservée

### Baseline

1. FastAPI reçoit la question et les articles sélectionnés.
2. E5 transforme la question en vecteur.
3. Qdrant renvoie les chunks.
4. Le prompt contient les passages sous les références serveur S1, S2, etc.
5. GPT‑6 Luna rédige la réponse.
6. Le contrat déterministe vérifie notamment langue et références.
7. L'API renvoie réponse, citations, texte complet des chunks et métriques LLM.

En absence de preuve, la réponse peut être une abstention sans appel LLM.

### LangGraph

Le graphe existant conserve ses nœuds d'analyse, retrieval, sélection des
preuves, planification factuelle, génération, validation et correction.
Une comparaison peut déclencher un appel de planification des requêtes.
La rédaction et l'éventuelle correction utilisent ensuite le même modèle API.

Le plan factuel actuel est principalement déterministe : il ne devient pas un
nouvel appel API dans ce milestone. Les vérifications restent obligatoires.
Une référence valide ne garantit toujours pas, à elle seule, que chaque
affirmation est sémantiquement soutenue : cette limite ne change pas.

Le gateway ne permet pas encore au LLM d'exécuter des outils. La Responses API
prépare cette évolution, mais la prochaine étape devra définir les schémas et
l'exécuteur d'outils explicitement.

## 4. Configuration

Copier `.env.example` vers `.env` à la racine du projet et renseigner la clé
localement. Ne jamais partager ce fichier ni coller sa valeur dans le chat.

| Variable | Défaut | Rôle |
|---|---|---|
| OPENAI_API_KEY | vide | Secret obligatoire pour un appel génératif |
| OPENAI_MODEL | gpt-6-luna | Modèle commun à tous les appels |
| OPENAI_REASONING_EFFORT | low | Compromis initial pour l'expérimentation |
| OPENAI_TIMEOUT_SECONDS | 60 | Délai par tentative réseau |
| OPENAI_MAX_RETRIES | 1 | Une retransmission maximum gérée par le SDK |
| OPENAI_MAX_OUTPUT_TOKENS | 4096 | Plafond serveur de sortie par appel |

Les efforts acceptés sont `none`, `low`, `medium`, `high`, `xhigh` et `max`.
Le défaut reste volontairement `low`, pas l'effort implicite du fournisseur.
Pour expérimenter sans raisonnement interne, choisir `none` explicitement.

Le SDK n'envoie ni `temperature`, ni `top_p`, ni les anciens paramètres
OpenVINO. Le comportement API n'est pas présenté comme déterministe.

Compose lit `.env` et transmet seulement les variables déclarées au service
`api`. Le processus Python ne charge pas automatiquement un fichier `.env`
lorsqu'il est lancé hors Docker : il lit ses variables d'environnement.

Après modification de la clé ou des options, recréer le conteneur API :
un simple redémarrage ne remplace pas nécessairement son environnement.

Le timeout est par tentative. Une requête avec retransmission peut dépasser
60 secondes. Une question LangGraph peut également contenir plusieurs appels.

## 5. Budget de tokens : changement important

L'ancien `max_new_tokens` concernait la génération locale.
`max_output_tokens` limite la sortie API, raisonnement inclus ; ce n'est pas
une limite séparée sur les seuls mots visibles.

Les budgets demandés par les rôles sont :

| Rôle | Budget demandé |
|---|---|
| Planification des recherches | 2048 |
| Première réponse non comparative | 3072 |
| Réponse comparative | 4096 |
| Correction | 4096 |
| Test API isolé | 2048 |

Le budget effectif est le minimum entre ce budget et le plafond serveur.
Augmenter OPENAI_MAX_OUTPUT_TOKENS ne relève donc pas à lui seul un budget
plus petit défini par un rôle. Si un rôle manque de tokens, réduire l'effort
ou adapter son budget explicitement, puis mesurer.

Une sortie `incomplete` n'est jamais publiée comme réponse complète.
Le gateway signale une erreur explicite plutôt que de masquer une troncature.
Aucune nouvelle génération automatique n'est ajoutée pour ce cas, afin
d'éviter des appels facturés non maîtrisés.

## 6. Explication des fichiers modifiés ou ajoutés

### app/settings.py

Définit OpenAISettings, lit les variables et valide les valeurs numériques.
La représentation de cet objet exclut la clé. public_configuration() produit
uniquement les paramètres utilisables dans les rapports et comparaisons.

Le chemin de l'ancien modèle subsiste pour le downloader historique, mais
aucun appel génératif actif ne le consulte.

### app/llm_gateway.py

C'est l'unique frontière avec OpenAI.

- _read_settings() transforme les erreurs de configuration en erreurs contrôlées.
- _get_client() réutilise le client HTTP, applique timeout et limite de retries.
- generate_chat() envoie les instructions et le message utilisateur à Responses.
- _translate_api_error() masque le corps brut des erreurs du fournisseur.
- capture_llm_calls() ouvre une collecte propre à une question ou un cas.
- _record_call() conserve durée et usage, sans question, preuves ni réponse.
- summarize_llm_calls() agrège les appels effectivement observés.

ContextVar évite de mélanger les métriques de deux requêtes. Les tests vérifient
également la propagation de cette collecte dans le graphe LangGraph réel.

### app/rag_service.py et app/agentic_rag.py

Adaptent les budgets des appels au nouveau gateway. Les prompts, détection
des intentions, sélection des preuves et politiques de validation ne sont pas
réécrits ici. Cela permet d'isoler l'effet du changement de modèle.

### app/schemas.py et app/main.py

Le statut LLM décrit maintenant une configuration API, pas des fichiers locaux.
Les réponses /ask et /agentic/ask ajoutent un champ llm avec les métriques.
Un gestionnaire commun transforme les erreurs LLM en réponses HTTP structurées.

La propriété configured indique qu'une clé non vide est présente, pas qu'elle
est valide. access_verified reste false sur /llm/status : ce endpoint ne teste
jamais l'accès distant, même après un smoke test réussi.

### Chargement différé des bibliothèques de retrieval

app/embeddings.py, app/hybrid_retrieval.py et app/document_ingestion.py
importent leurs bibliothèques lourdes au moment où elles sont utilisées.
Cela permet de vérifier l'API et de tester les contrats sans charger Docling,
PyTorch ou un modèle. Les opérations de retrieval et conversion conservent
leurs implémentations.

### Docker et secrets

requirements.txt ajoute le SDK openai 2.54.0 et retire les trois dépendances
OpenVINO. Le Dockerfile ne configure plus le linker natif OpenVINO.

compose.yaml transmet la configuration OpenAI et conserve les volumes.
model_cache reste nécessaire pour les embeddings et le reranker. Les anciens
poids Qwen éventuellement présents sont conservés, mais ne sont plus chargés.

.gitignore protège .env et ses variantes. .dockerignore les exclut du contexte
de build. La documentation de ce milestone et le template vide peuvent être
versionnés. Les autres milestones historiques ne sont pas réécrits.

scripts/download_openvino_model.py est conservé comme commande historique
autonome. Il n'importe plus le gateway actif et n'est plus nécessaire.

### Interface

La bibliothèque affiche le modèle API configuré et un état de configuration.
Chaque réponse peut afficher le nombre d'appels LLM et de tokens.
Les citations et le texte complet des chunks restent accessibles.

Cette migration n'ajoute pas de mémoire conversationnelle côté modèle :
l'interface conserve son fonctionnement actuel, avec questions indépendantes.

### scripts/check_openai.py

Sans option, vérifie uniquement la configuration : aucun réseau ni facturation.
Avec --live, demande une petite réponse réelle et affiche son usage.
C'est un test du gateway, pas une évaluation du RAG ou de Qdrant.
Un appel logique peut être retransmis selon OPENAI_MAX_RETRIES.

### Scripts d'évaluation

evaluate_rag.py et evaluate_agentic_rag.py passent par le même gateway.
Ils enregistrent configuration et usage par cas dans de nouveaux fichiers :

- /reports/rag-evaluation-gpt6-luna.json ;
- /reports/agentic-rag-evaluation-gpt6-luna.json.

Le pipeline agentique est identifié par
`langgraph_fact_plan_intent_multifacet_response_contract_v6_openai_v1`.

La reprise refuse un ancien pipeline, une autre configuration LLM, un autre
top-k ou un benchmark dont le hash a changé. Cela évite de mélanger Qwen et
Luna, ou plusieurs paramètres, dans un même rapport.

Aucune campagne du benchmark n'a été exécutée pour ce milestone.

### Fichiers de tests

- tests/openai_fixtures.py fournit les réponses du fournisseur et le client SDK
  avec un transport HTTP en mémoire : aucune connexion réelle n'est possible.
- tests/test_llm_gateway.py vérifie paramètres, configuration, erreurs,
  confidentialité des métadonnées et limites de tokens.
- tests/test_api_llm.py teste les endpoints FastAPI et le vrai graphe LangGraph,
  en simulant seulement OpenAI et le retrieval.
- tests/test_check_openai.py vérifie que le script reste hors ligne par défaut
  et que --live déclenche l'appel explicitement.
- tests/test_evaluation_checkpoint.py couvre la compatibilité de reprise :
  pipeline, configuration, top-k et contenu du benchmark.

## 7. Métriques et limites de leur interprétation

Chaque appel expose :

- identifiant et modèle retournés par le fournisseur ;
- statut et durée ;
- budget de sortie ;
- tokens d'entrée, de sortie et total ;
- tokens de raisonnement et entrée en cache lorsque disponibles.

Les tokens de raisonnement sont inclus dans les tokens de sortie : ne pas les
additionner une seconde fois. Les tokens d'entrée en cache sont une partie de
l'entrée, pas des tokens supplémentaires.

Une utilisation absente reste null, pas zéro. Un vrai cas sans appel a zéro
appel et zéro token. Les retransmissions internes au SDK ne sont pas listées
individuellement ; la durée les inclut mais la télémétrie n'est pas une facture
exhaustive. Aucun prix en dollars n'est figé dans le code. Pour les coûts réels,
utiliser le suivi de consommation du fournisseur.

## 8. Erreurs HTTP

| Code applicatif | HTTP | Interprétation |
|---|---|---|
| missing_api_key | 503 | Clé absente |
| invalid_llm_configuration | 503 | Option invalide |
| invalid_api_key | 503 | Clé rejetée |
| model_access_denied / model_unavailable | 503 | Modèle inaccessible |
| upstream_rate_limit | 503 | Quota ou débit OpenAI |
| llm_timeout | 504 | Délai dépassé |
| llm_connection_error | 502 | Échec de connexion |
| incomplete_llm_response | 502 | Sortie incomplète |
| empty_llm_response / llm_refusal / failed_llm_response | 502 | Pas de texte exploitable |
| upstream_api_error | 502 | Autre échec distant |

Les réponses incluent detail et code. Elles n'incluent pas le corps brut des
erreurs API. Une défaillance technique n'est pas présentée comme une abstention
scientifique due à un manque de preuves.

## 9. Installation et premier test, à faire localement

Depuis PowerShell :

```powershell
Set-Location "C:\Users\abden\Documents\Codex\2026-07-30\je\work\evidentia"
if (!(Test-Path -LiteralPath .env)) {
    Copy-Item -LiteralPath .env.example -Destination .env
}
notepad .env
```

Renseigner OPENAI_API_KEY, enregistrer, puis démarrer Docker Desktop.
Ne pas coller cette clé dans le chat. Ne pas afficher `docker compose config`
sans précaution : sa sortie interpolée peut contenir la clé.

```powershell
docker compose up -d --build api
docker compose run --rm --no-deps api python -m scripts.check_openai
docker compose run --rm --no-deps api python -m scripts.check_openai --live
```

Seule la dernière commande fait un appel LLM réel. Puis ouvrir l'interface
http://localhost:8000/ui/ et poser, par exemple :

> Selon l'article CLIP, comment les représentations d'image et de texte
> sont-elles utilisées pour la classification zero-shot ?

Les articles indexés précédemment doivent être conservés dans Qdrant.
Le test isolé ne vérifie pas leur présence ; une réponse dans l'interface
vérifie aussi le parcours de retrieval actuel.

Pour les tests automatisés, qui n'appellent pas OpenAI :

```powershell
docker compose run --rm --no-deps api python -m unittest discover -s tests -v
```

L'évaluation complète sera une action distincte, après validation du premier
appel réel et du comportement sur quelques questions. Elle implique des coûts
API et ne sera pas lancée automatiquement.

## 10. Vérification effectuée et limites restantes

79 tests passent avec Python 3.12 et les dépendances légères isolées, dont
46 tests déjà présents et 33 nouveaux contrôles/tests de migration.
Le SDK réel utilise un transport HTTP en mémoire, sans accès à OpenAI.
Cela vérifie la sérialisation des requêtes, les erreurs, les métriques,
la baseline, les citations/chunks et la boucle de correction LangGraph.

La syntaxe JavaScript et les 39 fichiers source Python ont aussi été vérifiés.
La configuration Compose a été validée avec le template sans secret.
Le script de configuration exécuté en mode hors ligne confirme le modèle et
signale l'absence de clé.

Docker n'était pas démarré dans l'environnement de travail : l'image finale
n'a pas été reconstruite ni démarrée ici. Aucun appel réel n'a vérifié la
validité d'une clé, l'accès du compte à gpt-6-luna, la qualité des réponses ou
le gain de latence réel. Ces points restent à vérifier localement.

## 11. Prochaine étape

Après le smoke test et une mesure de référence avec Luna, introduire le client
de découverte arXiv. Ne pas modifier simultanément le moteur, le retrieval,
l'ingestion et les prompts : les gains deviendraient impossibles à attribuer.

## Sources officielles consultées

- [GPT‑6 Luna : modèle et capacités](https://developers.openai.com/api/docs/models/gpt-6-luna).
- [Migration GPT‑6 : paramètres et Responses API](https://developers.openai.com/api/docs/guides/latest-model).
- [Raisonnement : budgets et réponses incomplètes](https://developers.openai.com/api/docs/guides/reasoning).
- [SDK Python et bibliothèques](https://developers.openai.com/api/docs/libraries).
