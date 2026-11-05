# Plan — Acquisition dynamique d'articles arXiv par tool calling

## Contexte

Evidentia répond aujourd'hui sur un corpus figé de 5 articles (CLIP, DINOv2, SAM,
BLIP-2, LLaVA), préparés à la main (`scripts/download_arxiv_papers.py` →
`POST /documents/{id}/convert` → `POST /documents/{id}/index`) et décrits en dur
dans `DOCUMENT_CATALOG` ([app/agentic_rag.py:81](app/agentic_rag.py#L81)).
Objectif : étendre l'outil à n'importe quel article arXiv **sans tout télécharger**.
Un agent à appels d'outils (OpenAI Responses API, comme le reste du projet) cherche
sur arXiv les articles utiles à la question, les ingère (PDF → Docling → chunks →
Qdrant) et les rattache à la **conversation**. Ils restent disponibles tant que la
conversation existe, et disparaissent quand l'utilisateur la supprime.

Décisions validées avec l'utilisateur :
- **Acquisition automatique** par l'agent, avec un budget borné ; l'UI montre les
  articles ajoutés et permet d'en retirer.
- **Tout devient dynamique** : une nouvelle conversation part d'un corpus vide. Les
  5 articles historiques deviennent de simples articles « en cache » (épinglés) ; le
  benchmark les utilise via un périmètre fixe explicite.
- **Streaming SSE** de la progression (recherche, téléchargement, conversion,
  indexation, réponse).
- **Mémoire minimale** : l'agent d'acquisition voit les 3 dernières questions et la
  liste des articles de la conversation ; le RAG lui-même reste mono-question.

Contrainte du projet à respecter : les profils de récupération et les résultats
du benchmark doivent rester **reproductibles à l'identique** (cf. README §5.3).
Le pipeline LangGraph existant n'est donc pas réécrit : on ajoute une étape
d'acquisition **en amont**, et on rend le catalogue et le périmètre paramétrables.

---

## Architecture cible

```text
UI (conversation id = UUID localStorage)
  │  POST /conversations/{id}/ask   (SSE)
  ▼
conversation_service.ask_stream()
  ├─ 1. source_discovery.run_discovery()   ← boucle tool calling (≤ 6 appels LLM)
  │      outils : search_arxiv, lookup_arxiv_ids, add_paper_to_conversation,
  │               finish_source_selection
  │      add_paper → paper_ingestion.ingest_arxiv_paper()  (cache partagé)
  ├─ 2. catalogue de la conversation (registre SQLite) → périmètre = ses document_ids
  └─ 3. answer_agentic_question(..., corpus_document_ids, catalog, focus_document_ids)
          (graphe LangGraph existant, inchangé dans sa logique)
```

### Modèle de stockage

- **Cache d'articles partagé** (clé = `arxiv_id` + version) : un article converti
  pour une conversation est réutilisé instantanément par une autre. Pas de
  re-conversion Docling.
- **Une seule collection Qdrant** : on garde `arxiv_chunks_e5_small`. Les nouveaux
  articles ont `document_id = "arxiv-<id><version>"` (ex. `arxiv-2304.02643v1`) ;
  les 5 historiques gardent leurs ids (`clip-2021`…), donc aucune réindexation.
- **Isolation par conversation** : toute recherche est filtrée par la liste des
  `document_id` de la conversation (filtre déjà supporté par
  `_document_filter` dans [app/qdrant_gateway.py:208](app/qdrant_gateway.py#L208)).
  **Règle absolue : plus aucune recherche avec `document_ids=None`** dans les
  chemins conversationnels, sinon fuite entre conversations.
- **Registre SQLite** (stdlib `sqlite3`, WAL, `/data/state/evidentia.sqlite3`) :
  qui possède quoi, état d'ingestion, mémoire minimale.
- **Garbage collection** : un article non épinglé qui n'est plus rattaché à aucune
  conversation est supprimé (points Qdrant, artefacts, PDF, ligne du registre).

---

## Les outils exposés au LLM (contrat précis)

Tous déclarés au format Responses API :
`{"type": "function", "name": ..., "description": ..., "parameters": {...}, "strict": true}`
(avec `strict: true`, tous les champs sont `required` et les optionnels sont
`["type", "null"]` ; `additionalProperties: false` partout).
Les sorties sont du JSON sérialisé renvoyé comme `function_call_output`. **Les
erreurs métier ne lèvent pas d'exception** : elles reviennent comme
`{"ok": false, "error": "<code>", "message": "..."}` pour que le modèle s'adapte.

### 1. `search_arxiv`
Recherche plein texte dans l'API arXiv (`http://export.arxiv.org/api/query`).

| Paramètre | Type | Contrainte |
|---|---|---|
| `query` | string | 3–200 caractères, en anglais ; mots-clés, titre ou nom de méthode |
| `max_results` | integer | 1–8 |
| `category` | string \| null | ex. `cs.CV` ; validé par regex `^[a-z\-]+(\.[A-Z]{2})?$` |
| `sort_by` | `"relevance"` \| `"submittedDate"` | |

Construction de la requête arXiv : `search_query=all:"<query>"` (+ `AND cat:<category>`),
`sortBy`, `max_results`. Sortie :
```json
{"ok": true, "results": [
  {"arxiv_id": "2304.02643", "latest_version": "v4", "title": "...",
   "authors": ["A. Kirillov", "E. Mintun", "N. Ravi"], "published": "2023-04-05",
   "primary_category": "cs.CV", "abstract": "<≤ 600 caractères>",
   "in_conversation": false, "cached": true}
]}
```
`in_conversation` / `cached` viennent du registre (rattachement par `arxiv_id` sans
version). Le résumé est marqué comme donnée non fiable dans le prompt système.

### 2. `lookup_arxiv_ids`
Métadonnées exactes pour des identifiants connus (l'utilisateur cite
« 2304.02643 », ou l'agent veut vérifier une version).

| Paramètre | Type | Contrainte |
|---|---|---|
| `arxiv_ids` | array[string] | 1–5 éléments, regex `^\d{4}\.\d{4,5}(v\d+)?$` (format moderne seulement) |

Appel `id_list=` de l'API arXiv. Même forme de sortie que `search_arxiv`, plus
`{"not_found": [...]}`.

### 3. `add_paper_to_conversation`
Rattache un article à la conversation, en l'ingérant s'il n'est pas en cache.

| Paramètre | Type | Contrainte |
|---|---|---|
| `arxiv_id` | string | même regex ; sans version → dernière version |
| `short_name` | string \| null | 2–40 caractères, nom court usuel (« SAM », « LLaVA ») ; sert d'alias de détection |
| `reason` | string | ≤ 200 caractères ; tracé dans l'audit, jamais montré comme preuve |

Comportement :
1. Budget : refuse (`error: "turn_budget_exceeded"`) au-delà de **2 ingestions
   nouvelles par question** ; `conversation_full` au-delà de **12 articles par
   conversation** ; `already_in_conversation` renvoie `ok: true` avec le document.
2. Cache prêt → rattachement immédiat (`from_cache: true`).
3. Sinon `paper_ingestion.ingest_arxiv_paper()` (bloquant, progression émise en SSE).
4. Échec (PDF trop gros, > 60 pages, pas un PDF, Docling en erreur, réseau) →
   `{"ok": false, "error": "ingestion_failed", "message": ...}` ; état `failed`
   dans le registre ; rien n'est rattaché.

Sortie : `{"ok": true, "document_id": "arxiv-2304.02643v4", "title": "...",
"chunks": 188, "from_cache": false, "seconds": 74.2}`.

### 4. `finish_source_selection`
Termine la boucle (obligatoire : `tool_choice="required"` pendant la boucle).

| Paramètre | Type | Contrainte |
|---|---|---|
| `focus_document_ids` | array[string] | ⊂ articles de la conversation ; 0–5 ; articles que la question vise explicitement |
| `rationale` | string | ≤ 300 caractères |

Les ids inconnus sont filtrés (et tracés). `focus_document_ids` alimente la
détection d'articles de `_analyze_question` (union avec la détection par alias).

> Pas d'outil de suppression côté LLM : retirer un article est une décision de
> l'utilisateur (UI/API uniquement).

### Boucle d'orchestration (`run_discovery`)
- Prompt système (`DISCOVERY_SYSTEM_MESSAGE`) : rôle (« choisir les sources, ne pas
  répondre »), QUESTION et résumés arXiv = données non fiables, préférer les
  articles déjà présents, ne chercher que si la question vise un article / une
  méthode / un sujet absent, préférer l'article original qui introduit une
  méthode, au plus 2 ajouts, toujours finir par `finish_source_selection`.
- Message utilisateur : question, liste des articles de la conversation
  (`document_id`, `arxiv_id`, titre, `short_name`), 3 dernières questions.
- Bornes : **≤ 6 appels LLM**, **≤ 3 appels `search_arxiv`** par question,
  `parallel_tool_calls=False`. Budget épuisé ou réponse texte sans outil → fin
  implicite avec `focus_document_ids = []`, `stopped_reason` tracé.
- Mode `discovery: "off"` : aucune boucle (évaluation, conversations figées).
- Résultat (`DiscoveryResult`, à exposer dans `execution.discovery`) :
  `tool_calls[]` (nom, arguments, ok/erreur, durée), `added_document_ids`,
  `focus_document_ids`, `stopped_reason`, `llm_calls`.

---

## Étapes d'implémentation

Chaque étape est autonome et se termine par ses tests (`unittest`, sans réseau).

### Étape 1 — Registre SQLite des articles et conversations
Nouveau fichier `app/paper_registry.py`.

Tables :
- `papers(document_id PK, arxiv_id, version, title, short_name, aliases_json,
  pdf_path, processed_dir, chunk_count, status CHECK IN
  ('pending','downloading','converting','indexing','ready','failed'),
  error, pinned INTEGER, created_at, updated_at)` ; index unique `(arxiv_id, version)`.
- `conversations(id PK, created_at, last_active_at)`.
- `conversation_papers(conversation_id, document_id, added_at, added_by
  CHECK IN ('agent','user','seed'), reason, PRIMARY KEY(conversation_id, document_id))`.
- `conversation_turns(conversation_id, position, question, created_at)` (mémoire
  minimale : on n'en lit que les 3 dernières).

API (fonctions pures, une connexion par appel, `PRAGMA journal_mode=WAL`,
`foreign_keys=ON`) : `init_db()`, `upsert_paper`, `set_paper_status`,
`get_paper(document_id)`, `find_paper(arxiv_id, version|None)`,
`ensure_conversation(id)`, `attach_paper`, `detach_paper`,
`list_conversation_papers(id)`, `record_turn`, `recent_questions(id, n=3)`,
`delete_conversation(id) -> list[str]` (documents devenus orphelins),
`orphan_documents()`, `stale_conversations(older_than)`.

Validation : `conversation_id` = UUID (regex), sinon 422 côté API.

**Seed des 5 articles historiques** : `seed_legacy_papers()` lit
`data/sources/arxiv_papers.json` + `DOCUMENT_CATALOG` actuel et insère les 5 lignes
`status='ready', pinned=1`, avec **exactement** les titres et alias regex
actuels (`aliases_json` stocke les regex telles quelles, ex. `\bblip[ -]?2\b`).
Appelé au démarrage (`@app.on_event("startup")` / lifespan), idempotent.

Paramètres (`app/settings.py`) : `STATE_DB_PATH` (défaut
`/data/state/evidentia.sqlite3`), `ARXIV_CACHE_DIR` (défaut `/data/cache`),
`CONVERSATION_TTL_DAYS` (30), `MAX_PAPERS_PER_CONVERSATION` (12),
`MAX_NEW_PAPERS_PER_TURN` (2), `ARXIV_MAX_PDF_MB` (40), `ARXIV_MAX_PAGES` (60).

Tests : `tests/test_paper_registry.py` (DB temporaire : rattachement, orphelins,
épinglés jamais orphelins, seed idempotent, alias identiques à l'ancien catalogue).

### Étape 2 — Client arXiv
Nouveau fichier `app/arxiv_client.py`.
- `search(query, max_results, category, sort_by) -> list[ArxivPaperMeta]` et
  `lookup(ids) -> list[ArxivPaperMeta]` via `httpx` (déjà présent via `openai`),
  User-Agent `Evidentia/0.1 (local research RAG prototype)` (même valeur que le
  script), timeout 20 s.
- Parsing Atom avec `xml.etree.ElementTree` (namespaces `atom`, `arxiv`) ; extraire
  id + version depuis `<id>http://arxiv.org/abs/2304.02643v4</id>`.
- **Politesse arXiv** : verrou global + au moins 3 s entre deux requêtes API ;
  cache mémoire TTL 1 h des recherches.
- `download_pdf(arxiv_id, version, destination)` : URL
  `https://arxiv.org/pdf/<id><version>`, écriture atomique `.part`, taille max
  `ARXIV_MAX_PDF_MB` (couper le flux), contrôle `%PDF`, SHA-256.
- **Réutilisation** : déplacer `download_pdf`, `ensure_pdf`, `sha256` de
  [scripts/download_arxiv_papers.py](scripts/download_arxiv_papers.py) dans ce
  module et faire importer le script depuis `app.arxiv_client` (le script reste
  sans dépendance externe : garder une version `urllib` ou accepter `httpx` —
  choisir de garder `urllib` dans les helpers partagés pour ne rien casser côté hôte).

Tests : `tests/test_arxiv_client.py` avec `httpx.MockTransport` (flux Atom réel
enregistré dans le test, id sans résultat, PDF non-PDF, PDF trop gros).

### Étape 3 — Ingestion générique d'un article
Refactor de [app/document_ingestion.py](app/document_ingestion.py) + nouveau
`app/paper_ingestion.py`.

Dans `document_ingestion.py`, extraire sans changer le comportement des 5 articles :
- `convert_pdf(pdf_path, output_dir, source_metadata, *, fast=False) -> DoclingDocument`
  (`convert_source` devient un wrapper ; `fast=True` = `PdfPipelineOptions(do_ocr=False)`
  et `max_num_pages=ARXIV_MAX_PAGES` pour les nouveaux articles seulement).
- `build_chunks(document_id, title, document) -> list[EvidenceChunk]` (wrapper de
  `_build_evidence_chunks`, réutilise `_get_chunker`, `_stable_chunk_id`,
  `_page_number`, `_section_name`).

`paper_ingestion.ingest_arxiv_paper(arxiv_id, version|None, short_name|None,
on_progress: Callable[[str, dict], None]) -> PaperRecord` :
1. `lookup` → version définitive, titre ; `document_id = f"arxiv-{id}{version}"`.
2. Si `find_paper` → `ready` : retour immédiat.
3. Verrou par `document_id` (`dict[str, threading.Lock]`) pour qu'une double demande
   concurrente n'ingère qu'une fois ; relire le registre après acquisition.
4. `downloading` → `download_pdf` vers `ARXIV_CACHE_DIR/raw/<document_id>.pdf`.
5. `converting` → `convert_pdf(..., fast=True)` vers
   `ARXIV_CACHE_DIR/processed/<document_id>/` (document.md, document.json,
   conversion-manifest.json, chunks.json — mêmes artefacts qu'aujourd'hui).
6. `indexing` → `embed_passages` ([app/embeddings.py](app/embeddings.py)) +
   `index_chunks(..., collection_name=ARXIV_COLLECTION_NAME)` + `clear_bm25_cache()`.
7. Alias : `short_name` normalisé (`_normalize` de agentic_rag, à déplacer dans un
   petit module partagé `app/text_normalization.py`) → `\b<re.escape>\b` ; plus le
   préfixe du titre avant « : » s'il fait ≤ 30 caractères. Pas d'alias de 1–2 lettres.
8. `ready` ; toute exception → `failed` + message sûr, artefacts partiels supprimés.

`on_progress(stage, payload)` est appelé à chaque étape (alimente le SSE).

Dans [app/qdrant_gateway.py](app/qdrant_gateway.py) :
- `delete_document_chunks(collection_name, document_id)` (filtre `document_id`).
- `_ensure_collection` crée aussi un index de payload `keyword` sur `document_id`
  (`create_payload_index`, idempotent ; à appeler aussi sur une collection existante).

Dans [app/hybrid_retrieval.py:130](app/hybrid_retrieval.py#L130) : passer
`get_bm25_index` à `maxsize=64`.

Tests : `tests/test_paper_ingestion.py` (Docling, embeddings et Qdrant simulés :
transitions d'états, cache hit, verrou, échec → `failed` sans rattachement).

### Étape 4 — Catalogue dynamique et périmètre obligatoire dans le pipeline
Objectif : brancher le graphe sur le registre **sans changer les résultats** quand
le catalogue = les 5 articles et qu'aucun focus n'est fourni.

[app/agentic_rag.py](app/agentic_rag.py) :
- Nouveau type `CatalogEntry = TypedDict(title: str, aliases: tuple[str, ...])`.
- `answer_agentic_question(..., corpus_document_ids: list[str] | None = None,
  catalog: dict[str, CatalogEntry] | None = None, focus_document_ids: list[str] = ())`.
  Par défaut (`None`) : catalogue et périmètre = les 5 articles épinglés
  (`legacy_catalog()` lu depuis le registre, repli sur la constante actuelle si
  la DB est absente — garder `DOCUMENT_CATALOG` comme constante de seed).
- Nouveaux champs d'état : `catalog`, `corpus_document_ids` (périmètre dur),
  `focus_document_ids`. **Ne pas** fusionner `corpus_document_ids` dans
  `selected_document_ids` : `_analyze_question` utilise `selected_ids` pour router
  les comparaisons (`2 <= len(selected_ids) <= 5`), ce qui changerait le routage.
- Remplacer chaque usage de `DOCUMENT_CATALOG` par `state["catalog"]` :
  `_analyze_question` (L323-330), `_retrieve_global` (L370),
  `_build_source_query` (L1015), `_plan_retrieval_queries` (L1035). Passer le
  catalogue en argument à ces deux helpers.
- `_analyze_question` : `detected = alias_matches ∪ focus_document_ids` (ordre :
  ordre du catalogue, puis focus), restreint à `allowed_ids`.
- Helper `_search_scope(state) -> list[str]` = `selected_document_ids or
  corpus_document_ids`. L'utiliser dans `_retrieve_global` (L434),
  `fallback_search` de `_recover_candidates` (L732) et `_coverage_document_ids`
  (L693). Jamais `None`.
- Court-circuit : si `corpus_document_ids == []`, renvoyer directement une
  abstention (« Aucun article n'est encore rattaché à cette conversation »)
  sans appel LLM ni Qdrant (nœud ou garde dans `answer_agentic_question`).

[app/rag_service.py:57](app/rag_service.py#L57) (`answer_question`, baseline) :
même paramètre `corpus_document_ids`, filtre jamais `None`.

[app/main.py](app/main.py) : `/ask`, `/agentic/ask`, `/arxiv/search`,
`/arxiv/documents` gardent leur contrat mais sont limités au corpus épinglé
(`/arxiv/documents` liste les articles épinglés ; documenter qu'ils sont les
endpoints « corpus historique / évaluation »).

Évaluation ([scripts/rag_evaluation_common.py](scripts/rag_evaluation_common.py),
[scripts/evaluate_agentic_rag.py](scripts/evaluate_agentic_rag.py) et les autres
`evaluate_*`) : appeler avec le périmètre épinglé explicite et `discovery` absent ;
`scope_document_ids` reste l'équivalent de `selected_document_ids`. Vérifier que
l'empreinte de code du rapport (`--resume`) change bien mais pas les sorties.

Tests : `tests/test_dynamic_catalog.py` — parité : pour un échantillon de questions
du benchmark (EN30 + heldout), `_analyze_question` avec `legacy_catalog()` donne
exactement la même `strategy` et les mêmes `detected_document_ids` que l'ancienne
constante ; focus fourni → union ; périmètre vide → abstention sans appel ;
aucune recherche sans filtre (mock de `search_chunks` qui échoue si
`document_ids is None`).

### Étape 5 — Tool calling dans la passerelle LLM + agent de découverte
[app/llm_gateway.py](app/llm_gateway.py) : nouvelle fonction
`run_tool_loop(*, system_message, user_message, tools, execute_tool,
max_model_calls, max_output_tokens=2048, on_event=None) -> ToolLoopResult`.
- Même client, mêmes réglages et même traduction d'erreurs que `generate_chat` ;
  chaque appel passe par `_record_call` (donc compté dans `execution.llm`).
- `client.responses.create(model, instructions, input=items, tools=tools,
  tool_choice="required", parallel_tool_calls=False, reasoning, max_output_tokens,
  store=False)`.
- Avec `store=False`, rejouer l'historique : `items += response.output` (tous les
  items, y compris `reasoning` éventuels) puis, pour chaque item
  `type == "function_call"`, `items.append({"type": "function_call_output",
  "call_id": item.call_id, "output": json.dumps(result)})`.
- Arguments : `json.loads(item.arguments)` ; JSON invalide → sortie
  `{"ok": false, "error": "invalid_arguments"}` (pas d'exception).
- Arrêt : `execute_tool` signale la fin (`finish_source_selection`), ou budget
  d'appels atteint, ou réponse sans `function_call`.
- `ToolLoopResult(final_call: dict | None, calls: list[ToolCallRecord],
  stopped_reason: Literal["finished","budget","no_tool_call"])`.

Nouveau `app/source_discovery.py` : schémas JSON des 4 outils (section ci-dessus),
`DISCOVERY_SYSTEM_MESSAGE`, `run_discovery(conversation_id, question,
on_event) -> DiscoveryResult`, exécuteurs d'outils (appel `arxiv_client`,
`paper_registry`, `paper_ingestion`), compteurs de budget par tour.

[app/schemas.py](app/schemas.py) : `ArxivPaperMeta`, `ConversationPaper`,
`DiscoveryToolCall`, `DiscoveryAudit` ; ajout de
`discovery: DiscoveryAudit | None = None` dans `AgenticExecution`.

[tests/openai_fixtures.py](tests/openai_fixtures.py) : ajouter
`function_call_body(name, arguments, call_id)` (forme Responses réelle :
`{"type":"function_call","id":..,"call_id":..,"name":..,"arguments":"<json>","status":"completed"}`).

Tests : `tests/test_tool_loop.py` (séquence search → add → finish ; JSON
invalide ; budget ; réponse texte ; vérifie que les `function_call_output`
renvoyés ont le bon `call_id`) et `tests/test_source_discovery.py` (budget de 2
ajouts, `conversation_full`, focus filtré, `in_conversation` / `cached`).

### Étape 6 — Service de conversation, API et SSE
Nouveau `app/conversation_service.py` :
`ask_stream(conversation_id, question, limit, pipeline, discovery) -> Iterator[str]`
1. `ensure_conversation`, `record_turn`.
2. Si `discovery == "auto"` : `run_discovery(...)`, chaque événement → SSE.
3. Catalogue de la conversation depuis le registre (seulement les `ready`).
4. `answer_agentic_question(corpus_document_ids=..., catalog=...,
   focus_document_ids=...)` (ou baseline), dans `capture_llm_calls()` englobant
   **aussi** la découverte pour un total de tokens unique.
5. Événement final `answer` = `AgenticAskResponse` complet (avec `discovery`).

Exécution : le travail tourne dans un thread (`contextvars.copy_context().run`,
indispensable pour `capture_llm_calls` qui repose sur un `ContextVar`) et pousse
dans une `queue.Queue` ; le générateur lit la file et formate
`event: <type>\ndata: <json>\n\n`. Si le client se déconnecte, l'ingestion
continue (le résultat reste en cache) ; un keep-alive `: ping` toutes les 15 s.

Événements SSE : `discovery.started`, `discovery.tool_call` {name, arguments},
`discovery.tool_result` {name, ok, résumé}, `ingestion.progress` {document_id,
stage}, `paper.added` {ConversationPaper}, `answer.started`, `answer`
{payload}, `error` {detail, code}, `done`.

Endpoints ([app/main.py](app/main.py)) :
| Méthode | Chemin | Rôle |
|---|---|---|
| `PUT` | `/conversations/{id}` | Crée/touche la conversation (idempotent). |
| `GET` | `/conversations/{id}/papers` | Articles rattachés + état. |
| `POST` | `/conversations/{id}/papers` | Ajout manuel `{arxiv_id}` (même ingestion, `added_by='user'`) — optionnel mais peu coûteux. |
| `DELETE` | `/conversations/{id}/papers/{document_id}` | Détache ; GC si orphelin. |
| `DELETE` | `/conversations/{id}` | Supprime conversation + tours ; GC des orphelins. |
| `POST` | `/conversations/{id}/ask` | `StreamingResponse(media_type="text/event-stream")`, corps `{question, limit, pipeline: "agentic"|"baseline", discovery: "auto"|"off"}`. |

GC (`paper_registry` + `paper_ingestion.purge_document`) : supprime points Qdrant,
dossier `processed`, PDF, ligne `papers` ; jamais un `pinned` ; jamais un article
en cours d'ingestion (`status != ready/failed`). Au démarrage : suppression des
conversations inactives depuis `CONVERSATION_TTL_DAYS` (les conversations
localStorage effacées sans appel serveur finissent ainsi nettoyées), puis GC.

Tests : `tests/test_conversation_api.py` avec `fastapi.testclient.TestClient`
(SSE lu en flux, ordre des événements, suppression → GC, UUID invalide → 422,
article épinglé jamais purgé).

### Étape 7 — Interface
[app/static/conversations.js](app/static/conversations.js) :
- Ajouter `remove(id)` au store (et choisir la conversation suivante) ; garder
  `HISTORY_KEY` v1 (pas de nouveau champ obligatoire).

[app/static/app.js](app/static/app.js) :
- Remplacer le sélecteur d'articles global (`fetch("/arxiv/documents")`, L178) par
  un panneau « Articles de cette conversation » alimenté par
  `GET /conversations/{id}/papers` à chaque activation de conversation : titre,
  arXiv id (lien `https://arxiv.org/abs/<id>`), badge « ajouté par l'agent »,
  bouton retirer (`DELETE .../papers/{document_id}`). Les cases à cocher restent
  le `selected_document_ids` optionnel.
- Bouton supprimer une conversation dans la liste → confirmation →
  `DELETE /conversations/{id}` puis `history.remove(id)`.
- Envoi (L237) : `fetch(POST /conversations/{id}/ask)` puis lecture de
  `response.body.getReader()` et parsing SSE manuel (EventSource ne fait pas de
  POST). Le message de chargement affiche les étapes en direct (« Recherche arXiv :
  "segment anything" », « Téléchargement 2304.02643v4 », « Conversion… », « Indexation
  (188 passages) », « Rédaction de la réponse »). À `answer`, rendu actuel inchangé
  (`renderAssistantMessage`) ; à `paper.added`, rafraîchir le panneau.
- Réglage engrenage : « Recherche automatique d'articles » (auto/off).
- Supprimer la condition `selectedDocumentIds.size === 0` qui bloque l'envoi
  (L225) : une conversation vide est l'état normal.
- Dans les détails techniques : afficher `execution.discovery` (outils appelés).

Tests : étendre [tests/ui.test.mjs](tests/ui.test.mjs) (store `remove`, parseur SSE
découpé dans une fonction pure `parseSseChunk` exportée, état vide autorisé).

### Étape 8 — Déploiement, docs
- [compose.yaml](compose.yaml) : ajouter `./data/cache:/data/cache` et
  `./data/state:/data/state` (en écriture) ; `./data/raw` reste `:ro` ; nouvelles
  variables de l'étape 1 avec défauts. `.env.example` idem. `.gitignore` :
  `data/cache/`, `data/state/`.
- Vérifier que le conteneur a accès réseau à `export.arxiv.org` et `arxiv.org`
  (déjà le cas pour OpenAI).
- `docs/milestones/25-acquisition-dynamique-arxiv.md` (format des jalons existants)
  et mise à jour du README : §1.2 (« ne télécharge pas à la volée » devient faux),
  §2.2, §2.4 (nouvelle étape amont), §4.3–4.5 (nouveaux endpoints).

---

## Fichiers touchés (résumé)

Nouveaux : `app/paper_registry.py`, `app/arxiv_client.py`,
`app/paper_ingestion.py`, `app/source_discovery.py`,
`app/conversation_service.py`, `app/text_normalization.py`, tests associés,
`docs/milestones/25-...md`.
Modifiés : `app/agentic_rag.py`, `app/rag_service.py`, `app/document_ingestion.py`,
`app/qdrant_gateway.py`, `app/hybrid_retrieval.py`, `app/llm_gateway.py`,
`app/schemas.py`, `app/settings.py`, `app/main.py`, `app/static/app.js`,
`app/static/conversations.js`, `scripts/download_arxiv_papers.py`, scripts
d'évaluation, `tests/openai_fixtures.py`, `compose.yaml`, `.env.example`,
`.gitignore`, `README.md`.

## Points d'attention pour l'implémentation

- **Parité du benchmark** avant tout : l'étape 4 doit passer les tests de parité
  avant d'aller plus loin ; aucun profil existant ne change.
- **Isolation** : tout appel `search_chunks` / `retrieve_multifacet_candidates` des
  chemins conversationnels reçoit un périmètre explicite.
- **Injection de prompt** : titres/résumés arXiv et passages sont des données ; le
  prompt de découverte le dit, et aucun outil ne permet de supprimer ou de sortir
  du périmètre.
- **Coût** : la découverte ajoute au moins 1 appel LLM par question (souvent 1 :
  `finish` direct quand les articles sont déjà là).
- Docling sur CPU reste lent (≈ 30 s à quelques minutes) : c'est la raison du SSE
  et du cache partagé ; ne pas lancer deux conversions en parallèle (verrou global
  de conversion en plus du verrou par document, pour borner la RAM).

## Vérification de bout en bout

1. `docker compose up -d --build`, puis
   `docker compose run --rm --no-deps -v "${PWD}/data/evaluation:/app/data/evaluation:ro" api python -m unittest discover -s tests -q`
   (anciens 196 tests + nouveaux) et `node --test tests/ui.test.mjs`.
2. Parité : `scripts.evaluate_agentic_rag` sur `rag_english30.draft.jsonl` avec
   `candidate_recovery_v2` ; résultats dans les mêmes marges que le rapport
   courant de `reports/current/` (3 passes, règle du README §5.3).
3. Manuel via l'UI `http://127.0.0.1:8000/ui/` :
   - Nouvelle conversation, « How does Segment Anything generate masks from a
     point prompt? » → l'agent trouve 2304.02643, cache hit (épinglé), réponse
     citée sans conversion.
   - « What is the training objective of Mamba? » → recherche, téléchargement,
     conversion visibles en direct, réponse citant `arxiv-2312.00752v*`.
   - Question de suivi « And how does it compare to S4? » → réutilise Mamba,
     ajoute au plus un article.
   - Autre conversation : ne voit pas Mamba dans ses passages (isolation) ;
     même question → cache hit instantané.
   - Retirer l'article puis supprimer la conversation → `GET .../papers` vide,
     points Qdrant et dossier `data/cache/processed/arxiv-2312.00752v*` supprimés
     s'il n'est plus utilisé ailleurs ; les 5 épinglés intacts.
4. `curl -N -X POST http://127.0.0.1:8000/conversations/<uuid>/ask -H 'Content-Type: application/json' -d '{"question":"...","limit":5,"pipeline":"agentic","discovery":"auto"}'`
   pour inspecter le flux SSE brut.
