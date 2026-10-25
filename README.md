# Evidentia

Evidentia est un prototype d'assistant de lecture scientifique. Il répond à des
questions sur un petit corpus d'articles arXiv, **uniquement à partir de passages
retrouvés dans ces articles**, cite chaque affirmation et s'abstient quand les
passages ne permettent pas de répondre.

Ce document décrit l'outil de A à Z : ce qu'il fait, comment il fonctionne, comment
l'installer, comment il est évalué et où en sont les résultats (état au
25 octobre 2026, branche `fix/coverage-recovery-v2`).

---

## 1. En bref

### 1.1 Ce que fait Evidentia

- Il répond en anglais ou en français à une question sur cinq articles de vision
  par ordinateur et de vision-langage : CLIP, DINOv2, Segment Anything (SAM),
  BLIP-2 et LLaVA.
- Il suit une approche **RAG** (*Retrieval-Augmented Generation*, génération
  augmentée par la recherche) : il retrouve d'abord des extraits des articles, puis
  demande à un modèle de langage (LLM) de rédiger une réponse **à partir de ces
  seuls extraits**.
- Chaque phrase factuelle porte une référence `[S1]`, `[S2]`… vers un passage
  affiché sous la réponse, avec l'article, la page et la section d'origine.
- Si les passages ne suffisent pas, il **s'abstient** (« I cannot answer from the
  retrieved evidence alone. ») au lieu d'inventer.
- Il expose chaque décision intermédiaire (articles détectés, requêtes, passages
  retenus, contrôles de couverture, corrections) pour qu'une réponse soit auditable.

### 1.2 Ce qu'il ne fait pas

- Il ne répond pas avec des connaissances générales : une question hors corpus doit
  aboutir à une abstention, même si le LLM « connaît » la réponse.
- Il ne lit pas les figures ni les images : seul le texte extrait des PDF (tableaux
  compris, sous forme de texte) est utilisé.
- Il ne garde pas de mémoire conversationnelle : l'historique de l'interface est un
  historique d'affichage, chaque question est traitée indépendamment.
- Il ne télécharge pas d'articles à la volée : le corpus est fixé et versionné
  (identifiants arXiv épinglés).
- C'est un prototype local (API liée à `127.0.0.1`), pas un service multi-utilisateur.

---

## 2. Fonctionnement de bout en bout

### 2.1 Vocabulaire

| Terme | Définition |
|---|---|
| **Chunk** | Fragment d'article (au plus 350 tokens) qui sert d'unité de recherche et de citation. |
| **Embedding** | Vecteur numérique qui représente le sens d'un texte ; deux textes proches ont des vecteurs proches. |
| **Recherche dense** | Recherche par similarité d'embeddings (sens), dans la base vectorielle Qdrant. |
| **BM25** | Score lexical classique : il favorise les chunks qui contiennent les mots exacts de la requête, pondérés par leur rareté. Utile pour les noms propres, chiffres et identifiants (`ViT-G/14`, `1e-4`). |
| **RRF** (*Reciprocal Rank Fusion*) | Fusion de plusieurs classements en sommant `1 / (60 + rang)` pour chaque liste ; n'utilise que les rangs, donc combine des scores incomparables (dense et BM25). |
| **Cross-encoder** | Modèle qui lit ensemble la question et un passage et prédit leur pertinence ; plus précis qu'un embedding, utilisé pour reclasser une courte liste de candidats. |
| **LangGraph** | Bibliothèque d'orchestration : le pipeline est un graphe de nœuds explicites (analyse, recherche, vérification, génération…). |
| **Abstention** | Réponse qui refuse explicitement de conclure faute de preuve. |

### 2.2 Corpus

| Identifiant | Article | arXiv | Chunks |
|---|---|---|---:|
| `clip-2021` | Learning Transferable Visual Models From Natural Language Supervision | 2103.00020 | 453 |
| `dinov2-2023` | DINOv2: Learning Robust Visual Features without Supervision | 2304.07193 | 144 |
| `sam-2023` | Segment Anything | 2304.02643 | 188 |
| `blip2-2023` | BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models | 2301.12597 | 92 |
| `llava-2023` | Visual Instruction Tuning | 2304.08485 | 61 |

Le catalogue est `data/sources/arxiv_papers.json`. Les PDF (`data/raw/`) et les
artefacts dérivés (`data/processed/`) restent locaux et ne sont pas versionnés.

### 2.3 Préparation du corpus (une fois)

1. **Téléchargement** : `scripts/download_arxiv_papers.py` télécharge les versions
   épinglées des PDF et écrit `data/raw/download-manifest.json` (empreintes SHA-256).
2. **Conversion** : [Docling](https://github.com/docling-project/docling) convertit
   chaque PDF en Markdown et en JSON structuré (titres, sections, pages, tableaux) :
   `data/processed/<article>/document.md` et `document.json`.
3. **Découpage** : le `HybridChunker` de Docling découpe le document en chunks d'au
   plus 350 tokens, en respectant la structure et en préfixant chaque chunk du titre
   de sa section. Chaque chunk garde `document_id`, page, section, langue et un
   identifiant stable : `chunks.json`.
4. **Indexation** : chaque chunk est encodé avec l'embedding local
   `intfloat/multilingual-e5-small` (révision épinglée, préfixes `passage:` /
   `query:`), puis stocké dans la collection Qdrant `arxiv_chunks_e5_small`
   (distance cosinus). L'index BM25 est construit en mémoire, par article, à partir
   des mêmes chunks.

### 2.4 Répondre à une question : le pipeline LangGraph

```text
question
  │
  ▼
analyze_question ── détection déterministe des articles nommés (alias : « CLIP »,
  │                 « SAM », « BLIP-2 »…) et des marqueurs de comparaison
  ├── 0 ou 1 article ──► retrieve_global
  │                       · 1 article : dense + BM25 multi-requêtes, fusion RRF,
  │                         reclassement cross-encoder contre la question entière
  │                       · aucun article : recherche dense sur tout le corpus
  │                         (+ requête traduite en anglais si la question ne l'est pas)
  └── ≥ 2 articles ────► plan_source_retrieval (le LLM écrit une requête par article)
                          └─► retrieve_source × N, en parallèle (dense + BM25 + RRF
                              + cross-encoder, restreint à un article)
  ▼
select_evidence ── top-k (5 en évaluation) global, ou tourniquet équitable entre articles
  ▼
check_evidence_coverage ── 1) le LLM décompose la QUESTION seule en 1 à 4 exigences
  │                           atomiques (sans voir les passages)
  │                        2) le LLM vérifie chaque exigence contre les passages, avec
  │                           une citation exacte contrôlée localement
  ├── tout est couvert ou verdict inconnu ──────────────────────────────┐
  └── lacune confirmée ─► recover_candidates                             │
                           · candidats de la réserve reclassés par lacune│
                           · vérification des extraits exacts            │
                           · recherche de secours si la lacune persiste  │
                           · ajout de 0 à 2 passages, jamais de retrait  │
  ▼ ◄───────────────────────────────────────────────────────────────────┘
plan_answer ── plan factuel déterministe (pour certaines intentions seulement)
  ▼
generate_answer ── le LLM rédige à partir des seuls passages, avec citations [Sn]
  ▼
validate_answer ── contrat déterministe : langue, citations valides, réponse complète,
  │                cohérence avec l'abstention ; langue incertaine → 1 appel classifieur
  ├── valide ─────────────────────────────────────────────┐
  └── invalide ─► correct_answer (une seule fois) ─► validate_answer
  ▼ ◄─────────────────────────────────────────────────────┘
finalize_answer ── publie la réponse et ses citations, ou s'abstient
```

Points importants :

- **Recherche multi-facettes.** Quand une question exprime une intention reconnue
  (échelle numérique, comparaison chiffrée, mécanisme de prompt, connecteur entre
  modèles, mécanisme d'inférence, capacités, objectif d'entraînement), des requêtes
  supplémentaires en anglais sont ajoutées. Leurs classements dense et BM25 sont
  fusionnés par RRF. Avec les profils de récupération, tous les résultats de la
  requête d'origine restent dans la réserve, même au-delà de la coupure.
- **Réserve de candidats.** Les passages fusionnés mais non sélectionnés forment une
  réserve dans laquelle la récupération de couverture peut puiser sans nouvelle
  recherche.
- **Couverture plutôt que pertinence.** Cinq passages pertinents peuvent quand même
  omettre un chiffre ou un des articles d'une comparaison. Les exigences sont
  fixées à partir de la question seule, pour que le vérificateur ne puisse pas les
  ajuster à ce qui a été retrouvé.
- **Contrat de réponse.** Une réponse n'est publiée que si elle est dans la langue de
  la question (anglais, français ou chinois détectés), si chaque citation renvoie à
  un passage fourni, si elle n'est pas tronquée et si elle ne mélange pas réponse et
  marqueur d'abstention. Sinon le graphe fait **une** correction, puis s'abstient si
  le contrat échoue encore. Aucune boucle n'est possible.
- **Citations.** Les références `[Sn]` sont résolues côté serveur vers les chunks
  réellement fournis au LLM ; une référence inventée invalide la réponse.
- **Modèles.** Embeddings et cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`,
  révision épinglée) tournent localement sur CPU. Les appels génératifs passent par
  l'API OpenAI Responses avec le modèle `gpt-6-luna`, un effort de raisonnement fixé
  à `none`, au plus 4 096 tokens de sortie, un timeout de 60 s et une seule relance.

L'endpoint `/ask` fournit en plus une **baseline** sans graphe : recherche dense, puis
génération et contrat de réponse. Il sert de point de comparaison.

---

## 3. Profils de récupération

Le comportement de recherche et de vérification est choisi par un **profil**
(`RAG_RETRIEVAL_PROFILE`). Chaque profil ajoute une étape au précédent. Les anciens
profils sont conservés à l'identique pour pouvoir reproduire les ablations.

| Profil | Ce qu'il ajoute |
|---|---|
| `intent_baseline` | Recherche multi-facettes ; les correspondances lexicales d'intention priment sur la pertinence. |
| `question_rerank` | Le cross-encoder reclasse tous les candidats contre la question complète. |
| `original_query` | La question d'origine reste la première requête et ses résultats sont protégés de la fusion. |
| `coverage_retry` | Un juge LLM de couverture et, si besoin, une recherche complémentaire unique. |
| `coverage_diagnostic` | Exigences tirées de la question seule puis vérifiées, sans aucune récupération (diagnostic). |
| `candidate_recovery` | Récupération depuis la réserve : 2 lacunes max, 3 candidats par lacune, 2 passages ajoutés max (16 000 caractères), recherche dense de secours si une lacune n'a aucun candidat. |
| `coverage_answer_plan` | `candidate_recovery` + liens exigence → citation transmis au prompt de réponse (ablation, non retenue). |
| **`candidate_recovery_v2`** | **Profil par défaut.** Voir ci-dessous. |

`candidate_recovery_v2` corrige des défauts génériques observés dans les traces de
`candidate_recovery` :

1. **Filtre bibliographique.** Les chunks des sections `References` / `Bibliography`
   (éventuellement numérotées) sont retirés après la recherche, avant la fusion et le
   reclassement. Ils ressemblaient à beaucoup de requêtes sans jamais prouver quoi
   que ce soit. Aucune réindexation n'est nécessaire.
2. **Articles du plan d'exigences.** Le plan ne porte que sur les articles détectés
   dans la question. Avant, un chunk incident d'un autre article rendait le plan
   invalide.
3. **Recherche hybride.** Un article détecté passe toujours par dense + BM25 + RRF,
   même avec une seule requête, et la question traduite en anglais sert de requête
   supplémentaire si besoin. La recherche de secours d'une lacune liée à un article
   est elle aussi hybride.
4. **Recherche de secours systématique.** Après la vérification de la réserve, chaque
   lacune encore sans preuve déclenche une recherche, puis une seconde vérification
   limitée à ces lacunes. Le budget total reste de 2 passages ajoutés.
5. **Vérificateur cohérent.** Pendant la récupération, une lacune ne peut être
   fermée que par un passage **ajouté**, jamais par un passage initial déjà jugé
   insuffisant.
6. **Réserve élargie.** 8 candidats proposés par lacune (au lieu de 3).
7. **Consignes de rédaction.** Recopier noms de modèles, jeux de données,
   identifiants et valeurs exactement comme dans la source, avec leur provenance
   (« X from Y »). Si la question porte principalement sur une entité absente des
   passages, s'abstenir ; la correction d'une abstention jugée prématurée rappelle
   que l'abstention reste permise.

Aucune règle propre à un article ni aucun fait du benchmark n'est codé dans le
pipeline ou les prompts.

**Pourquoi c'est le défaut.** Il fait au moins aussi bien que `candidate_recovery`,
mesuré sur trois passes de chaque profil, et ne régresse sur aucun cas (section 6).
Le défaut est le même partout : `compose.yaml`, `.env.example` et
`app/retrieval_policy.py`.

---

## 4. Installation et utilisation

### 4.1 Prérequis

- Docker avec Docker Compose (testé sous Windows 11 avec Docker 29).
- Une clé d'API OpenAI donnant accès au modèle configuré.
- Python 3 sur l'hôte, uniquement pour le script de téléchargement des PDF (aucune dépendance).
- Node.js, facultatif, pour les tests de l'interface.

### 4.2 Configuration

```bash
cp .env.example .env      # puis renseigner OPENAI_API_KEY dans .env
```

`.env` n'est jamais versionné. Variables utiles :

| Variable | Valeur par défaut | Rôle |
|---|---|---|
| `OPENAI_API_KEY` | (vide) | Clé d'API, à renseigner localement. |
| `OPENAI_MODEL` | `gpt-6-luna` | Modèle génératif. |
| `RAG_RETRIEVAL_PROFILE` | `candidate_recovery_v2` | Profil de la section 3. |
| `OPENAI_REASONING_EFFORT` | `none` | **Fixé à `none` par `compose.yaml`**, quelle que soit la valeur de `.env`. |
| `OPENAI_TIMEOUT_SECONDS`, `OPENAI_MAX_RETRIES`, `OPENAI_MAX_OUTPUT_TOKENS` | 60, 1, 4096 | Limites des appels API. |

### 4.3 Démarrage

```bash
docker compose up -d --build        # API sur http://127.0.0.1:8000, Qdrant sur 127.0.0.1:6333
docker compose run --rm api python -m scripts.download_reranker_model   # une fois
docker compose run --rm --no-deps api python -m scripts.check_openai     # configuration, sans appel payant
docker compose run --rm --no-deps api python -m scripts.check_openai --live   # un petit appel réel
```

Préparation du corpus, une fois :

```bash
python -m scripts.download_arxiv_papers                 # sur l'hôte, depuis la racine du dépôt
curl -X POST http://127.0.0.1:8000/documents/clip-2021/convert
curl -X POST http://127.0.0.1:8000/documents/clip-2021/index
# … même chose pour dinov2-2023, sam-2023, blip2-2023, llava-2023
```

Le modèle d'embedding est téléchargé au premier usage et mis en cache dans le volume
`model_cache`. Les données Qdrant persistent dans le volume `qdrant_storage`.

### 4.4 Interfaces

- **Conversation** : `http://127.0.0.1:8000/ui/`. Questions, réponses en Markdown,
  citations cliquables qui ouvrent le passage complet, historique local des
  conversations. L'engrenage permet de choisir les articles, le pipeline (LangGraph
  ou baseline) et d'afficher les détails techniques (traces, tokens).
- **Revue du benchmark** : `http://127.0.0.1:8000/review/`. Filtres, édition et
  validation humaine des cas du benchmark défini par `BENCHMARK_PATH` (English30
  dans `compose.yaml`). Les écritures sont atomiques, avec copie de sauvegarde.

### 4.5 API

| Méthode et chemin | Rôle |
|---|---|
| `GET /health`, `GET /ready` | Le processus vit ; Qdrant est joignable. |
| `GET /llm/status` | Configuration du LLM, sans appel réseau ni coût. |
| `POST /agentic/ask` | **Pipeline principal** (LangGraph). Corps : `{"question": "...", "limit": 3, "document_ids": null}` (`limit` de 1 à 5, 5 en évaluation). Renvoie la réponse, les citations, les passages et `execution` (profil, articles détectés, requêtes, couverture, récupération, contrat, étapes chronométrées). |
| `POST /ask` | Baseline sans graphe. |
| `GET /arxiv/search?query=...&limit=3` | Recherche dense brute dans le corpus. |
| `GET /arxiv/documents` | Articles indexés. |
| `POST /documents/{id}/convert`, `POST /documents/{id}/index` | Préparation du corpus. |
| `GET /evaluation/cases`, `PATCH /evaluation/cases/{id}` | API de l'interface de revue. |
| `GET /search`, `POST /demo/index` | Collection de démonstration des premiers jalons. |

Exemple :

```bash
curl -s -X POST http://127.0.0.1:8000/agentic/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many GPUs are used to train the ViT-B and ViT-L models of SAM?"}'
# → "ViT-B and ViT-L are trained across 128 GPUs [S1]." (profil candidate_recovery_v2)
```

### 4.6 Tests

```bash
docker compose run --rm --no-deps -v "${PWD}/data/evaluation:/app/data/evaluation:ro" \
  api python -m unittest discover -s tests -q     # 196 tests, sans réseau ni coût
node --test tests/ui.test.mjs                     # 18 tests de l'interface
```

Les tests n'appellent jamais l'API OpenAI : les réponses sont simulées par un
transport HTTP en mémoire (`tests/openai_fixtures.py`) et la recherche est simulée.
Le montage de `data/evaluation` sert à un test qui relit les étiquettes de langue
du benchmark.

---

## 5. Méthode d'évaluation

### 5.1 Lancer une évaluation

```bash
docker compose run --rm --no-deps --env RAG_RETRIEVAL_PROFILE=candidate_recovery_v2 api \
  python -m scripts.evaluate_agentic_rag \
  --benchmark /data/evaluation/rag_english30.draft.jsonl \
  --output /reports/mon-rapport.json
# Sous-ensemble anglais du heldout :
#   --benchmark /data/evaluation/rag_heldout.draft.jsonl --language en
```

Options : `--limit` (top-k, 5 par défaut), `--case-id` (répétable), `--max-cases`,
`--language en|fr` et `--resume`. `--resume` reprend un rapport interrompu, mais
seulement si le pipeline, le code source, le modèle, le profil, le top-k, le
benchmark et le filtre de langue sont identiques. Chaque cas est exécuté
séquentiellement et le rapport est réécrit après chaque cas. Le rapport conserve
tout : réponse, passages, trace, appels LLM et tokens, empreintes du code et du
benchmark.

### 5.2 Métriques

Un cas « répondable » attend une réponse ; un cas d'abstention attend un refus.

| Métrique | Définition |
|---|---|
| **Réussite d'un cas** (`passed`) | Abstention attendue : le système s'abstient. Sinon : pas d'abstention **et** couverture des faits ≥ 0,5 **et** précision et rappel des citations par document égaux à 1. |
| `end_to_end_pass_rate` | Part des cas réussis. |
| `abstention_accuracy` | Part des cas où « s'abstenir ou non » correspond à l'attendu. |
| `mean_fact_coverage` | Sur les cas répondables : part des groupes de faits attendus (`required_answer_terms`) présents dans la réponse. Un groupe est couvert si l'une de ses formulations acceptées y figure, après normalisation (casse, accents, ponctuation). |
| `mean_citation_document_precision` | Part des articles cités qui font partie des articles attendus. |
| `mean_citation_document_recall` | Part des articles attendus qui sont cités. |
| `mean_retrieved_document_recall` | Part des articles attendus présents parmi les passages fournis au LLM. |
| `mean_selected_evidence_recall` | Part des fragments de preuve attendus (`evidence_targets`) présents dans les passages fournis au LLM. |
| `mean_latency_seconds` | Temps moyen de bout en bout par question, appels API compris. |
| Appels LLM | Nombre moyen d'appels au modèle par question (planification, couverture, génération, correction, langue). |

### 5.3 Règles suivies

- Benchmarks **relus par un humain** (`review_status: verified` pour tous les cas
  actifs). Les preuves sont ancrées sur des phrases réellement présentes dans les
  chunks ; `scripts/validate_benchmark.py --benchmark <fichier>` vérifie leur
  présence.
- **Aucune règle propre au benchmark** dans le code ou les prompts : les corrections
  doivent être génériques et derrière un profil, pour que les anciens résultats
  restent reproductibles.
- **Plusieurs passes** : la sortie de l'API n'est pas déterministe, donc une
  décision repose sur 3 passes du candidat **et** 3 de la référence, lancées dans
  les mêmes conditions. Une régression est un cas que la référence réussit 3 fois
  sur 3 et que le candidat rate au moins une fois.
- `reasoning_effort=none`, top-k = 5, même modèle pour toutes les comparaisons.
- **Anglais uniquement** : depuis le 25 octobre 2026, par décision du projet,
  l'évaluation porte uniquement sur des cas en anglais.

### 5.4 Limites des métriques

- La couverture des faits est **lexicale** : une paraphrase correcte mais non prévue
  n'est pas comptée, et une mention correcte d'un terme ne prouve pas que la phrase
  qui l'entoure est juste.
- La réussite tolère une réponse incomplète (couverture ≥ 0,5).
- Les contrôles de couverture et de langue reposent en partie sur un LLM, lui-même
  faillible.
- Les échantillons sont petits (30 et 20 questions) : un cas vaut 3,3 ou 5 points de
  pourcentage.

---

## 6. Benchmarks (anglais)

Les fichiers sont dans `data/evaluation/` (brouillons relus, versionnés). Le
détail des fichiers, des règles de relecture et de l'archive de l'ancien
benchmark est dans [`data/evaluation/README.md`](data/evaluation/README.md).

### 6.1 Structure d'un cas

Chaque ligne JSONL est un cas :

| Champ | Contenu |
|---|---|
| `id`, `category`, `language` | Identifiant, catégorie (`factoid`, `method`, `comparison`, `abstention`), langue. |
| `question` | Question posée telle quelle au système. |
| `scope_document_ids` | Filtre optionnel d'articles (`null` partout ici : tout le corpus est interrogé). |
| `expected_document_ids` | Articles qui doivent fonder et être cités dans la réponse. |
| `evidence_targets` | Phrases exactes du corpus qui prouvent la réponse, par article. |
| `reference_answer` | Réponse de référence relue. |
| `required_answer_terms` | Groupes de faits ; chaque groupe liste des formulations acceptées. |
| `should_abstain` | `true` si le corpus ne permet pas de répondre. |
| `review_status`, `notes` | Statut de relecture humaine, remarques. |

### 6.2 English30 — `rag_english30.draft.jsonl`

30 cas en anglais, tous relus, dont 5 comparaisons entre deux articles :

| Catégorie | Cas | Exemple |
|---|---:|---|
| `factoid` | 12 | « What minibatch size do the CLIP authors use for training? » |
| `method` | 7 | « How are the smaller DINOv2 models obtained? » |
| `comparison` | 5 | « Do CLIP and DINOv2 both rely on image captions during pre-training? » |
| `abstention` | 6 | « How many V100 GPUs did the LLaVA authors use for training? » |

Articles attendus (cas répondables) : CLIP 7, DINOv2 7, LLaVA 6, SAM 5, BLIP-2 4.

Les abstentions sont de trois sortes : une information absente des articles
(nombre de GPU V100 de LLaVA, langage de l'implémentation de SAM, fenêtre de contexte
de LLaVA), une entité hors corpus (Mistral-7B, Claude 3) ou un fait externe
attribué à tort à un article (prix de l'API GPT-4 « selon LLaVA »).

Composition : 20 cas `heldout-…-en`, identiques au sous-ensemble anglais du heldout,
et 10 cas `english20-…`, centrés sur des détails chiffrés d'entraînement
(minibatch, GPU, learning rate, nombre de paramètres…).

Exemple complet (cas répondable) :

```json
{"id": "heldout-clip-prompt-ensemble-en", "category": "method", "language": "en",
 "question": "How do CLIP's authors improve zero-shot ImageNet classifiers beyond using a single text prompt?",
 "expected_document_ids": ["clip-2021"],
 "evidence_targets": [{"document_id": "clip-2021", "match_phrases": ["we ensemble 80 different context prompts"]}],
 "reference_answer": "They ensemble 80 different context prompts, which improves zero-shot performance on ImageNet.",
 "required_answer_terms": [["80"], ["prompt"], ["ensemble", "ensembling"]],
 "should_abstain": false, "review_status": "verified"}
```

### 6.3 Heldout anglais — `rag_heldout.draft.jsonl --language en`

Les 20 cas `language = "en"` du fichier heldout, sélectionnés à la lecture par
`--language en` (le fichier n'est pas modifié) : 6 `method`, 5 `factoid`,
5 `comparison`, 4 `abstention`.

**Attention** : ces 20 cas sont aussi dans English30. Les deux mesures ne sont donc
pas indépendantes. English30 ajoute les 10 cas `english20`, qui concentrent les
échecs historiques.

---

## 7. Résultats

Configuration commune sauf mention contraire : `gpt-6-luna`, top-k 5, rapports dans
`reports/`.

### 7.1 Décision actuelle : `candidate_recovery_v2` contre `candidate_recovery`

3 passes de chaque profil, le 25 octobre 2026, `reasoning_effort=none`, deux files
d'exécution en parallèle (mêmes conditions d'API ; latences absolues gonflées par la
charge).

| Benchmark | Profil | Réussis (passes 1/2/3) | Pass rate moyen | Abstention | Couverture des faits | Latence | Appels LLM |
|---|---|---|---:|---:|---:|---:|---:|
| English30 | `candidate_recovery` | 29 / 29 / 29 | 0,967 | 0,967 | 0,898 | 22,3 s | 3,76 |
| English30 | **`candidate_recovery_v2`** | **30 / 30 / 30** | **1,000** | 1,000 | 0,937 | 26,2 s | 4,20 |
| Heldout anglais | `candidate_recovery` | 20 / 20 / 20 | 1,000 | 1,000 | 0,927 | 24,4 s | 3,95 |
| Heldout anglais | **`candidate_recovery_v2`** | 20 / 20 / 20 | 1,000 | 1,000 | 0,892 | 28,3 s | 4,40 |

Rapports : `reports/current/m24/agentic-rag-{english30|heldout-en}-{candidate-recovery|recovery-v2c}-20261025-run{1,2,3}.json`.

- Aucune régression : aucun cas réussi 3/3 par la référence n'est raté par v2.
- `english20-blip2-peak-learning-rate` : référence 0/3, abstention à tort parce que
  la lacune reste sans preuve ; v2 3/3 (« peak learning rate of 1e-4 »).
- Coût : environ +0,45 appel LLM et +4 s par question.

### 7.2 Historique (cas anglais)

| Date | Version | Benchmark | Résultat (anglais) | Rapport |
|---|---|---|---|---|
| 30/09/2026 | v9, `candidate_recovery`, effort `low` | ancien benchmark mixte, 22 cas anglais | 21/22 | `archive/m21-coverage-v9/agentic-rag-v9-candidate-coverage-final-30-20260930.json` |
| 07/10/2026 | v9, effort `none` | idem | 22/22 | `archive/m21-coverage-v9/agentic-rag-v9-effort-none-20261007.json` |
| 09/10/2026 | variantes exploratoires de `candidate_recovery` | English30 | `verified` 28/30, `primary-pool` 28/30, `widened` 29/30, `keep-primary-hits` 29/30 | `archive/m24-recovery-v2/agentic-rag-english30-*-candidate-recovery-20261009.json` |
| 24/10/2026 | v2, 1 passe | English30 | 30/30 | `archive/m24-recovery-v2/agentic-rag-english30-recovery-v2-20261024.json` |
| 25/10/2026 | v2b, 3 passes | English30 / heldout anglais | 29, 30, 30 / 20, 20, 20 | `current/m24/agentic-rag-{english30|heldout-en}-recovery-v2b-20261025-run*.json` |
| 25/10/2026 | v2 final, 3 passes | English30 / heldout anglais | 30, 30, 30 / 20, 20, 20 | voir 7.1 |

`keep-primary-hits` (09/10) est le comportement actuel de la réserve de candidats
pour les profils de récupération.

### 7.3 Échecs historiques résolus

| Cas | Problème | Correction |
|---|---|---|
| `english20-blip2-peak-learning-rate` | Abstention : la valeur n'était ni dans le top 5 ni ajoutée par la récupération. | Réserve élargie et recherche de secours (v2). |
| `english20-clip-minibatch` | Échec dans les variantes `verified`, `primary-pool` et `widened` du 09/10. | Réussi depuis `keep-primary-hits` (tous les résultats de la requête d'origine restent dans la réserve), puis dans toutes les passes v2. |
| `english20-sam-vit-b-l-gpus` | v2b : une fois sur trois, la réponse niait un fait pourtant cité (« the supplied passages do not state… »), à cause d'une consigne de repli trop permissive. | Consigne retirée (v2 final) : 3/3. |
| Encodeurs d'images de BLIP-2 (question anglaise testée hors benchmark) | La preuve (« ViT-L/14 from CLIP and ViT-G/14 from EVA-CLIP ») n'était jamais retrouvée par la recherche dense seule. | Recherche hybride pour un article détecté : le passage arrive en S1, 3/3. |
| Abstention Midjourney / CLIP (question anglaise testée hors benchmark) | La correction d'une abstention jugée prématurée produisait une réponse partielle sur CLIP. | Règle d'abstention si le sujet principal est absent, rappelée lors de la correction : 3/3 abstentions. |

### 7.4 Points faibles connus

- `heldout-dinov2-sam-data-scale-en` : réussi, mais avec une couverture des faits de
  0,5 sur les 6 passes v2. La récupération ajoute pour SAM un passage d'ablation
  (« approximately 100M masks ») au lieu de l'introduction (« more than 1 billion
  masks »). Le vérificateur accepte une preuve qui couvre l'exigence (« une échelle
  de données SAM ») sans que ce soit le meilleur passage. C'est la principale cause
  de la baisse de couverture sur le heldout anglais (0,927 → 0,892).
- `heldout-sam-ambiguous-prompt-en`, `heldout-llava-gpt4-data-en` : couverture
  variable (0,67 ou 1,0) d'une passe à l'autre, avec les deux profils.

### 7.5 Limites des résultats

- 30 et 20 questions, dont 20 communes : les intervalles d'incertitude sont larges.
- La décision repose sur 3 passes contre 3 ; une différence d'un cas reste fragile.
- Les benchmarks sont des brouillons relus, pas un jeu de test externe.
- La latence augmente (environ +4 s par question à charge égale).

---

## 8. Structure du dépôt

```text
app/
  main.py                FastAPI : routes, interfaces statiques
  agentic_rag.py         graphe LangGraph (nœuds de la section 2.4)
  retrieval_policy.py    profils de récupération et profil par défaut
  hybrid_retrieval.py    BM25, RRF, cross-encoder, filtre bibliographique
  query_expansion.py     intentions et requêtes multi-facettes
  query_translation.py   requête anglaise pour une question non anglaise
  candidate_recovery.py  plan d'exigences, vérification, récupération de couverture
  evidence_coverage.py   juge de couverture (profil coverage_retry) et citations exactes
  fact_planning.py       plan factuel déterministe
  rag_service.py         prompt de réponse, baseline, finalisation
  response_contract.py   contrat de langue, citations, abstention
  language_verification.py  classification LLM d'une langue incertaine
  llm_gateway.py         appels à l'API Responses, comptage des tokens
  embeddings.py, qdrant_gateway.py, document_ingestion.py   corpus et index
  evaluation.py, benchmark_review.py   benchmark, métriques, revue humaine
  static/, review_static/  interfaces conversation et revue
scripts/                 téléchargements, vérification API, évaluations
tests/                   tests unitaires et d'intégration sans réseau (+ ui.test.mjs)
data/sources/            catalogue des articles (versionné)
data/evaluation/         benchmarks
reports/                 rapports d'évaluation (index : reports/README.md)
  current/m24/           base de la décision actuelle
  archive/<jalon>/       campagnes précédentes, par jalon
docs/milestones/         journal détaillé de chaque étape (en français)
compose.yaml, Dockerfile, .env.example, requirements.txt
```

## 9. Historique des jalons

| # | Jalon |
|---|---|
| 00 | Premier conteneur Docker |
| 01 | Conteneur actif et exécution en arrière-plan |
| 02 | Première image applicative et premier service web |
| 03 | Environnement Compose et disponibilité de Qdrant |
| 04 | Première recherche sémantique |
| 05 | Article arXiv, Docling et indexation RAG |
| 06 | RAG génératif local avec OpenVINO (remplacé depuis par l'API) |
| 07 | Corpus multi-source et interface de sélection |
| 08 | Benchmark et évaluation du RAG |
| 09 | Interface de revue humaine du benchmark |
| 10 | Baseline d'évaluation vérifiée |
| 11 | Première amélioration RAG avec LangGraph |
| 12 | Évaluation des comparaisons LangGraph |
| 13 | Retrieval hybride, RRF et cross-encoder |
| 14 | Contrat de langue, citations et correction |
| 15 | Retrieval multi-facettes guidé par l'intention |
| 16 | Plan factuel sourcé et campagne finale reproductible |
| 17 | Correctifs de généralisation (v6) |
| 18 | Migration du LLM vers GPT-6 Luna via API |
| 19 | Correction du retrieval SAM et ablation successive |
| 20 | Contrôle de langue à trois états |
| 21 | Récupération de preuves guidée par la couverture (v9) |
| 22 | Interface épurée, Markdown et conversations locales |
| 23 | Logo vectoriel |
| 24 | Récupération de couverture v2 et évaluation en anglais uniquement |

Le détail de chaque étape, y compris les essais abandonnés, est dans
`docs/milestones/`.

## 10. Limites connues et pistes

- **Choix du meilleur passage** : le vérificateur accepte la première preuve
  suffisante. Comparer plusieurs preuves candidates pour une exigence chiffrée
  améliorerait des cas comme l'échelle de données SAM/DINOv2.
- **Latence et coût** : la seconde vérification et la réserve élargie ajoutent
  environ un demi-appel LLM et quelques secondes par question.
- **Questions sans article nommé** : elles passent encore par une recherche dense
  seule sur tout le corpus. BM25 n'existe que par article.
- **Évaluation** : agrandir le benchmark anglais avec des questions réellement
  indépendantes d'English30.
- **Corpus** : cinq articles, texte seulement, pas de figures.

## 11. Licence

Apache License 2.0 — voir `LICENSE`.
