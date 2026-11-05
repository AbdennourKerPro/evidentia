# Milestone 25 — Agent à appels d'outils et articles arXiv à la demande

Date : 5 novembre 2026. Plan : [`plans/acquisition-dynamique-arxiv.md`](../../plans/acquisition-dynamique-arxiv.md).

## 1. Pourquoi cette étape

Le corpus était limité à cinq articles préparés à la main. Télécharger tout arXiv
n'a pas de sens : on récupère donc, pendant la conversation, uniquement les articles
utiles à la question, on les prépare pour le RAG et on les garde tant que la
conversation existe.

## 2. Ce qui a été ajouté

| Module | Rôle |
|---|---|
| `app/paper_registry.py` | Registre SQLite : articles, conversations, rattachements, 3 dernières questions. Seed des cinq articles historiques (épinglés, alias identiques). |
| `app/arxiv_client.py` | Recherche et métadonnées arXiv (3 s entre deux appels, cache 1 h), téléchargement atomique de PDF (taille max, contrôle `%PDF`). |
| `app/paper_ingestion.py` | PDF → Docling (sans OCR) → chunks → embeddings → Qdrant. Verrou par article, verrou global de conversion, nettoyage en cas d'échec, suppression des orphelins. |
| `app/llm_gateway.py` (`run_tool_loop`) | Boucle d'appels d'outils bornée (même client, mêmes métriques que `generate_chat`). |
| `app/source_discovery.py` | Les quatre outils et le prompt de l'agent de sélection des sources. |
| `app/conversation_service.py` | Découverte + réponse, flux SSE, nettoyage. |
| `app/agentic_rag.py` | Catalogue, périmètre de recherche et articles ciblés passés en paramètres. |

## 3. Garanties

- **Isolation** : toutes les recherches de ce chemin sont filtrées par les articles de
  la conversation ; les endpoints classiques sont limités aux cinq articles
  historiques, donc un article ajouté en mode agent n'y apparaît jamais.
- **Reproductibilité** : sans paramètres, le graphe utilise le catalogue historique
  et le même ordre de détection ; un test compare le routage du catalogue du
  registre à l'ancienne constante.
- **Budgets** : appels LLM, recherches, téléchargements et taille de conversation sont
  bornés, et les erreurs d'outils reviennent au modèle sans exception.
- **Conversation vide** : abstention immédiate, sans appel LLM ni recherche.

## 4. Choix notables

- Un identifiant arXiv sans version réutilise l'article déjà prêt (par exemple SAM
  v1, épinglé) plutôt que de télécharger la dernière version.
- L'ordre de détection des articles suit l'ordre de rattachement à la conversation.
- Le script `scripts/download_arxiv_papers.py` est inchangé ; le client de l'application
  a ses propres fonctions de téléchargement.

## 5. Suivis et formules (mode agent)

Deux défauts observés lors d'une vraie conversation (« What is the training objective in
SimCLR ? », puis « Can you give me the formula ? », puis « Explain the different terms. ») :

- **Question de suivi sans nom.** « Explain the different terms. » était recherchée telle
  quelle et s'achevait en abstention. `app/question_rewriting.py` la transforme en question
  autonome à partir des trois dernières questions et des titres des articles (un appel LLM,
  seulement pour les suivis ; repli sur la question d'origine en cas de doute ou d'erreur).
  L'historique garde ce que l'utilisateur a écrit ; l'interface affiche la reformulation.
- **Formule écrite de mémoire sous une citation.** Docling n'a pas décodé l'équation de la
  loss (`<!-- formula-not-decoded -->`), mais le modèle l'a réécrite et citée. Le contrôle
  `app/formula_fidelity.py` compare chaque équation de la réponse aux passages fournis
  (normalisation, blocs communs d'au moins cinq caractères, seuil de 50 %). Une équation
  absente des passages ajoute l'erreur `unsupported_formula` : le brouillon est corrigé une
  fois (retirer l'équation, expliquer les termes avec les passages, dire que l'équation n'est
  pas disponible), sinon le graphe s'abstient. Ce contrôle est actif seulement en mode agent.

Décodage des formules : les articles ajoutés par l'agent sont convertis avec
`do_formula_enrichment` (modèle CodeFormula de Docling, téléchargé une fois dans le cache
Hugging Face). Les équations deviennent du LaTeX dans les passages, donc récupérables et
citables ; mesuré sur SimCLR : 0 formule non décodée sur 103 chunks, 1 min 28 s en tout
(téléchargement du modèle compris). Le manifeste de conversion note `formula_enrichment` ;
un article converti avant ce changement est automatiquement reconverti à sa prochaine
utilisation (il reste rattaché à ses conversations). Les cinq articles historiques ne
changent pas. Le décodage est imparfait : l'équation numérotée de SimCLR porte un préfixe
parasite (`\int _ { S } ^ { n - 1 }`) et `sim` y devient `\sin`.

Limite du contrôle : c'est une heuristique. Les définitions courtes (moins de 20 caractères
normalisés) ne sont pas vérifiées, car le texte aplati des PDF inverse indices et exposants.

## 6. Limites connues

- Les articles ne sont pas évalués par le benchmark : il reste limité aux cinq
  articles historiques, en mode classique.
- Docling sur CPU met de 30 s à quelques minutes par article ; la progression est
  affichée en direct.
- La mémoire conversationnelle reste limitée aux trois dernières questions (jamais les
  réponses) : elles servent à l'agent de sélection et à la reformulation des suivis.
