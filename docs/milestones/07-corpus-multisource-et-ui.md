# Milestone 07 — Corpus multi-source et interface de sélection

## Objectif

Faire évoluer Evidentia d'un lecteur d'article unique vers un explorateur de
corpus scientifique. L'utilisateur doit pouvoir voir les articles réellement
indexés, limiter une question à une ou plusieurs sources, puis vérifier les
chunks utilisés dans chaque réponse.

## Corpus initial

Le corpus cible comporte cinq articles complémentaires :

| Source interne | Article | Sujet principal |
|---|---|---|
| `clip-2021` | *Learning Transferable Visual Models From Natural Language Supervision* | Apprentissage image-texte et zero-shot. |
| `dinov2-2023` | *DINOv2: Learning Robust Visual Features without Supervision* | Représentations visuelles auto-supervisées. |
| `sam-2023` | *Segment Anything* | Segmentation par prompts et transfert zero-shot. |
| `blip2-2023` | *BLIP-2* | Pont léger entre encodeur d'images gelé et LLM gelé. |
| `llava-2023` | *Visual Instruction Tuning* | VLM et instruction tuning multimodal. |

Chaque document est conservé sous un `source_id` stable et dans une version
arXiv explicite. Ils sont tous stockés dans la collection Qdrant
`arxiv_chunks_e5_small`; le champ de payload `document_id` les distingue.

## Nouvelles capacités API

### Bibliothèque dynamique

`GET /arxiv/documents` parcourt la collection Qdrant sans charger les vecteurs.
Il lit seulement les payloads `document_id` et `title`, regroupe les points par
article et compte leurs chunks. Il retourne par exemple :

```json
{
  "collection": "arxiv_chunks_e5_small",
  "documents": [
    {
      "document_id": "sam-2023",
      "title": "Segment Anything",
      "indexed_chunks": 318
    }
  ]
}
```

L'interface ne connaît donc pas a priori les noms des articles. Après une
nouvelle indexation, son rechargement suffit à afficher la nouvelle source.

### Recherche filtrée

`POST /ask` accepte maintenant un champ facultatif :

```json
{
  "question": "How is SAM promptable?",
  "limit": 3,
  "document_ids": ["sam-2023"]
}
```

Lorsque `document_ids` est absent, Qdrant recherche dans tout le corpus. Lors
qu'il contient une ou plusieurs sources, Evidentia construit un filtre Qdrant
`MatchAny` sur le payload `document_id`. Seuls les chunks de ces articles sont
donc candidats à la récupération et au contexte envoyé au LLM.

```text
Sélection dans l'UI
        |
        v
document_ids dans POST /ask
        |
        v
Qdrant Filter(document_id IN sélection)
        |
        v
Chunks récupérés -> prompt LLM -> citations
```

Une sélection multiple est particulièrement utile pour les questions de
comparaison. Une sélection unique évite qu'un article voisin apporte un
contexte non désiré à une question précise.

## Interface conversationnelle

L'UI locale à `/ui/` possède désormais trois zones :

| Zone | Responsabilité |
|---|---|
| Barre latérale | Liste dynamique des articles, leur nombre de chunks et cases de sélection. |
| Fil de conversation | Questions, réponses, périmètre interrogé et citations sous la réponse. |
| Compositeur fixe | Saisie de la prochaine question, désactivée lorsqu'aucun article n'est sélectionné. |

Sur petit écran, la bibliothèque devient un panneau latéral ouvrable par le
bouton `Corpus`; elle ne réduit donc pas la zone de lecture de la conversation.

Les citations affichent un titre raccourci et la page. Un clic ouvre le bloc de
preuves de la réponse, déplie le chunk concerné et le met visuellement en
évidence. Le panneau de preuves montre aussi les chunks récupérés mais non
cités : cette distinction permet d'observer ce que le modèle a reçu et ce qu'il
a choisi de référencer.

## État conversationnel

L'interface conserve un historique visuel de messages afin de lire plusieurs
réponses dans le même écran. Les requêtes restent indépendantes côté serveur :
aucun échange précédent n'est ajouté au prompt du LLM. Cette limite est
volontaire. Une mémoire de conversation fiable devra être introduite plus tard
avec un état LangGraph et des règles explicites de reprise de contexte.

## Validation réalisée

- syntaxe Python de l'API validée ;
- contrat vérifié entre les identifiants HTML et le JavaScript de l'interface ;
- l'UI utilise des ressources versionnées et l'API applique `Cache-Control:
  no-store` à `/ui`, pour éviter le mélange HTML/CSS/JavaScript observé lors
  de l'itération précédente ;
- les articles seront lus dynamiquement par l'endpoint après reconstruction du
  conteneur API.

## Questions d'évaluation manuelle

Le corpus permet maintenant de tester quatre comportements différents :

1. question ciblée à une source ;
2. comparaison entre deux ou trois sources ;
3. provenance de chaque affirmation ;
4. abstention hors du corpus.

Ces tests serviront de base à un benchmark versionné avant l'introduction de
LangGraph et du fine-tuning.
