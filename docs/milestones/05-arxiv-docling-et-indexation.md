# Milestone 05 — Article arXiv, Docling et indexation RAG

## Objectif

Faire passer Evidentia d'un corpus de démonstration écrit à la main à un vrai
article scientifique PDF. Le résultat attendu est une recherche sémantique
traçable : chaque passage retrouvé doit conserver son article, son titre, sa
section et sa page d'origine.

## Source indexée

| Champ | Valeur |
|---|---|
| Source interne | `clip-2021` |
| Article | *Learning Transferable Visual Models From Natural Language Supervision* |
| arXiv | `2103.00020v1` |
| Domaine | Computer vision et apprentissage multimodal image-texte |
| PDF local | `data/raw/clip_2103.00020v1.pdf` |
| Langue déclarée | Anglais (`en`) |

La version `v1` est explicitement conservée dans le nom du PDF et dans le
manifeste de téléchargement : on sait donc exactement quel document a été
indexé.

## Architecture validée

```text
PDF arXiv (monté en lecture seule dans l'API)
        |
        v
Docling DocumentConverter
        |
        +--> document.md       (lecture humaine)
        +--> document.json     (structure Docling complète)
        |
        v
HybridChunker + tokenizer de multilingual-e5-small
        |
        +--> chunks.json       (passages inspectables, avec provenance)
        |
        v
multilingual-e5-small : "passage: <chunk>"
        |
        v
Qdrant : arxiv_chunks_e5_small
        |
        v
multilingual-e5-small : "query: <question>"
        |
        v
GET /arxiv/search : passages, scores et provenance
```

La conversion et l'indexation sont volontairement deux responsabilités
différentes. Convertir extrait la structure d'un PDF. Indexer transforme des
passages en vecteurs et les rend recherchables. Ainsi, un problème de recherche
peut être diagnostiqué en inspectant `chunks.json`, sans devoir relire les
vecteurs de Qdrant.

## Conversion Docling

La route suivante exécute la conversion :

```text
POST /documents/{source_id}/convert
```

Pour `clip-2021`, elle produit :

- `data/processed/clip-2021/document.md` : version Markdown pratique à lire ;
- `data/processed/clip-2021/document.json` : représentation structurée,
  réutilisable par Docling ;
- `data/processed/clip-2021/conversion-manifest.json` : lien entre les
  artefacts et la source précise convertie.

Docling ne se contente pas d'extraire une chaîne de caractères. Il identifie
notamment les éléments de mise en page et leur provenance, ce qui permet de
rattacher un chunk à une page.

### Dépendances système

L'image `python:3.12-slim` ne contient pas les bibliothèques graphiques dont
OpenCV, utilisé par Docling pour l'analyse de document, a besoin. Le
`Dockerfile` installe donc `libgl1`, `libglib2.0-0` et `libxcb1`. Ce sont des
bibliothèques de l'image API ; elles ne concernent ni Qdrant ni le poste
Windows hôte.

## Chunking

`HybridChunker` part d'abord de la structure du document, puis ajuste les
passages à la fenêtre du tokenizer E5. La limite choisie est de **350 tokens**.
Elle laisse une marge sous la limite du modèle et évite de perdre une partie
d'un long passage au moment de l'embedding.

Le texte stocké est obtenu avec `chunker.contextualize(...)`. Il contient le
texte du passage et le contexte structurel utile, par exemple ses titres de
section. Les métadonnées suivantes sont conservées dans chaque chunk :

| Champ | Origine / rôle |
|---|---|
| `id` | Hash stable de `source_id:position`, utilisé comme identifiant Qdrant. |
| `document_id` | `clip-2021`, pour connaître l'article source. |
| `title` | Titre déclaré dans le manifeste de téléchargement. |
| `page` | Première page de provenance Docling connue pour ce passage. |
| `section` | Hiérarchie de titres Docling, rendue lisible avec `>` . |
| `language` | `en`, car cet article est en anglais. |
| `text` | Passage contextualisé envoyé à E5 et affiché à la recherche. |

Le fichier `data/processed/clip-2021/chunks.json` contient les **453 chunks**
produits pour cet article. Il ne contient aucun vecteur : les vecteurs restent
dans Qdrant, tandis que ce fichier permet de contrôler le résultat du
chunking.

L'identifiant est reproductible pour une même source et une même position. Une
réindexation identique réalise donc un *upsert* : Qdrant remplace le point au
même identifiant au lieu d'ajouter un doublon. Si les règles de chunking sont
modifiées de façon importante, une étape future de réindexation complète devra
aussi supprimer les anciens chunks de cette source.

## Collections Qdrant

Deux collections sont maintenant séparées :

| Collection | Contenu | Rôle |
|---|---|---|
| `evidence_chunks_e5_small` | Les 6 passages synthétiques initiaux | Démonstration contrôlée. |
| `arxiv_chunks_e5_small` | Les 453 passages de l'article CLIP | Corpus scientifique réel. |

Cette séparation évite de mélanger des données de test et des articles. Le nom
inclut aussi le modèle d'embeddings : changer de modèle signifie changer
d'espace vectoriel et donc créer une autre collection, pas insérer ses vecteurs
dans celle-ci.

## Routes ajoutées

| Route | Rôle |
|---|---|
| `POST /documents/{source_id}/index` | Recharge `document.json`, produit `chunks.json`, encode les chunks avec E5 et les insère dans Qdrant. |
| `GET /arxiv/search?query=...&limit=...` | Cherche uniquement dans la collection des articles arXiv. |

La route d'indexation retourne le nombre de chunks, le modèle utilisé, la
collection et le chemin de l'artefact `chunks.json`.

## Validation observée

La recherche suivante a été exécutée avec succès :

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/arxiv/search?query=How%20does%20CLIP%20learn%20from%20images%20and%20text%3F"
```

Le service a renvoyé des résultats depuis `arxiv_chunks_e5_small`, dont le
premier avait un score d'environ **0,8977**, le `document_id` `clip-2021` et le
titre de l'article CLIP. La collection est également visible dans le tableau de
bord Qdrant.

Cela valide la chaîne complète : PDF réel → structure Docling → chunks
contextualisés → vecteurs E5 → Qdrant → récupération sémantique.

## Limites actuelles

- La recherche renvoie les preuves, mais aucun LLM ne rédige encore de réponse.
- Il n'y a qu'un article arXiv et aucune recherche hybride mot-clé + vecteur.
- Le score de similarité n'est pas une probabilité ni une mesure de véracité.
- Les figures et tableaux sont conservés par Docling, mais ils ne sont pas
  encore interprétés par un VLM.

## Suite

Le prochain jalon introduira un LLM local pour générer une réponse uniquement à
partir des passages retrouvés, avec citations page/section et comportement
explicite d'abstention lorsque les preuves sont insuffisantes. C'est le passage
de la récupération sémantique au RAG génératif.
