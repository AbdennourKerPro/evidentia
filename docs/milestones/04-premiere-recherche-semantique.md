# Milestone 04 — Première recherche sémantique

## Objectif

Construire un pipeline RAG minimal qui transforme un corpus de textes en
vecteurs, les stocke dans Qdrant, puis retrouve les passages sémantiquement
proches d'une question. À ce stade, aucun LLM génératif ne rédige encore de
réponse : le résultat est volontairement la liste des preuves retrouvées.

## Chaîne complète réalisée

```text
Corpus contrôlé
    │
    ├── chunks avec métadonnées (source, titre, page, langue, texte)
    │
    ├── préfixe "passage: "
    │
    ├── modèle multilingual-e5-small
    │       └── vecteur dense normalisé de 384 dimensions
    │
    └── Qdrant : collection `evidence_chunks_e5_small`

Question utilisateur
    │
    ├── préfixe "query: "
    ├── même modèle d'embeddings
    ├── recherche par similarité cosinus dans Qdrant
    └── résultats : texte + provenance + score
```

## Corpus de démonstration

`app/demo_corpus.py` contient six chunks synthétiques, en français et en
anglais, sur des notes de recherche concernant la chaleur urbaine, des capteurs
de température, la couverture arborée et la rétention d'eau.

Le corpus est synthétique afin de vérifier sans ambiguïté :

- l'encodage multilingue ;
- la structure des métadonnées ;
- l'indexation idempotente ;
- la recherche vectorielle ;
- la conservation de la provenance.

Les vrais PDF et les documents multimodaux seront ajoutés plus tard, une fois
ce socle vérifié.

## Modèle d'embeddings

Le pipeline utilise `intfloat/multilingual-e5-small`.

| Propriété | Choix |
|---|---|
| Langues | Multilingue, donc adapté aux questions et sources françaises / anglaises |
| Taille de vecteur | 384 dimensions |
| Passage | Préfixe obligatoire `passage: ` |
| Question | Préfixe obligatoire `query: ` |
| Similarité | Cosinus |
| Chargement | Différé au premier appel et conservé en mémoire du processus |
| Révision du modèle | Épinglée dans le code pour la reproductibilité |

Les vecteurs sont normalisés. Avec des vecteurs de norme unitaire, leur produit
scalaire correspond à la similarité cosinus ; Qdrant peut donc comparer leur
direction sémantique plutôt que leur longueur.

## Structure d'un point Qdrant

Chaque chunk devient un `PointStruct` avec :

- un `id` stable ;
- un vecteur dense ;
- un `payload` contenant le texte et la provenance.

Un `upsert` avec le même identifiant remplace le point existant. L'indexation
de démonstration peut donc être relancée sans créer de doublons.

La collection est nommée `evidence_chunks_e5_small`. Le nom inclut le modèle :
un second modèle d'embeddings, par exemple Qwen3-Embedding, devra disposer de
sa propre collection car ses vecteurs n'ont pas la même dimension ni le même
espace sémantique.

## Routes API ajoutées

| Route | Rôle |
|---|---|
| `POST /demo/index` | Encode les six chunks et les insère dans Qdrant. |
| `GET /search?query=...&limit=...` | Encode la question et renvoie les chunks les plus proches. |

La réponse de recherche expose volontairement le `score`, le `text`, le
`document_id`, le `title`, la `page`, la `section` et la `language`. Une couche
LLM future recevra ces mêmes éléments afin de générer une réponse avec des
citations contrôlables.

## Commandes exécutées et résultat observé

### Indexation

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/demo/index
```

Résultat :

```text
collection               indexed_chunks embedding_model
----------               -------------- -------------------------------
evidence_chunks_e5_small 6              intfloat/multilingual-e5-small
```

La collection Qdrant a été créée et six points ont été persistés.

### Recherche

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/search?query=How%20were%20the%20sensors%20calibrated%3F"
```

Le premier résultat observé est le chunk `id=3`, issu de
`sensor-protocol-2025`, section `Calibration`, avec un score d'environ
`0.8569`. Son texte décrit la comparaison des capteurs avec un instrument de
référence avant le déploiement et après la récupération.

Le résultat correspond à la question alors que le corpus contient aussi des
documents sur la chaleur, la végétation et les jardins de pluie. Cela valide la
recherche par sens et non une simple recherche de mots exacts.

## Persistance

Deux volumes Docker sont maintenant utilisés :

| Volume | Contenu | Raison |
|---|---|---|
| `qdrant_storage` | Collections, vecteurs, index, payloads | Les données de recherche survivent au conteneur Qdrant. |
| `model_cache` | Fichiers téléchargés depuis Hugging Face | Le modèle n'est pas retéléchargé à chaque recréation de l'API. |

`docker compose down` conserve ces volumes. `docker compose down -v` les
supprime explicitement et imposerait une nouvelle indexation et un nouveau
téléchargement du modèle.

## Ce qui est validé

- communication API → modèle d'embeddings → Qdrant ;
- corpus bilingue ;
- indexation idempotente ;
- recherche vectorielle ;
- provenance détaillée dans les résultats ;
- visualisation de la collection dans le tableau de bord Qdrant.

## Ce qui ne l'est pas encore

- extraction de texte depuis de vrais PDF ;
- découpage automatique de documents ;
- filtres de métadonnées, recherche hybride et reranking ;
- génération de réponses par LLM ;
- agents LangGraph, VLM, SFT/DPO et évaluation formelle.

## Suite

Le prochain jalon fera passer Evidentia d'un corpus codé en dur à de vrais
documents locaux. Nous définirons d'abord le format d'entrée, les règles de
chunking et un petit jeu d'évaluation avant d'introduire l'extraction de PDF.
