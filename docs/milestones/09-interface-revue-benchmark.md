# Milestone 09 — Interface de revue humaine du benchmark

## Objectif

Remplacer l'édition manuelle du fichier JSONL par une interface locale conçue
pour relire les 30 cas du benchmark, examiner leurs preuves, corriger les champs
éditoriaux et enregistrer une décision de validation traçable.

L'interface est disponible à `http://localhost:8000/review/`. Un lien
`Évaluer le RAG` est également présent dans la barre supérieure du chat.

## Parcours de revue

La page possède trois niveaux :

| Zone | Responsabilité |
|---|---|
| File latérale | Progression, recherche, filtres par état et catégorie, sélection d'un cas. |
| Éditeur central | Question, vérité terrain, termes attendus, preuves et notes. |
| Barre d'action | Enregistrement en brouillon ou validation humaine. |

Chaque cas affiche :

- son identifiant stable, sa langue et sa catégorie ;
- le périmètre documentaire prévu ;
- la question modifiable ;
- la réponse de référence modifiable ;
- les groupes de termes acceptés, une ligne par fait et `|` entre variantes ;
- les documents attendus en lecture seule ;
- le texte intégral du chunk contenant chaque fragment de preuve ;
- des notes libres de revue ;
- son état `À vérifier` ou `Vérifié`.

Les cas d'abstention masquent la réponse et les termes, car leur vérité terrain
est précisément l'absence de réponse justifiée par le corpus.

## API de revue

### Lecture

`GET /evaluation/cases` charge le JSONL, calcule la progression et enrichit
chaque cible avec le premier chunk correspondant dans `data/processed`.

Les identifiants de chunks sont renvoyés sous forme de chaînes. Qdrant utilise
des entiers 64 bits, tandis que JavaScript ne représente exactement que les
entiers jusqu'à `2^53 - 1`. Une chaîne évite donc qu'un identifiant soit arrondi
dans l'interface.

### Écriture

`PATCH /evaluation/cases/{case_id}` accepte seulement :

- `question` ;
- `reference_answer` ;
- `required_answer_terms` ;
- `notes` ;
- `review_status`.

Les identifiants, la catégorie, la langue, les documents attendus, les cibles
de preuve et la règle d'abstention ne sont pas modifiables depuis l'UI. Cette
séparation protège la structure du benchmark pendant la revue éditoriale.

## Sécurité de l'écriture

Une sauvegarde suit cet ordre :

1. verrouillage du processus pour éviter deux écritures concurrentes ;
2. nettoyage des espaces et alternatives vides ;
3. reconstruction et validation Pydantic du cas complet ;
4. pour une validation, vérification que toutes les preuves existent encore ;
5. sérialisation de l'ensemble du benchmark dans un fichier temporaire ;
6. relecture et validation du fichier temporaire ;
7. copie de la version courante vers `rag_benchmark.backup.jsonl` ;
8. remplacement atomique du fichier canonique.

Le montage Compose `./data/evaluation:/data/evaluation` est donc inscriptible,
contrairement à l'étape précédente. En contrepartie, le port de l'API est lié à
`127.0.0.1:8000` : le service d'écriture reste accessible uniquement depuis ce
PC, comme Qdrant.

## État dans le navigateur

Le JavaScript conserve seulement l'état d'affichage : file filtrée, cas actif,
modifications non enregistrées et requête en cours. La source de vérité reste le
JSONL côté serveur.

Changer de cas avec des modifications non sauvegardées déclenche une demande de
confirmation. La progression est recalculée après chaque sauvegarde. Les
touches `Alt + ←` et `Alt + →` permettent de naviguer dans la file filtrée.

## Fichiers

| Fichier | Rôle |
|---|---|
| `app/benchmark_review.py` | Chargement enrichi, contrat de modification et écriture atomique. |
| `app/main.py` | Routes de revue et montage de l'interface statique. |
| `app/review_static/index.html` | Structure accessible de la page. |
| `app/review_static/review.css` | Mise en page desktop et mobile. |
| `app/review_static/review.js` | Filtres, navigation, édition, validation et progression. |
| `data/evaluation/rag_benchmark.backup.jsonl` | Dernière version avant la sauvegarde la plus récente. |

## Validation réalisée

- syntaxe Python validée ;
- `GET /evaluation/cases` retourne 30 cas et 31 extraits de preuve ;
- une sauvegarde neutre par `PATCH` conserve le statut et le contenu ;
- le backup est créé ;
- le validateur complet du benchmark passe après la sauvegarde ;
- l'interface charge 30 éléments, sélectionne le premier cas, affiche ses
  champs et son chunk de preuve ;
- les identifiants de chunks sont protégés contre l'arrondi JavaScript.

## Note d'environnement Docker

Pendant ce milestone, BuildKit a cessé de produire des journaux avant même la
lecture du Dockerfile, alors que le moteur et les conteneurs continuaient de
répondre. Les clients de build bloqués ont été arrêtés sans toucher aux volumes.

Pour terminer la validation, les fichiers ont été déployés dans le conteneur
API, puis le conteneur testé a été enregistré comme image locale
`evidentia-app:dev`. Une recréation `--no-build` depuis cette image a confirmé
que l'interface et ses routes survivent sans copies manuelles. Le Dockerfile et
le contexte source restent la définition reproductible ; Docker Desktop devra
être redémarré avant la prochaine reconstruction normale avec `--build`.
