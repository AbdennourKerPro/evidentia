# Benchmarks d'évaluation Evidentia

Ce dossier contient les jeux de questions qui servent à évaluer le RAG. Depuis le
25 octobre 2026, l'évaluation porte uniquement sur des cas en anglais.

| Fichier | Rôle | Cas évalués |
|---|---|---|
| `rag_english30.draft.jsonl` | **Benchmark principal**, défaut des scripts et de l'interface de revue | 30 cas anglais |
| `rag_heldout.draft.jsonl` | Jeu heldout ; évalué avec `--language en` | 20 cas anglais |
| `archive/rag_benchmark.jsonl` | Ancien benchmark mixte français/anglais (jalons 08 à 21), conservé pour reproduire les anciens rapports | — |

Les fichiers `*.backup.jsonl` sont des copies de sécurité créées par l'interface de
revue à chaque enregistrement. Ils sont ignorés par git.

## English30

30 cas, tous relus (`review_status: verified`) : 12 `factoid`, 7 `method`,
5 `comparison` (deux articles chacun) et 6 `abstention`.

- 20 cas `heldout-…-en` : identiques aux 20 cas anglais du heldout.
- 10 cas `english20-…` : détails chiffrés d'entraînement (minibatch, GPU, learning
  rate, paramètres…) ; ce sont eux qui concentrent les échecs historiques.

Les abstentions couvrent trois situations : une information absente des articles,
une entité hors corpus, ou un fait externe attribué à tort à un article.

## Heldout anglais

`rag_heldout.draft.jsonl` contient aussi des cas en français, qui ne sont plus
évalués. Le sous-ensemble anglais est sélectionné à la lecture, sans modifier le
fichier :

```bash
docker compose run --rm --no-deps api python -m scripts.evaluate_agentic_rag \
  --benchmark /data/evaluation/rag_heldout.draft.jsonl --language en \
  --output /reports/mon-rapport.json
```

Il compte 20 cas : 6 `method`, 5 `factoid`, 5 `comparison`, 4 `abstention`.
**Ces 20 cas sont aussi dans English30** : les deux mesures ne sont pas
indépendantes.

## Structure d'un cas

Une ligne JSON par cas :

- `id`, `category`, `language`, `question` ;
- `scope_document_ids` : filtre optionnel d'articles appliqué avant la recherche
  (`null` : tout le corpus) ;
- `expected_document_ids` : articles qui doivent fonder la réponse et être cités ;
- `evidence_targets` : phrases exactes du corpus qui prouvent la réponse ;
- `reference_answer` : réponse de référence relue ;
- `required_answer_terms` : groupes de faits ; chaque groupe liste des formulations
  acceptées pour la métrique de couverture ;
- `should_abstain` : `true` si le corpus ne permet pas de répondre ;
- `review_status`, `notes`.

Les preuves sont des fragments de texte plutôt que des identifiants de chunks : si
le découpage change, le benchmark retrouve toujours la preuve textuelle.

## Règles de relecture

Un cas n'est utilisé comme vérité terrain qu'après relecture humaine
(`needs_human_review` → `verified`), par exemple dans l'interface
`http://127.0.0.1:8000/review/`. Il faut vérifier que :

1. la question n'est pas ambiguë ;
2. `reference_answer` est fidèle aux passages de preuve ;
3. `required_answer_terms` couvre les faits importants sans imposer une
   formulation unique ;
4. pour une abstention, l'information est réellement absente des cinq articles.

`scripts/validate_benchmark.py --benchmark <fichier>` contrôle le schéma, les
identifiants en double et la présence de chaque fragment de preuve dans les chunks
Docling.

Aucune règle propre à un cas du benchmark ne doit être ajoutée au code ou aux
prompts : les corrections restent génériques.
