# Benchmark RAG Evidentia

`rag_benchmark.jsonl` est un premier **benchmark semi-synthétique** de 30 cas
construits à partir des cinq articles du corpus. Il couvre les questions
factuelles, les méthodes, les comparaisons entre articles et l'abstention.

## Ce qui est automatisé

- validation du schéma de chaque ligne ;
- détection des identifiants dupliqués ;
- vérification que chaque fragment de preuve existe réellement dans les chunks
  Docling ;
- calcul des métriques de retrieval et de génération ;
- conservation des réponses et des chunks récupérés dans des rapports JSON.

## Ce qui exige une vérification humaine

Le champ `review_status` vaut initialement `needs_human_review`. Pour chaque cas,
il faut confirmer trois éléments :

1. la question ne contient pas d'ambiguïté ;
2. `reference_answer` est fidèle au passage de preuve ;
3. les groupes `required_answer_terms` couvrent les faits importants sans
   imposer une formulation artificiellement unique.

Après validation, remplacer `needs_human_review` par `verified` sur la ligne.
Il n'est pas nécessaire de créer toutes les questions à la main. En revanche,
une référence non relue peut contenir les mêmes erreurs qu'un générateur et ne
constitue donc pas une vérité terrain fiable.

## Structure d'un cas

- `scope_document_ids` : filtre optionnel appliqué avant la recherche ; `null`
  signifie que tout le corpus est interrogé ;
- `expected_document_ids` : documents qui doivent contribuer à la réponse ;
- `evidence_targets` : fragments stables recherchés dans les chunks attendus ;
- `reference_answer` : réponse de référence relue ;
- `required_answer_terms` : groupes d'expressions alternatives utilisés par la
  métrique déterministe de couverture ;
- `should_abstain` : indique que le corpus ne permet pas de répondre.

Les fragments de preuve sont préférés comme vérité terrain aux seuls IDs de
chunks : si le découpage change légèrement, le benchmark peut encore retrouver
la preuve textuelle au lieu de devenir entièrement obsolète.
