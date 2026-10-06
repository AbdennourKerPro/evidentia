# Milestone 14 — Contrat de langue, citations et correction LangGraph

## But technique

Transformer une génération libre en sortie contrôlée. Le LLM reste chargé de
rédiger ; du code déterministe décide si son brouillon peut être montré à
l'utilisateur.

Cette séparation est importante : demander « réponds en français » dans un
prompt est une préférence, alors que vérifier la langue après génération est
une contrainte exécutable.

## Composants

### `app/response_contract.py`

Ce module ne charge aucun modèle. Il contient :

- `detect_question_language()` : détection légère anglais/français/chinois à
  partir des mots-outils, accents et scripts Unicode ;
- `validate_answer()` : orchestration de tous les contrôles ;
- `resolve_citations()` : résolution des références individuelles ou groupées ;
- `AnswerValidation` : résultat immuable avec langue, citations, abstention et
  codes d'erreur ;
- `describe_validation_issues()` : transformation des codes en instructions
  de correction lisibles par Qwen.

La détection n'est pas un classifieur linguistique général. Elle est adaptée
aux langues actuellement utilisées dans le benchmark et permet surtout de
rejeter les erreurs évidentes sans dépendance supplémentaire.

### `app/rag_service.py`

La génération est maintenant découpée en deux fonctions :

- `generate_answer_draft()` construit le prompt initial ou correctif ;
- `finalize_answer()` produit une réponse acceptée ou une abstention localisée.

Le prompt comparatif reçoit une table explicite des références disponibles par
article, par exemple :

```text
- blip2-2023: [S1], [S3], [S5]
- llava-2023: [S2], [S4]
```

Le LLM sait ainsi exactement comment satisfaire la couverture multi-source.

### `app/agentic_rag.py`

Trois nœuds s'ajoutent au graphe :

1. `validate_answer` applique le contrat ;
2. `correct_answer` réécrit une seule fois avec les erreurs détectées ;
3. `finalize_answer` expose la réponse ou s'abstient.

L'arête conditionnelle `validate_answer → correct_answer|finalize_answer`
illustre une boucle agentique contrôlée. Le booléen `correction_attempted`
empêche un second tour.

### `app/schemas.py`

La réponse `execution` expose désormais :

- `expected_language` ;
- `correction_attempted` ;
- `response_contract_valid`.

Ces champs et les étapes du graphe rendent chaque décision visible dans l'UI et
dans les rapports.

### `tests/test_response_contract.py`

Treize tests `unittest` s'exécutent sans Qdrant, embedding ou LLM. Ils prennent
environ une milliseconde dans le conteneur.

### `scripts/evaluate_agentic_rag.py`

Le rapport ajoute `response_contract_valid_rate` et `correction_rate`. Le nom du
pipeline est fixé à `langgraph_hybrid_rrf_response_contract_v3`.

## Politique de sûreté

Une réponse est affichée seulement si :

- son script et ses marqueurs correspondent à la langue de la question ;
- elle paraît terminée et ses crochets sont équilibrés ;
- toutes ses références sont dans la fenêtre de preuves ;
- chaque article comparé est cité ;
- elle ne mélange pas une réponse avec `INSUFFICIENT_EVIDENCE`.

Après un premier échec, Qwen reçoit le brouillon précédent comme texte non
fiable, la liste des erreurs et les mêmes preuves. Après un deuxième échec, le
serveur renvoie une abstention dans la langue attendue.

## Résultat

Sur les cinq comparaisons :

- passage : 2/5 en v2, 3/5 en v3 ;
- couverture factuelle : 0,480 → 0,570 ;
- précision et rappel documentaire des citations : 1,000 ;
- contrats valides : 5/5 ;
- corrections finales : 0/5 ;
- latence moyenne : 135,85 s.

Le validateur prend environ 0,2 ms. Le coût vient des réponses plus longues,
pas des contrôles Python.

Le rapport détaillé est `reports/response-contract-v3-summary.md` et la preuve
machine est `reports/agentic-rag-response-contract-v3-final.json`.

## Limites

- une langue courte ou très technique peut contenir trop peu de mots-outils ;
- le contrat ne vérifie pas que chaque proposition individuelle possède sa
  citation la plus proche ;
- citer un article ne garantit pas que le fait est exact ;
- une correction locale peut presque doubler la génération pour une question ;
- le seuil de complétude est syntaxique, pas sémantique.

## Étape suivante

Créer une stratégie de retrieval dédiée aux questions factuelles numériques :
détection de l'intention, requêtes contenant les entités et unités attendues,
puis fusion de plusieurs facettes. L'objectif immédiat est de retrouver
LVD-142M et SA-1B dans le top-5 de la comparaison sur l'échelle des données.
