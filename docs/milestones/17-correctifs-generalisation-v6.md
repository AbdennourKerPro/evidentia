# Milestone 17 — Correctifs de généralisation et pipeline v6

## Objectif

La campagne v5 de 30 cas a atteint 27 réussites sur 30. Les trois échecs ont
été analysés à partir des preuves réellement sélectionnées et des traces du
contrat. Cette itération les corrige sans mémoriser un article, une valeur
numérique ou une réponse de benchmark.

Les corrections portent sur trois capacités générales :

- expliquer un mécanisme d'inférence ;
- couvrir fidèlement des faits atomiques déjà prouvés ;
- restituer une comparaison quantitative dans la langue demandée.

Le pipeline est versionné
`langgraph_fact_plan_intent_multifacet_response_contract_v6`. Un rapport v5 ne
doit jamais être repris avec `--resume` sous cette version, car cela mélangerait
deux comportements différents.

## Diagnostic v5

| Cas | Preuve sélectionnée | Cause de l'échec |
|---|---|---|
| `clip-zero-shot-classifier-en` | document correct, mais passages d'introduction | le passage décrivant score, température et softmax n'était pas dans les cinq chunks |
| `blip2-bottleneck-fr` | toutes les preuves utiles présentes | le LLM omettait `requêtes apprenables` et utilisait une paraphrase française imprécise |
| `blip2-efficiency-en` | la phrase contenant tâche, gain et ratio présente | un brouillon dans une mauvaise langue restait invalide après correction |

Le rappel de document de 1,00 n'était donc pas suffisant pour diagnostiquer le
premier cas : retrouver le bon PDF ne signifie pas retrouver le bon passage.

## 1. Intentions réutilisables

`app/query_expansion.py` ajoute deux intentions activées par les termes de la
question, sans connaître le corpus :

- `inference_mechanism` : pour les questions de classification/prédiction
  zero-shot ; la facette cherche les entrées candidates, le score, la
  normalisation et la décision ;
- `quantitative_comparison` : pour les compromis d'efficacité ou de coût ; la
  facette cherche méthodes, tâche, différence de performance et ratio de
  ressources.

Le reranker privilégie ensuite les sections qui décrivent ce mécanisme ou ce
résultat. Aucun nom de modèle, article, score ou valeur du benchmark ne figure
dans ces règles.

## 2. Plan factuel du mécanisme d'inférence

`app/fact_planning.py` peut construire, seulement lorsque les passages les
contiennent, une checklist de :

- noms/classes candidats encodés comme texte ;
- similarité cosinus ;
- température ;
- softmax.

Comme pour les plans antérieurs, les faits viennent exclusivement des chunks
sélectionnés et chaque fait conserve ses références autorisées. Le prompt ne
doit citer ni les identifiants internes `F1`, `F2`, etc. ; ils sont également
supprimés défensivement dans `app/rag_service.py` avant validation et affichage.

## 3. Complément factuel déterministe

Après l'unique correction LLM, `repair_corrected_answer_from_evidence()` ne
s'active que si le brouillon est publiable mais qu'il manque encore des faits du
plan. Il ajoute alors une phrase courte provenant de l'affirmation planifiée et
sa première référence autorisée.

Cette étape ne synthétise aucune information nouvelle : le fait et la citation
ont déjà été extraits des preuves. Elle évite qu'une paraphrase du LLM fasse
échouer une réponse pourtant parfaitement soutenue par les chunks. Les
synonymes français incorrects (`entrave d'information`, `LLM froid`,
`encodeur ... froid`) ont été retirés des alternatives acceptées.

## 4. Réparation des comparaisons numériques

Si, après correction, le seul défaut est une réponse anglaise attendue mais
produite dans une autre langue, et qu'une intention
`quantitative_comparison` est présente, le système peut extraire une unique
phrase anglaise déjà présente dans une preuve. La phrase doit contenir :

- au moins un nombre ;
- des termes de comparaison ;
- des termes de ressources ou de paramètres.

Elle est retournée avec sa référence `[Sx]`. Le mécanisme est borné à une preuve
dont `language == "en"` ; il ne traduit pas, ne calcule pas et n'invente pas de
valeur. Les autres langues conservent la correction LLM normale.

## 5. Nouvelle observabilité de l'évaluation

`scripts/rag_evaluation_common.py` enregistre maintenant :

- `selected_evidence_recall` par cas ;
- `matched_evidence_targets` ;
- `mean_selected_evidence_recall` dans l'agrégat.

Cette mesure vérifie que les chunks fournis au LLM contiennent effectivement la
phrase cible revue dans le benchmark. Elle complète, sans le remplacer, le
rappel des documents.

## Tests et validation

Les tests déterministes couvrent :

- détection/reranking de l'inférence et des comparaisons quantitatives ;
- extraction des quatre faits d'un mécanisme de décision ;
- retrait des synonymes français imprécis ;
- complément d'un fait manquant avec citation locale ;
- récupération d'une comparaison numérique anglaise depuis une preuve anglaise ;
- suppression des identifiants internes de checklist.

Résultat : **46/46 tests réussis dans Docker**.

La validation bout-en-bout ciblée a été effectuée dans un nouveau rapport v6 :

```powershell
docker compose run --rm api python -m scripts.evaluate_agentic_rag `
  --case-id clip-zero-shot-classifier-en `
  --case-id blip2-bottleneck-fr `
  --case-id blip2-efficiency-en `
  --output /reports/agentic-rag-v6-three-fixes-final.json
```

Résultat dans `reports/agentic-rag-v6-three-fixes-final.json` :

- **3/3 réussites** ;
- couverture factuelle moyenne : **1,00** ;
- précision/rappel des documents cités : **1,00 / 1,00** ;
- rappel moyen des preuves sélectionnées : **1,00** ;
- contrat de réponse et contrat factuel valides : **1,00** ;
- aucun identifiant interne de type `[F1]` dans la réponse exposée.

Puis, après revue, la campagne complète repart de zéro avec :

```powershell
docker compose run --rm api python -m scripts.evaluate_agentic_rag `
  --output /reports/agentic-rag-v6-final-30.json `
  --resume
```

Le fichier v6 sera checkpointé après chaque cas. L'option `--resume` ne
réutilisera que ses propres checkpoints v6.
