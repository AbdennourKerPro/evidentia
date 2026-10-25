# Évaluation LangGraph v3 — contrat de réponse

## Objectif

La v2 améliorait les chunks fournis au LLM, mais certaines réponses restaient
dans la mauvaise langue, tronquées ou ne citaient qu'un seul article d'une
comparaison. La v3 ajoute un contrat déterministe après la génération et une
seule tentative corrective optionnelle.

Le validateur ne juge pas la vérité scientifique avec un second LLM. Il vérifie
des propriétés observables :

- langue attendue à partir de la question ;
- absence d'un script inattendu, notamment une réponse chinoise à une question
  française ;
- réponse non tronquée et crochets équilibrés ;
- citations `[S…]` résolubles par le serveur ;
- au moins une citation provenant de chaque article comparé ;
- abstention non mélangée avec une réponse ordinaire.

Les citations individuelles `[S1] [S2]` et groupées `[S1, S2]` sont acceptées.
Une forme tronquée comme `[S4` reste rejetée.

## Graphe v3

```text
generate_answer
      |
validate_answer
      |
      +-- valide --------------------> finalize_answer
      |
      +-- invalide, aucun essai -----> correct_answer
                                            |
                                      validate_answer
                                            |
                                      finalize_answer
```

La branche corrective est limitée à un passage. Après la deuxième validation,
une réponse encore invalide devient une abstention localisée dans la langue de
la question.

## Protocole final

- benchmark : `data/evaluation/rag_benchmark.jsonl` ;
- cas : cinq comparaisons vérifiées ;
- retrieval : dense E5 + BM25 + RRF, équilibré par article ;
- générateur : Qwen2.5-7B-Instruct INT4 avec OpenVINO CPU ;
- fenêtre : cinq chunks ;
- pipeline : `langgraph_hybrid_rrf_response_contract_v3` ;
- rapport : `reports/agentic-rag-response-contract-v3-final.json`.

## Résultats agrégés

| Mesure | LangGraph hybride v2 | Contrat v3 |
|---|---:|---:|
| Réussite end-to-end | 2/5 (0,400) | **3/5 (0,600)** |
| Couverture factuelle | 0,480 | **0,570** |
| Précision documentaire des citations | 0,800 | **1,000** |
| Rappel documentaire des citations | 0,700 | **1,000** |
| Documents attendus dans les chunks | 1,000 | 1,000 |
| Bonne décision réponse/abstention | 0,800 | **1,000** |
| Contrat de réponse valide | non mesuré | **1,000** |
| Taux de correction | non applicable | 0,000 |
| Latence moyenne | 104,62 s | 135,85 s |

Le prompt explicite a suffi à produire cinq premiers brouillons valides dans la
campagne finale. La boucle corrective n'a donc ajouté aucun appel LLM à cette
mesure. Elle a néanmoins été exercée pendant les smoke tests, où elle a corrigé
une mauvaise langue et révélé une citation de source manquante.

L'augmentation de latence vient surtout de réponses comparatives plus longues :
le budget maximal est passé de 350 à 450 tokens pour éviter les sorties
tronquées.

Décomposition moyenne :

- planification : 21,85 s ;
- une branche retrieval : 91,5 ms ;
- génération : 113,88 s ;
- validation : 0,2 ms.

## Résultats par cas

| Cas | v2 → v3 | Couverture v3 | Contrat | Diagnostic |
|---|---:|---:|---:|---|
| CLIP–DINOv2, supervision | échec → **réussite** | 0,60 | valide | Les deux signaux de supervision et les deux sources sont cités. |
| BLIP-2–LLaVA, pont | réussite → **réussite** | 0,60 | valide, français | Q-Former et projection linéaire sont distingués. |
| CLIP–SAM, prompts | réussite → échec | 0,40 | valide | Réponse bien sourcée mais trop générale pour trois groupes factuels. |
| Échelle des données | échec → échec | 0,25 | valide | CLIP 400M est présent ; les valeurs LVD-142M et SA-1B manquent aux chunks. |
| Composants gelés | échec → **réussite** | 1,00 | valide | Les composants gelés et les deux ponts entraînés sont couverts et cités. |

Le contrat ne transforme donc pas une preuve absente en fait correct. Il
garantit la forme et la provenance de la réponse, tandis que le benchmark
continue de mesurer le contenu attendu.

## Vérifications techniques

Treize tests unitaires déterministes couvrent :

- détection français, anglais et chinois ;
- réponse multi-source valide en français et en anglais ;
- source obligatoire non citée ;
- script inattendu ;
- référence hors plage ;
- citation tronquée ;
- citation groupée valide ;
- abstention prématurée ou mélangée à une réponse.

Commande :

```powershell
docker compose run --rm api python -m unittest -v tests.test_response_contract
```

## Décision

Conserver la v3 : elle améliore à la fois le passage end-to-end, la couverture
factuelle et la provenance, sans correction supplémentaire dans la campagne
finale. La boucle corrective reste une protection de secours.

La prochaine itération doit revenir au retrieval ciblé pour les deux échecs :

1. détecter les questions numériques et rechercher explicitement valeurs,
   tailles de datasets et unités ;
2. générer plusieurs sous-requêtes par facette pour CLIP–SAM ;
3. évaluer ensuite la v3 sur les 25 questions répondables, pas seulement les
   cinq comparaisons.
