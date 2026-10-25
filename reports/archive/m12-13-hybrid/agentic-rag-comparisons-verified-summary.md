# Évaluation LangGraph — cinq comparaisons vérifiées

## Protocole

- benchmark : `data/evaluation/rag_benchmark.jsonl` ;
- cas : les cinq entrées de catégorie `comparison` ;
- pipeline : `langgraph_balanced_per_source_v1` ;
- embedding : `intfloat/multilingual-e5-small` ;
- générateur et planificateur : Qwen2.5-7B-Instruct INT4 avec OpenVINO ;
- collection : `arxiv_chunks_e5_small` ;
- `top_k` final : 5 chunks ;
- candidats : 20 chunks par article avant reranking ;
- exécution : séquentielle sur CPU local.

Commande exécutée :

```powershell
docker exec evidentia-api-1 python -u -m scripts.evaluate_agentic_rag `
  --limit 5 `
  --case-id compare-clip-dinov2-supervision-en `
  --case-id compare-blip2-llava-bridge-fr `
  --case-id compare-clip-sam-prompts-en `
  --case-id compare-training-data-scale-en `
  --case-id compare-frozen-components-en `
  --output /reports/agentic-rag-comparisons-verified.json
```

## Résultats agrégés

| Mesure | Baseline sur les 5 comparaisons | LangGraph v1 |
|---|---:|---:|
| Réussite end-to-end | 0/5, soit 0 % | 2/5, soit 40 % |
| Couverture factuelle moyenne | 8 % | 32 % |
| Rappel documentaire des chunks | 73,3 % | 100 % |
| Précision documentaire des citations | 20 % | 60 % |
| Rappel documentaire des citations | 20 % | 60 % |
| Décision réponse/abstention | 20 % | 60 % |
| Latence moyenne | 46,21 s | 84,05 s |

La latence agentique est 1,82 fois celle de la baseline sur ce sous-ensemble.
La campagne complète a duré environ 438,5 secondes, soit 7 min 18 s.

Décomposition moyenne :

- planification et embedding des sous-requêtes : 22,95 s ;
- une branche Qdrant : 73,5 ms ;
- génération et validation : 60,99 s.

## Résultats par cas

| Cas | Passage automatique | Couverture | Chunks cibles présents | Diagnostic |
|---|---:|---:|---:|---|
| CLIP–DINOv2, supervision | oui | 0,60 | 2/2 | Les deux objectifs sont récupérés et les deux articles cités. |
| BLIP-2–LLaVA, pont visuel/langage | non, abstention | 0,00 | 1/2 | La preuve LLaVA est présente, mais le passage BLIP-2 sur le Q-Former manque. |
| CLIP–SAM, rôle des prompts | non | 0,20 | 0/2 | Les deux articles sont cités, mais les chunks décrivant les templates CLIP et les types de prompts SAM manquent. |
| Échelle des données, trois articles | non, abstention | 0,00 | 0/3 | Les documents sont équilibrés, mais aucun des trois passages chiffrés n'entre dans le top-5 final. |
| BLIP-2–LLaVA, composants gelés | oui | 0,80 | 1/2 | Des preuves alternatives suffisent au seuil, mais le Q-Former est omis. |

Au total, seulement 4 des 11 chunks cibles validés sont présents dans les
fenêtres finales, malgré un rappel documentaire de 100 %.

## Interprétation

L'hypothèse « une recherche filtrée et équilibrée par article corrige la
comparaison » est partiellement validée. Le graphe résout la monopolisation du
top-k par une seule source : chaque comparaison reçoit tous les articles
attendus. Cette amélioration fait passer deux cas qui échouaient avec la
baseline.

Elle ne résout pas encore la sélection du meilleur passage à l'intérieur de
chaque article. Le planificateur produit des questions cohérentes mais souvent
générales. Le reranking lexical privilégie les bonnes sections sans toujours
isoler le chunk qui contient le mécanisme ou la valeur attendue. Avec cinq
places et trois articles, une seule place est garantie à la troisième source,
ce qui rend la comparaison des tailles de données particulièrement fragile.

Les deux abstentions sont conformes aux chunks réellement fournis au LLM : le
garde-fou refuse de reconstruire les preuves absentes. Elles sont toutefois
incorrectes vis-à-vis du corpus complet et du benchmark. L'erreur appartient
donc principalement au retrieval/reranking, en amont de la génération.

## Limite du score automatique

Le cas `compare-frozen-components-en` est compté comme réussi parce que :

- la couverture factuelle atteint 0,8, au-dessus du seuil 0,5 ;
- les deux documents attendus sont cités ;
- aucune source étrangère n'est citée.

La réponse omet néanmoins le Q-Former de BLIP-2 et présente trop globalement la
projection linéaire de LLaVA. Le taux de 40 % doit donc être interprété comme le
taux de passage de la métrique actuelle, pas comme 40 % de réponses humaines
parfaites. Une revue humaine ou un juge calibré devra compléter les termes
explicites.

## Décision technique suivante

La priorité reste le retrieval. Avant de modifier le prompt de réponse ou de
faire du SFT, il faut comparer hors LLM plusieurs variantes sur les chunks
cibles validés :

1. reranking lexical actuel ;
2. plusieurs sous-requêtes par source avec fusion de rangs ;
3. cross-encoder multilingue sur les 20 candidats ;
4. allocation dynamique du nombre de chunks selon le nombre de sources.

L'évaluation retrieval-only est rapide et permettra de choisir une variante
sur les 25 cas répondables avant de relancer une campagne Qwen coûteuse.
