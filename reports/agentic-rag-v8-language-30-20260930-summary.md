# Validation complète v8 — contrôle de langue

Date : 30 septembre 2026. Campagne terminée : 30/30 cas, sans modification de code pendant l'évaluation.

## Résultat principal

**28/30 réussites (93,3 %), contre 27/30 (90 %) en v7.**

Deux gains et une perte par rapport à v7 :

- `llava-scienceqa-fr` : échec → réussite ; gain directement cohérent avec la correction de langue et le replay à texte identique de la milestone 20.
- `compare-blip2-llava-bridge-fr` : échec → réussite ; couverture des termes de 40 % à 60 %. Ce cas n'a pas sollicité le juge de langue : ne pas attribuer ce gain à la correction linguistique.
- `compare-clip-sam-prompts-en` : réussite → échec ; couverture des termes de 60 % à 40 %. Les textes des cinq chunks sélectionnés sont exactement identiques entre les deux campagnes ; la réponse générée varie. Ce n'est pas une preuve de régression provoquée par le nouveau contrôle de langue.

## Conditions de comparaison

- Modèle : `gpt-6-luna`, Responses API, effort `low`.
- Profil : `question_rerank` ; top-k = 5.
- Configuration LLM, benchmark et profil identiques à v7.
- 25 questions répondables et 5 abstentions attendues.
- Une exécution par cas, avec au maximum une correction interne : pas de répétitions dans cette campagne complète.
- Pipeline : `langgraph_language_contract_v8_openai_v3`.
- Benchmark SHA-256 : `31917579b4753cf27d3eca756dbf7b8ef23917f6bfcb1bdb6a7b2414a4aab74b`.
- Aucun changement de benchmark, retrieval, prompt de génération ou dépendances pendant cette validation.
- Rapport v7 conservé : `agentic-rag-v7-question-rerank-30-20260930.json`.
- Nouveau rapport : `agentic-rag-v8-language-30-20260930.json`.

Commande exécutée depuis le dossier de l'application :

```powershell
docker compose run --rm --no-deps api python -m scripts.evaluate_agentic_rag --limit 5 --output /reports/agentic-rag-v8-language-30-20260930.json
```

Le script a enregistré un checkpoint après chaque cas, puis un rapport complet.
L'image déployée contient la correction de langue et avait passé 124 tests unitaires et d'intégration avant cette campagne ; cette suite n'a pas été relancée ici.

## Indicateurs

| Mesure | v7 | v8 |
|---|---:|---:|
| Réussites | 27/30 | 28/30 |
| Couverture moyenne des groupes de termes attendus, sur 25 questions répondables | 82,4 % | 82,3 % |
| Rappel moyen des preuves cibles sélectionnées | 79,3 % | 81,3 % |
| Précision et rappel moyens des articles cités | 92 % | 96 % |
| Rappel des articles retrouvés | 100 % | 100 % |
| Décision répondre/s'abstenir correcte, sur les 30 cas | 28/30 | 29/30 |
| Abstentions attendues réussies | 5/5 | 5/5 |
| Contrats de réponse valides | 28/30 | 29/30 |
| Cas ayant nécessité une correction | 9/30 | 8/30 |
| Appels LLM logiques | 44 | 45 |
| Tokens cumulés | 80 461 | 80 976 |
| Latence moyenne par cas | 7,49 s | 6,10 s |

La latence médiane v8 est 4.43 s. Le premier cas prend 21.33 s, incluant le chargement local des modèles. Ces moyennes ne permettent pas d'attribuer un gain de vitesse à la modification : caches, CPU, réponses et réseau ne sont pas contrôlés.

Usage v8 : 73 882 tokens d'entrée et 7 094 de sortie, dont le raisonnement comptabilisé par la passerelle. Il ne s'agit pas d'un coût monétaire estimé.

Résultats par catégorie : méthodes 12/12, faits ponctuels 7/8, comparaisons 4/5, abstentions 5/5.

## Validation de la langue

Deux appels de classification ont été nécessaires, tous deux acceptés sans réécriture :

- ScienceQA en français : réponse « Après fine-tuning sur ScienceQA, LLaVA combiné à GPT-4 atteint une précision de 92,53 % [S3, S4]. » ; vérification environ 0,97 s.
- Efficacité BLIP-2 en anglais : réponse chiffrée sur VQAv2, 8,7 % et 54× moins de paramètres entraînables ; vérification environ 1,44 s.

Aucune erreur `language_mismatch` ou `language_uncertain` n'apparaît dans les traces de validation de cette campagne. Cela ne certifie pas la langue de toutes les réponses hors de ce panel.

ScienceQA couvre 2/3 groupes attendus : valeur et ScienceQA. La mention explicite « état de l'art » manque toujours. Le cas passe le seuil existant de 50 % ; sa réussite n'est pas une couverture exhaustive.

## Deux échecs restants

### 1. Taille des données CLIP — `clip-data-scale-fr`

La preuve des 400 millions de paires n'est pas dans les cinq chunks retenus : rappel des passages cibles 0. Le modèle renvoie `INSUFFICIENT_EVIDENCE`, puis le finaliseur une abstention française. Le contrat local est valide, mais la question était répondable dans le corpus : l'évaluation de bout en bout échoue.

Ce défaut persiste depuis v7. Il concerne la sélection des preuves, pas la langue.

### 2. Rôle des prompts CLIP/SAM — `compare-clip-sam-prompts-en`

La réponse anglaise explique les classes/templates CLIP, mais ne couvre pas complètement les types de prompts SAM attendus. Elle obtient 2/5 groupes de termes, avec un rappel des passages cibles de 0,5. La source décrivant explicitement les prompts clairsemés et denses n'est toujours pas sélectionnée.

En v7, les mêmes textes de chunks avaient donné une réponse comportant notamment « points », qui franchissait le seuil avec 3/5 groupes. La réponse v8 parle notamment d'un « single foreground point », mais le groupe lexical attendu « points » n'est pas reconnu. Les boîtes et les masques restent absents des deux réponses. La différence réussite/échec combine donc un manque documentaire réel, une variation de formulation et la sensibilité du score lexical au singulier/pluriel ; elle n'établit pas à elle seule une détérioration sémantique de la solution.

La planification des requêtes a produit des formulations différentes, mais les passages finaux sont identiques. Aucun appel de vérification LLM de langue sur ce cas : la langue anglaise est acceptée par l'heuristique. La correction déclenchée porte sur des citations de faits planifiés, pas sur la langue.

## Fragilités que le score global masque

- BLIP-2/LLaVA passe à 60 % de couverture des termes, mais le rappel des preuves cibles reste **0**. Les extraits utilisés ne permettent toujours pas de nommer précisément les encodeurs visuels et modèles de langage demandés. C'est une réussite partielle selon les critères actuels, pas la résolution du retrieval architectural.
- CLIP/électricité s'abstient correctement, mais garde `premature_abstention` dans les contrats. Ce défaut existait avant : les preuves ne contiennent pas la consommation électrique et une liste de faits planifiés rend néanmoins l'abstention localement « prématurée ». Le contrat final n'est donc pas valide sur ce cas, même si la décision produit attendue est correcte.
- Le validateur des faits planifiés atteint 100 % de conformité, mais il ne couvre que les faits effectivement planifiés et n'atteste pas que toutes les demandes sont couvertes.
- Précision/rappel des articles cités ne signifient pas vérification de chaque affirmation. Le scoring par groupes de termes reste à confronter à une lecture humaine.

## Détail des 30 cas

Les colonnes de couverture sont des mesures lexicales et de rappel des passages cibles, non un jugement de vérité. Les abstentions n'ont pas de couverture factuelle applicable.

| Cas | v7 | v8 | Groupes de termes couverts v8 | Preuves cibles v8 |
|---|---|---|---:|---:|
| clip-training-objective-en | ✓ | ✓ | 100.0 % | 100.0 % |
| clip-zero-shot-classifier-en | ✓ | ✓ | 100.0 % | 100.0 % |
| clip-data-scale-fr | ✗ | ✗ | 0.0 % | 0.0 % |
| clip-prompt-ensembling-en | ✓ | ✓ | 50.0 % | 100.0 % |
| dinov2-objectives-en | ✓ | ✓ | 100.0 % | 100.0 % |
| dinov2-curation-en | ✓ | ✓ | 50.0 % | 100.0 % |
| dinov2-frozen-features-fr | ✓ | ✓ | 75.0 % | 0.0 % |
| dinov2-distillation-en | ✓ | ✓ | 100.0 % | 100.0 % |
| sam-architecture-en | ✓ | ✓ | 100.0 % | 100.0 % |
| sam-prompt-types-fr | ✓ | ✓ | 100.0 % | 100.0 % |
| sam-ambiguity-en | ✓ | ✓ | 100.0 % | 100.0 % |
| sam-dataset-scale-en | ✓ | ✓ | 100.0 % | 100.0 % |
| blip2-two-stages-en | ✓ | ✓ | 100.0 % | 100.0 % |
| blip2-bottleneck-fr | ✓ | ✓ | 100.0 % | 100.0 % |
| blip2-efficiency-en | ✓ | ✓ | 100.0 % | 100.0 % |
| blip2-capabilities-en | ✓ | ✓ | 100.0 % | 100.0 % |
| llava-synthetic-data-en | ✓ | ✓ | 100.0 % | 100.0 % |
| llava-architecture-en | ✓ | ✓ | 100.0 % | 100.0 % |
| llava-feature-alignment-en | ✓ | ✓ | 60.0 % | 100.0 % |
| llava-scienceqa-fr | ✗ | ✓ | 66.7 % | 100.0 % |
| compare-clip-dinov2-supervision-en | ✓ | ✓ | 80.0 % | 50.0 % |
| compare-blip2-llava-bridge-fr | ✗ | ✓ | 60.0 % | 0.0 % |
| compare-clip-sam-prompts-en | ✓ | ✗ | 40.0 % | 50.0 % |
| compare-training-data-scale-en | ✓ | ✓ | 75.0 % | 33.3 % |
| compare-frozen-components-en | ✓ | ✓ | 100.0 % | 100.0 % |
| abstain-clip-electricity-en | ✓ | ✓ | — | — |
| abstain-sam-mars-en | ✓ | ✓ | — | — |
| abstain-blip2-radiology-en | ✓ | ✓ | — | — |
| abstain-dinov2-future-benchmark-fr | ✓ | ✓ | — | — |
| abstain-llava-local-latency-fr | ✓ | ✓ | — | — |

## Conclusion et suite

La correction linguistique est confirmée dans la campagne complète. Le score monte d'un cas net, mais la stabilité documentaire et la couverture exhaustive restent les prochains enjeux : sélectionner la preuve numérique CLIP et assurer la couverture de chaque aspect demandé dans les comparaisons.

Les gains/pertes non linguistiques doivent être répétés avant toute attribution causale. Ne pas modifier les questions ou les seuils pour faire disparaître les échecs sans revue humaine des critères. Aucun correctif supplémentaire n'a été appliqué pendant cette campagne.

L'analyse par traces, par cas et par régressions est cadrée par OpenAI Docs : [bonnes pratiques d'évaluation](https://developers.openai.com/api/docs/guides/evaluation-best-practices). Les chiffres ci-dessus proviennent exclusivement des rapports locaux.

