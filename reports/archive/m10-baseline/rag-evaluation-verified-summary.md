# Rapport d'évaluation RAG — benchmark vérifié

Date locale : 14 août 2026  
Benchmark : 30/30 cas vérifiés humainement  
Collection : `arxiv_chunks_e5_small`  
Retrieval : `intfloat/multilingual-e5-small`, `top_k=5`  
Génération : `OpenVINO/Qwen2.5-7B-Instruct-int4-ov`, CPU

## Résultat global

| Mesure | Résultat |
|---|---:|
| Taux de réussite end-to-end | 70,0 % (21/30) |
| Réussite sur les cas répondables | 64,0 % (16/25) |
| Couverture factuelle moyenne | 59,0 % |
| Décision réponse/abstention correcte | 80,0 % (24/30) |
| Abstentions hors corpus correctes | 100 % (5/5) |
| Faux refus sur questions répondables | 24,0 % (6/25) |
| Précision documentaire des citations | 76,0 % sur les 25 cas répondables |
| Rappel documentaire des citations | 76,0 % sur les 25 cas répondables |
| Rappel des documents dans le contexte | 94,7 % |

La précision documentaire de 76 % inclut les six faux refus, dont la valeur est
zéro. Parmi les 19 réponses effectivement acceptées par le validateur, aucune
ne cite un document extérieur à la vérité terrain.

## Retrieval

| Mesure | Résultat |
|---|---:|
| Document recall@5 | 94,7 % |
| Evidence recall@5 | 76,0 % |
| MRR | 0,580 |
| Latence moyenne embedding + Qdrant | environ 0,42 s |

Le rappel documentaire élevé ne garantit pas que le passage précis soit
retrouvé. Par exemple, un article attendu peut être présent dans le top-5 avec
des chunks proches du sujet mais insuffisants pour établir le fait demandé.

## Résultats par catégorie

| Catégorie | Réussite | Diagnostic principal |
|---|---:|---|
| Méthodes | 11/12 (91,7 %) | Bon comportement sur les architectures mono-article. |
| Faits | 5/8 (62,5 %) | Deux faux refus et une réponse dans la mauvaise langue. |
| Comparaisons | 0/5 (0 %) | Manque de diversification des sources dans le top-5. |
| Abstentions | 5/5 (100 %) | Aucun fait hors corpus n'est présenté comme certain. |

## Résultats par langue sur les cas répondables

| Langue | Réussite |
|---|---:|
| Anglais | 14/19 (73,7 %) |
| Français | 2/6 (33,3 %) |

Un cas français DINOv2 reçoit une réponse chinoise, malgré la consigne de
répondre dans la langue de la question. Le cas BLIP-2 français produit une
réponse pertinente mais incomplète selon les quatre faits vérifiés.

## Les neuf échecs end-to-end

| Cas | Retrieval de la preuve | Cause observée |
|---|---:|---|
| `clip-data-scale-fr` | 0 % | Le bon article est présent, mais pas le passage chiffré ; faux refus. |
| `dinov2-frozen-features-fr` | 100 % | Réponse chinoise à une question française. |
| `blip2-bottleneck-fr` | 0 % | Réponse partielle ; un seul groupe de faits sur quatre est couvert lexicalement. |
| `blip2-capabilities-en` | 100 % | Sortie rejetée par le validateur de citations, donc abstention de sécurité. |
| `compare-clip-dinov2-supervision-en` | 0 % | Documents présents, mais aucun des deux passages de référence. |
| `compare-blip2-llava-bridge-fr` | 0 % | Les cinq chunks proviennent de BLIP-2 ; LLaVA est absent. |
| `compare-clip-sam-prompts-en` | 50 % | Les deux documents sont présents, mais un seul passage cible ; réponse incomplète. |
| `compare-training-data-scale-en` | 0 % | SAM est absent et les passages chiffrés précis ne sont pas retrouvés. |
| `compare-frozen-components-en` | 50 % | Les cinq chunks proviennent de BLIP-2 ; LLaVA est absent. |

## Biais de la recherche sur les comparaisons

La recherche dense globale autorise plusieurs voisins du même article :

- CLIP/DINOv2 supervision : 4 chunks DINOv2, 1 chunk CLIP ;
- BLIP-2/LLaVA bridge : 5 chunks BLIP-2 ;
- CLIP/SAM prompts : 4 chunks SAM, 1 chunk CLIP ;
- échelle des données : 3 chunks CLIP, 2 chunks DINOv2, aucun SAM ;
- composants gelés : 5 chunks BLIP-2.

Augmenter simplement `top_k` peut accroître le bruit et le prompt. La correction
prioritaire est une recherche diversifiée ou décomposée par source : détecter
les articles concernés, effectuer une sous-requête par article, puis fusionner
et reranker les preuves.

## Latence

| Mesure | Valeur |
|---|---:|
| Durée de la campagne | 24 min 6 s |
| Somme des latences par cas | 23 min 55 s |
| Latence moyenne | 47,84 s |
| Médiane | 46,67 s |
| Minimum | 26,32 s |
| Maximum | 72,69 s |

Le retrieval représente environ 0,42 seconde en moyenne ; la génération locale
domine donc la latence end-to-end.

## Limites de la métrique factuelle

`fact_coverage` recherche des groupes d'expressions explicitement acceptées.
La mesure est reproductible, mais une paraphrase peut être sémantiquement bonne
sans contenir la séquence exacte. Le cas DINOv2 sur la curation illustre cette
limite : la réponse explique l'absence de métadonnées avec une formulation non
exactement contiguë et n'obtient que 0,5.

La prochaine version devra conserver cette mesure déterministe, puis ajouter un
juge sémantique calibré sur un sous-ensemble relu humainement. Il ne devra pas
remplacer la vérité terrain humaine.

## Reproductibilité

- SHA-256 benchmark :
  `31917579B4753CF27D3ECA756DBF7B8EF23917F6BFCB1BDB6A7B2414A4AAB74B`
- SHA-256 rapport retrieval :
  `15CF2A64E58214827F180387AE1410822B039DBDF31FE547110744F860E290F0`
- SHA-256 rapport RAG :
  `E03CB2CDD6A2BE304B1CC2FE3C01ECA425EA99C977032D9D8865BC39FCB3C6D1`

Les rapports JSON complets conservent chaque question, réponse, citation,
preuve récupérée, métrique et latence.
