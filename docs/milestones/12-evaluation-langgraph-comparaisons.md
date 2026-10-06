# Milestone 12 — Évaluation des cinq comparaisons LangGraph

## Objectif

Vérifier sur tous les cas comparatifs du benchmark si le graphe LangGraph v1
corrige la faiblesse structurante de la baseline, qui obtenait 0 réussite sur
5 comparaisons.

Le code, le benchmark vérifié et le `top_k=5` sont conservés. Seule la stratégie
de récupération change : décomposition par article, branches Qdrant filtrées,
reranking et fusion équilibrée.

## Résultat principal

LangGraph obtient 2 réussites sur 5, contre 0 sur 5 pour la baseline. La
couverture factuelle moyenne passe de 8 % à 32 % et le rappel documentaire des
chunks de 73,3 % à 100 %.

Ce résultat confirme que le fan-out par source résout la première couche du
problème : les articles attendus sont tous représentés. Il montre aussi que
représenter un document ne suffit pas. Seuls 4 des 11 chunks cibles relus sont
présents dans les fenêtres finales.

## Coût

- durée totale : environ 7 min 18 s ;
- latence moyenne : 84,05 s par comparaison ;
- baseline sur les mêmes cas : 46,21 s ;
- multiplicateur : 1,82 ;
- planification moyenne : 22,95 s ;
- génération moyenne : 60,99 s ;
- recherche Qdrant moyenne par branche : 73,5 ms.

Le surcoût vient du second appel Qwen utilisé pour planifier. LangGraph et
Qdrant ne constituent pas le goulot d'étranglement.

## Diagnostic des trois échecs

### BLIP-2 contre LLaVA

La recherche retrouve le passage LLaVA sur la projection linéaire mais manque
le passage BLIP-2 décrivant explicitement le Q-Former comme goulot
d'étranglement. Qwen reçoit donc une comparaison asymétrique et s'abstient.

### CLIP contre SAM

La fenêtre contient les deux articles, mais aucun des deux chunks cibles sur
les 80 context prompts de CLIP et les prompts sparse/dense de SAM. Qwen produit
une réponse générale correctement citée, mais elle ne répond pas aux
différences techniques attendues.

### Échelle des données

Le graphe détecte les trois articles et réserve au moins un chunk à chacun.
Cependant, les passages contenant 400 millions de paires, LVD-142M et plus
d'un milliard de masques ne sont pas sélectionnés. Sans ces nombres, le modèle
s'abstient conformément au garde-fou.

## Attention à la métrique

Le cas sur les composants gelés passe avec 0,8 de couverture et les deux
documents cités. Une lecture humaine révèle néanmoins l'absence du Q-Former et
une généralisation imprécise de la projection linéaire. Le seuil de couverture
à 0,5 accepte donc des réponses utiles mais incomplètes.

La prochaine évolution de l'évaluation devra ajouter une catégorie de passage
strict ou une revue sémantique, tout en conservant les métriques déterministes
pour l'auditabilité.

## Décision

Ne pas modifier Qwen et ne pas commencer le SFT à ce stade. Les erreurs
observées viennent d'abord de l'absence des bons chunks. La prochaine
expérience doit être une évaluation retrieval-only de plusieurs rerankers sur
les 25 questions répondables : elle sera beaucoup plus rapide et isolera le
meilleur mécanisme avant une nouvelle campagne end-to-end.

Le rapport JSON complet est
`reports/agentic-rag-comparisons-verified.json`. Son résumé humain détaillé est
`reports/agentic-rag-comparisons-verified-summary.md`.

Cette expérience a été poursuivie au milestone 13 avec BM25, RRF et un
cross-encoder. Voir
`docs/milestones/13-retrieval-hybride-bm25-rrf-cross-encoder.md`.
