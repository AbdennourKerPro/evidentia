# Milestone 16 — Plan factuel sourcé et campagne finale reproductible

## Objectif

Cette itération traite les échecs qui subsistaient après l'amélioration du
retrieval. Dans ces cas, les bons passages étaient retrouvés, mais le LLM
paraphrasait ou omettait une partie des faits attendus. Le problème se situait
donc dans la construction et la validation de la réponse.

L'objectif est double :

- rendre explicites les faits importants contenus dans les preuves avant la
  rédaction ;
- rendre l'évaluation complète de 30 cas relançable, observable et reprenable
  après une interruption.

La campagne de 30 cas n'a volontairement pas été exécutée à ce milestone. Les
cas ciblés et les cinq comparaisons ont été utilisés comme garde de régression.

## Architecture v5

```text
question
  -> analyse de la question et détection des intentions
  -> plan de recherche par article
  -> retrieval dense + BM25 multi-facettes
  -> fusion RRF et reranking guidé par l'intention
  -> sélection équilibrée des preuves
  -> plan factuel déterministe à partir des preuves sélectionnées
  -> génération guidée par le plan et un guide de couverture
  -> validation langue + citations + complétude factuelle
  -> correction unique si nécessaire
  -> publication ou abstention
```

L'identifiant de pipeline enregistré dans les rapports est
`langgraph_fact_plan_intent_multifacet_response_contract_v5`.

## 1. Nouvelle intention `training_objective`

`app/query_expansion.py` détecte désormais les questions portant sur la
supervision, l'objectif de pré-entraînement, la loss, le contraste, la
self-supervision et les réseaux teacher/student.

Cette intention produit une facette de recherche spécialisée et favorise les
passages de méthode qui décrivent :

- les paires image-texte et l'objectif contrastif ;
- la self-supervision ;
- les réseaux teacher/student ;
- les objectifs image-level et patch-level.

Les sections seulement apparentées, comme `Related Work`, sont pénalisées. Les
règles sont génériques : aucune valeur attendue du benchmark n'est copiée dans
les requêtes.

## 2. Plan factuel fondé sur les preuves

Le nœud LangGraph `plan_answer`, placé après `select_evidence`, appelle
`plan_evidence_facts()` dans `app/fact_planning.py`.

Le planificateur inspecte uniquement les chunks effectivement sélectionnés. Il
construit des objets `PlannedFact` contenant :

- un identifiant de trace (`F1`, `F2`, etc.) ;
- une affirmation atomique ;
- une ancre lexicale attendue et ses équivalents contrôlés ;
- les références autorisées (`S1`, `S2`, etc.) ;
- l'article auquel le fait appartient.

Le préfixe par `document_id` est essentiel dans une comparaison : il empêche de
transférer un objectif de CLIP vers DINOv2, ou un composant de BLIP-2 vers
LLaVA.

Une première expérimentation utilisait le LLM pour générer un plan JSON. Elle a
été abandonnée : elle ajoutait environ 200 secondes, pouvait produire des faits
redondants et rendait le contrat moins reproductible. Le planificateur final est
déterministe, rapide, testable et auditable. Il ne s'active que pour des
intentions dont les motifs ont une preuve explicite.

## 3. Deux niveaux de contrat

`app/response_contract.py` distingue désormais :

- les défauts bloquants : mauvaise langue, citation invalide, réponse tronquée
  ou source interdite ;
- les défauts factuels consultatifs : une ancre planifiée ou sa citation locale
  manque encore après l'unique correction.

Une réponse avec un défaut bloquant est rejetée et le système s'abstient. Une
réponse seulement incomplète sur le plan factuel peut rester publiable, mais
`fact_contract_valid=false` le signale dans la trace. Cette distinction évite de
remplacer une réponse utile et correctement sourcée par une abstention totale,
tout en conservant l'information nécessaire à l'évaluation.

Les champs exposés dans `execution` sont notamment :

- `fact_plan` et `fact_planning_fallback` ;
- `covered_planned_facts` ;
- `correction_attempted` ;
- `response_contract_valid` et `response_publishable` ;
- `fact_contract_valid` et `contract_issues`.

L'interface affiche également ce plan et son statut dans la trace LangGraph.

## 4. Politique de risque pour les comparaisons

Une comparaison architecturale peut extraire trop de faits stricts à partir de
plusieurs articles et surcharger le petit modèle local. Dans le chemin
`per_source`, le plan strict `architecture_bridge` est donc désactivé ; le guide
de couverture reste actif et demande explicitement le nom de l'encodeur visuel,
du modèle de langage et du type de pont.

Pour `training_objective`, le plan strict reste actif, car l'association de
chaque objectif à son article est précisément le risque à contrôler.

## 5. Génération et correction

Le prompt de génération reçoit le plan factuel et le guide correspondant aux
intentions détectées. Il demande une réponse concise, sans répétition, et une
citation dans la même phrase que chaque fait planifié.

La taille maximale est de 600 nouveaux tokens pour une comparaison et de 400
pour une question globale. L'unique passe corrective peut aller jusqu'à 650 ou
500 tokens. Une seule correction borne la latence et évite une boucle
agentique infinie.

## 6. Évaluateur avec checkpoints

`scripts/evaluate_agentic_rag.py` écrit le rapport après chaque cas. L'option
`--resume` recharge les cas déjà terminés si le rapport appartient au même
pipeline. Elle refuse un rapport d'une autre version afin de ne pas mélanger
des résultats incompatibles.

Le rapport ajoute :

- le taux de conformité stricte du contrat de réponse ;
- le taux de conformité factuelle ;
- le taux de correction ;
- le nombre de cas ayant un plan factuel ;
- le taux de fallback du planificateur ;
- la couverture moyenne des faits planifiés.

Commande à exécuter plus tard depuis la racine du projet :

```powershell
docker compose run --rm api python -m scripts.evaluate_agentic_rag `
  --output /reports/agentic-rag-v5-final-30.json `
  --resume
```

Au premier lancement, le fichier n'existe pas et les 30 cas sont calculés. Si
le processus est interrompu, la même commande reprend uniquement les cas
absents. Il ne faut pas changer de code ou reconstruire l'image entre deux
reprises d'une même campagne, sinon les checkpoints ne représenteraient plus
exactement la même solution.

## 7. Validation avant la campagne complète

Les 38 tests unitaires passent dans l'image Docker.

Les deux échecs persistants hors comparaison passent désormais :

| Cas | Résultat | Couverture factuelle |
|---|---:|---:|
| `dinov2-frozen-features-fr` | réussite | 0,50 |
| `blip2-bottleneck-fr` | réussite | 0,50 |

Les cinq comparaisons de garde passent avec la version v5 :

| Cas | Couverture factuelle |
|---|---:|
| `compare-clip-dinov2-supervision-en` | 1,00 |
| `compare-blip2-llava-bridge-fr` | 0,60 |
| `compare-clip-sam-prompts-en` | 0,60 |
| `compare-training-data-scale-en` | 1,00 |
| `compare-frozen-components-en` | 0,80 |

Le contrôle humain du premier cas confirme que les faits sont maintenant bien
attribués : paires image-texte et objectif contrastif pour CLIP ;
self-supervision et teacher/student pour DINOv2.

Cette validation ciblée ne permet pas encore d'annoncer le score final sur les
30 cas. Ce score devra provenir exclusivement de
`reports/agentic-rag-v5-final-30.json` une fois la campagne exécutée.

## 8. Fichiers principaux

- `app/fact_planning.py` : extraction déterministe et rendu du plan ;
- `app/query_expansion.py` : intentions, facettes et reranking ;
- `app/agentic_rag.py` : nœud LangGraph et politique de risque ;
- `app/rag_service.py` : prompts de génération/correction ;
- `app/response_contract.py` : validation bloquante et consultative ;
- `app/schemas.py` : schémas de plan et d'observabilité ;
- `scripts/evaluate_agentic_rag.py` : campagne et reprise par checkpoints ;
- `tests/` : 38 tests déterministes.

## Limites connues

- la couverture factuelle du benchmark repose encore sur des groupes de termes
  explicites et reste plus lexicale qu'une évaluation sémantique humaine ;
- les règles du planificateur couvrent les intentions prioritaires, pas tout
  concept scientifique imaginable ;
- la passe corrective augmente fortement la latence sur CPU/OpenVINO ;
- un score de garde ne remplace pas la campagne complète ni un audit humain
  d'un échantillon des réponses.
