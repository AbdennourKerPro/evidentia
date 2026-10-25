# Rapports d'évaluation

Chaque rapport JSON conserve la configuration (modèle, profil, top-k, empreintes
du code et du benchmark), les agrégats et, cas par cas, la réponse, les passages,
la trace et les appels LLM. Les JSON sont stockés avec Git LFS depuis le
25 octobre 2026 (`git lfs install` avant de cloner pour obtenir leur contenu).

- `current/` : rapports qui fondent la configuration actuelle.
- `archive/<jalon>/` : campagnes précédentes, rangées par jalon (`docs/milestones/`).

Les campagnes antérieures au 08/10 portent sur l'ancien benchmark mixte
`rag_benchmark.jsonl`, désormais dans `data/evaluation/archive/`.

Les nouveaux rapports sont écrits à la racine de `reports/` par les scripts
d'évaluation ; on les range ensuite dans le dossier de leur jalon.

## `current/m24/` — jalon 24

Décision actuelle : `candidate_recovery_v2` (v2b, v2c) contre `candidate_recovery`, 3 passes chacun, cas anglais.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-english30-candidate-recovery-20261025-run1.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-english30-candidate-recovery-20261025-run2.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-english30-candidate-recovery-20261025-run3.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-english30-recovery-v2b-20261025-run1.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery_v2 | 29/30 réussis |
| `agentic-rag-english30-recovery-v2b-20261025-run2.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery_v2 | 30/30 réussis |
| `agentic-rag-english30-recovery-v2b-20261025-run3.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery_v2 | 30/30 réussis |
| `agentic-rag-english30-recovery-v2c-20261025-run1.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery_v2 | 30/30 réussis |
| `agentic-rag-english30-recovery-v2c-20261025-run2.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery_v2 | 30/30 réussis |
| `agentic-rag-english30-recovery-v2c-20261025-run3.json` | 2026-10-25 | rag_english30.draft.jsonl | candidate_recovery_v2 | 30/30 réussis |
| `agentic-rag-heldout-en-candidate-recovery-20261025-run1.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery | 20/20 réussis |
| `agentic-rag-heldout-en-candidate-recovery-20261025-run2.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery | 20/20 réussis |
| `agentic-rag-heldout-en-candidate-recovery-20261025-run3.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery | 20/20 réussis |
| `agentic-rag-heldout-en-recovery-v2b-20261025-run1.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery_v2 | 20/20 réussis |
| `agentic-rag-heldout-en-recovery-v2b-20261025-run2.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery_v2 | 20/20 réussis |
| `agentic-rag-heldout-en-recovery-v2b-20261025-run3.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery_v2 | 20/20 réussis |
| `agentic-rag-heldout-en-recovery-v2c-20261025-run1.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery_v2 | 20/20 réussis |
| `agentic-rag-heldout-en-recovery-v2c-20261025-run2.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery_v2 | 20/20 réussis |
| `agentic-rag-heldout-en-recovery-v2c-20261025-run3.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery_v2 | 20/20 réussis |

## `archive/m24-recovery-v2/` — jalon 24

Variantes de récupération du 08–09/10 et première passe v2 du 24/10.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-english30-keep-primary-hits-candidate-recovery-20261009.json` | 2026-10-09 | rag_english30.draft.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-english30-main-candidate-recovery-20261009.json` | 2026-10-09 | rag_english30.draft.jsonl | candidate_recovery | 4/4 réussis |
| `agentic-rag-english30-primary-pool-candidate-recovery-20261009.json` | 2026-10-09 | rag_english30.draft.jsonl | candidate_recovery | 28/30 réussis |
| `agentic-rag-english30-recovery-v2-20261024.json` | 2026-10-24 | rag_english30.draft.jsonl | candidate_recovery_v2 | 30/30 réussis |
| `agentic-rag-english30-verified-candidate-recovery-20261009.json` | 2026-10-09 | rag_english30.draft.jsonl | candidate_recovery | 28/30 réussis |
| `agentic-rag-english30-widened-candidate-recovery-20261009.json` | 2026-10-09 | rag_english30.draft.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-heldout-draft-20261008.json` | 2026-10-08 | rag_heldout.draft.jsonl | question_rerank | 35/40 réussis |
| `agentic-rag-heldout-draft-candidate-recovery-20261008.json` | 2026-10-08 | rag_heldout.draft.jsonl | candidate_recovery | 34/40 réussis |
| `agentic-rag-heldout-draft-candidate-recovery-gapsearch-20261009.json` | 2026-10-09 | rag_heldout.draft.jsonl | candidate_recovery | 35/40 réussis |
| `agentic-rag-heldout-draft-main-358445c-candidate-recovery-20261009.json` | 2026-10-09 | rag_heldout.draft.jsonl | candidate_recovery | 37/40 réussis |
| `agentic-rag-heldout-draft-point2-english-rewrite-20261009.json` | 2026-10-09 | rag_heldout.draft.jsonl | candidate_recovery | 18/19 réussis |
| `agentic-rag-heldout-draft-pr2-faefd18-20261009.json` | 2026-10-09 | rag_heldout.draft.jsonl | candidate_recovery | 37/40 réussis |
| `agentic-rag-heldout-draft-recovery-v2-20261024.json` | 2026-10-24 | rag_heldout.draft.jsonl | candidate_recovery_v2 | 38/40 réussis |
| `agentic-rag-heldout-en-candidate-recovery-20261025.json` | 2026-10-25 | rag_heldout.draft.jsonl (en) | candidate_recovery | 20/20 réussis |

## `archive/m21-coverage-v9/` — jalon 21

Récupération guidée par la couverture (v9) et ablation de ses variantes.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-v9-candidate-coverage-30-20260930.json` | 2026-09-30 | rag_benchmark.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-v9-candidate-coverage-final-30-20260930-summary.md` |  |  |  | résumé |
| `agentic-rag-v9-candidate-coverage-final-30-20260930.json` | 2026-09-30 | rag_benchmark.jsonl | candidate_recovery | 29/30 réussis |
| `agentic-rag-v9-effort-none-20261007.json` | 2026-10-07 | rag_benchmark.jsonl | candidate_recovery | 30/30 réussis |
| `coverage-v9-01-diagnostic.json` | 2026-09-30 |  |  | 2/4 réussis |
| `coverage-v9-02-candidates.json` | 2026-09-30 |  |  | 7/11 réussis |
| `coverage-v9-03-fixed-requirements.json` | 2026-09-30 |  |  | 9/11 réussis |
| `coverage-v9-04-scoped-requirements.json` | 2026-09-30 |  |  | 8/11 réussis |
| `coverage-v9-05-atomic-candidates.json` | 2026-09-30 |  |  | 11/11 réussis |
| `coverage-v9-06-answer-plan.json` | 2026-09-30 |  |  | 8/11 réussis |
| `coverage-v9-07-mechanism-requirements.json` | 2026-09-30 |  |  | 10/13 réussis |
| `coverage-v9-08-completeness-verifier.json` | 2026-09-30 |  |  | 12/13 réussis |
| `coverage-v9-09-source-fair-budget.json` | 2026-09-30 |  |  | 13/13 réussis |

## `archive/m20-language-v8/` — jalon 20

Contrôle de langue à trois états (v8).

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-v8-language-30-20260930-summary.md` |  |  |  | résumé |
| `agentic-rag-v8-language-30-20260930.json` | 2026-09-30 | rag_benchmark.jsonl | question_rerank | 28/30 réussis |
| `language-v8-scienceqa-targeted-20260930.json` | 2026-09-30 |  |  | 5/5 réussis |

## `archive/m19-ablation-sam/` — jalon 19

Ablation successive du retrieval (cas SAM) et campagne v7.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-v7-question-rerank-30-20260930-summary.md` |  |  |  | résumé |
| `agentic-rag-v7-question-rerank-30-20260930.json` | 2026-09-30 | rag_benchmark.jsonl | question_rerank | 27/30 réussis |
| `retrieval-sam-00-baseline.json` | 2026-09-29 |  |  | 5/6 réussis |
| `retrieval-sam-01-question-rerank.json` | 2026-09-29 |  |  | 6/6 réussis |
| `retrieval-sam-02-original-query.json` | 2026-09-29 |  |  | 5/6 réussis |
| `retrieval-sam-03-coverage-retry.json` | 2026-09-29 |  |  | 6/6 réussis |

## `archive/m18-gpt6-luna/` — jalon 18

Première évaluation après migration vers GPT-6 Luna.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-evaluation-gpt6-luna.json` | 2026-09-29 | rag_benchmark.jsonl |  | 28/30 réussis |

## `archive/m17-v6/` — jalon 17

Correctifs de généralisation (v6).

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-v6-final-30.json` | 2026-08-15 | rag_benchmark.jsonl |  | 29/30 réussis |
| `agentic-rag-v6-three-fixes-final.json` | 2026-08-15 | rag_benchmark.jsonl |  | 3/3 réussis |
| `agentic-rag-v6-three-fixes.json` | 2026-08-15 | rag_benchmark.jsonl |  | 3/3 réussis |

## `archive/m16-17-fact-plan/` — jalon 16–17

Plan factuel sourcé (v5) et ses garde-fous, campagne finale v5.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-fact-plan-v5-blip2-final.json` | 2026-08-14 | rag_benchmark.jsonl |  | 0/1 réussis |
| `agentic-rag-fact-plan-v5-blip2-publishable.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-fact-plan-v5-dinov2-v3.json` | 2026-08-14 | rag_benchmark.jsonl |  | 0/1 réussis |
| `agentic-rag-fact-plan-v5-dinov2-v4.json` | 2026-08-14 | rag_benchmark.jsonl |  | 0/1 réussis |
| `agentic-rag-fact-plan-v5-dinov2-v5.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-fact-plan-v5-two-persistent-final.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/2 réussis |
| `agentic-rag-fact-plan-v5-two-persistent-v2.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/2 réussis |
| `agentic-rag-fact-plan-v5-two-persistent.json` | 2026-08-14 | rag_benchmark.jsonl |  | 0/2 réussis |
| `agentic-rag-v5-blip2-llava-guard-v2.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-v5-clip-dinov2-attribution-guard.json` | 2026-08-15 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-v5-comparisons-final-guard.json` | 2026-08-14 | rag_benchmark.jsonl |  | 3/4 réussis |
| `agentic-rag-v5-comparisons-guard.json` | 2026-08-14 | rag_benchmark.jsonl |  | 3/5 réussis |
| `agentic-rag-v5-comparisons-remaining-guard.json` | 2026-08-14 | rag_benchmark.jsonl |  | 3/3 réussis |
| `agentic-rag-v5-final-30.json` | 2026-08-15 | rag_benchmark.jsonl |  | 27/30 réussis |
| `agentic-rag-v5-supervision-guard.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/2 réussis |
| `fact-plan-v5-guard-summary.md` |  |  |  | résumé |

## `archive/m15-multifacet/` — jalon 15

Retrieval multi-facettes guidé par l'intention.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-intent-multifacet-regression-seven.json` | 2026-08-14 | rag_benchmark.jsonl |  | 5/7 réussis |
| `agentic-rag-intent-multifacet-two-failures.json` | 2026-08-14 | rag_benchmark.jsonl |  | 2/2 réussis |
| `intent-multifacet-retrieval-two-failures.json` | 2026-08-14 | rag_benchmark.jsonl |  | voir le fichier |
| `intent-multifacet-v4-summary.md` |  |  |  | résumé |
| `multifacet-retrieval-two-failures-v2.json` | 2026-08-14 | rag_benchmark.jsonl |  | voir le fichier |
| `multifacet-retrieval-two-failures.json` | 2026-08-14 | rag_benchmark.jsonl |  | voir le fichier |

## `archive/m14-contract/` — jalon 14

Contrat de langue et de citations (v3).

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-response-contract-v3-clip-dinov2-final.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-response-contract-v3-comparisons.json` | 2026-08-14 | rag_benchmark.jsonl |  | 3/5 réussis |
| `agentic-rag-response-contract-v3-final.json` | 2026-08-14 | rag_benchmark.jsonl |  | 3/5 réussis |
| `response-contract-v3-summary.md` |  |  |  | résumé |

## `archive/m12-13-hybrid/` — jalon 12–13

Comparaisons LangGraph et retrieval hybride BM25/RRF/cross-encoder.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-comparisons-verified-summary.md` |  |  |  | résumé |
| `agentic-rag-comparisons-verified.json` | 2026-08-14 | rag_benchmark.jsonl |  | 2/5 réussis |
| `agentic-rag-hybrid-comparisons-verified.json` | 2026-08-14 | rag_benchmark.jsonl |  | 2/5 réussis |
| `hybrid-retrieval-comparisons-no-cross.json` | 2026-08-14 | rag_benchmark.jsonl |  | voir le fichier |
| `hybrid-retrieval-comparisons-summary.md` |  |  |  | résumé |
| `hybrid-retrieval-comparisons.json` | 2026-08-14 | rag_benchmark.jsonl |  | voir le fichier |

## `archive/m11-14-smoke/` — jalon 11–14

Essais ponctuels (1 ou 2 cas).

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `agentic-rag-contract-french-smoke-v2.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-contract-french-smoke.json` | 2026-08-14 | rag_benchmark.jsonl |  | 0/1 réussis |
| `agentic-rag-smoke-v2.json` | 2026-08-14 | rag_benchmark.jsonl |  | 1/1 réussis |
| `agentic-rag-smoke.json` | 2026-08-14 | rag_benchmark.jsonl |  | 0/1 réussis |

## `archive/m10-baseline/` — jalon 08–10

Baseline de retrieval et de RAG.

| Fichier | Date | Benchmark | Profil | Résultat |
|---|---|---|---|---|
| `rag-evaluation-verified-summary.md` |  |  |  | résumé |
| `rag-evaluation-verified.json` | 2026-08-13 | rag_benchmark.jsonl |  | 21/30 réussis |
| `rag-smoke-evaluation.json` | 2026-08-12 | rag_benchmark.jsonl |  | 1/1 réussis |
| `retrieval-evaluation-verified.json` | 2026-08-13 | rag_benchmark.jsonl |  | document_recall_at_k 0.947, evidence_recall_at_k 0.760 |
| `retrieval-evaluation.json` | 2026-08-12 | rag_benchmark.jsonl |  | document_recall_at_k 0.947, evidence_recall_at_k 0.760 |
