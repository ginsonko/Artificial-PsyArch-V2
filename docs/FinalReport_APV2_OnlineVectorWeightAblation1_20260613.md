# OnlineVector-WeightAblation-1 Final Report

Date: 2026-06-13
Route: AP-Core bottom-loop mechanism hardening
Cycle: 设计 -> 审查完善 -> 落地 -> 严谨验收测试 -> 最终汇总报告

## 1. Design

Stage 6A.4 proved the online learned vector coordinate exists, persists, and is exposed via recall audit. OnlineVector-WeightAblation-1 asks the next question: in recall competition, is this branch *useful* (does it improve experience-neighborhood recall) and *bounded* (does it avoid dominating exact SA/energy evidence)?

Design doc: `docs/Design_APV2_OnlineVectorWeightAblation1_20260613.md`

## 2. Review And Refinement

A pre-implementation probe surfaced a real and important implementation detail that the design now records faithfully:

- The `online_learned_vector` contribution (`learned_vector_score`) is added to the recall score **only** in the audit/exact path (`audit_recall` -> `_exact_hot_recall` -> `_score_snapshot_exact`).
- The main `recall()` path currently adds only the learned_**similarity** branch (`learned_score * learned_weight`).

The experiment was therefore aligned to verify `learned_vector_score` through the audit path and `learned_score` through the main path. A second probe detail: `learned_vector_score` is always *computed* and exposed on the audit row, but only enters the total score scaled by `min(0.22, learned_weight)`, so at `learned_weight=0` the field is non-zero yet contributes nothing. Acceptance therefore keys on the change in `score` across weights, not the raw field value.

Both details are recorded as project memory so AttentionBand-6A.5 can decide deliberately whether to wire learned vectors into the main recall path.

## 3. Implementation

- Experiment: `experiments/online_vector_weight_ablation_1.py`
- Strict test: `tests/test_online_vector_weight_ablation_1.py`
- No runtime mechanism was modified. The experiment only constructs three `learned_weight` settings and observes the existing scoring formula.

Scene (three snapshots, no surface-token overlap with the `alpha/beta` query except B_exact):

- `A_neighbor` (mem-1): subject `gamma` is a trained online neighbor of query subject `alpha`; no surface overlap.
- `A_unrelated` (mem-2): no surface overlap, no learned experience.
- `B_exact` (mem-3): exact label/energy match with the query (strong main evidence), no learned experience.

## 4. Validation

Commands:

```text
python experiments/online_vector_weight_ablation_1.py
python -m pytest tests/test_online_vector_weight_ablation_1.py tests/test_online_learned_vector_boundary.py -q
```

Result: experiment `passed=true`; tests `6 passed`.

Audit-path recall scores across the three weights:

| weight | neighbor score | neighbor lvs | neighbor rank | unrelated score | unrelated rank | exact score | exact rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| off (0.0) | 0.6848 | 0.4413 | 1 | 0.6522 | 2 | 14.6954 | 0 |
| default (0.28) | 1.0460 | 0.4413 | 1 | 0.6768 | 2 | 14.9660 | 0 |
| high (0.9) | 1.6308 | 0.4413 | 1 | 0.6768 | 2 | 15.1495 | 0 |

Main-recall path (learned_similarity only), neighbor score: off `0.6848` -> default `0.9489` -> high `1.5336`.

## 5. What This Proves

Four acceptance checks all pass:

- **H1 (improves neighborhood recall, audit path):** with the learned-vector branch weighted in, the neighbor score rises from `0.6848` (off) to `1.0460` (default), and the neighbor ranks above the unrelated snapshot. The online learned vector improves recall for an experience neighbor that shares no surface tokens with the query.
- **H1b (improves recall, main path):** the learned-similarity branch raises the neighbor score from `0.6848` to `0.9489` at default weight.
- **H2 (does not dominate):** even at high weight `0.9`, the exact label/energy match (`B_exact`, score ~15) stays rank 0, far above the learned-only neighbor. Learned vectors do not override SA/energy main evidence.
- **H3 (monotonic and bounded):** neighbor score is monotonic non-decreasing across off -> default -> high (`0.6848 <= 1.0460 <= 1.6308`), off adds no learned mass to the score, and the learned-vector branch is bounded by `min(0.22, learned_weight)`.

Together with Stage 6A.4, this sharpens the bottom-loop claim: the online learned vector is a useful, bounded, ablatable auxiliary branch that improves experience-neighborhood recall without competing away exact SA/energy evidence. It does not modify the runtime and does not claim an open-world dialogue base.

## 6. Paper/FAQ Use

Recommended paper phrasing:

In an AP-Core ablation, disabling the online learned branch (`learned_weight=0`) removes its contribution to recall scoring; at the default weight the branch raises the recall score of an experience neighbor that shares no surface tokens with the query, while an exact label/energy match still outranks any learned-only candidate even at high weight. This supports the claim that the learned vector is a bounded auxiliary neighborhood signal, not a dominating answer route.

Recommended boundary phrasing:

This is bottom-loop mechanism evidence on the audit recall path. The learned vector currently enters live ranking through the audit/exact path; wiring it into the main recall path is a deliberate next-step decision (AttentionBand-6A.5), not assumed here.

## 7. Artifacts

- `outputs/online_vector_weight_ablation_1_20260613_193916/online_vector_weight_ablation_1_report.json`
- `outputs/online_vector_weight_ablation_1_20260613_193916/online_vector_weight_ablation_1_report.md`

## 8. Next AP-Core Experiments

- `OnlineVector-NegativePressure-1`: repeatedly over-predict a wrong subject, verify negative pressure pushes it away from the real context.
- `TransitionIsolation-1`: train A->B, verify directed successor score rises while symmetric concept similarity does not.
- `AttentionBand-6A.5`: decide whether learned-vector neighborhoods should influence the main recall/attention path, not only the audit path.
