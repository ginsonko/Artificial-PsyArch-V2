# TransitionIsolation-1 Final Report

Date: 2026-06-13
Route: AP-Core bottom-loop mechanism hardening
Cycle: 设计 -> 审查完善 -> 落地 -> 严谨验收测试 -> 最终汇总报告

## 1. Design

AP separates "B usually follows A" (succession/transition) from "A and B are the same kind" (concept similarity). Imitation and sequence learning need the former to be strong, but that must not collapse A and B into synonyms. This experiment verifies that training A->B transitions raises the directed successor score while the symmetric concept similarity of A and B is left untouched, and the successor stays directed.

Design doc: `docs/Design_APV2_TransitionIsolation1_20260613.md`

## 2. Implementation

- Experiment: `experiments/transition_isolation_1.py`
- Strict test: `tests/test_transition_isolation_1.py`
- No runtime mechanism modified.

`observe_transition_pair(a, b)` writes only `a.transition_counts[b]` and transition support; it does not call the vector-pull path, so concept vectors do not move. `learned_transition` reads the directed successor channel, while `learned_vector_similarity`/`learned_similarity` read the separate concept channels.

## 3. Validation

Commands:

```text
python experiments/transition_isolation_1.py
python -m pytest tests/test_transition_isolation_1.py tests/test_online_learned_vector_boundary.py -q
```

Result: experiment `passed=true`; tests `5 passed` (+ boundary `2 passed`).

Direct isolation (before vs after training A->B):

| quantity | before | after |
|---|---:|---:|
| successor `learned_transition(A->B)` | 0.0000 | 0.9412 |
| concept vector `learned_vector_similarity(A,B)` | 0.1840 | 0.1840 |
| co-occurrence `learned_similarity(A,B)` | 0.0000 | 0.0000 |
| reverse `learned_transition(B->A)` | - | 0.0000 |

`pair_evidence(A,B)`: `transition_raw=32.0`, `positive_raw=0.0`.

Recall-side successor (`successors` API): source A -> top successor B, `learned_transition_score=0.2422`.

## 4. What This Proves

Six acceptance checks pass:

- **Successor strengthens:** `learned_transition(A->B)` rises from `0.0` to `0.9412`.
- **Concept vector not contaminated:** `learned_vector_similarity(A,B)` is exactly unchanged (`0.1840 -> 0.1840`).
- **Co-occurrence similarity not contaminated:** `learned_similarity(A,B)` stays `0.0`.
- **Directed:** forward `0.9412` vastly exceeds reverse `0.0`.
- **Pure transition evidence:** `transition_raw=32`, `positive_raw=0` — no co-occurrence mixed in.
- **Recall-side visible:** the `successors` API exposes a positive `learned_transition_score`, so the successor gain is auditable in recall.

This closes the online-vector ablation trio. WeightAblation-1 showed the branch *adds* bounded neighborhood signal; NegativePressure-1 showed it *prunes* wrong residue; TransitionIsolation-1 shows succession and concept similarity stay *separate channels*. Together they demonstrate that the online learned layer is useful, bounded, self-correcting, and structurally clean.

## 5. Paper/FAQ Use

Recommended paper phrasing:

Training A->B transitions raised the directed successor score from 0 to 0.94 while the symmetric concept similarity of A and B stayed exactly constant, and the reverse transition remained near zero. This supports the claim that AP learns succession without collapsing it into concept similarity: imitation and sequence learning are strengthened without contaminating which objects count as the same kind.

Recommended boundary phrasing:

This is bottom-loop mechanism evidence. A strong A->B successor does not imply A and B are synonyms; concept similarity is decided by the separate vector/co-occurrence channels.

## 6. Artifacts

- `outputs/transition_isolation_1_20260613_211028/transition_isolation_1_report.json`
- `outputs/transition_isolation_1_20260613_211028/transition_isolation_1_report.md`
