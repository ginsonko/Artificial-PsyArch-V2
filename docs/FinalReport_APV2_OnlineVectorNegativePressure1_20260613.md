# OnlineVector-NegativePressure-1 Final Report

Date: 2026-06-13
Route: AP-Core bottom-loop mechanism hardening
Cycle: 设计 -> 审查完善 -> 落地 -> 严谨验收测试 -> 最终汇总报告

## 1. Design

AP claims that wrong predictive residue should be pruned by negative cognitive pressure rather than persisting as a "thought stamp" that keeps polluting recall. This experiment tests that mechanism on the online learned vector: when a wrong subject is repeatedly over-predicted under a context (each mismatch producing negative pressure), is it pushed away from the real context while the correct association survives?

Design doc: `docs/Design_APV2_OnlineVectorNegativePressure1_20260613.md`

## 2. Review And Refinement

A probe surfaced a real design choice. Applying symmetric `observe_negative_pair` to the *context* damages all of that context's positive associations, because `learned_similarity` divides by a support term that includes `negative_support`. That is collateral damage, not targeted pruning.

The correct model of "over-predicting a wrong subject" is **directed** `observe_negative_anchor(subject, context)`: it moves only the wrong subject's vector away from the context and never moves or penalizes the context itself. The experiment uses this directed form, consistent with the prior finding that the learned vector (not learned_similarity) is the Stage 6A.4 protagonist.

## 3. Implementation

- Experiment: `experiments/online_vector_negative_pressure_1.py`
- Strict test: `tests/test_online_vector_negative_pressure_1.py`
- No runtime mechanism modified.

Scene:
- `C_real -> S_good`: strong correct association (repeated positive co-occurrence).
- `C_real -> S_wrong`: weak stale residue from early noise.
- Then 10 ticks of `observe_negative_anchor(S_wrong, C_real)` (repeated over-prediction mismatch / negative pressure).

## 4. Validation

Commands:

```text
python experiments/online_vector_negative_pressure_1.py
python -m pytest tests/test_online_vector_negative_pressure_1.py tests/test_online_learned_vector_boundary.py -q
```

Result: experiment `passed=true`; tests `5 passed` (+ boundary `2 passed`).

Direct learned-vector similarity:

| quantity | value |
|---|---:|
| correct C_real-S_good before pressure | 0.8790 |
| correct C_real-S_good after pressure | 0.8504 (drop 0.0286) |
| wrong residue S_wrong-C_real before pressure | 0.3598 |
| wrong residue S_wrong-C_real after pressure | -0.5937 (drop 0.9535) |

`pair_evidence(S_wrong, C_real)`: `negative_raw=20.0`, `source_negative_support=20.0`.

Audit-path recall competition (query = `C_real`):

| snapshot | learned_vector_score |
|---|---:|
| correct subject (S_good) | 0.8136 |
| wrong residue (S_wrong) | 0.0 |

## 5. What This Proves

Five acceptance checks pass:

- **Wrong residue pruned:** the wrong subject's similarity to the real context is driven from positive `0.3598` to negative `-0.5937` — a `0.95` drop. The residue is not merely weakened; it is pushed to the far side.
- **Useful experience preserved:** the correct association barely moves (`0.8790 -> 0.8504`, drop `0.0286`). Pressure is targeted at the wrong residue, not the context.
- **White-box auditable:** `negative_raw` and `source_negative_support` are exposed as readable evidence.
- **Correct subject wins recall:** in audit-path competition under `C_real`, the correct subject snapshot contributes `0.8136` of learned-vector score versus `0.0` for the wrong residue.

Together with WeightAblation-1, this strengthens the bottom-loop story: the online learned branch not only *adds* useful neighborhood signal (WeightAblation) but also *removes* wrong residue under negative pressure (NegativePressure), and both effects are bounded, targeted, and auditable.

## 6. Paper/FAQ Use

Recommended paper phrasing:

Under repeated over-prediction of a wrong subject, AP-native negative cognitive pressure pushed the wrong subject's learned-vector similarity to its real context from positive to negative (about a 0.95 drop), while the correct association lost only 0.03. In audit-path recall the correct subject then dominated the pruned residue. This supports the claim that wrong predictive residue is actively pruned rather than left as a permanent bias, and that the pruning is targeted rather than indiscriminate.

Recommended boundary phrasing:

This is bottom-loop mechanism evidence using directed negative anchors on the learned vector. Symmetric negative pressure on a context would also damage its correct associations; the targeted directed form is the AP-native model for "the prediction was wrong, the context was not".

## 7. Artifacts

- `outputs/online_vector_negative_pressure_1_20260613_210139/online_vector_negative_pressure_1_report.json`
- `outputs/online_vector_negative_pressure_1_20260613_210139/online_vector_negative_pressure_1_report.md`

## 8. Next AP-Core Experiment

- `TransitionIsolation-1`: train A->B transitions and verify directed successor score rises while symmetric concept similarity does not.
