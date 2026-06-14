# APV2.2 AP-Core Dynamics Experiment Plan (2026-06-10)

This note separates AP-side mechanism experiments from GL-side learning
experiments.

## Boundary

AP-Core experiments validate bottom-loop dynamics:

- energy flow in state pool
- short-term narrative slot injection and interruption/resumption traces
- B/Bn recall and residual absorption
- C/Cn successor lag shaping
- reward/punishment-derived repair states
- persistence reload of AP memory structures
- module-specific ablations

GL experiments validate learning protocol and curriculum outcomes:

- scaffold phases
- state-to-response-family learning
- teacher-off/cold retest
- open-world validation
- skill packages and learning evidence

GL replay evidence must not be described as AP-Core runtime proof unless it has
a separate AP-Core no-leakage audit.

## AP Experiment Matrix

1. FeedbackOverride-1
   - Same token receives reward and punishment evidence.
   - Expected AP result: punishment-dominant text is not a positive payload; it
     becomes a repair context.
   - Purpose: prove reward/punishment is the base feedback layer and repair is
     derived, not a second feedback system.

2. PersistenceReload-1
   - Write state and short-term-slot snapshots through the persistence adapter.
   - Warm-load into a fresh memory store.
   - Expected AP result: B recall, C successor prediction, and short-term-slot
     snapshots survive reload.
   - Purpose: prove the bottom loop writes durable AP memory artifacts.

3. NegativeFeedback-Ablation-1
   - Compare normal repair conversion with a controlled ablation that disables
     the negative-text detector.
   - Expected AP result: normal path suppresses raw punished text; ablated path
     lets it remain as positive text payload.
   - Purpose: prove the repair behavior is caused by the AP feedback projection
     layer, not by an unrelated scorer artifact.

4. ShortTermInterruptionRecovery-1
   - Run a stable focus sequence, interrupt with another sequence, then return
     to the original sequence.
   - Expected AP result: focus buffer records interruption/resumption and
     short-term-slot packets continue to inject narrative energy.
   - Purpose: validate the humanlike "continuous fragments plus jumps" design.

5. ResidualDepth-1
   - Use a query with multiple separable components and several partial memory
     objects.
   - Expected AP result: one winner per round; matched SA mass decreases;
     later rounds shift toward unabsorbed components.
   - Purpose: validate the resonance/absorption interpretation of B recall.

6. SuccessorPeakAblation-1
   - Compare lag-shaped successor kernels with lag shaping disabled.
   - Expected AP result: shaped path has next-tick peak and decaying tail;
     ablated path flattens the kernels.
   - Purpose: show successor bias is a time-energy distribution, not an
     answer table or fixed n-gram gate.

## Reporting Rule

Every AP experiment should report:

- design intent
- observed mechanism trace
- pass conditions
- ablation result where applicable
- boundary flags: no answer table, no regex route, no student-side LLM, no
  hidden solver, no full-sentence action macro.

## Final Result

Status: passed.

- Runner: `experiments/apv22_apcore_dynamics.py`
- Latest evidence directory: `outputs/apv22_apcore_dynamics_20260610_203001/`
- JSON: `outputs/apv22_apcore_dynamics_20260610_203001/apv22_apcore_dynamics.json`
- Markdown report: `outputs/apv22_apcore_dynamics_20260610_203001/apv22_apcore_dynamics_report.md`
- Summary: `6 pass / 0 partial / 0 fail`
- Guard tests: `tests/test_apv22_apcore_dynamics.py`
- Joint regression:
  `pytest -q tests/test_apv22_apcore_dynamics.py tests/test_apv22_core_loop_acceptance.py tests/test_p1j16_short_term_memory_window.py tests/test_prediction_energy_budget_semantics.py tests/test_phase1_text_chain.py`
- Joint regression result: `152 passed, 1 warning`

Observed AP-Core mechanism evidence:

- FeedbackOverride-1: later punishment-dominant evidence for the same token
  suppresses positive raw text payload and emits the repair state.
- PersistenceReload-1: warm-load restored state B recall, C successor prediction,
  and short-term-slot memory recall from the persistence adapter.
- NegativeFeedback-Ablation-1: disabling only the negative-text detector made
  punished raw text leak back into positive payload, proving the normal repair
  behavior is module-specific.
- ShortTermInterruptionRecovery-1: white-box focus packets produced interruption
  and resumption traces; short-term-slot rows remained readable from state pool
  and persisted in `short_term_slot` memory.
- ResidualDepth-1: separable query components were absorbed over multiple B
  rounds: `A/B`, then `C`, then `E`, with declining residual mass.
- SuccessorPeakAblation-1: shaped kernels were `1.0`, `0.42`, `0.1101`; the
  lag-shaping ablation flattened kernels to `1.0`, `1.0`, `1.0`.

Interpretation: APV2.2 now has a stronger AP-Core-only evidence packet for the
bottom-loop dynamics. This does not replace GL learning experiments; it gives GL
a better base and gives the AP paper/design notes a cleaner mechanism-level
evidence layer.
