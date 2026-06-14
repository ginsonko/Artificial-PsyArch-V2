# APV2 P2 Stress Mechanism Evidence Final Report

Date: 2026-06-11

## 1. Design

This round focused on AP-Core bottom-loop mechanism stress evidence. The goal was to do useful AP-side work while ACG-j third-party artifacts and GL OpenWorld-Foundation evidence are pending.

The planned suite followed the required cycle:

1. design;
2. review and refinement;
3. implementation;
4. rigorous acceptance testing;
5. final report.

Scope boundary:

- included: AP-Core runtime mechanisms: residual B recall, short-term narrative slot, focus continuity, MemoryStore recall, artifact freeze, same-machine clean-room rerun;
- excluded: GL learning proof, Skill38/OpenWorld, product-shell/desktop-pet proof, ACG-j rerun, open-world dialogue maturity.

## 2. Review And Refinement

Design docs:

- `docs/Design_APV2_P2StressMechanismEvidence_20260611.md`
- `docs/Review_APV2_P2StressMechanismEvidence_20260611.md`

The review made three important choices:

- Do not require a fixed winner order in residual recall. MemoryStore combines several scoring channels, so the invariant is roundwise absorption and residual mass decline, not a hand-coded sequence.
- Do not require perfect narrative replay in long-run stability. The invariant is resumability, readback, and slot/state recall under interruptions.
- Do not hide capacity clipping in short-term slots. Low-capacity cases should remain bounded and interpretable, and clipping should be explicitly recorded.

## 3. Implementation

Created:

- `experiments/apv2_p2_stress_mechanism_evidence.py`
- `tests/test_apv2_p2_stress_mechanism_evidence.py`

Created evidence reports:

- `outputs/apv2_p2_stress_mechanism_evidence_20260611_021433/apv2_p2_stress_mechanism_evidence.json`
- `outputs/apv2_p2_stress_mechanism_evidence_20260611_021433/apv2_p2_stress_mechanism_evidence_report.md`
- `outputs/apv2_p2_stress_mechanism_evidence_20260611_021433/artifact_manifest.json`

Implementation correction:

- `memory/short_term/slot_packet.py` now composes packet rows with a minimum narrative skeleton instead of simple prefix truncation. This preserves summary, item/order evidence, and continuity under low capacity.

Why this correction matters:

- The first P2 run exposed that low-capacity short-term slots could fill with summary + item rows and drop order/continuity rows entirely.
- That behavior contradicted the intended role of the slot as a tick-level narrative inner-sense packet.
- The corrected composer keeps the packet bounded while preserving structural channels under pressure.

## 4. Acceptance Results

### ResidualDepth-Stress-1

Result: pass.

Key observations:

- winners: `pair_GH`, `pair_AB`, `pair_CD`, `pair_EF`, `pair_IJ`, `distractor_mixed`, `distractor_CY`, `distractor_AX`;
- 8 B winners returned;
- 7 residual trace rounds recorded;
- residual mass declined round by round:
  - before: `14.3126`, `10.1431`, `6.3442`, `3.8148`, `2.2067`, `1.3052`, `0.9265`;
  - after: `10.1431`, `6.3442`, `3.8148`, `2.2067`, `1.3052`, `0.9265`, `0.7448`;
- drained label count: `10`.

Interpretation:

- The residual-recall process shows the intended roundwise absorption behavior.
- Supported paired memories dominate early; distractors appear later after much of the supported query mass has been absorbed.

### LongRun-Stability-1

Result: pass.

Key observations:

- tick count: `12`;
- interruptions: `4`;
- resumptions: `5`;
- final slot virtual mass: `0.4791`;
- state recall winners include `garden_main_memory`;
- slot recall winners include recent/resumed slot memories.

Interpretation:

- The short-term narrative slot remains resumable under repeated interruption.
- The result supports AP-Core continuity and readback mechanics, not broad open-world dialogue maturity.

### ShortTermSlot-Grid-1

Result: pass.

Key observations:

- grid cases: `108/108 pass`;
- virtual mass range: `0.8706-4.1611`;
- capacity-clipped cases: `81`;
- capacity respected in every case;
- order coefficients monotonic in every case.

Interpretation:

- The short-term slot remains bounded and interpretable across capacity, virtual-budget, order-decay, and continuity-gain perturbations.
- The grid does not prove final parameter optimality; it proves the packet mechanics remain stable across the tested basin.

## 5. Reproducibility Update

After adding P2, the public-freeze package was regenerated.

Final freeze:

- output dir: `outputs/apv2_public_freeze_candidate_20260611_022942`
- file count: `163`
- zip bytes: `1,114,899`
- zip SHA-256: `f14b78e982410e8e997bdcd3912e1512b3d3b419d5b17ad1744a1634bd19caed`
- secret-like findings: `0`
- missing requested inputs: `0`

Final same-machine clean-room rerun:

- output dir: `outputs/apv2_clean_room_rerun_20260611_022953`
- commands: `2`
- passed: `2`
- failed: `0`

The clean-room pytest command included P0, P1, P2, figure, and freeze tests and reported `17 passed, 1 warning`. The warning is the known legacy Python `audioop` deprecation warning.

## 6. Paper And Index Updates

Updated:

- `docs/APV2_MainPaper_ArchitectureRuntime_Draft_20260611.md`
- `docs/APV2_MainPaper_Supplement_Index_20260611.md`
- `docs/Plan_APV2_NextEvidencePriorities_AfterSkill38AndThirdPartyAudit_20260611.md`
- `scripts/check_apv2_mainpaper_runtime_draft.py`
- `scripts/build_apv2_public_freeze_candidate.py`
- `tests/test_apv2_public_freeze_clean_room.py`

The main paper now treats P2 as current AP-Core runtime evidence rather than a future plan.

## 7. Validation Commands

Executed:

```powershell
python experiments\apv2_p2_stress_mechanism_evidence.py
python -m pytest tests\test_apv2_p2_stress_mechanism_evidence.py -q
python -m pytest tests\test_apv2_bottom_loop_p0_materials.py tests\test_apv2_p1_hardening_materials.py tests\test_apv2_p2_stress_mechanism_evidence.py -q
python -m pytest tests\test_apv2_public_freeze_clean_room.py -q
python scripts\build_apv2_public_freeze_candidate.py
python scripts\run_apv2_clean_room_rerun.py --freeze-dir outputs\apv2_public_freeze_candidate_20260611_022942
python scripts\check_apv2_mainpaper_runtime_draft.py
```

Observed:

- P2 script: 3 pass / 0 partial / 0 fail.
- P2 tests: 4 passed.
- P0/P1/P2 targeted tests: 12 passed, 1 warning.
- public-freeze tests: 3 passed.
- same-machine clean-room rerun: 2/2 commands passed.

## 8. Conclusion

This round completed the currently available AP-side work that did not depend on ACG-j or GL. The AP-Core mechanism evidence is now stronger in three ways:

- residual B recall has deeper mixed-query stress evidence;
- short-term narrative continuity has a longer interruption/resumption probe;
- short-term slot packet mechanics have a 108-case grid and a capacity-pressure fix.

The expected bottom-loop logic was observed: one-B-per-round residual absorption, decreasing residual mass, resumable narrative slot behavior, bounded short-term-slot virtual-energy injection, and reproducible AP-Core paper artifacts.

The next AP-side priority is no longer another local toy probe. The strongest next steps are independent clean-machine/VM rerun, ACG-j artifact freeze after the author responds, and later absorption of GL OpenWorld-Foundation only after GL finishes its own teacher-off/cold/ablation evidence.
