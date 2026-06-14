# APV2.1 STP-v2 Process Anchor Transfer v0.4 Final Report

Date: 2026-06-08

## 1. Cycle

This run followed the requested loop:

1. Design: `docs/Design_APV21_STPv2_ProcessAnchorTransfer_v0_4_20260608.md`
2. Review and refinement: `docs/Review_APV21_STPv2_ProcessAnchorTransfer_v0_4_Design_20260608.md`
3. Theory hardening: `docs/APV21_CognitiveFeeling_LegitimacyStandard_v0_1_20260608.md`
4. Implementation: `scripts/run_stpv2_process_anchor_transfer_v04.py`
5. Strict validation: `tests/test_stpv2_process_anchor_transfer_v04.py`
6. Final evidence package: `outputs/stpv2_process_anchor_transfer_v04_20260608/`

## 2. Purpose

v0.3 showed causal contribution inside one controlled STP-v2 setting. v0.4 asks whether the same endogenous process-anchor vocabulary can transfer across external surfaces, and whether matched-energy fake feelings fail when they lack process origin.

This directly supports the cognitive-feeling legitimacy standard:

- valid cognitive feelings must come from AP-internal process variables;
- they must be visible before the decision;
- they must not encode private labels or target answers;
- they must be causally testable;
- useful feelings should have cross-surface potential.

## 3. Boundary

This is a strict-core controlled cross-surface transfer experiment. It is not a full APV2.1 open-world runtime completion claim.

Student-side API calls: `0`  
Student-side LLM: `false`  
Hidden solver: `false`  
Record count: `3780`  
Seeds: `2026060841, 2026060842, 2026060843, 2026060844, 2026060845`

## 4. Experimental Setup

Training domain:

- D1 text relation domain only.

Transfer domains:

- D1 text relation held-out;
- D2 symbol/shape relation;
- D3 draft/buffer repair.

The primary process route learns two D1-trained action heads:

- relation-trigger head: `teacher_context + correction_event`;
- local-repair head: `mismatch + low_grasp`.

D2/D3 receive no per-domain retuning in the process route.

Controls:

- D1 surface keyword baseline;
- domain-specific surface adapter upper bound, explicitly non-AP-native;
- energy-matched random sham feeling;
- energy-matched permuted sham feeling;
- no-mismatch ablation;
- no-teacher/correction ablation.

## 5. Main Results

| Group | Macro accuracy | D1 text | D2 symbol/shape | D3 draft repair | Trigger accuracy | Repair accuracy | Replace event |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 process-anchor transfer | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| P1 D1 surface keyword baseline | 0.591 | 0.772 | 0.500 | 0.500 | 0.675 | 0.500 | 0.000 |
| P2 domain-specific adapter upper bound | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 |
| P3 random sham feeling | 0.509 | 0.500 | 0.528 | 0.500 | 0.529 | 0.488 | 0.511 |
| P4 permuted sham feeling | 0.482 | 0.722 | 0.556 | 0.167 | 0.571 | 0.385 | 0.577 |
| P5 no mismatch | 0.759 | 0.778 | 1.000 | 0.500 | 1.000 | 0.500 | 0.000 |
| P6 no teacher+correction | 0.741 | 0.722 | 0.500 | 1.000 | 0.500 | 1.000 | 0.500 |

## 6. Interpretation

### 6.1 Cross-surface transfer

P0 reaches `1.000` on D1/D2/D3. Because the process route trains only on D1 and uses the same process-anchor action heads on D2/D3, the result supports controlled cross-surface transfer.

### 6.2 Surface binding risk

P1 drops to `0.500` on D2 and `0.500` on D3, with macro accuracy `0.591`. This supports the teaching-protocol warning: surface markers learned in one domain do not provide robust transfer when the external form changes.

### 6.3 Sham-feeling falsification

P3 random sham feeling reaches macro accuracy `0.509`; P4 permuted sham feeling reaches `0.482`. Both use matched or reused energy scales, but lack case-level process origin. This supports:

> process origin matters; the effect is not explained by adding any high-energy feature.

### 6.4 Targeted causal roles

P5 no-mismatch keeps relation triggering high but drops D3 repair to `0.500` and replace events to `0.000`. This localizes `mismatch` to repair/revision behavior.

P6 no-teacher/correction keeps D3 repair high but drops trigger accuracy to `0.500`. This localizes `teacher_context/correction_event` to relation/paradigm triggering.

These results sharpen v0.3's conclusion: different cognitive feelings control different process stages.

## 7. Validation

Command:

```powershell
python -m pytest tests\test_stpv2_process_anchor_audit_v01.py tests\test_stpv2_process_anchor_runtime_bridge_v02.py tests\test_stpv2_process_anchor_ablation_v03.py tests\test_stpv2_process_anchor_transfer_v04.py -q
```

Result:

```text
20 passed, 1 warning in 59.56s
```

The warning is the legacy Python `audioop` deprecation warning from `legacy_apv2\sensors\hearing_sensor_v1.py`; it is unrelated to this experiment.

v0.4 single-test run:

```text
5 passed in 1.56s
```

Formal output run:

```text
schema_id: stpv2_process_anchor_transfer/v0.4
output_dir: outputs\stpv2_process_anchor_transfer_v04_20260608
validation_passed: true
record_count: 3780
```

## 8. Artifacts

Output directory:

`outputs/stpv2_process_anchor_transfer_v04_20260608/`

Main files:

- `summary.json`
- `records.json`
- `private_examiner_key.json`
- `artifact_manifest.json`
- `STPV2_ProcessAnchorTransfer_v04_report_zh.md`
- `stpv2_process_anchor_transfer_v04_showcase_zh.html`

Artifact hashes from local manifest:

| File | SHA-256 |
|---|---|
| `summary.json` | `6784daf857c006ed221dea160db9f22691d7148e30ab7001b585e1f141944cd0` |
| `records.json` | `125ead806d0edab6c9cf1a4df19b5d46643600d6dc914bfc7e94cf6393b47a15` |
| `private_examiner_key.json` | `f18aa5aa5cb84ca7360f3c62ac99e25598f2671590c7f381def37c8410862bbb` |
| `STPV2_ProcessAnchorTransfer_v04_report_zh.md` | `cb8f3005d199f7766dd97d10c653d5e5346453b013374d1d845eeb946e980ba5` |
| `stpv2_process_anchor_transfer_v04_showcase_zh.html` | `2382cf912a7688c305e7efbb21bc98af004169c68ad5879f78e734c0e8eb8e27` |

Manifest boundary:

`local manifest/hash freeze only; current workspace is not a git repository`

## 9. Paper-Ready Claim

Recommended claim:

> In controlled cross-surface transfer, D1-trained process-anchor action heads transferred to symbol/shape and draft-buffer domains with 1.000 macro accuracy, while D1 surface markers and matched-energy sham feelings failed to recover the same behavior. Targeted ablations further localized mismatch to local repair and teacher/correction anchors to relation triggering. This supports the claim that valid AP cognitive feelings must be process-grounded, temporally legal, and causally testable.

Recommended limitation:

> These results establish controlled transfer and sham-feeling falsification for selected process anchors. They do not prove complete open-world language understanding, full APV2.1 runtime autonomy, or general superiority over LLMs.

