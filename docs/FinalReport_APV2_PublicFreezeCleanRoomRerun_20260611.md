# APV2 Public Freeze And Clean-Room Rerun Final Report

Date: 2026-06-11

## 1. Design

This round targeted the AP-Core architecture/runtime paper evidence layer. The goal was to harden traceability and reproducibility without mixing in GL learning proof, DPP/Skill37 open-dialogue proof, or desktop-pet/product-shell evidence.

The designed evidence package has two parts:

1. `APV2-PublicFreezeCandidate-1`: a local hash-frozen AP-Core paper-material package.
2. `APV2-CleanRoomRerun-1`: a same-machine staging-copy rerun from that frozen package.

This is stronger than running scripts in the live workspace, but it is still not a DOI artifact, public repo release, independent clean-machine rerun, or third-party replication.

## 2. Review And Refinement

The main risk was hidden dependency on the original workspace. The package therefore includes the runtime dependencies needed by P0/P1 checks: `config`, `core`, `memory`, `channels`, `sensors`, selected `legacy_apv2/sensors`, `education`, and `learning_events.py`. GL and product-shell directories remain excluded from the AP-Core paper package.

The second risk was overclaiming. The design/review docs and reports explicitly label the result as local public-freeze candidate plus same-machine clean-room rerun.

Design docs:

- `docs/Design_APV2_PublicFreezeCleanRoomRerun_20260611.md`
- `docs/Review_APV2_PublicFreezeCleanRoomRerun_20260611.md`

## 3. Implementation

Created:

- `scripts/build_apv2_public_freeze_candidate.py`
- `scripts/run_apv2_clean_room_rerun.py`
- `tests/test_apv2_public_freeze_clean_room.py`

Updated:

- `docs/APV2_MainPaper_ArchitectureRuntime_Draft_20260611.md`

The paper now records that material-level freeze/rerun evidence exists, while preserving the boundary that this is not public DOI-level or independent third-party replication.

## 4. Artifact Freeze Result

Final freeze output:

- output dir: `outputs/apv2_public_freeze_candidate_20260611_010141`
- package dir: `outputs/apv2_public_freeze_candidate_20260611_010141/package`
- manifest: `outputs/apv2_public_freeze_candidate_20260611_010141/package/public_freeze_manifest.json`
- report: `outputs/apv2_public_freeze_candidate_20260611_010141/apv2_public_freeze_candidate_report.md`
- archive: `outputs/apv2_public_freeze_candidate_20260611_010141/apv2_public_freeze_candidate.zip`
- zip SHA-256: `8591c11523f0cd71edb437353d8b102da9490aa50e4f28937c9408a0c894726d`

Summary:

| item | value |
|---|---:|
| file count | 156 |
| total bytes | 3,259,348 |
| zip bytes | 1,093,823 |
| missing requested inputs | 0 |
| secret-like findings | 0 |
| repository status | not a git repository |

Category summary:

| category | files | bytes |
|---|---:|---:|
| AP-Core runtime dependency | 98 | 2,007,324 |
| Frozen paper evidence output | 22 | 853,844 |
| Paper documentation | 14 | 100,054 |
| Runtime support dependency | 7 | 91,550 |
| Paper reproducibility script | 4 | 44,416 |
| AP-Core paper test | 4 | 14,111 |
| Other included file | 4 | 91,959 |
| AP-Core paper experiment | 2 | 55,682 |
| Package metadata | 1 | 408 |

## 5. Clean-Room Rerun Result

Final rerun output:

- output dir: `outputs/apv2_clean_room_rerun_20260611_010150`
- stage: `outputs/apv2_clean_room_rerun_20260611_010150/stage`
- JSON: `outputs/apv2_clean_room_rerun_20260611_010150/clean_room_rerun.json`
- report: `outputs/apv2_clean_room_rerun_20260611_010150/clean_room_rerun_report.md`

Commands:

| command | returncode | seconds | result |
|---|---:|---:|---|
| `python scripts\check_apv2_mainpaper_runtime_draft.py` | 0 | 0.157 | pass |
| `python -m pytest tests\test_apv2_publication_figures_supplement.py tests\test_apv2_bottom_loop_p0_materials.py tests\test_apv2_p1_hardening_materials.py tests\test_apv2_public_freeze_clean_room.py -q` | 0 | 21.623 | pass |

Clean-room summary: 2/2 commands passed.

## 6. Acceptance Tests

Validation commands run in the live workspace:

```powershell
python scripts\check_apv2_mainpaper_runtime_draft.py
python -m pytest tests\test_apv2_publication_figures_supplement.py tests\test_apv2_bottom_loop_p0_materials.py tests\test_apv2_p1_hardening_materials.py tests\test_apv2_public_freeze_clean_room.py -q
```

Results:

- main paper check: PASS, 24,840 chars checked
- pytest: 13 passed, 1 warning

The warning is the known Python `audioop` deprecation warning from `legacy_apv2/sensors/hearing_sensor_v1.py`; it is unrelated to the AP-Core paper-material checks.

## 7. Reviewer Score Update

Previous reviewer-style score: 7.75 / 10.

After this round, my updated estimate is **8.2 / 10**.

Why it improves:

- The paper no longer merely plans local artifact hardening; it has a 156-file hash-frozen AP-Core paper package.
- The package has a zip hash, manifest, report, command list, and secret-like scan result.
- The paper checks and P0/P1/figure/freeze tests pass from a separate staging copy.
- The main paper now states this positively and with correct boundaries.

Why it is not 9+ yet:

- The workspace is not a git repository, so there is no commit/tag/hash.
- The rerun is same-machine, not independent clean-machine reproduction.
- There is no public DOI/archive host.
- There is no third-party replication.
- Broader stress/OOD/baseline evidence is still limited.

## 8. Full-Score Evidence Gap Matrix

| target | current status | local feasibility | score impact |
|---|---|---|---|
| Public repo/archive with release tag | not available in this checkout | feasible if we create a clean public repo/archive | high |
| Dependency lock or container | not yet frozen | feasible locally with `requirements.txt`/lockfile or Docker-style notes | medium-high |
| Independent clean-machine rerun | same-machine staging rerun passed | needs a second machine/VM or clean environment | high |
| Third-party replication | not available | needs external collaborator | very high |
| ResidualDepth-Stress-1 | not yet added beyond P0 probe | feasible locally | medium |
| LongRun-Stability-1 with sleep/wake/task switch | partial adjacent long-run evidence exists | feasible locally but longer runtime | medium-high |
| ShortTermSlot capacity/energy ablation grid | P0 has conservative sensitivity | feasible locally | medium |
| Baseline comparison | related work prose only | feasible at small scale; costly for strong LLM baselines | medium |
| GL learning teacher-off/cold retest | owned by GL route | wait for GL | high for companion learning paper |
| Product-shell long autonomous demo | separate product evidence | wait for desktop-pet/product route | medium for public communication, low for AP-Core proof |

## 9. Can We Push Toward Full Marks?

Yes, but not by writing more prose alone. The strongest next evidence is:

1. Create a public-style artifact release candidate with a stable archive hash and a minimal dependency lock.
2. Run the same package in a fresh VM or second Windows/Python environment.
3. Add `ResidualDepth-Stress-1` and `LongRun-Stability-1` to show the bottom loop remains interpretable under heavier pressure.
4. Keep GL learning evidence separate until GL finishes teacher-off/cold retest/no-leakage validation.
5. Convert the long technical report into a supplement with stable artifact anchors.

The team can produce items 1, 3, and 5 locally. Item 2 needs a cleaner environment or VM. Item 4 depends on GL. Third-party replication requires outside help, but it would be the single strongest move for a 9+ reviewer score.

## 10. Conclusion

This round successfully closed the biggest immediate reproducibility gap short of true public release. APV2 now has a paper-specific local freeze candidate and a passing same-machine clean-room rerun. The evidence supports a stronger reviewer-facing claim: the AP-Core runtime paper materials are traceable, hash-frozen, and locally rerunnable from a copied package.

The remaining path to a top-tier score is clear: public release identity, independent rerun, stronger stress tests, and external replication.

