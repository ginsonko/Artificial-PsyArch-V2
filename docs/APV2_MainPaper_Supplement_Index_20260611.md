# APV2 Main Paper Supplement Index

Date: 2026-06-11

This index connects the concise APV2 architecture/runtime main paper to the long technical report and local evidence artifacts. It is an index, not a second copy of the full report.

## 1. Source Relationship

| item | path | role |
|---|---|---|
| Main paper draft | `docs/APV2_MainPaper_ArchitectureRuntime_Draft_20260611.md` | concise architecture/runtime manuscript |
| Master technical report | `docs/APV21_PublicPaper_InitialDraft_v1_0n_20260610.md` | long-form source, appendix material, full evidence context |
| P0 report | `outputs/apv2_bottom_loop_p0_materials_20260611_021431/apv2_bottom_loop_p0_materials_report.md` | parameter sensitivity, order ablation, defaults, tick trace |
| P1 report | `outputs/apv2_p1_hardening_materials_20260611_021432/apv2_p1_hardening_materials_report.md` | long-run recovery, rhythm replay, persistence reload, artifact freeze |
| P2 report | `outputs/apv2_p2_stress_mechanism_evidence_20260611_021433/apv2_p2_stress_mechanism_evidence_report.md` | residual-depth stress, long-run stability, short-term-slot parameter grid |
| Public freeze report | `outputs/apv2_public_freeze_candidate_20260611_022942/apv2_public_freeze_candidate_report.md` | 163-file local AP-Core paper-material freeze candidate |
| Clean-room rerun report | `outputs/apv2_clean_room_rerun_20260611_022953/clean_room_rerun_report.md` | same-machine staging-copy rerun, 2/2 commands passed |
| Third-party ACG-j audit report | `docs/ThirdParty_ACGj_ArtificialPsyArch_ArtifactAudit_20260611.md` | authorized third-party repo/artifact audit, commit and `.ap.zip` hash |
| Third-party ACG-j Rust rerun report | `docs/ThirdParty_ACGj_LocalRustRerun_Report_20260611.md` | local Rust rerun of independent AP-inspired bounded mechanisms |
| Third-party ACG-j rerun summary | `outputs/thirdparty_acgj_rerun_20260611/rerun_summary.json` | machine-readable rerun status and key metrics |
| Third-party ACG-j public-freeze-ready bundle | `outputs/thirdparty_acgj_public_freeze_20260611/acgj_public_freeze_ready_report.md` | source archive, `.ap` artifact copy, manifest, logs, and clean-copy 8/8 rerun |
| Controlled pilot baseline results | `docs/APV21_Paper_BaselinePilotResults_LBF1_LongRun_v0_2_20260606.md` | RepeatMap v0.5 / LBF1 / LongRun v0.2 controlled pilot vs real LLM/agent, provenance and records hash |
| AP-Core KeySuite report | `paper_artifacts/apv21_20260605/KEY_SUITE_REPORT.md` | Canonical-KeySuite-1 8/8 PASS, claim matrix, rerunnable script |
| AP-Core STP-v2 transfer report | `docs/FinalReport_APV21_STPv2_ProcessAnchorTransfer_v0_4_20260608.md` | process-anchor cross-surface transfer, sham-feeling falsification, targeted causal ablation |

## 2. Figure Inventory

| id | title | svg | png |
|---|---|---|---|
| F1 | APV2 runtime loop | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f1_apv2_runtime_loop.svg` | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f1_apv2_runtime_loop.png` |
| F2 | State pool vs short-term narrative slot | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f2_state_pool_vs_short-term_narrative_slot.svg` | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f2_state_pool_vs_short-term_narrative_slot.png` |
| F3 | Residual B recall absorption | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f3_residual_b_recall_absorption.svg` | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f3_residual_b_recall_absorption.png` |
| F4 | Successor lag and rhythm replay | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f4_successor_lag_and_rhythm_replay.svg` | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f4_successor_lag_and_rhythm_replay.png` |
| F5 | Evidence layer split | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f5_evidence_layer_split.svg` | `outputs/apv2_publication_figures_supplement_20260611_002510/figures/f5_evidence_layer_split.png` |
| F-EV1 | APV2 evidence stack (mechanism + pressure dynamics / baseline / task / third-party) | `outputs/apv2_press_evidence_figures_20260614_015121/f_ev1_evidence_panorama.svg` | `outputs/apv2_press_evidence_figures_20260614_015121/f_ev1_evidence_panorama.png` |
| F-EV2 | Controlled pilot cost vs capability (RepeatMap v0.5) | `outputs/apv2_press_evidence_figures_20260614_015121/f_ev2_baseline_cost_vs_capability.svg` | `outputs/apv2_press_evidence_figures_20260614_015121/f_ev2_baseline_cost_vs_capability.png` |

Manifest: `outputs/apv2_publication_figures_supplement_20260611_002510/figure_manifest.json`

## 3. Main Paper To Technical Report Map

| main paper section | technical report support | evidence artifact | boundary note |
|---|---|---|---|
| Abstract | Summary and APV2 bottom-loop additions | P0/P1/P2 reports | Claims runtime mechanisms, not full open-world mastery |
| 1. Introduction | Chapter 1.0-1.8 | technical report chapter 1 | APV2 is positioned as continuous cognition runtime |
| 2. Minimal objects | Chapter 2.2-2.8 | technical report chapter 2 | External fields are ordinary SA, not AP-native feelings |
| 3. Runtime architecture | Chapter 3.0-3.5 | F1/F2 figures | GL and product shell stay outside AP-Core proof |
| 4. Core mechanisms | Chapter 4.1-4.10.1 | P0/P1/P2 reports, F3/F4 | Residual recall and successor lag are mechanisms, not macro routes |
| 5. P0/P1/P2 evidence | Chapter 6.12 and recent P0/P1/P2 reports | P0/P1/P2 JSON/report artifacts | Evidence supports AP-Core bottom-loop dynamics |
| 5.4 third-party Rust rerun | Chapter 9.1 artifact boundary and third-party reproduction notes | ACG-j audit/rerun reports and summary JSON | Evidence supports cross-implementation portability of AP-style bounded mechanisms |
| 5.7 controlled pilot baseline | LBF1/RepeatMap/LongRun baseline cold-saves and pilot result drafts | baseline pilot result draft and `outputs/` artifacts | Controlled pilot candidate vs real LLM/agent, not a final benchmark |
| 5.8 KeySuite and STP-v2 | KeySuite methods cold-save and STP-v2 v0.1-v0.4 final reports | KeySuite report/results JSON and STP-v2 transfer report | AP-Core task evidence and process-feeling causal testability, adjacent to runtime proof |
| 6. Related systems | Chapter 7 and 8.2-8.5 | technical report discussion | LLMs are complementary carriers/teachers/tools |
| 7. Evidence boundary | Chapter 1.6, 5, 6.11, 9 | F5 evidence layer figure | DPP/Skill37/product shell are adjacent evidence lines |
| 8. Conclusion | Chapter 8 and 9 | technical report synthesis | Keep the claim positive but bounded |

## 4. P0/P1/P2 Evidence Map

| evidence | result | manuscript use |
|---|---|---|
| `APV2-BottomLoop-ParamSensitivity-1` | `16/16 pass` | bottom-loop mechanisms remain qualitatively stable under conservative parameter perturbations |
| `ShortTermSlot-OrderAblation-1` | full-order margin `18.2466`, without-order margin `9.0539` | order is a soft bias, not a hard gate |
| `LongRun-InterruptionRecovery-1` | interruptions `2`, resumptions `2`, final slot virtual mass `1.3715` | short-term narrative traces can recover after controlled interruption |
| `RhythmSuccessor-Replay-1` | lag 1 `1.0`, lag 2 `0.42`, lag 4 `0.172` | successor shaping has a next-tick peak and decaying tail |
| `PersistenceBackend-Reload-1` | warm-load loaded `3`, JSONL SHA-256 recorded | MemoryStore crosses a real local file persistence boundary |
| `ArtifactFreeze-1` | local manifest `12` entries | local pre-public traceability exists |
| `ResidualDepth-Stress-1` | 8 winners, 7 residual rounds, mass `14.3126 -> 0.7448` | residual B recall remains roundwise and inspectable under a larger mixed query |
| `LongRun-Stability-1` | interruptions `4`, resumptions `5`, final slot virtual mass `0.4791` | short-term narrative continuity remains recoverable under a longer interruption sequence |
| `ShortTermSlot-Grid-1` | `108/108 pass`, virtual mass `0.8706-4.1611`, clipped cases `81` | slot packets remain bounded, monotonic, and capacity-limited across a broader grid |
| `DoubleEnergyBalance-PressureDynamics-1` | stress `text_commit` near 0, `text_replace` / `replay_episode` rise; removing pressure anchor or mismatch weakens effect | pressure reshapes action competition toward reread / replace / replay |
| `DoubleEnergyBalance-PressureDynamics-Sweep-1` | clean sweep keeps `text_commit` positive while falling monotonically; stress sweep suppresses `text_commit` to `0` and lifts replay / revision with rising pressure | pressure effect is a reproducible curve, not a one-off point |
| `OnlineVector-WeightAblation-1` | audit-path neighbor score off `0.6848` -> default `1.0460` -> high `1.6308`; exact match stays rank 0 (score ~15) even at high weight | the online learned branch improves experience-neighborhood recall, is monotonic and bounded, and does not dominate exact SA/energy evidence |
| `OnlineVector-NegativePressure-1` | wrong residue similarity `0.3598` -> `-0.5937` under negative pressure; correct association drop only `0.0286`; audit recall correct `0.8136` vs wrong `0.0` | negative cognitive pressure prunes wrong predictive residue in a targeted way without damaging useful experience |
| `TransitionIsolation-1` | `learned_transition(A->B)` `0.0` -> `0.9412`; concept similarity exactly unchanged `0.1840 -> 0.1840`; reverse `0.0` | succession strengthens without contaminating concept similarity; the transition channel stays directed and separate |
| `APV2-PublicFreezeCandidate-1` | 163 files, zip SHA-256 `f14b78e982410e8e997bdcd3912e1512b3d3b419d5b17ad1744a1634bd19caed` | current AP-Core paper package is locally hash-frozen |
| `APV2-CleanRoomRerun-1` | 2/2 commands passed, including P0/P1/P2 tests | freeze package is rerunnable from a same-machine staging copy |
| `ACGj-RustRerun-1` | `cargo check` PASS; `cargo fmt --check` PASS; `cargo clippy ... -D warnings` PASS; `cargo test --lib` 84/84 PASS; core report commands PASS | independent Rust implementation reproduces AP-style training/audit mechanisms outside the original codebase |
| `ACGj-GeneratedMath-1` | train/holdout/generalization/reload accuracy `1.00`; untrained operator accuracy `0.00`; taint audit PASS | feedback learning, teacher-off generalization, skill reload, and boundary probes are jointly auditable |
| `ACGj-RelationWord-1` | teacher-off/cold-retest/reload/holdout accuracy `1.0`; controls `geometry_only=1.0`, `no_geometry=0.0`, `shuffled_expected=0.0` | relation-word learning can be tied to geometry features while separating label/path leakage |
| `ACGj-PublicFreezeReady-1` | source archive SHA-256 `F2C229584EB80F55B0C8791F741816129593A1D7F2235658F5807AD378481EC5`; `.ap.zip` SHA-256 `051589554123405652740729A02D5BD5A2B00EADDFA425AE2007BAC1EEAE7679`; clean-copy rerun `8/8` commands PASS | third-party evidence is packaged as a release-ready source-plus-artifact bundle |
| `RepeatMap-RealAPI-v0.5-fixed` | G1A/G1B holdout `1.0000±0.0000` at 0 token; real G3 `claude-opus-4-6` `~0.22-0.30`; real G4 `gpt-5.5-all`+memory/tool `0.83-0.85`; 1424 real calls; records SHA-256 `0bfc62535a43b8fb4b7c254e5587c529036cc14291c93474335789b9ef40ec7a` | controlled pilot candidate comparing AP-style endogenous feedback loop with real LLM/agent on cost and audit structure, not a final benchmark |
| `LongRun-UnknownRule-Learning-1-v0.2` | AP-style readapt gain `+0.2414`, external memory/tool baseline `+0.2826`, static/no-memory `+0.0000`; 9 seeds | AP-style continual learning shows learning gain, reload retention, and rule-switch readaptation in the same band as a strong external-tool baseline |
| `Canonical-KeySuite-1` | `8/8 PASS`, claim matrix fixed, rerunnable via `scripts/run_paper_key_suite.py` | AP-Core action-feedback-memory closed loop and controlled generalization are itemized and acceptance-tested |

## 7.1 F-EV1 / F-EV2 图的补充说明

F-EV1 是证据分层全景图, 不再只展示机制、对照、任务和第三方复现, 还把压力动力学和在线 learned vector 明确放入 AP-Core runtime 层: 高压力下的 `text_commit -> reread / replace / replay` 不是额外花絮, 而是底层动作竞争的一部分; learned vector 三证据则说明 AP 有一个有界、可审计、可定向 pruning 的邻域分支, 但它不改写主 SA/energy 定义。现在这条 A10 证据由单点实验 `DoubleEnergyBalance-PressureDynamics-1` 与扫参实验 `DoubleEnergyBalance-PressureDynamics-Sweep-1` 双层支撑, 单点负责因果触发, sweep 负责曲线复现。F-EV2 仍然保留成本-能力对照, 作为与机制图并列的受控基线图, 不承担压力动力学证明。

图像源已更新到最新 pressure evidence 包: `outputs/apv2_press_evidence_figures_20260614_015121/`。这使 F-EV1 / F-EV2 与 A10 证据、图注和正文引用保持一致。
| `STP-v2-ProcessAnchorTransfer-v0.4` | P0 cross-surface transfer `1.000` on D1/D2/D3, surface-keyword baseline `0.591`, sham feelings `0.509`/`0.482`; 3780 records, `20 passed` | cognitive feelings are process-grounded, temporally legal, and causally testable; not explained by adding any high-energy feature |
| `OnlineVector-WeightAblation-1` | audit-path neighbor score off `0.6848` -> default `1.0460` -> high `1.6308`; exact match stays rank 0 (score ~15) even at high weight | the online learned branch improves experience-neighborhood recall, is monotonic and bounded, and does not dominate exact SA/energy evidence |
| `OnlineVector-NegativePressure-1` | wrong residue similarity `0.3598` -> `-0.5937` under negative pressure; correct association drop only `0.0286`; audit recall correct `0.8136` vs wrong `0.0` | negative cognitive pressure prunes wrong predictive residue in a targeted way without damaging useful experience |
| `TransitionIsolation-1` | `learned_transition(A->B)` `0.0` -> `0.9412`; concept similarity exactly unchanged `0.1840 -> 0.1840`; reverse `0.0` | succession strengthens without contaminating concept similarity; the transition channel stays directed and separate |

## 5. Boundary Notes

| line | correct use |
|---|---|
| AP-Core runtime | Use P0/P1/P2 to support bottom-loop mechanism claims |
| AP-Core tasks | Use KeySuite/STP only as adjacent taredacted-test-key context unless Paper 2 expands it |
| GL learning | Use DPP/Skill37 only after GL-side teacher-off/cold retest and no-leakage audit |
| Product shell | Use desktop pet or UI demos as product/integration evidence, not AP-Core proof |
| Third-party clean-room implementation | Use ACG-j as cross-implementation AP-style bounded-mechanism evidence, with repo/commit/artifact hash/rerun logs attached |
| Cognitive feelings | Treat them as process-grounded SA generated from internal process quantities |
| Forbidden substitutes | Do not replace learning with answer tables, regex routes, hidden solvers, student-side LLM, or full-sentence macros |

## 6. Next Supplement Work

1. Convert this index into a venue-specific appendix after choosing a target format.
2. Add static line-number anchors if the technical report is frozen.
3. Add public artifact freeze commit/tag/hash when the release package is ready.
4. Ask the ACG-j owner to publish a tag/release/archive for the already generated local public-freeze-ready bundle; optional second independent clean-machine or VM rerun remains useful.
5. Add GL learning evidence only as a separate appendix or companion paper once GL validation is complete.
