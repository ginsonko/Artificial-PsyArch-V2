# APV2 P1 Hardening Materials

- generated_at: `2026-06-11T02:14:32`
- scope: `AP-Core runtime hardening and publication materials`
- not_scope: `GL learning / DPP / Skill37 / desktop-pet product shell`
- all_passed: `True`
- pass: `4`
- partial: `0`
- fail: `0`

## Figures

- figure_count: `5`
- `outputs\apv2_p1_hardening_materials_20260611_021432\figures\apv2_runtime_loop.mmd` sha256=`55d4d6946228b6bf8f0253565e402a7e9dca00a78bf80e9cf549884452205b3c`
- `outputs\apv2_p1_hardening_materials_20260611_021432\figures\state_pool_vs_short_term_slot.mmd` sha256=`4739cf4223d6d45c774fc4a810b2d5753eed3d2d75df770b29e5b5306cb10f51`
- `outputs\apv2_p1_hardening_materials_20260611_021432\figures\residual_b_recall_absorption.mmd` sha256=`18cf64d3a0a16821df0500664ca4d00cb8615524984f415391ff4224ac378e2d`
- `outputs\apv2_p1_hardening_materials_20260611_021432\figures\successor_lag_rhythm_replay.mmd` sha256=`d40e9b4489815e1601f1430a0428a537c3b5020fdf1b1fb027fa13bb4de9a154`
- `outputs\apv2_p1_hardening_materials_20260611_021432\figures\evidence_layer_split.mmd` sha256=`a38796b562fcd183d1c152d857e96d9fd0b3708dc671a9d7688529e840be4409`

## Experiments

### LongRun-InterruptionRecovery-1

- verdict: `pass`
- design: White-box AP-Core focus packets are interrupted and resumed; short-term-slot and recall traces should recover the earlier narrative.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "white_box_focus_probe": true, "open_dialogue_learning_claim": false}`

- interruptions: `2`
- resumptions: `2`
- final_slot_virtual_mass: `1.3715`
- state_recall_winners: `["river_story", "river_successor", "alarm_distractor"]`

### RhythmSuccessor-Replay-1

- verdict: `pass`
- design: Periodic focus pulses should create phase expectation, slot rhythm rows, and next-tick successor peak with a decaying tail.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "music_performance_claim": false, "rhythm_replay_dynamics_probe": true}`

- phase_traces: `[{"tick": 13, "channels": {"family_key": "beat", "period_ticks": 4.0, "regularity": 1.0, "recurrence": 0.3333, "recovery_match": 1.0, "groove": 0.1389, "phase_expectation": 0.0111, "fatigue": 0.0}, "items": []}, {"tick": 14, "channels": {"family_key": "beat", "period_ticks": 4.0, "regularity": 1.0, "recurrence": 0.3333, "recovery_match": 1.0, "groove": 0.1389, "phase_expectation": 0.1353, "fatigue": 0.0}, "items": []}, {"tick": 15, "channels": {"family_key": "beat", "period_ticks": 4.0, "regularity": 1.0, "recurrence": 0.3333, "recovery_match": 1.0, "groove": 0.1389, "phase_expectation": 0.6065, "fatigue": 0.0}, "items": ["rhythmfelt::phase_expectation"]}, {"tick": 16, "channels": {"family_key": "beat", "period_ticks": 4.0, "regularity": 1.0, "recurrence": 0.3333, "recovery_match": 1.0, "groove": 0.1297, "phase_expectation": 1.0, "fatigue": 0.12}, "items": ["rhythmfelt::phase_expectation"]}]`
- successor_shaped: `[{"lag": 1, "kernel": 1.0, "predicted_labels": ["text::next_beat"]}, {"lag": 2, "kernel": 0.42, "predicted_labels": ["text::second_tail"]}, {"lag": 4, "kernel": 0.172, "predicted_labels": ["text::period_tail"]}]`

### PersistenceBackend-Reload-1

- verdict: `pass`
- design: A real local JSONL persistence boundary should support cold MemoryStore reload and restore state, successor, and slot recall.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "local_file_persistence_boundary": true, "postgres_production_claim": false}`

- persistence_path: `H:\AP原型实验第二期\APV2.1版本原型测试\outputs\apv2_p1_hardening_materials_20260611_021432\jsonl_persistence\apv2_memory.jsonl`
- sha256: `19a0d88ba4ce8eacb01fe488ae72207a427c50f85bd8cc7bf4a30f74a50e60d7`
- warm_load: `{"schema_id": "apv21_warm_load/v1", "loaded": 3, "skipped": 0, "indexed": 0, "pending_index_jobs": 0, "process_indexes": true, "replay_learning": false, "policy": "bounded_hot_window_loaded_from_authoritative_persistence;no_full_history_load"}`

### ArtifactFreeze-1

- verdict: `pass`
- design: Local pre-public artifact freeze records exact files, sizes, and SHA-256 hashes for paper-material traceability.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "public_doi_claim": false, "local_pre_public_freeze": true}`

- manifest_path: `H:\AP原型实验第二期\APV2.1版本原型测试\outputs\apv2_p1_hardening_materials_20260611_021432\artifact_freeze_manifest.json`
- entry_count: `12`
