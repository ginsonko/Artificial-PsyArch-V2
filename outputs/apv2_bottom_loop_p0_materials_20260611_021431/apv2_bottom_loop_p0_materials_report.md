# APV2 Bottom-Loop P0 Publication Materials

- generated_at: `2026-06-11T02:14:31`
- scope: `AP-Core bottom-loop publication materials`
- not_scope: `GL learning / DPP / Skill37 / desktop-pet product shell`
- all_passed: `True`
- pass: `2`
- partial: `0`
- fail: `0`

## Experiments

### APV2-BottomLoop-ParamSensitivity-1

- verdict: `pass`
- design: Conservative perturbations around APV2 bottom-loop defaults should preserve qualitative mechanism traces.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

- pass_rate: `1.0` (16/16)

### ShortTermSlot-OrderAblation-1

- verdict: `pass`
- design: Short-term-slot order should provide a soft recall advantage, while item overlap still permits nonmatching order recall.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "order_is_soft_bias_not_hard_gate": true, "item_overlap_recall_still_allowed": true}`

- full_order_scores: `{"slot_ABC": 36.1733, "slot_CBA": 17.9267, "margin": 18.2466}`
- without_order_rows_scores: `{"slot_ABC": 21.0494, "slot_CBA": 11.9955, "margin": 9.0539}`
- margin_drop_without_order_rows: `9.1927`

## Default Parameters

### Short-Term Slot

| item | value |
|---|---|
| `enabled` | `true` |
| `capacity` | `32` |
| `base_virtual_budget` | `0.72` |
| `item_real_fraction` | `0.06` |
| `item_min_virtual` | `0.02` |
| `item_max_virtual` | `0.14` |
| `item_rank_decay` | `0.86` |
| `item_order_decay` | `0.92` |
| `summary_ratio` | `0.18` |
| `order_ratio` | `0.16` |
| `continuity_ratio` | `0.14` |
| `rhythm_ratio` | `0.1` |
| `load_floor` | `0.25` |
| `continuity_gain` | `0.35` |
| `order_gain` | `0.28` |
| `rhythm_gain` | `0.22` |
| `working_memory_fill_limit` | `8` |
| `focus_merge_limit` | `32` |

### Memory And Bottom-Loop Policies

| item | value |
|---|---|
| `recall_top_k` | `5` |
| `predict_top_k` | `5` |
| `prediction_energy_scale` | `0.55` |
| `max_snapshots_per_kind` | `256` |
| `candidate_limit` | `256` |
| `core_item_limit` | `1024` |
| `query_feature_limit` | `1024` |
| `scoring_candidate_limit` | `96` |
| `temporal_applicability_enabled` | `true` |
| `temporal_floor` | `0.18` |

### Residual B Recall

| item | value |
|---|---|
| `round_policy` | `"one_b_winner_per_round"` |
| `max_rounds` | `"min(top_k * 2, 12)"` |
| `matched_label_scale` | `"max(0.08, 1 - match_efficiency * 0.82)"` |
| `unmatched_label_scale` | `"max(0.48, 1 - match_efficiency * 0.20)"` |
| `energy_semantic` | `"matched query SA mass is absorbed into the selected B object for the next round only"` |

### Successor Lag Kernel

| item | value |
|---|---|
| `lag_1` | `1.0` |
| `lag_2` | `0.42` |
| `lag_ge_3` | `"max(0.08, 0.42 * (0.64 ** (lag - 2)))"` |
| `semantic` | `"next_tick_peak_then_sharp_drop_and_tail"` |

## Representative Tick Trace

- trace_source: `sun_moon_star_teacher_off_probe`
- tick_index: `9`
- short_term_slot_virtual_mass: `1.8849`
- short_term_slot_first_labels: `["short_term_slot::summary", "short_term_slot::item::text::sun", "short_term_slot::item::text::moon", "short_term_slot::item::feeling::dissonance", "short_term_slot::item::expectation_pressure::pressure", "short_term_slot::item::feeling::surprise", "short_term_slot::item::text_revision_opportunity::start_empty_draft", "short_term_slot::item::text_action::draft_state", "short_term_slot::item::runtimefelt::simplicity", "short_term_slot::order::0::text::sun", "short_term_slot::order::1::text::moon", "short_term_slot::order::2::feeling::dissonance", "short_term_slot::order::3::expectation_pressure::pressure", "short_term_slot::order::4::feeling::surprise", "short_term_slot::order::5::text_revision_opportunity::start_empty_draft", "short_term_slot::order::6::text_action::draft_state"]`
- fast_residual_b_recall: `[{"memory_id": "mem-25", "source_text": "star", "score": 142.7148, "round_index": null, "matched_labels": [], "residual_mass_before": null, "residual_mass_after": null}, {"memory_id": "mem-10", "source_text": "sun", "score": 139.8634, "round_index": null, "matched_labels": [], "residual_mass_before": null, "residual_mass_after": null}, {"memory_id": "mem-22", "source_text": "moon", "score": 118.4728, "round_index": null, "matched_labels": [], "residual_mass_before": null, "residual_mass_after": null}, {"memory_id": "mem-16", "source_text": "star", "score": 72.294, "round_index": null, "matched_labels": [], "residual_mass_before": null, "residual_mass_after": null}]`
- fast_residual_b_recall_probe: `{"winners": ["AB", "CD", "E", "A"], "round_count": 3, "mass_declines": true, "distinct_winners": true, "covers_late_component": true, "absorption_trace": [{"schema_id": "residual_b_recall_round/v1", "round_index": 1, "winner_memory_id": "mem-1", "winner_score": 7.6549, "match_efficiency": 0.985, "matched_labels": ["text::A", "text::B"], "drained_labels": ["text::A", "text::B"], "residual_mass_before": 4.922, "residual_mass_after": 2.3188, "policy": "one_b_per_round_matched_sa_absorption"}, {"schema_id": "residual_b_recall_round/v1", "round_index": 2, "winner_memory_id": "mem-2", "winner_score": 3.3991, "match_efficiency": 0.7628, "matched_labels": ["text::C"], "drained_labels": ["text::C"], "residual_mass_before": 2.3188, "residual_mass_after": 1.5182, "policy": "one_b_per_round_matched_sa_absorption"}, {"schema_id": "residual_b_recall_round/v1", "round_index": 3, "winner_memory_id": "mem-3", "winner_score": 3.2757, "match_efficiency": 0.7553, "matched_labels": ["text::E"], "drained_labels": ["text::E"], "residual_mass_before": 1.5182, "residual_mass_after": 0.9479, "policy": "one_b_per_round_matched_sa_absorption"}]}`
- fast_successor_lag_predictions: `[{"source_memory_id": "mem-10", "successor_memory_id": "mem-13", "lag": 1, "kernel": 1.0, "predicted_labels": ["text::moon", "text::star", "text::sun", "text_action::draft_state", "text_action::write::moon"], "predicted_virtual_energy": [0.6366, 0.2544, 0.7715, 0.6669, 0.1904]}, {"source_memory_id": "mem-22", "successor_memory_id": "mem-25", "lag": 1, "kernel": 1.0, "predicted_labels": ["text::moon", "text::star", "text::sun", "text_action::draft_state", "text_action::write::moon"], "predicted_virtual_energy": [0.6084, 0.3106, 0.6787, 0.4929, 0.2358]}, {"source_memory_id": "mem-16", "successor_memory_id": "mem-19", "lag": 1, "kernel": 1.0, "predicted_labels": ["text::moon", "text::star", "text::sun", "text_action::draft_state", "text_action::write::moon"], "predicted_virtual_energy": [0.4702, 0.2846, 0.7065, 0.5381, 0.1731]}]`
- attention_selected_labels: `["text::sun", "text::moon", "feeling::dissonance", "expectation_pressure::pressure", "feeling::surprise", "text_revision_opportunity::start_empty_draft", "text_action::draft_state", "runtimefelt::simplicity"]`
