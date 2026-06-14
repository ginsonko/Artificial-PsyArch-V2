# APV2 P2 Stress Mechanism Evidence

- generated_at: `2026-06-11T02:14:33`
- scope: `AP-Core runtime mechanism stress evidence`
- not_scope: `GL learning / Skill38 / OpenWorld / ACG-j rerun / desktop-pet product shell`
- all_passed: `True`
- pass: `3`
- partial: `0`
- fail: `0`

## Experiments

### ResidualDepth-Stress-1

- verdict: `pass`
- design: Larger mixed AP-Core state query should recall one B winner per round and absorb matched SA mass before later rounds.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "third_party_replication_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "one_b_winner_per_round": true, "language_learning_claim": false}`

- winners: `["pair_GH", "pair_AB", "pair_CD", "pair_EF", "pair_IJ", "distractor_mixed", "distractor_CY", "distractor_AX"]`
- trace_round_count: `7`
- mass_before: `[14.3126, 10.1431, 6.3442, 3.8148, 2.2067, 1.3052, 0.9265]`
- mass_after: `[10.1431, 6.3442, 3.8148, 2.2067, 1.3052, 0.9265, 0.7448]`

### LongRun-Stability-1

- verdict: `pass`
- design: Repeated AP-Core focus interruptions should preserve a resumable short-term narrative slot without exact replay forcing.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "third_party_replication_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "white_box_focus_probe": true, "open_dialogue_learning_claim": false}`

- tick_count: `12`
- interruptions: `4`
- resumptions: `5`
- final_slot_virtual_mass: `0.4791`
- state_recall_winners: `["garden_main_memory", "garden_successor_memory", "alarm_interrupt_memory", "math_interrupt_memory"]`

### ShortTermSlot-Grid-1

- verdict: `pass`
- design: Broader short-term-slot parameter grid should remain bounded, monotonic, and capacity-limited.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "product_shell_scope": false, "third_party_replication_scope": false, "answer_table_lookup": false, "keyword_hard_gate": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "parameter_optimality_claim": false, "bounded_packet_mechanics_probe": true}`

- pass_rate: `1.0` (108/108)
- virtual_mass_min/max: `0.8706` / `4.1611`
- capacity_clipped_count: `81`
