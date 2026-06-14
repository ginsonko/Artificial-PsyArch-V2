# APV2.2 AP-Core Dynamics Report

- generated_at: `2026-06-14T02:20:59`
- scope: `AP-Core bottom-loop dynamics`
- not_scope: `GL curriculum learning / skill-package generalization`
- all_passed: `True`
- pass: `8`
- partial: `0`
- fail: `0`

## Experiments

### FeedbackOverride-1

- verdict: `pass`
- design: Punishment-dominant later evidence overrides a rewarded same-token positive payload by deriving repair context.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

### PersistenceReload-1

- verdict: `pass`
- design: Reloaded authoritative memory restores state B recall, C successor prediction, and short-term-slot memory objects.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

### NegativeFeedback-Ablation-1

- verdict: `pass`
- design: Disable only the negative-text detector and verify punished raw text leaks back into positive payload.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

### ShortTermInterruptionRecovery-1

- verdict: `pass`
- design: Stable focus, interruption, and return should create observable interruption/resumption traces while slot packets keep narrative energy online.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

### ResidualDepth-1

- verdict: `pass`
- design: A multi-component query should be absorbed over multiple B rounds rather than reduced to a single winner.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

### SuccessorPeakAblation-1

- verdict: `pass`
- design: Lag-shaped successor energy should have a next-tick peak; disabling lag shaping should flatten the distribution.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false}`

### DoubleEnergyBalance-PressureDynamics-1

- verdict: `pass`
- design: Higher cognitive pressure should shift action competition toward reread / replace / replay and away from direct commit; removing pressure anchors or mismatch evidence should weaken the effect.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "runtime_mechanism_modified": false, "ap_core_full_proof_claimed": false}`

### DoubleEnergyBalance-PressureDynamics-Sweep-1

- verdict: `pass`
- design: Two pressure sweeps should show the same underlying shape in both clean and stressed regimes: clean drafts stay commit-dominant while commit weakens with pressure, and stressed drafts shift toward replay / revision as pressure rises.
- boundary: `{"ap_core_scope_only": true, "gl_learning_protocol_scope": false, "answer_table_lookup": false, "regex_route": false, "student_side_llm": false, "hidden_solver": false, "full_sentence_macro": false, "runtime_mechanism_modified": false, "ap_core_full_proof_claimed": false}`
