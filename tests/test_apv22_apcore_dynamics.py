from __future__ import annotations

from experiments.apv22_apcore_dynamics import (
    run_double_energy_balance_pressure_dynamics,
    run_double_energy_balance_pressure_sweep,
    run_feedback_override,
    run_negative_feedback_ablation,
    run_persistence_reload,
    run_residual_depth,
    run_short_term_interruption_recovery,
    run_successor_peak_ablation,
)


def _assert_pass(result: dict) -> None:
    assert result["verdict"] == "pass", result


def test_feedback_override_projects_punishment_into_repair_state() -> None:
    result = run_feedback_override()
    _assert_pass(result)
    labels = result["observed"]["payload_labels_after_punishment"]
    assert "text::same" not in labels
    assert "text_revision_opportunity::negative_feedback::same" in labels


def test_persistence_reload_restores_state_successor_and_slot_memory() -> None:
    result = run_persistence_reload()
    _assert_pass(result)
    observed = result["observed"]
    assert observed["warm_load"]["loaded"] >= 3
    assert "cue context" in observed["bn_source_texts"]
    assert "text::outcome" in observed["predicted_labels"]
    assert observed["slot_bn_source_texts"] == ["slot cue context"]


def test_negative_feedback_ablation_leaks_punished_text_without_detector() -> None:
    result = run_negative_feedback_ablation()
    _assert_pass(result)
    observed = result["observed"]
    assert observed["normal_suppressed"] is True
    assert observed["ablated_leaks_bad"] is True


def test_short_term_interruption_recovery_records_resume_and_slot_injection() -> None:
    result = run_short_term_interruption_recovery()
    _assert_pass(result)
    observed = result["observed"]
    assert observed["recent_interruptions"]
    assert observed["recent_resumptions"]
    assert observed["slot_virtual_mass"] > 0.0
    assert observed["state_readback_labels"]
    assert observed["slot_memory_labels"]


def test_residual_depth_absorbs_separable_query_components() -> None:
    result = run_residual_depth()
    _assert_pass(result)
    trace = result["observed"]["absorption_trace"]
    assert trace[0]["matched_labels"] == ["text::A", "text::B"]
    assert trace[1]["matched_labels"] in (["text::C"], ["text::E"])
    assert trace[2]["matched_labels"] in (["text::E"], ["text::A"])
    assert result["observed"]["mass_declines"] is True


def test_successor_peak_ablation_flattens_lag_kernel() -> None:
    result = run_successor_peak_ablation()
    _assert_pass(result)
    shaped = result["observed"]["shaped"]
    assert [row["kernel"] for row in shaped] == [1.0, 0.42, 0.1101]
    assert result["observed"]["flat_kernels"] == [1.0, 1.0, 1.0]


def test_double_energy_balance_pressure_dynamics_shifts_commit_to_revision_and_replay() -> None:
    result = run_double_energy_balance_pressure_dynamics()
    _assert_pass(result)
    observed = result["observed"]
    checks = observed["checks"]
    assert checks["baseline_commit_has_positive_drive"] is True
    assert checks["stress_commit_drops_to_zero_or_near_zero"] is True
    assert checks["stress_reread_or_replace_outruns_commit"] is True
    assert checks["stress_anchor_boosts_replay"] is True
    assert checks["stress_mismatch_boosts_replace"] is True
    assert checks["stress_commit_ready_lower_than_baseline"] is True
    assert observed["cases"][0]["goal_alignment"] == 0.0
    assert observed["cases"][1]["goal_alignment"] == 0.0
    stress = next(row for row in observed["cases"] if row["name"] == "stress")
    baseline = next(row for row in observed["cases"] if row["name"] == "baseline")
    assert stress["candidate_drives"]["action::text_commit"] <= 0.01
    assert max(
        stress["candidate_drives"]["action::text_reread"],
        stress["candidate_drives"]["action::text_replace"],
        stress["candidate_drives"]["action::replay_episode"],
    ) > stress["candidate_drives"]["action::text_commit"]
    assert baseline["candidate_drives"]["action::text_commit"] > 0.0


def test_double_energy_balance_pressure_sweep_shows_clean_and_stressed_shapes() -> None:
    result = run_double_energy_balance_pressure_sweep()
    _assert_pass(result)
    observed = result["observed"]
    assert observed["checks"]["clean_commit_nonincreasing"] is True
    assert observed["checks"]["clean_commit_remains_positive"] is True
    assert observed["checks"]["stress_commit_suppressed"] is True
    assert observed["checks"]["stress_replay_outruns_clean_peak_replay"] is True
    assert observed["checks"]["stress_peak_revision_outruns_clean_peak"] is True
