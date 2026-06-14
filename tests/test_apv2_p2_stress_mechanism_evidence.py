from __future__ import annotations

import json
from pathlib import Path

from experiments.apv2_p2_stress_mechanism_evidence import (
    run_longrun_stability,
    run_p2_stress_suite,
    run_residual_depth_stress,
    run_short_term_slot_grid,
    write_outputs,
)


def test_residual_depth_stress_keeps_roundwise_absorption() -> None:
    result = run_residual_depth_stress()

    assert result["experiment"] == "ResidualDepth-Stress-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["row_count"] >= 7
    assert observed["trace_round_count"] >= 6
    assert len(observed["supported_winners"]) >= 5
    assert observed["distinct_winners"] is True
    assert observed["mass_declines_each_traced_round"] is True
    assert observed["drained_label_count"] >= 8
    assert all(winner.startswith("pair_") for winner in observed["winner_source_texts"][:5])
    assert result["boundary"]["one_b_winner_per_round"] is True
    assert result["boundary"]["student_side_llm"] is False


def test_longrun_stability_preserves_resumable_narrative_slot() -> None:
    result = run_longrun_stability()

    assert result["experiment"] == "LongRun-Stability-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["tick_count"] >= 12
    assert len(observed["recent_interruptions"]) >= 2
    assert len(observed["recent_resumptions"]) >= 2
    assert observed["final_slot_virtual_mass"] > 0.45
    assert len(observed["main_final_labels"]) >= 3
    assert observed["readback"]["available"] is True
    assert observed["replay_candidates"]
    assert "garden_main_memory" in observed["state_recall_winners"]
    assert any(str(winner).startswith("slot_") for winner in observed["slot_recall_winners"])
    assert result["boundary"]["open_dialogue_learning_claim"] is False


def test_short_term_slot_grid_is_bounded_and_monotonic() -> None:
    result = run_short_term_slot_grid()

    assert result["experiment"] == "ShortTermSlot-Grid-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["case_count"] == 108
    assert observed["pass_rate"] == 1.0
    assert observed["virtual_mass_min"] >= 0.05
    assert observed["virtual_mass_max"] <= 8.0
    assert observed["capacity_clipped_count"] > 0
    for case in observed["cases"]:
        assert case["passed"] is True, case
        assert case["observed"]["capacity_respected"] is True
        assert case["observed"]["order_coeff_monotonic"] is True
    assert result["boundary"]["parameter_optimality_claim"] is False


def test_p2_stress_suite_writes_artifacts_and_boundaries(tmp_path: Path) -> None:
    result = run_p2_stress_suite(output_dir=tmp_path / "p2")

    assert result["schema_id"] == "apv2_p2_stress_mechanism_evidence/v1"
    assert result["summary"]["all_passed"] is True, result
    assert result["summary"]["pass"] == 3
    assert result["route_split"]["this_suite"] == "AP-Core runtime mechanism stress evidence"
    for key in ("answer_table_lookup", "keyword_hard_gate", "regex_route", "student_side_llm", "hidden_solver", "full_sentence_macro"):
        assert result["boundary"][key] is False

    artifacts = write_outputs(result, output_dir=tmp_path / "p2")
    json_path = Path(artifacts["json_path"])
    md_path = Path(artifacts["markdown_path"])
    manifest_path = Path(artifacts["manifest_path"])
    assert json_path.exists()
    assert md_path.exists()
    assert manifest_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["all_passed"] is True
    report = md_path.read_text(encoding="utf-8")
    assert "ResidualDepth-Stress-1" in report
    assert "LongRun-Stability-1" in report
    assert "ShortTermSlot-Grid-1" in report
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["validation_passed"] is True

