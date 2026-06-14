from __future__ import annotations

import json
from pathlib import Path

from experiments.apv2_bottom_loop_p0_materials import (
    collect_default_parameter_table,
    run_bottom_loop_param_sensitivity,
    run_p0_materials_suite,
    run_short_term_slot_order_ablation,
    write_outputs,
)


def test_apv2_bottom_loop_param_sensitivity_preserves_core_traces() -> None:
    result = run_bottom_loop_param_sensitivity()

    assert result["experiment"] == "APV2-BottomLoop-ParamSensitivity-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["pass_rate"] == 1.0
    assert observed["case_count"] >= 12

    cases = {case["case"]: case for case in observed["cases"]}
    assert cases["slot_default"]["observed"]["slot_virtual_mass"] > 0.0
    assert cases["residual_default_scale"]["observed"]["mass_declines"] is True
    assert cases["residual_default_scale"]["observed"]["distinct_winners"] is True
    assert cases["successor_default_scale"]["observed"]["kernels"][0] == 1.0
    assert cases["successor_default_scale"]["observed"]["kernels"][0] > cases["successor_default_scale"]["observed"]["kernels"][1]
    assert cases["below_threshold"]["observed"]["repair_present"] is False
    assert cases["at_threshold"]["observed"]["repair_present"] is True
    assert cases["above_threshold"]["observed"]["repair_present"] is True


def test_apv2_short_term_slot_order_ablation_is_soft_bias_not_hard_gate() -> None:
    result = run_short_term_slot_order_ablation()

    assert result["experiment"] == "ShortTermSlot-OrderAblation-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["full_order_scores"]["slot_ABC"] > observed["full_order_scores"]["slot_CBA"]
    assert observed["without_order_rows_scores"]["slot_ABC"] > observed["without_order_rows_scores"]["slot_CBA"]
    assert observed["margin_drop_without_order_rows"] > 0.0
    assert observed["reversed_query_scores"]["slot_CBA"] > observed["reversed_query_scores"]["slot_ABC"]
    assert observed["wrong_order_still_recalled"] is True
    assert result["boundary"]["order_is_soft_bias_not_hard_gate"] is True


def test_apv2_bottom_loop_default_parameter_table_contains_publication_values() -> None:
    params = collect_default_parameter_table()

    assert params["schema_id"] == "apv2_bottom_loop_default_parameters/v1"
    assert params["short_term_slot"]["capacity"] == 32
    assert params["short_term_slot"]["base_virtual_budget"] == 0.72
    assert params["short_term_slot"]["item_order_decay"] == 0.92
    assert params["memory"]["recall_top_k"] == 5
    assert params["memory"]["prediction_energy_scale"] == 0.55
    assert params["successor_lag_kernel"]["lag_1"] == 1.0
    assert params["successor_lag_kernel"]["lag_2"] == 0.42
    assert params["residual_b_recall"]["round_policy"] == "one_b_winner_per_round"
    assert params["negative_text_feedback"]["positive_text_prediction_allowed"] is False


def test_apv2_bottom_loop_p0_materials_suite_outputs_trace_and_boundaries(tmp_path: Path) -> None:
    result = run_p0_materials_suite()

    assert result["schema_id"] == "apv2_bottom_loop_p0_publication_materials/v1"
    assert result["summary"]["all_passed"] is True
    assert result["summary"]["pass"] == 2
    assert result["route_split"]["this_suite"] == "AP-Core bottom-loop publication materials"
    for key in ("answer_table_lookup", "regex_route", "student_side_llm", "hidden_solver", "full_sentence_macro"):
        assert result["boundary"][key] is False

    trace = result["representative_tick_trace"]
    assert trace["short_term_slot"]["virtual_mass"] > 0.0
    assert trace["fast_system"]["successor_lag_predictions"]
    probe = trace["fast_system"]["residual_b_recall_probe"]
    assert probe["mass_declines"] is True
    assert probe["distinct_winners"] is True

    artifacts = write_outputs(result, output_root=tmp_path / "apv2_p0")
    json_path = Path(artifacts["json_path"])
    md_path = Path(artifacts["markdown_path"])
    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["all_passed"] is True
    report = md_path.read_text(encoding="utf-8")
    assert "APV2-BottomLoop-ParamSensitivity-1" in report
    assert "ShortTermSlot-OrderAblation-1" in report
    assert "fast_residual_b_recall_probe" in report

