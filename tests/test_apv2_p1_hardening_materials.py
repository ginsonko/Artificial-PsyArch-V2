from __future__ import annotations

import json
from pathlib import Path

from experiments.apv2_p1_hardening_materials import (
    run_longrun_interruption_recovery,
    run_p1_hardening_suite,
    run_persistence_backend_reload,
    run_rhythm_successor_replay,
    write_outputs,
)


def test_longrun_interruption_recovery_restores_narrative_slot_and_recall() -> None:
    result = run_longrun_interruption_recovery()

    assert result["experiment"] == "LongRun-InterruptionRecovery-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["recent_interruptions"]
    assert observed["recent_resumptions"]
    assert observed["final_slot_virtual_mass"] > 0.0
    assert any("text::river" in label for label in observed["final_slot_labels"])
    assert "river_story" in observed["state_recall_winners"]
    assert observed["slot_recall_winners"]
    assert result["boundary"]["open_dialogue_learning_claim"] is False


def test_rhythm_successor_replay_emits_phase_slot_and_lag_peak() -> None:
    result = run_rhythm_successor_replay()

    assert result["experiment"] == "RhythmSuccessor-Replay-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert any("rhythmfelt::phase_expectation" in row["items"] for row in observed["phase_traces"])
    assert "short_term_slot::rhythm" in observed["slot_labels"]
    kernels = [row["kernel"] for row in observed["successor_shaped"]]
    assert kernels[0] == 1.0
    assert kernels[0] > kernels[1] > kernels[2]
    assert observed["flat_kernels"] == [1.0, 1.0, 1.0]
    assert result["boundary"]["music_performance_claim"] is False


def test_persistence_backend_reload_uses_real_jsonl_boundary(tmp_path: Path) -> None:
    result = run_persistence_backend_reload(tmp_path)

    assert result["experiment"] == "PersistenceBackend-Reload-1"
    assert result["verdict"] == "pass", result
    observed = result["observed"]
    assert observed["exists"] is True
    assert observed["bytes"] > 0
    assert len(observed["sha256"]) == 64
    assert observed["warm_load"]["loaded"] >= 3
    assert "cue context" in observed["bn_source_texts"]
    assert "text::outcome" in observed["predicted_labels"]
    assert observed["slot_bn_source_texts"] == ["slot cue context"]
    assert result["boundary"]["postgres_production_claim"] is False


def test_p1_hardening_suite_writes_report_manifest_and_figures(tmp_path: Path) -> None:
    result = run_p1_hardening_suite(output_dir=tmp_path / "p1")

    assert result["schema_id"] == "apv2_p1_hardening_materials/v1"
    assert result["summary"]["all_passed"] is True, result
    assert result["summary"]["pass"] == 4
    assert result["figures"]["figure_count"] >= 5
    for key in ("answer_table_lookup", "regex_route", "student_side_llm", "hidden_solver", "full_sentence_macro"):
        assert result["boundary"][key] is False

    artifacts = write_outputs(result, output_dir=tmp_path / "p1")
    json_path = Path(artifacts["json_path"])
    md_path = Path(artifacts["markdown_path"])
    manifest_path = tmp_path / "p1" / "artifact_freeze_manifest.json"
    assert json_path.exists()
    assert md_path.exists()
    assert manifest_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["summary"]["all_passed"] is True
    report = md_path.read_text(encoding="utf-8")
    assert "LongRun-InterruptionRecovery-1" in report
    assert "RhythmSuccessor-Replay-1" in report
    assert "PersistenceBackend-Reload-1" in report
    assert "ArtifactFreeze-1" in report

