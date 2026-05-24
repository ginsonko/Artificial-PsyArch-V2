# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any
import shutil

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.runtime_v2 import RuntimeV2
from observatory_v2.agent_sandbox import AgentSandboxV1
from observatory_v2.config import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "conditioned_learning_probe"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _action_rules_payload(*, mode: str) -> dict[str, Any]:
    if mode == "left_bias":
        left_drive = 0.58
        right_drive = 0.46
    elif mode == "right_bias":
        left_drive = 0.46
        right_drive = 0.58
    elif mode == "probe_bias":
        left_drive = 0.56
        right_drive = 0.46
    else:
        raise ValueError(f"unsupported rules mode: {mode}")

    return {
        "schema_id": "innate_rules_v2",
        "schema_version": "1.0",
        "rules": [
            {
                "rule_id": "rule::cmd_alpha_press_left",
                "enabled": True,
                "priority": 180,
                "display_name": "口令甲提供按左键候选",
                "family": "action_drive",
                "conditions": [{"metric": "text.ngram::口令甲", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "add_action_drive",
                        "action_id": "action::press_key",
                        "reason": "条件训练_甲_left",
                        "params": {"key": "left"},
                        "formula": {"kind": "constant", "value": left_drive},
                    }
                ],
            },
            {
                "rule_id": "rule::cmd_alpha_press_right",
                "enabled": True,
                "priority": 178,
                "display_name": "口令甲提供按右键候选",
                "family": "action_drive",
                "conditions": [{"metric": "text.ngram::口令甲", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "add_action_drive",
                        "action_id": "action::press_key",
                        "reason": "条件训练_甲_right",
                        "params": {"key": "right"},
                        "formula": {"kind": "constant", "value": right_drive},
                    }
                ],
            },
            {
                "rule_id": "rule::cmd_beta_press_left",
                "enabled": True,
                "priority": 176,
                "display_name": "口令乙提供按左键候选",
                "family": "action_drive",
                "conditions": [{"metric": "text.ngram::口令乙", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "add_action_drive",
                        "action_id": "action::press_key",
                        "reason": "条件训练_乙_left",
                        "params": {"key": "left"},
                        "formula": {"kind": "constant", "value": left_drive},
                    }
                ],
            },
            {
                "rule_id": "rule::cmd_beta_press_right",
                "enabled": True,
                "priority": 174,
                "display_name": "口令乙提供按右键候选",
                "family": "action_drive",
                "conditions": [{"metric": "text.ngram::口令乙", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "add_action_drive",
                        "action_id": "action::press_key",
                        "reason": "条件训练_乙_right",
                        "params": {"key": "right"},
                        "formula": {"kind": "constant", "value": right_drive},
                    }
                ],
            },
            {
                "rule_id": "rule::prediction_mismatch_dissonance",
                "enabled": True,
                "priority": 150,
                "display_name": "预测落空提升违和感",
                "family": "cognitive_feeling",
                "conditions": [{"metric": "state.prediction_mismatch_mass", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "dissonance",
                        "formula": {"kind": "mul", "metric": "state.prediction_mismatch_mass", "factor": 0.6, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "预测与实际不一致，违和感上升"},
                ],
            },
            {
                "rule_id": "rule::prediction_match_correctness",
                "enabled": True,
                "priority": 148,
                "display_name": "预测命中提升正确感",
                "family": "cognitive_feeling",
                "conditions": [{"metric": "state.prediction_match_mass", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "correctness",
                        "formula": {"kind": "mul", "metric": "state.prediction_match_mass", "factor": 0.35, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "预测命中，正确感上升"},
                ],
            },
            {
                "rule_id": "rule::feedback_reward_expectation",
                "enabled": True,
                "priority": 146,
                "display_name": "奖励信号提升期待",
                "family": "emotion",
                "conditions": [{"metric": "feedback.reward", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "expectation",
                        "formula": {"kind": "mul", "metric": "feedback.reward", "factor": 1.0, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "奖励信号抬升期待"},
                ],
            },
            {
                "rule_id": "rule::feedback_punishment_pressure",
                "enabled": True,
                "priority": 144,
                "display_name": "惩罚信号提升压力",
                "family": "emotion",
                "conditions": [{"metric": "feedback.punishment", "op": ">", "value": 0.0}],
                "effects": [
                    {
                        "type": "set_emotion_floor",
                        "channel": "pressure",
                        "formula": {"kind": "mul", "metric": "feedback.punishment", "factor": 1.0, "min": 0.0, "max": 1.0},
                    },
                    {"type": "append_rule_log", "message": "惩罚信号抬升压力"},
                ],
            },
        ],
    }


def _tuner_payload() -> dict[str, Any]:
    return {
        "schema_id": "auto_tuner_v2",
        "schema_version": "1.0",
        "enabled": True,
        "profiles": [
            {
                "profile_id": "conditioned_learning_probe",
                "enabled": True,
                "display_name": "条件化奖惩学习实验基线",
                "when": [{"metric": "metrics.logic_ms", "op": "<", "value": 500.0}],
                "adjustments": [
                    {"target": "attention.focus_gain", "value": 1.05},
                    {"target": "sampling.increment_budget", "value": 18.0},
                    {"target": "prediction.successor_bias_gain", "value": 1.08},
                ],
            }
        ],
    }


def _choose_action(action_drives: list[dict[str, Any]], *, tick_index: int) -> dict[str, Any]:
    sandbox = AgentSandboxV1(enabled=False, dry_run=True, screenshot_enabled=False, max_actions_per_tick=1)
    sandbox_result = sandbox.evaluate_action_drives(tick_index=tick_index, action_drives=action_drives)
    selected = list(sandbox_result.get("selected_actions", []) or [])
    return selected[0] if selected else {}


def _build_probe_steps(*, repeats: int, include_feedback: bool) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for _ in range(repeats):
        steps.append(
            {
                "command": "口令 甲",
                "expected_action": "press_key",
                "expected_signature": "press_key:left",
                "mode": "left_bias" if include_feedback else "probe_bias",
                "label": "口令甲",
            }
        )
        steps.append(
            {
                "command": "口令 甲",
                "expected_action": "press_key",
                "expected_signature": "press_key:left",
                "mode": "right_bias" if include_feedback else "probe_bias",
                "label": "口令甲",
            }
        )
        steps.append(
            {
                "command": "口令 乙",
                "expected_action": "press_key",
                "expected_signature": "press_key:right",
                "mode": "left_bias" if include_feedback else "probe_bias",
                "label": "口令乙",
            }
        )
        steps.append(
            {
                "command": "口令 乙",
                "expected_action": "press_key",
                "expected_signature": "press_key:right",
                "mode": "right_bias" if include_feedback else "probe_bias",
                "label": "口令乙",
            }
        )
    return steps


def _run_sequence(
    *,
    runtime: RuntimeV2,
    steps: list[dict[str, Any]],
    start_tick_index: int,
    apply_feedback: bool,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    tick_index = int(start_tick_index)
    prev_feedback = {"reward": 0.0, "punishment": 0.0}
    original_rules = copy.deepcopy(runtime.rules_engine.export_rules())
    try:
        for step in steps:
            runtime.rules_engine.save_rules(_action_rules_payload(mode=str(step["mode"])))
            tick = runtime.process_text_tick(text=str(step["command"]), tick_index=tick_index)
            planner_selected = list(((tick.get("rules_result", {}) or {}).get("planned_selected_actions_preview", []) or []))
            selected = _choose_action(
                planner_selected,
                tick_index=tick_index,
            )
            selected_action = str(selected.get("action_name", "") or "")
            selected_signature = f"{selected_action}:{str(((selected.get('params', {}) or {}).get('key', '')) or ((selected.get('params', {}) or {}).get('text', '')))}".rstrip(":")
            expected_action = str(step["expected_action"])
            expected_signature = str(step.get("expected_signature", expected_action) or expected_action)
            reward = 0.0
            punishment = 0.0
            if apply_feedback:
                if selected_signature == expected_signature:
                    reward = 1.0
                else:
                    punishment = 1.0
                runtime_action_effects = runtime.apply_selected_actions([selected], runtime_tick=tick)
                runtime.apply_action_feedback(
                    tick_index=tick_index,
                    selected_actions=[selected] if selected_action else [],
                    emotion_channels=dict((tick.get("rules_result", {}) or {}).get("emotion_channels", {}) or {}),
                    runtime_action_effects=runtime_action_effects,
                    external_feedback={"reward": reward, "punishment": punishment},
                )
                runtime.inject_feedback_signals(
                    tick_index=tick_index,
                    feedback={"reward": reward, "punishment": punishment},
                    provenance={
                        "selected_action_ids": [str(selected.get("action_id", "") or "")] if selected_action else [],
                        "selected_action_names": [selected_action] if selected_action else [],
                        "focus_memory_id": str((tick.get("focus_memory", {}) or {}).get("memory_id", "") or ""),
                        "exact_memory_id": str((tick.get("exact_memory", {}) or {}).get("memory_id", "") or ""),
                    },
                    source_type="conditioned_experiment_feedback",
                    channel="conditioned_experiment_feedback",
                    meta_extra={"expected_action": expected_action},
                )

            emotion = dict((tick.get("rules_result", {}) or {}).get("emotion_channels", {}) or {})
            context_bias = list(tick.get("action_learning_context_bias_summary", []) or [])
            rows.append(
                {
                    "tick_index": tick_index,
                    "command": str(step["command"]),
                    "label": str(step["label"]),
                    "rules_mode": str(step["mode"]),
                    "expected_action": expected_action,
                    "expected_signature": expected_signature,
                    "selected_action": selected_action,
                    "selected_signature": selected_signature,
                    "selected_drive": _round4(float(selected.get("drive", 0.0) or 0.0)),
                    "selected_firmness": _round4(float(selected.get("firmness", 0.0) or 0.0)),
                    "selected_actuator_id": str(selected.get("actuator_id", "") or ""),
                    "reward": _round4(reward),
                    "punishment": _round4(punishment),
                    "prev_feedback_reward": _round4(float(prev_feedback.get("reward", 0.0) or 0.0)),
                    "prev_feedback_punishment": _round4(float(prev_feedback.get("punishment", 0.0) or 0.0)),
                    "emotion_expectation": _round4(float(emotion.get("expectation", 0.0) or 0.0)),
                    "emotion_pressure": _round4(float(emotion.get("pressure", 0.0) or 0.0)),
                    "emotion_correctness": _round4(float(emotion.get("correctness", 0.0) or 0.0)),
                    "emotion_dissonance": _round4(float(emotion.get("dissonance", 0.0) or 0.0)),
                    "bias_summary": list(tick.get("action_learning_bias_summary", []) or []),
                    "context_bias_summary": context_bias[:8],
                    "planner_top": list(((tick.get("rules_result", {}) or {}).get("planned_action_drives", []) or [])[:4]),
                    "planner_selected_preview": list(((tick.get("rules_result", {}) or {}).get("planned_selected_actions_preview", []) or [])[:2]),
                    "planner_actuator_reports": list(((tick.get("rules_result", {}) or {}).get("action_actuator_reports", []) or [])[:4]),
                    "rule_logs": [str(item.get("message", "") or "") for item in ((tick.get("rules_result", {}) or {}).get("rule_logs", []) or [])[:8]],
                }
            )
            prev_feedback = {"reward": reward, "punishment": punishment}
            tick_index += 1
    finally:
        runtime.rules_engine.save_rules(original_rules)
    return rows, tick_index


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return _round4(sum(float(row.get(field, 0.0) or 0.0) for row in rows) / max(1, len(rows)))


def _action_accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    hit = sum(1 for row in rows if str(row.get("selected_signature", "")) == str(row.get("expected_signature", "")))
    return _round4(hit / max(1, len(rows)))


def _per_label_accuracy(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = sorted({str(row.get("label", "")) for row in rows if str(row.get("label", ""))})
    result: dict[str, float] = {}
    for label in labels:
        bucket = [row for row in rows if str(row.get("label", "")) == label]
        result[label] = _action_accuracy(bucket)
    return result


def _context_bias_tail(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    tail = rows[-1]
    return list(tail.get("context_bias_summary", []) or [])


def _build_summary(*, baseline_probe_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], trained_probe_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reward_follow_rows = [row for row in training_rows if float(row.get("prev_feedback_reward", 0.0) or 0.0) > 0.0]
    punishment_follow_rows = [row for row in training_rows if float(row.get("prev_feedback_punishment", 0.0) or 0.0) > 0.0]
    baseline_probe_accuracy = _action_accuracy(baseline_probe_rows)
    trained_probe_accuracy = _action_accuracy(trained_probe_rows)
    alpha_probe_rows = [row for row in trained_probe_rows if str(row.get("label", "")) == "口令甲"]
    beta_probe_rows = [row for row in trained_probe_rows if str(row.get("label", "")) == "口令乙"]
    alpha_left_hits = sum(1 for row in alpha_probe_rows if str(row.get("selected_signature", "")) == "press_key:left")
    beta_right_hits = sum(1 for row in beta_probe_rows if str(row.get("selected_signature", "")) == "press_key:right")
    return {
        "baseline_probe_accuracy": baseline_probe_accuracy,
        "baseline_probe_accuracy_by_label": _per_label_accuracy(baseline_probe_rows),
        "trained_probe_accuracy": trained_probe_accuracy,
        "trained_probe_accuracy_by_label": _per_label_accuracy(trained_probe_rows),
        "training_reward_total": _round4(sum(float(row.get("reward", 0.0) or 0.0) for row in training_rows)),
        "training_punishment_total": _round4(sum(float(row.get("punishment", 0.0) or 0.0) for row in training_rows)),
        "reward_follow_expectation_mean": _mean(reward_follow_rows, "emotion_expectation"),
        "reward_follow_pressure_mean": _mean(reward_follow_rows, "emotion_pressure"),
        "punishment_follow_expectation_mean": _mean(punishment_follow_rows, "emotion_expectation"),
        "punishment_follow_pressure_mean": _mean(punishment_follow_rows, "emotion_pressure"),
        "punishment_follow_dissonance_mean": _mean(punishment_follow_rows, "emotion_dissonance"),
        "reward_follow_correctness_mean": _mean(reward_follow_rows, "emotion_correctness"),
        "alpha_probe_left_hits": alpha_left_hits,
        "alpha_probe_count": len(alpha_probe_rows),
        "beta_probe_right_hits": beta_right_hits,
        "beta_probe_count": len(beta_probe_rows),
        "trained_context_bias_tail": _context_bias_tail(training_rows),
        "passed_conditioned_action_learning": trained_probe_accuracy >= 0.9 and baseline_probe_accuracy <= 0.6,
        "passed_punishment_pressure_signal": _mean(punishment_follow_rows, "emotion_pressure") > _mean(punishment_follow_rows, "emotion_expectation"),
        "passed_reward_expectation_signal": _mean(reward_follow_rows, "emotion_expectation") >= _mean(reward_follow_rows, "emotion_pressure"),
    }


def run_probe(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        isolated_root = Path(tmpdir) / "isolated_repo"
        shutil.copytree(REPO_ROOT / "config", isolated_root / "config")
        config = load_config(
            overrides={
                "executor_enabled": False,
                "executor_dry_run": True,
                "executor_screenshot_enabled": False,
                "text_sensor_budget": 18,
                "short_term_successor_tail_limit": 10,
                "state_pool_residual_unit_limit": 24,
            }
        )

        baseline_runtime = RuntimeV2(config=config, repo_root=isolated_root)
        trained_runtime = RuntimeV2(config=config, repo_root=isolated_root)
        original_baseline_rules = copy.deepcopy(baseline_runtime.rules_engine.export_rules())
        original_trained_rules = copy.deepcopy(trained_runtime.rules_engine.export_rules())
        original_baseline_tuner = copy.deepcopy(baseline_runtime.rules_engine.export_tuner())
        original_trained_tuner = copy.deepcopy(trained_runtime.rules_engine.export_tuner())
        try:
            baseline_runtime.rules_engine.save_tuner(_tuner_payload())
            trained_runtime.rules_engine.save_tuner(_tuner_payload())

            baseline_probe_rows, _ = _run_sequence(
                runtime=baseline_runtime,
                steps=_build_probe_steps(repeats=6, include_feedback=False),
                start_tick_index=0,
                apply_feedback=False,
            )
            training_rows, next_tick_index = _run_sequence(
                runtime=trained_runtime,
                steps=_build_probe_steps(repeats=8, include_feedback=True),
                start_tick_index=0,
                apply_feedback=True,
            )
            trained_probe_rows, _ = _run_sequence(
                runtime=trained_runtime,
                steps=_build_probe_steps(repeats=6, include_feedback=False),
                start_tick_index=next_tick_index,
                apply_feedback=False,
            )
        finally:
            baseline_runtime.rules_engine.save_rules(original_baseline_rules)
            trained_runtime.rules_engine.save_rules(original_trained_rules)
            baseline_runtime.rules_engine.save_tuner(original_baseline_tuner)
            trained_runtime.rules_engine.save_tuner(original_trained_tuner)

    summary = _build_summary(
        baseline_probe_rows=baseline_probe_rows,
        training_rows=training_rows,
        trained_probe_rows=trained_probe_rows,
    )
    payload = {
        "schema_id": "conditioned_learning_probe/v1",
        "schema_version": "1.0",
        "summary": summary,
        "baseline_probe_rows": baseline_probe_rows,
        "training_rows": training_rows,
        "trained_probe_rows": trained_probe_rows,
    }
    _write_json(output_root / "conditioned_learning_probe_report.json", payload)
    return payload


def main() -> None:
    output_root = DEFAULT_OUTPUT_ROOT
    if len(sys.argv) >= 2:
        output_root = Path(sys.argv[1]).resolve()
    report = run_probe(output_root)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
