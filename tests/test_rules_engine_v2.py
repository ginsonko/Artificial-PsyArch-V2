# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from iesm.rules_engine_v2 import RulesEngineV2


class RulesEngineV2Tests(unittest.TestCase):
    def test_save_rules_returns_warnings_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            result = engine.save_rules(
                {
                    "schema_id": "wrong_schema",
                    "rules": [
                        {
                            "rule_id": "dup",
                            "display_name": "",
                            "conditions": [{"metric": "unknown.metric", "op": "???", "value": "x"}],
                            "effects": [{"type": "inject_sa", "sa_label": "", "formula": {"kind": "bad"}}],
                        },
                        {
                            "rule_id": "dup",
                            "conditions": [],
                            "effects": [],
                        },
                    ],
                }
            )
            payload = result["payload"]
            warnings = result["warnings"]
            stats = result["stats"]
            self.assertEqual(payload["schema_id"], "innate_rules_v2")
            self.assertEqual(len(payload["rules"]), 2)
            self.assertNotEqual(payload["rules"][0]["rule_id"], payload["rules"][1]["rule_id"])
            self.assertGreaterEqual(len(warnings), 5)
            self.assertEqual(stats["rule_count"], 2)
            self.assertGreaterEqual(stats["always_on_rule_count"], 1)
            self.assertGreaterEqual(stats["noop_rule_count"], 1)

    def test_save_tuner_returns_warnings_and_preserves_unknown_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            result = engine.save_tuner(
                {
                    "profiles": [
                        {
                            "profile_id": "dup",
                            "display_name": "",
                            "when": [{"metric": "", "op": "bad", "value": "oops"}],
                            "adjustments": [{"target": "custom.target", "value": "oops"}],
                        },
                        {
                            "profile_id": "dup",
                            "when": [],
                            "adjustments": [],
                        },
                    ]
                }
            )
            payload = result["payload"]
            warnings = result["warnings"]
            stats = result["stats"]
            self.assertEqual(payload["schema_id"], "auto_tuner_v2")
            self.assertEqual(len(payload["profiles"]), 2)
            self.assertNotEqual(payload["profiles"][0]["profile_id"], payload["profiles"][1]["profile_id"])
            self.assertEqual(payload["profiles"][0]["adjustments"][0]["target"], "custom.target")
            self.assertGreaterEqual(len(warnings), 5)
            self.assertEqual(stats["profile_count"], 2)
            self.assertGreaterEqual(stats["always_on_profile_count"], 1)
            self.assertGreaterEqual(stats["empty_adjustment_profile_count"], 1)

    def test_simulate_accepts_draft_rules_and_tuner_without_saving(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            context = {
                "tick_index": 3,
                "state_top": [{"sa_label": "text::今天", "energy": 1.5}],
                "state_pool_summary": {"state_pool_size": 2, "residual_summary": {"count": 0, "total_unresolved_mass": 0.0}},
                "bn_list": [{"memory_id": "mem_1", "score": 0.9}],
                "c_star": {"items": [{"sa_label": "text::冷", "energy": 0.7}]},
                "runtime_metrics": {"logic_ms": 42.0},
            }
            draft_rules = {
                "schema_id": "innate_rules_v2",
                "schema_version": "1.0",
                "rules": [
                    {
                        "rule_id": "rule::draft_only",
                        "enabled": True,
                        "priority": 999,
                        "display_name": "草稿专用规则",
                        "family": "draft_test",
                        "conditions": [{"metric": "bn.count", "op": ">", "value": 0}],
                        "effects": [
                            {"type": "append_rule_log", "message": "draft fired"},
                            {
                                "type": "add_action_drive",
                                "action_id": "action::draft_probe",
                                "reason": "draft",
                                "formula": {"kind": "constant", "value": 0.66},
                            },
                        ],
                    }
                ],
            }
            draft_tuner = {
                "schema_id": "auto_tuner_v2",
                "schema_version": "1.0",
                "enabled": True,
                "profiles": [
                    {
                        "profile_id": "profile::draft_only",
                        "enabled": True,
                        "display_name": "草稿调参档",
                        "when": [{"metric": "metrics.logic_ms", "op": ">", "value": 1.0}],
                        "adjustments": [{"target": "attention.focus_gain", "value": 1.23}],
                    }
                ],
            }
            simulated = engine.simulate(context, rules_payload=draft_rules, tuner_payload=draft_tuner)
            self.assertEqual([item["rule_id"] for item in simulated["rules_fired"]], ["rule::draft_only"])
            self.assertEqual(simulated["rule_logs"][0]["message"], "draft fired")
            self.assertEqual(simulated["action_drives"][0]["action_id"], "action::draft_probe")
            self.assertEqual(simulated["tuner_result"]["matched_profiles"][0]["profile_id"], "profile::draft_only")
            self.assertEqual(engine.export_rules()["rules"][0]["rule_id"], "rule::residual_dissonance")

    def test_metrics_and_emotion_channels_include_surprise(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            simulated = engine.simulate(
                {
                    "tick_index": 5,
                    "state_top": [],
                    "state_pool_summary": {
                        "state_pool_size": 2,
                        "residual_summary": {"count": 0, "total_unresolved_mass": 0.0},
                        "prediction_trace": {
                            "underprediction_mass": 0.4,
                            "overprediction_mass": 0.1,
                            "match_mass": 0.2,
                            "mismatch_mass": 0.5,
                        },
                    },
                    "bn_list": [],
                    "c_star": {"items": []},
                    "runtime_metrics": {},
                    "multimodal_summary": {"has_image": True},
                },
                rules_payload={
                    "schema_id": "innate_rules_v2",
                    "schema_version": "1.0",
                    "rules": [
                        {
                            "rule_id": "rule::surprise_probe",
                            "enabled": True,
                            "priority": 10,
                            "display_name": "surprise probe",
                            "family": "test",
                            "conditions": [{"metric": "state.prediction_underprediction_mass", "op": ">", "value": 0.0}],
                            "effects": [{"type": "set_emotion_floor", "channel": "surprise", "formula": {"kind": "metric", "metric": "state.prediction_underprediction_mass", "min": 0.0, "max": 1.0}}],
                        }
                    ],
                },
            )
            metrics = dict(simulated.get("metrics_snapshot", {}) or {})
            emotion = dict(simulated.get("emotion_channels", {}) or {})
            self.assertEqual(float(metrics.get("state.prediction_underprediction_mass", 0.0) or 0.0), 0.4)
            self.assertIn("emotion.surprise", metrics)
            self.assertGreater(float(emotion.get("surprise", 0.0) or 0.0), 0.0)

    def test_metrics_include_alignment_and_grasp_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            simulated = engine.simulate(
                {
                    "tick_index": 7,
                    "state_top": [],
                    "state_pool_summary": {
                        "state_pool_size": 3,
                        "residual_summary": {"count": 0, "total_unresolved_mass": 0.0},
                        "prediction_trace": {
                            "predicted_mass": 1.0,
                            "actual_mass": 1.0,
                            "match_mass": 0.8,
                            "overprediction_mass": 0.1,
                            "underprediction_mass": 0.2,
                            "mismatch_mass": 0.3,
                        },
                    },
                    "bn_list": [],
                    "c_star": {"items": []},
                    "runtime_metrics": {},
                }
            )
            metrics = dict(simulated.get("metrics_snapshot", {}) or {})
            self.assertGreater(float(metrics.get("state.prediction_alignment_score", 0.0) or 0.0), 0.0)
            self.assertGreater(float(metrics.get("state.prediction_grasp_score", 0.0) or 0.0), 0.0)
            self.assertGreater(float(metrics.get("state.prediction_underprediction_ratio", 0.0) or 0.0), 0.0)

    def test_temp_seed_uses_packaged_rule_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            expected = json.loads((Path(__file__).resolve().parents[1] / "config" / "innate_rules_v2.json").read_text(encoding="utf-8"))
            self.assertEqual(engine.export_rules(), expected)

    def test_action_drives_keep_distinct_parameterized_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            merged = engine._merge_action_drives(
                [
                    {"action_id": "action::move_gaze", "drive": 0.6, "reason": "left", "params": {"x": 0.1, "y": 0.5}},
                    {"action_id": "action::move_gaze", "drive": 0.7, "reason": "right", "params": {"x": 0.9, "y": 0.5}},
                ]
            )
            self.assertEqual(len(merged), 2)
            xs = sorted(float((item.get("params", {}) or {}).get("x", 0.0) or 0.0) for item in merged)
            self.assertEqual(xs, [0.1, 0.9])

    def test_packaged_rules_can_emit_auditory_focus_action_drives(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = RulesEngineV2(repo_root=Path(tmpdir))
            simulated = engine.simulate(
                {
                    "tick_index": 9,
                    "state_top": [],
                    "state_pool_summary": {
                        "state_pool_size": 4,
                        "residual_summary": {"count": 0, "total_unresolved_mass": 0.0},
                        "prediction_trace": {
                            "predicted_mass": 1.0,
                            "actual_mass": 1.0,
                            "match_mass": 0.7,
                            "overprediction_mass": 0.1,
                            "underprediction_mass": 0.1,
                            "mismatch_mass": 0.2,
                        },
                    },
                    "bn_list": [{"memory_id": "mem_audio", "score": 0.74}],
                    "c_star": {"items": [{"sa_label": "audio::mem_voice", "energy": 0.62}]},
                    "runtime_metrics": {
                        "audio_budget_used": 6.0,
                        "audio_memory_write_count": 2.0,
                        "audio_focus_priority_count": 2.0,
                        "audio_global_structure_count": 1.0,
                    },
                    "multimodal_summary": {
                        "has_audio": True,
                        "audio_window_count": 6,
                        "audio_memory_write_count": 2,
                        "audio_focus_priority_count": 2,
                        "audio_global_structure_count": 1,
                    },
                }
            )
            action_ids = [str(item.get("action_id", "") or "") for item in (simulated.get("action_drives", []) or [])]
            self.assertIn("action::continue_audio_focus", action_ids)


if __name__ == "__main__":
    unittest.main()
