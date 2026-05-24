# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class ApAgentCouplingPhase22Tests(unittest.TestCase):
    def test_ap_remains_primary_and_teacher_executor_only_gate_and_feedback(self) -> None:
        config = load_config(
            overrides={
                "executor_enabled": True,
                "executor_dry_run": True,
                "executor_screenshot_enabled": False,
                "autonomous_external_teacher_enabled": False,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_text_run(
                texts=["今天 天气 不错", "今天 天气 不错", "我 想 出门"],
                tick_interval_ms=0,
                reset_runtime=True,
                label="phase22 ap-agent coupling smoke",
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

            sidecar0 = app.get_tick_sidecar(result["run_id"], 0)
            sidecar1 = app.get_tick_sidecar(result["run_id"], 1)
            sidecar2 = app.get_tick_sidecar(result["run_id"], 2)

            self.assertTrue(sidecar0)
            self.assertTrue(sidecar1)
            self.assertTrue(sidecar2)

            rules_result_1 = dict(sidecar1.get("rules_result", {}) or {})
            teacher_review_1 = dict(sidecar1.get("teacher_review", {}) or {})
            sandbox_result_1 = dict(sidecar1.get("sandbox_result", {}) or {})
            teacher_feedback_1 = dict(sidecar1.get("teacher_feedback", {}) or {})
            planner_selected_1 = list(rules_result_1.get("planned_selected_actions_preview", []) or [])

            rule_action_names = [
                str(item.get("action_id", "") or "")
                for item in (rules_result_1.get("action_drives", []) or [])
                if str(item.get("action_id", "") or "")
            ]
            reviewed_action_names = [
                str(item.get("action_id", "") or "")
                for item in (teacher_review_1.get("scored_action_drives", []) or [])
                if str(item.get("action_id", "") or "")
            ]
            executed_action_names = [
                str(item.get("action_id", "") or "")
                for item in (sandbox_result_1.get("selected_actions", []) or [])
                if str(item.get("action_id", "") or "")
            ]

            self.assertIn("action::continue_focus", rule_action_names)
            self.assertIn("action::continue_focus", [str(item.get("action_id", "") or "") for item in planner_selected_1])
            self.assertIn("action::continue_focus", reviewed_action_names)
            self.assertIn("action::continue_focus", executed_action_names)
            self.assertTrue(bool(teacher_review_1.get("applied")))
            self.assertEqual(str(teacher_review_1.get("mode", "")), "heuristic")
            self.assertGreater(float(teacher_feedback_1.get("reward", 0.0) or 0.0), 0.0)
            self.assertEqual(len(teacher_review_1.get("planner_selected_action_drives", []) or []), len(planner_selected_1))

            self.assertFalse(bool((teacher_review_1.get("external_teacher_review", {}) or {}).get("applied", False)))

            rules_result_2 = dict(sidecar2.get("rules_result", {}) or {})
            action_drives_2 = list(rules_result_2.get("action_drives", []) or [])
            self.assertTrue(action_drives_2)
            continue_focus_row = next(
                (item for item in action_drives_2 if str(item.get("action_id", "") or "") == "action::continue_focus"),
                {},
            )
            self.assertGreater(float(continue_focus_row.get("learned_bias", 0.0) or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
