# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.runtime_v2 import RuntimeV2
from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class TunerLearningV2Tests(unittest.TestCase):
    def test_runtime_tuner_learning_feedback_changes_next_tick_controls(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.set_last_logic_ms(40.0)
        runtime.process_text_tick(text="today weather nice", tick_index=0)
        feedback = runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.7}],
            emotion_channels={"expectation": 0.9, "pressure": 0.0, "correctness": 0.8, "dissonance": 0.0},
            runtime_action_effects={"moved": True},
            external_feedback={"reward": 0.4},
        )
        self.assertIn("tuner_learning_feedback", feedback)
        tick = runtime.process_text_tick(text="today weather a_bit cold", tick_index=1)
        tuner_learning_summary = tick.get("tuner_learning_summary", {})
        self.assertGreaterEqual(len(tuner_learning_summary.get("applied_offsets", [])), 1)
        self.assertGreaterEqual(len(tuner_learning_summary.get("target_bias_summary", [])), 1)
        self.assertGreater(tick["runtime_controls"]["sampling.increment_budget"], 16.0)
        self.assertGreater(tick["runtime_controls"]["attention.focus_gain"], 1.0)

    def test_runtime_export_import_preserves_tuner_learning(self) -> None:
        config = load_config(overrides={"memory_vector_backend": "numpy_flat"})
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app.start_multimodal_run(
                items=[
                    {"text": "today weather nice", "external_feedback": {"reward": 0.4}},
                    {"text": "go outside", "external_feedback": {"reward": 0.2}},
                ],
                label="tuner-learning-export",
                tick_interval_ms=0,
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            exported = app.export_runtime()
            self.assertIn("tuner_learning", exported["runtime"])
            self.assertIn("last_control_feedback_context", exported["runtime"])

            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            restored.import_runtime(exported)
            restored_export = restored.export_runtime()
            self.assertIn("tuner_learning", restored_export["runtime"])
            self.assertEqual(
                restored_export["runtime"]["memory_store"]["index_summary"]["vector"]["requested_backend"],
                "numpy_flat",
            )
            self.assertEqual(
                restored_export["runtime"]["last_control_feedback_context"].get("runtime_controls", {}),
                exported["runtime"]["last_control_feedback_context"].get("runtime_controls", {}),
            )


if __name__ == "__main__":
    unittest.main()
