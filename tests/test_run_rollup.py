# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from observatory_v2.run_rollup import empty_rollup, update_rollup


class RunRollupTests(unittest.TestCase):
    def test_first_tick_zero_is_preserved_in_series_tail(self) -> None:
        rollup = empty_rollup(run_id="demo")
        next_rollup = update_rollup(
            rollup,
            summary={
                "run_id": "demo",
                "tick_index": 0,
                "state_top": [{"sa_label": "text::今", "energy": 1.2}],
                "state_pool_summary": {"residual_summary": {"count": 0}},
                "memory_index_summary": {"vector": {"vector_count": 3}},
                "bn_preview": [],
                "rules_preview": {"rules_fired": [], "emotion_channels": {}},
                "input_preview": "今天",
                "a_focus_preview": ["今天"],
            },
            metrics={
                "logic_ms": 12.3,
                "state_pool_size": 4,
                "bn_count": 2,
                "c_star_count": 1,
            },
        )
        self.assertEqual(next_rollup["tick_count"], 1)
        self.assertEqual(next_rollup["last_tick_index"], 0)
        self.assertEqual(next_rollup["series_tail"]["tick_index"], [0])

    def test_emotion_and_rule_series_are_recorded(self) -> None:
        rollup = empty_rollup(run_id="demo")
        next_rollup = update_rollup(
            rollup,
            summary={
                "run_id": "demo",
                "tick_index": 3,
                "state_top": [{"sa_label": "text::冷", "energy": 1.4}],
                "state_pool_summary": {"residual_summary": {"count": 2}},
                "memory_index_summary": {"vector": {"vector_count": 9}},
                "bn_preview": [],
                "rules_preview": {
                    "rules_fired": ["r1", "r2"],
                    "emotion_channels": {
                        "dissonance": 0.7,
                        "correctness": 0.2,
                        "expectation": 0.6,
                        "pressure": 0.1,
                    },
                    "rule_fired_count": 2,
                    "action_drive_count": 3,
                    "sandbox_action_count": 1,
                    "tuner_matched_count": 4,
                },
                "tuner_preview": {"matched_profiles": [{"profile_id": "p1"}]},
                "input_preview": "今天 天气",
                "a_focus_preview": ["今天", "天气"],
            },
            metrics={
                "logic_ms": 8.1,
                "state_pool_size": 6,
                "bn_count": 4,
                "c_star_count": 2,
            },
        )
        series = next_rollup["series_tail"]
        self.assertEqual(series["emotion_dissonance"], [0.7])
        self.assertEqual(series["emotion_correctness"], [0.2])
        self.assertEqual(series["emotion_expectation"], [0.6])
        self.assertEqual(series["emotion_pressure"], [0.1])
        self.assertEqual(series["rules_fired_count"], [2])
        self.assertEqual(series["action_drive_count"], [3])
        self.assertEqual(series["sandbox_action_count"], [1])
        self.assertEqual(series["tuner_matched_count"], [4])

    def test_runtime_stage_timing_last_is_preserved(self) -> None:
        rollup = empty_rollup(run_id="demo")
        next_rollup = update_rollup(
            rollup,
            summary={
                "run_id": "demo",
                "tick_index": 1,
                "state_top": [{"sa_label": "text::apple", "energy": 1.0}],
                "state_pool_summary": {"residual_summary": {"count": 1}},
                "memory_index_summary": {"vector": {"vector_count": 2}},
                "bn_preview": [],
                "rules_preview": {"rules_fired": [], "emotion_channels": {}},
                "input_preview": "apple",
                "a_focus_preview": ["apple"],
            },
            metrics={
                "logic_ms": 9.8,
                "state_pool_size": 3,
                "bn_count": 1,
                "c_star_count": 1,
                "runtime_stage_timing_ms": {
                    "01_text_competition_ms": 0.4,
                    "05_main_recall_prediction_ms": 1.8,
                    "09_total_runtime_ms": 4.2,
                },
            },
        )
        timing = dict(next_rollup.get("runtime_stage_timing_last", {}) or {})
        self.assertEqual(timing["01_text_competition_ms"], 0.4)
        self.assertEqual(timing["09_total_runtime_ms"], 4.2)


if __name__ == "__main__":
    unittest.main()
