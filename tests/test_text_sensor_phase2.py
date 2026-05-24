# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from sensors.text_sensor_v2 import TextSensorV2

REPO_ROOT = Path(__file__).resolve().parents[1]


class TextSensorPhase2Tests(unittest.TestCase):
    def test_budget_is_stable(self) -> None:
        sensor = TextSensorV2(budget_limit=4, fatigue_window=8, fatigue_threshold=2, max_suppression=0.75)
        packet = sensor.ingest("今天 天气 真 的 很 不 错 呢", tick_index=0)
        self.assertEqual(packet["budget_limit"], 4)
        self.assertEqual(packet["budget_used"], 4)
        self.assertGreater(packet["total_units"], packet["budget_used"])

    def test_repeated_input_triggers_fatigue(self) -> None:
        sensor = TextSensorV2(budget_limit=6, fatigue_window=8, fatigue_threshold=2, max_suppression=0.75)
        first = sensor.ingest("老虎 老虎 老虎", tick_index=0)
        sensor.ingest("老虎 老虎 老虎", tick_index=1)
        repeated = sensor.ingest("老虎 老虎 老虎", tick_index=2)
        self.assertGreater(sum(item["energy"] for item in first["sa_items"]), sum(item["energy"] for item in repeated["sa_items"]))
        self.assertGreater(repeated["fatigue_summary"]["suppressed_count"], 0)

    def test_different_inputs_produce_different_packets(self) -> None:
        sensor = TextSensorV2(budget_limit=6, fatigue_window=8, fatigue_threshold=2, max_suppression=0.75)
        packet_a = sensor.ingest("今天下雨", tick_index=0)
        packet_b = sensor.ingest("明天放晴", tick_index=1)
        self.assertNotEqual(packet_a["sa_flow"], packet_b["sa_flow"])

    def test_text_run_writes_sensor_and_sidecar_data(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_text_run(
                texts=["今天 天气 不错", "今天 天气 不错", "我想出门"],
                label="phase2 test",
                tick_interval_ms=5,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            tick = app.get_tick_summary(result["run_id"], 2)
            self.assertIn("sensor_summary", tick)
            self.assertIn("competition_summary", tick)
            self.assertIn("bn_preview", tick)
            self.assertIn("c_star_preview", tick)
            self.assertIn("rules_preview", tick)
            self.assertIn("short_term_preview", tick)
            self.assertIn("state_pool_summary", tick)
            self.assertIn("state_pool_sidecar_summary", tick)
            self.assertIn("r_state_heads", tick)
            self.assertEqual(tick["a_focus_preview"][:4], ["我", "想", "出", "门"])
            sidecar = app.get_tick_sidecar(result["run_id"], 2)
            self.assertIn("state_pool_sidecar", sidecar)
            self.assertIn("bn_list", sidecar)
            self.assertIn("c_star", sidecar)
            self.assertIn("rules_result", sidecar)
            self.assertIn("short_term_snapshot", sidecar)
            self.assertIn("hot_anchor_cache", sidecar["state_pool_sidecar"])
            self.assertIn("residual_bucket", sidecar["state_pool_sidecar"])


if __name__ == "__main__":
    unittest.main()
