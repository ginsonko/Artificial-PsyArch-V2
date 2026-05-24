# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeContinuityPhase16Tests(unittest.TestCase):
    def test_runtime_tick_index_continues_across_runs_while_run_tick_resets(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            first = app.start_text_run(texts=["今天 天气 不错", "我 想 出门"], label="first run", tick_interval_ms=0, reset_runtime=True)
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            first_sidecar = app.get_tick_sidecar(first["run_id"], 1)
            self.assertEqual(first_sidecar["tick_index"], 1)
            self.assertEqual(first_sidecar["runtime_tick_index"], 1)

            second = app.start_text_run(texts=["有点 冷", "算了 不说了"], label="second run", tick_interval_ms=0, reset_runtime=False)
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            second_tick0 = app.get_tick_sidecar(second["run_id"], 0)
            second_tick1 = app.get_tick_sidecar(second["run_id"], 1)
            self.assertEqual(second_tick0["tick_index"], 0)
            self.assertEqual(second_tick1["tick_index"], 1)
            self.assertGreaterEqual(second_tick0["runtime_tick_index"], 2)
            self.assertGreater(second_tick1["runtime_tick_index"], second_tick0["runtime_tick_index"])


if __name__ == "__main__":
    unittest.main()
