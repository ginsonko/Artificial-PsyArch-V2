# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_long_multimodal_dataset import build_dataset


class LongMultimodalDatasetGeneratorTests(unittest.TestCase):
    def test_build_dataset_emits_config_overrides_and_idle_ticks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "long_dataset.json"
            build_dataset(
                ticks=24,
                tick_interval_ms=0,
                vision_budget=48,
                vision_reconstruction_budget=512,
                preset="heavy",
                output_path=output_path,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "multimodal")
            self.assertEqual(len(payload["items"]), 24)
            self.assertEqual(payload["label"], "长程多模态连续场景_24tick_heavy")
            self.assertEqual(payload["config_overrides"]["vision_patch_budget"], 40)
            self.assertEqual(payload["config_overrides"]["vision_focus_patch_budget"], 20)
            self.assertEqual(payload["config_overrides"]["vision_reconstruction_patch_budget"], 1024)
            self.assertEqual(payload["config_overrides"]["hearing_window_budget"], 24)
            self.assertTrue(any((item.get("text", "") == "" and item.get("source_type") == "long_multimodal_scene::idle") for item in payload["items"]))
            multimodal_count = sum(1 for item in payload["items"] if item.get("image_b64") or item.get("audio_b64"))
            self.assertGreater(multimodal_count, 8)

    def test_build_dataset_supports_lighter_preset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "long_dataset_lighter.json"
            build_dataset(
                ticks=32,
                tick_interval_ms=0,
                vision_budget=48,
                vision_reconstruction_budget=512,
                preset="lighter",
                output_path=output_path,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            overrides = dict(payload.get("config_overrides", {}) or {})
            self.assertEqual(payload["label"], "长程多模态连续场景_32tick_lighter")
            self.assertEqual(overrides["vision_patch_budget"], 16)
            self.assertEqual(overrides["vision_focus_patch_budget"], 8)
            self.assertEqual(overrides["vision_raw_state_budget"], 80)
            self.assertEqual(overrides["vision_attention_boost_max_extra_raw_budget"], 48)
            self.assertEqual(overrides["hearing_window_budget"], 18)
            self.assertEqual(overrides["memory_candidate_limit"], 176)


if __name__ == "__main__":
    unittest.main()
