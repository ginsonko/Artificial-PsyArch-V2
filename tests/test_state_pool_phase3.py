# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest

from core.state_pool_v2 import StatePoolV2
from sensors.text_sensor_v2 import TextSensorV2


class StatePoolPhase3Tests(unittest.TestCase):
    def _build_runtime(self) -> tuple[TextSensorV2, StatePoolV2]:
        sensor = TextSensorV2(budget_limit=4, fatigue_window=8, fatigue_threshold=1, max_suppression=0.75)
        pool = StatePoolV2(
            decay=0.92,
            prune_threshold=0.05,
            recent_queue_limit=6,
            verbatim_window_chars=48,
            head_limit=5,
            items_per_head=4,
            anchor_cache_limit=5,
            residual_limit=5,
            handle_limit=7,
            residual_unit_limit=4,
            attention_object_fatigue_decay=0.8,
            attention_object_fatigue_step=0.25,
            attention_object_fatigue_gain=0.9,
            attention_object_fatigue_max=1.0,
            attention_object_min_multiplier=0.35,
        )
        return sensor, pool

    def test_residual_bucket_tracks_truncated_units_with_bound(self) -> None:
        sensor, pool = self._build_runtime()
        packet = sensor.ingest("今 天 天 气 真 的 很 不 错 呢", tick_index=0)
        result = pool.apply_text_packet(packet, tick_index=0)
        summary = pool.snapshot_summary()
        self.assertGreater(result["residual_truncated_count"], 0)
        self.assertGreater(summary["residual_summary"]["count"], 0)
        self.assertLessEqual(summary["residual_summary"]["count"], 5)

    def test_handle_ring_and_anchor_cache_remain_bounded(self) -> None:
        sensor, pool = self._build_runtime()
        for tick in range(20):
            packet = sensor.ingest(f"我 想 出 门 {tick}", tick_index=tick)
            pool.apply_text_packet(packet, tick_index=tick)
        sidecar = pool.snapshot_sidecar()
        summary = sidecar["state_pool_summary"]
        self.assertLessEqual(len(sidecar["hot_anchor_cache"]), 5)
        self.assertLessEqual(len(sidecar["handle_ring"]), 7)
        self.assertLessEqual(summary["anchor_summary"]["count"], 5)
        self.assertLessEqual(summary["handle_summary"]["count"], 7)

    def test_r_state_exposes_residual_head_without_second_energy_system(self) -> None:
        sensor, pool = self._build_runtime()
        pool.apply_text_packet(sensor.ingest("甲 乙 丙 丁 戊 己 庚", tick_index=0), tick_index=0)
        r_state = pool.read_r_state()
        head_ids = [head["head_id"] for head in r_state["heads"]]
        self.assertIn("head_residual", r_state["available_head_ids"])
        self.assertIn("head_anchor", head_ids)
        self.assertIn("head_residual", head_ids)
        residual_items = next(head["items"] for head in r_state["heads"] if head["head_id"] == "head_residual")
        self.assertTrue(all("unresolved_mass" in item for item in residual_items))
        global_items = next(head["items"] for head in r_state["heads"] if head["head_id"] == "head_global")
        self.assertTrue(all("energy" in item for item in global_items))
    def test_lazy_decay_is_applied_when_future_tick_arrives(self) -> None:
        sensor, pool = self._build_runtime()
        packet = sensor.ingest("apple", tick_index=0)
        pool.apply_text_packet(packet, tick_index=0)
        pool.apply_text_packet(sensor.ingest("", tick_index=3), tick_index=3)
        top = pool.snapshot_top(limit=4)
        apple = next(item for item in top if item.get("sa_label") == "text::apple")
        self.assertAlmostEqual(float(apple.get("energy", 0.0) or 0.0), round(0.92 ** 3, 4), places=4)

    def test_repeated_top_and_anchor_reads_are_stable_with_lazy_decay(self) -> None:
        sensor, pool = self._build_runtime()
        pool.apply_text_packet(sensor.ingest("alpha beta", tick_index=0), tick_index=0)
        top_first = pool.snapshot_top(limit=3)
        top_second = pool.snapshot_top(limit=3)
        anchor_first = pool.snapshot_summary()["anchor_summary"]["top"]
        anchor_second = pool.snapshot_summary()["anchor_summary"]["top"]
        self.assertEqual(top_first, top_second)
        self.assertEqual(anchor_first, anchor_second)

    def test_attention_focus_commit_builds_object_level_fatigue(self) -> None:
        sensor, pool = self._build_runtime()
        pool.apply_text_packet(sensor.ingest("apple", tick_index=0), tick_index=0)
        focus1 = pool.read_a_focus_with_bias(limit=1, commit=True)
        summary1 = pool.snapshot_summary()
        fatigue_top1 = list((summary1.get("attention_fatigue_summary", {}) or {}).get("top", []) or [])
        self.assertEqual(focus1.get("focus_units", []), ["apple"])
        self.assertGreater(len(fatigue_top1), 0)
        self.assertEqual(str(fatigue_top1[0].get("sa_label", "") or ""), "text::apple")
        self.assertAlmostEqual(float(fatigue_top1[0].get("fatigue", 0.0) or 0.0), 0.25, places=4)

        pool.apply_text_packet(sensor.ingest("apple", tick_index=1), tick_index=1)
        focus2 = pool.read_a_focus_with_bias(limit=1, commit=True)
        summary2 = pool.snapshot_summary()
        fatigue_top2 = list((summary2.get("attention_fatigue_summary", {}) or {}).get("top", []) or [])
        self.assertEqual(focus2.get("focus_units", []), ["apple"])
        self.assertGreater(float(fatigue_top2[0].get("fatigue", 0.0) or 0.0), 0.25)

    def test_attention_fatigue_recovers_across_idle_ticks(self) -> None:
        sensor, pool = self._build_runtime()
        pool.apply_text_packet(sensor.ingest("apple", tick_index=0), tick_index=0)
        pool.read_a_focus_with_bias(limit=1, commit=True)
        before = list((pool.snapshot_summary().get("attention_fatigue_summary", {}) or {}).get("top", []) or [])
        self.assertGreater(float(before[0].get("fatigue", 0.0) or 0.0), 0.0)

        pool.apply_text_packet(sensor.ingest("", tick_index=3), tick_index=3)
        after = list((pool.snapshot_summary().get("attention_fatigue_summary", {}) or {}).get("top", []) or [])
        self.assertGreater(len(after), 0)
        self.assertLess(float(after[0].get("fatigue", 0.0) or 0.0), float(before[0].get("fatigue", 0.0) or 0.0))

    def test_attention_fatigue_survives_export_import(self) -> None:
        sensor, pool = self._build_runtime()
        pool.apply_text_packet(sensor.ingest("apple", tick_index=0), tick_index=0)
        pool.read_a_focus_with_bias(limit=1, commit=True)
        payload = pool.export_payload()

        restored = self._build_runtime()[1]
        restored.import_payload(payload)
        summary = restored.snapshot_summary()
        fatigue_top = list((summary.get("attention_fatigue_summary", {}) or {}).get("top", []) or [])
        self.assertGreater(len(fatigue_top), 0)
        self.assertEqual(str(fatigue_top[0].get("sa_label", "") or ""), "text::apple")
        self.assertAlmostEqual(float(fatigue_top[0].get("fatigue", 0.0) or 0.0), 0.25, places=4)


if __name__ == "__main__":
    unittest.main()
