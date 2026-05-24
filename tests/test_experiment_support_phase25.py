# -*- coding: utf-8 -*-
from __future__ import annotations

from io import BytesIO
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config
from core.state_pool_v2 import StatePoolV2
from scripts.run_v2_experiment_suite import evaluate_text_run


class ExperimentSupportPhase25Tests(unittest.TestCase):
    def test_evaluate_text_run_reads_externalized_competition_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            chunks_dir = run_dir / "chunks"
            chunks_dir.mkdir(parents=True, exist_ok=True)

            summary_row = {
                "tick_index": 0,
                "input_preview": "冬天 的 天气 有点 冷",
                "a_focus_preview": ["冬天", "天气", "有点"],
                "state_top": [{"sa_label": "text::天", "energy": 2.0}],
            }
            sidecar_row = {
                "tick_index": 0,
                "competition_packet": {
                    "schema_id": "sidecar_ref/competition",
                    "run_id": "demo",
                    "tick_index": 0,
                    "kind": "competition",
                    "externalized": True,
                },
                "state_pool_snapshot": {
                    "prediction_trace": {
                        "predicted_texts": ["天", "冷"],
                        "actual_texts": ["冬天", "天气", "有点"],
                        "unexpected_labels": ["phrase::冬_天", "phrase::有_点"],
                        "missed_predicted_labels": ["text::冷"],
                        "mismatch_mass": 1.0,
                    }
                },
            }
            competition_row = {
                "tick_index": 0,
                "competition_summary": {
                    "phrase_hit_count": 3,
                    "phrase_hit_preview": ["冬天", "天气", "有点"],
                },
                "sa_items": [
                    {"sa_label": "phrase::冬_天", "family": "learned_text_phrase"},
                    {"sa_label": "phrase::天_气", "family": "learned_text_phrase"},
                    {"sa_label": "phrase::有_点", "family": "learned_text_phrase"},
                ],
            }

            (chunks_dir / "ticks_000000_000999.summary.jsonl").write_text(json.dumps(summary_row, ensure_ascii=False) + "\n", encoding="utf-8")
            (chunks_dir / "ticks_000000_000999.sidecar.jsonl").write_text(json.dumps(sidecar_row, ensure_ascii=False) + "\n", encoding="utf-8")
            (chunks_dir / "ticks_000000_000999.competition.jsonl").write_text(json.dumps(competition_row, ensure_ascii=False) + "\n", encoding="utf-8")

            evaluation = evaluate_text_run(run_dir)
            self.assertEqual(int(evaluation.get("phrase_hit_total", 0) or 0), 3)
            self.assertEqual(int(evaluation.get("dynamic_phrase_hit_count", 0) or 0), 3)
            self.assertEqual(list(evaluation.get("dynamic_phrase_labels", []) or []), ["phrase::冬_天", "phrase::天_气", "phrase::有_点"])

    def test_prediction_trace_ignores_raw_visual_samples_as_unexpected_items(self) -> None:
        runtime = RuntimeV2(config=load_config())
        pool = runtime.state_pool
        packet = {
            "normalized_text": "",
            "full_stream": {"units": []},
            "sa_items": [
                {"sa_label": "vision_mem::shape_a", "display_text": "视觉特征A", "energy": 0.8, "sa_kind": "visual_focus_feature_unit", "attributes": {"sample_role": "memory_feature"}},
            ],
            "state_pool_sa_items": [
                {"sa_label": "vision::raw_1", "display_text": "视觉采样1", "energy": 0.3, "sa_kind": "visual_sparse_sample_unit", "channel": "vision"},
                {"sa_label": "vision::raw_2", "display_text": "视觉采样2", "energy": 0.3, "sa_kind": "visual_sparse_sample_unit", "channel": "vision"},
                {"sa_label": "vision_mem::shape_a", "display_text": "视觉特征A", "energy": 0.8, "sa_kind": "visual_focus_feature_unit", "attributes": {"sample_role": "memory_feature"}},
            ],
        }
        result = pool.apply_text_packet(
            packet,
            tick_index=0,
            predicted_items=[
                {"sa_label": "vision_mem::shape_a", "display_text": "视觉特征A", "energy": 0.7, "sa_kind": "visual_focus_feature_unit", "attributes": {"sample_role": "memory_feature"}},
            ],
        )
        self.assertEqual(int(result.get("prediction_unexpected_count", -1)), 0)
        self.assertGreaterEqual(float(result.get("prediction_mismatch_mass", -1.0)), 0.0)
        self.assertGreaterEqual(float((pool.snapshot_summary().get("prediction_trace", {}) or {}).get("overprediction_mass", 0.0) or 0.0), 0.0)

    def test_dynamic_phrase_learning_emerges_after_repetition(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="冬天 天气 有点 冷", tick_index=0)
        runtime.process_text_tick(text="冬天 天气 有点 冷", tick_index=1)
        runtime.process_text_tick(text="冬天 天气 有点", tick_index=2)
        phrase_ids = {item.sa_id for item in runtime.sa_registry.all_prototypes()}
        self.assertTrue(any("冬天" in item or "天气" in item or "有点" in item for item in phrase_ids))

    def test_prediction_trace_records_match_and_mismatch(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="冬天 天气 有点 冷", tick_index=0)
        runtime.process_text_tick(text="冬天 天气 有点 冷", tick_index=1)
        tick = runtime.process_text_tick(text="冬天 天气 有点 凉", tick_index=2)
        prediction_trace = dict((tick.get("state_pool_summary", {}) or {}).get("prediction_trace", {}) or {})
        self.assertIn("unexpected_count", prediction_trace)
        self.assertIn("missed_count", prediction_trace)
        self.assertIn("overprediction_mass", prediction_trace)
        self.assertIn("underprediction_mass", prediction_trace)
        self.assertIn("committed_overprediction_mass", prediction_trace)
        self.assertIn("predicted_commitment_mass", prediction_trace)
        self.assertGreaterEqual(float(prediction_trace.get("mismatch_mass", 0.0) or 0.0), 0.0)

    def test_first_tick_does_not_fake_prediction_mismatch(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        prediction_trace = dict((tick.get("state_pool_summary", {}) or {}).get("prediction_trace", {}) or {})
        self.assertEqual(int(prediction_trace.get("missed_count", -1)), 0)
        self.assertEqual(float(prediction_trace.get("overprediction_mass", -1.0)), 0.0)
        self.assertGreater(float(prediction_trace.get("underprediction_mass", 0.0) or 0.0), 0.0)

    def test_rules_engine_can_see_dynamic_text_ngram_metrics(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="打开 记事本", tick_index=0)
        metrics = dict((tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertEqual(float(metrics.get("text.ngram::打开记事本", 0.0) or 0.0), 1.0)
        self.assertEqual(float(metrics.get("text.unit::打开", 0.0) or 0.0), 1.0)

    def test_feedback_metrics_flow_into_next_tick_rules(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.6}],
            emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
            runtime_action_effects={"moved": False},
            external_feedback={"reward": 0.35, "punishment": 0.1},
        )
        tick = runtime.process_text_tick(text="我 想 出门", tick_index=1)
        metrics = dict((tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertEqual(float(metrics.get("feedback.reward", 0.0) or 0.0), 0.35)
        self.assertGreaterEqual(float(metrics.get("feedback.punishment", 0.0) or 0.0), 0.1)
        self.assertEqual(float(metrics.get("feedback.external_punishment", 0.0) or 0.0), 0.1)

    def test_context_action_bias_separates_commands(self) -> None:
        runtime = RuntimeV2(config=load_config())
        notepad_context = {
            "normalized_text": "open notepad",
            "query_units": ["open", "notepad"],
            "context_keys": ["text::open_notepad"],
        }
        calc_context = {
            "normalized_text": "open calc",
            "query_units": ["open", "calc"],
            "context_keys": ["text::open_calc"],
        }
        for tick_index in range(6):
            runtime.action_learning.record_feedback(
                tick_index=tick_index,
                selected_actions=[{"action_id": "action::type_text", "action_name": "type_text", "drive": 0.6}],
                emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
                runtime_action_effects={"moved": False},
                external_feedback={"reward": 0.5},
                context_hints=notepad_context,
            )
            runtime.action_learning.record_feedback(
                tick_index=100 + tick_index,
                selected_actions=[{"action_id": "action::press_key", "action_name": "press_key", "drive": 0.6}],
                emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
                runtime_action_effects={"moved": False},
                external_feedback={"reward": 0.5},
                context_hints=calc_context,
            )
        scored_notepad = runtime.action_learning.score_action_drives(
            [
                {"action_id": "action::type_text", "action_name": "type_text", "drive": 0.5},
                {"action_id": "action::press_key", "action_name": "press_key", "drive": 0.5},
            ],
            context_hints=notepad_context,
        )
        scored_calc = runtime.action_learning.score_action_drives(
            [
                {"action_id": "action::type_text", "action_name": "type_text", "drive": 0.5},
                {"action_id": "action::press_key", "action_name": "press_key", "drive": 0.5},
            ],
            context_hints=calc_context,
        )
        self.assertEqual(scored_notepad["scored_action_drives"][0]["action_id"], "action::type_text")
        self.assertEqual(scored_calc["scored_action_drives"][0]["action_id"], "action::press_key")

    def test_context_instance_bias_can_override_global_instance_habit(self) -> None:
        runtime = RuntimeV2(config=load_config())
        left_instance = 'action::press_key::{"key":"left"}'
        right_instance = 'action::press_key::{"key":"right"}'
        alpha_context = {
            "normalized_text": "口令甲",
            "query_units": ["口令", "甲"],
            "context_keys": ["text::口令甲"],
        }
        beta_context = {
            "normalized_text": "口令乙",
            "query_units": ["口令", "乙"],
            "context_keys": ["text::口令乙"],
        }
        for tick_index in range(6):
            runtime.action_learning.record_feedback(
                tick_index=tick_index,
                selected_actions=[{"action_id": "action::press_key", "action_name": "press_key", "instance_id": left_instance, "drive": 0.6}],
                emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
                runtime_action_effects={"moved": False},
                external_feedback={"reward": 0.5},
                context_hints=alpha_context,
            )
        for tick_index in range(6):
            runtime.action_learning.record_feedback(
                tick_index=100 + tick_index,
                selected_actions=[{"action_id": "action::press_key", "action_name": "press_key", "instance_id": right_instance, "drive": 0.6}],
                emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
                runtime_action_effects={"moved": False},
                external_feedback={"reward": 0.5},
                context_hints=beta_context,
            )
        scored_beta = runtime.action_learning.score_action_drives(
            [
                {"action_id": "action::press_key", "action_name": "press_key", "instance_id": left_instance, "drive": 0.5},
                {"action_id": "action::press_key", "action_name": "press_key", "instance_id": right_instance, "drive": 0.5},
            ],
            context_hints=beta_context,
        )
        self.assertEqual(str((scored_beta["scored_action_drives"][0].get("params", {}) or {}).get("key", "") or ""), "")
        self.assertEqual(scored_beta["scored_action_drives"][0]["instance_id"], right_instance)

    def test_multimodal_sensor_features_include_reconstruction_signals(self) -> None:
        runtime = RuntimeV2(config=load_config())
        image = Image.new("RGB", (12, 12), color=(220, 40, 30))
        image_buf = BytesIO()
        image.save(image_buf, format="PNG")
        image_packet = runtime.vision_sensor.ingest_image_bytes(image_buf.getvalue(), tick_index=0, source_type="image_input")
        self.assertGreater(len(image_packet.get("patches", []) or []), 0)
        self.assertGreater(len(image_packet.get("raw_samples", []) or []), 0)
        self.assertIn("preview_image", image_packet)
        self.assertIn("fixation_buffer", image_packet)
        patch_attrs = dict(((image_packet.get("patches", []) or [])[0].get("attributes", {}) or {}))
        self.assertIn("avg_r", patch_attrs)
        self.assertIn("avg_g", patch_attrs)
        self.assertIn("avg_b", patch_attrs)
        self.assertIn("hue", patch_attrs)

        import wave
        import struct
        import math
        audio_buf = BytesIO()
        with wave.open(audio_buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            frames = []
            for i in range(1600):
                sample = int(12000 * math.sin(2 * math.pi * 440 * (i / 16000.0)))
                frames.append(struct.pack("<h", sample))
            wav.writeframes(b"".join(frames))
        audio_packet = runtime.hearing_sensor.ingest_wav_bytes(audio_buf.getvalue(), tick_index=0, source_type="audio_input")
        self.assertGreater(len(audio_packet.get("windows", []) or []), 0)
        win_attrs = dict(((audio_packet.get("windows", []) or [])[0].get("attributes", {}) or {}))
        self.assertIn("low_band", win_attrs)
        self.assertIn("mid_band", win_attrs)
        self.assertIn("high_band", win_attrs)
        self.assertIn("dominant_bin_ratio", win_attrs)

    def test_runtime_transient_reset_keeps_long_term_memory_only(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="apple red", tick_index=0)
        runtime.process_text_tick(text="apple red", tick_index=1)
        self.assertGreater(runtime.memory_store.count(), 0)
        self.assertGreaterEqual(int(runtime.state_pool.snapshot_summary().get("state_pool_size", 0) or 0), 1)
        runtime.reset_transient_state()
        self.assertGreater(runtime.memory_store.count(), 0)
        self.assertEqual(int(runtime.state_pool.snapshot_summary().get("state_pool_size", -1)), 0)
        self.assertEqual(len(runtime.short_term.snapshot()), 0)


if __name__ == "__main__":
    unittest.main()
