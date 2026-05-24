# -*- coding: utf-8 -*-
from __future__ import annotations

import unittest
from io import BytesIO
import math
import struct
import wave

from PIL import Image, ImageDraw

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config


class RuntimeChainPhase4To7Tests(unittest.TestCase):
    @staticmethod
    def _vision_probe_png(
        *,
        base_rect: tuple[int, int, int, int] | None = (68, 76, 108, 116),
        moving_rect: tuple[int, int, int, int] | None = None,
        size: tuple[int, int] = (192, 192),
    ) -> bytes:
        image = Image.new("RGB", size, color=(18, 18, 18))
        draw = ImageDraw.Draw(image)
        if base_rect is not None:
            draw.rectangle(base_rect, fill=(236, 236, 236))
        if moving_rect is not None:
            draw.rectangle(moving_rect, fill=(255, 255, 255))
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def test_phrase_competition_and_memory_recall_chain(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        tick1 = runtime.process_text_tick(text="今天 天气 不错", tick_index=1)
        tick2 = runtime.process_text_tick(text="我 想 出门", tick_index=2)

        phrase_preview = tick1["competition_summary"]["phrase_hit_preview"]
        self.assertTrue(any("今天天气" in item or "天气不错" in item for item in phrase_preview))
        self.assertGreaterEqual(len(tick2["bn_list"]), 1)
        self.assertIn("items", tick2["c_star"])
        self.assertGreaterEqual(tick2["memory_count"], 3)

    def test_short_term_focus_chain_and_rules_injection(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="这 句话 值得 想想", tick_index=0)
        tick1 = runtime.process_text_tick(text="这 句话 值得 想想", tick_index=1)
        tick2 = runtime.process_text_tick(text="我 想 继续 看", tick_index=2)

        self.assertGreaterEqual(len(tick2["short_term_snapshot"]), 1)
        self.assertIn("emotion_channels", tick1["rules_result"])
        self.assertIn("rules_fired", tick1["rules_result"])
        self.assertIn("state_pool_summary", tick2)

    def test_mixed_recall_score_breakdown_is_visible(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="今天 天气 有点 冷", tick_index=0)
        runtime.process_text_tick(text="今天 天气 有点 冷", tick_index=1)
        tick2 = runtime.process_text_tick(text="今天 天气 有点", tick_index=2)
        self.assertGreaterEqual(len(tick2["bn_list"]), 1)
        top = tick2["bn_list"][0]
        self.assertIn("score_breakdown", top)
        self.assertIn("bigram_overlap", top["score_breakdown"])
        self.assertIn("candidate_sources", top)

    def test_logic_ms_feedback_and_tuner_adjustments_apply_to_runtime_controls(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.set_last_logic_ms(220.0)
        tick = runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        tuner = tick["rules_result"]["tuner_result"]
        matched_ids = [item.get("profile_id", "") for item in tuner.get("matched_profiles", [])]
        self.assertIn("high_load_guard", matched_ids)
        controls = tick["runtime_controls"]
        self.assertEqual(controls["sampling.increment_budget"], 32.0)
        self.assertEqual(controls["attention.focus_gain"], 1.05)
        self.assertEqual(controls["prediction.successor_bias_gain"], 1.05)
        self.assertEqual(controls["state.current_input_gain"], 0.9)
        self.assertEqual(controls["state.history_suppression_gain"], 1.1)
        self.assertEqual(controls["state.prediction_suppression_gain"], 1.15)
        self.assertEqual(tick["logic_feedback"]["previous_tick_logic_ms"], 220.0)
        applied_targets = {item.get("target", "") for item in tick.get("applied_tuner_adjustments", [])}
        self.assertIn("sampling.increment_budget", applied_targets)

    def test_tuner_resets_to_baseline_when_load_recovers(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.set_last_logic_ms(220.0)
        runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        runtime.set_last_logic_ms(40.0)
        tick = runtime.process_text_tick(text="我 想 出门", tick_index=1)
        matched_ids = [item.get("profile_id", "") for item in tick["rules_result"]["tuner_result"].get("matched_profiles", [])]
        self.assertIn("baseline_default", matched_ids)
        self.assertEqual(tick["runtime_controls"]["sampling.increment_budget"], 48.0)
        self.assertEqual(tick["runtime_controls"]["prediction.successor_bias_gain"], 1.18)
        self.assertEqual(tick["runtime_controls"]["state.current_input_gain"], 1.0)

    def test_runtime_apply_selected_actions_can_move_visual_gaze(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.vision_sensor.move_gaze(0.5, 0.5)
        image_packet = {
            "patches": [
                {
                    "sa_label": "vision::0_0",
                    "energy": 0.2,
                    "coords": {"cx": 0.2, "cy": 0.2},
                    "attributes": {"brightness": 0.2},
                },
                {
                    "sa_label": "vision::1_1",
                    "energy": 0.9,
                    "coords": {"cx": 0.75, "cy": 0.25},
                    "attributes": {"brightness": 0.9},
                },
            ]
        }
        effects = runtime.apply_selected_actions(
            [{"action_name": "continue_focus", "params": {}}],
            runtime_tick={"image_packet": image_packet},
        )
        self.assertTrue(effects["moved"])
        self.assertAlmostEqual(effects["gaze_center_after"]["x"], 0.75, places=2)
        self.assertAlmostEqual(effects["gaze_center_after"]["y"], 0.25, places=2)
        self.assertIn("attention_boost", effects)
        self.assertTrue(bool((effects.get("attention_boost", {}) or {}).get("active", False)))

    def test_runtime_apply_selected_actions_sets_next_tick_visual_attention_boost(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.vision_sensor.move_gaze(0.4, 0.4)
        image_packet = {
            "focus_priority_samples": [
                {
                    "sa_label": "vision::focus",
                    "energy": 0.9,
                    "coords": {"cx": 0.7, "cy": 0.3},
                    "attributes": {"brightness": 0.9},
                }
            ]
        }
        effects = runtime.apply_selected_actions(
            [{"action_name": "continue_focus", "params": {}, "firmness_norm": 1.0}],
            runtime_tick={"image_packet": image_packet},
        )
        boost = dict(runtime.vision_sensor.attention_boost_snapshot() or {})
        self.assertTrue(bool(boost.get("active", False)))
        self.assertEqual(str(boost.get("source_action", "") or ""), "continue_focus")
        self.assertAlmostEqual(float((boost.get("target_gaze", {}) or {}).get("x", 0.0) or 0.0), 0.7, places=2)
        self.assertIn("attention_boost", effects)
        modulation = dict(effects.get("attention_modulation", {}) or {})
        self.assertGreater(float(modulation.get("attention_lock", 0.0) or 0.0), 0.0)
        modulated_controls = dict(modulation.get("modulated_controls", {}) or {})
        self.assertGreater(
            float(modulated_controls.get("state.history_suppression_gain", 0.0) or 0.0),
            float(runtime.runtime_controls_snapshot().get("state.history_suppression_gain", 0.0) or 0.0),
        )

    def test_runtime_apply_selected_actions_can_move_audio_focus(self) -> None:
        runtime = RuntimeV2(config=load_config())
        audio_packet = {
            "focus_priority_samples": [
                {
                    "sa_label": "audio::focus",
                    "energy": 0.9,
                    "coords": {"freq_center_hz": 880.0},
                    "attributes": {"focus_priority": 0.9, "dominant_hz": 880.0},
                }
            ]
        }
        effects = runtime.apply_selected_actions(
            [{"action_name": "continue_audio_focus", "params": {}, "firmness_norm": 1.0}],
            runtime_tick={"audio_packet": audio_packet},
        )
        self.assertTrue(bool(effects.get("audio_moved", False)))
        self.assertAlmostEqual(float((effects.get("audio_focus_after", {}) or {}).get("center_hz", 0.0) or 0.0), 880.0, places=1)
        self.assertTrue(bool((effects.get("audio_attention_boost", {}) or {}).get("active", False)))

    def test_runtime_hearing_sensor_packet_exports_audio_focus_fields(self) -> None:
        runtime = RuntimeV2(config=load_config())
        audio_buf = BytesIO()
        with wave.open(audio_buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            frames = []
            for i in range(1600):
                sample = int(12000 * math.sin(2 * math.pi * 880 * (i / 16000.0)))
                frames.append(struct.pack("<h", sample))
            wav.writeframes(b"".join(frames))
        packet = runtime.hearing_sensor.ingest_wav_bytes(audio_buf.getvalue(), tick_index=0, source_type="audio_input")
        self.assertIn("audio_focus", packet)
        self.assertIn("attention_boost", packet)
        self.assertIn("focus_priority_samples", packet)

    def test_multimodal_tick_builds_hearing_feelings_and_query_spacetime(self) -> None:
        runtime = RuntimeV2(config=load_config())
        audio_buf = BytesIO()
        with wave.open(audio_buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            frames = []
            for i in range(2400):
                sample = int(11000 * math.sin(2 * math.pi * 880 * (i / 16000.0)))
                frames.append(struct.pack("<h", sample))
            wav.writeframes(b"".join(frames))
        audio_packet = runtime.hearing_sensor.ingest_wav_bytes(audio_buf.getvalue(), tick_index=0, source_type="audio_input")
        tick = runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("声音", tick_index=0, source_type="multimodal_input"),
            audio_packet=audio_packet,
            source_type="multimodal_input",
        )
        feeling_items = list(tick.get("channel_feeling_items", []) or [])
        labels = [str(item.get("sa_label", "") or "") for item in feeling_items]
        self.assertTrue(any(label.startswith("hearingfelt::") for label in labels))
        query_spacetime = dict(tick.get("query_spacetime", {}) or {})
        self.assertIn("hearing_confidence", query_spacetime)
        self.assertTrue(
            any(
                key in query_spacetime
                for key in (
                    "hearing_timbre_center",
                    "hearing_noise_center",
                    "hearing_pitch_stability_center",
                    "hearing_percussive_center",
                )
            )
        )
        self.assertIn("hearing_feeling_trace", tick)

    def test_runtime_tick_exposes_stage_timing_breakdown(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="apple banana", tick_index=0)
        timing = dict(tick.get("runtime_stage_timing_ms", {}) or {})
        self.assertIn("01_text_competition_ms", timing)
        self.assertIn("05_main_recall_prediction_ms", timing)
        self.assertIn("09_total_runtime_ms", timing)
        self.assertGreaterEqual(float(timing.get("09_total_runtime_ms", 0.0) or 0.0), 0.0)
        logic_feedback = dict(tick.get("logic_feedback", {}) or {})
        self.assertIn("runtime_stage_timing_ms", logic_feedback)

    def test_runtime_residual_target_prefers_dynamic_motion_summary(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.vision_sensor.move_gaze(0.4, 0.4)
        image_packet = {
            "focus_priority_samples": [
                {
                    "sa_label": "vision::focus_static",
                    "energy": 0.92,
                    "coords": {"cx": 0.72, "cy": 0.28},
                    "attributes": {"brightness": 0.95},
                }
            ],
            "dynamic_motion_samples": [
                {
                    "sa_label": "vision_dyn::trk_0001",
                    "energy": 0.66,
                    "coords": {"cx": 0.2, "cy": 0.78},
                    "attributes": {
                        "dynamic_objectness": 0.82,
                        "motion_speed": 0.24,
                        "motion_surprise": 0.21,
                        "motion_coherence": 0.74,
                        "temporal_persistence": 0.58,
                    },
                }
            ],
        }
        effects = runtime.apply_selected_actions(
            [{"action_name": "inspect_residual", "params": {}, "firmness_norm": 1.0}],
            runtime_tick={"image_packet": image_packet},
        )
        boost = dict(runtime.vision_sensor.attention_boost_snapshot() or {})
        self.assertTrue(bool(boost.get("active", False)))
        self.assertEqual(str(boost.get("source_action", "") or ""), "inspect_residual")
        self.assertAlmostEqual(float((boost.get("target_gaze", {}) or {}).get("x", 0.0) or 0.0), 0.2, places=2)
        self.assertAlmostEqual(float((boost.get("target_gaze", {}) or {}).get("y", 0.0) or 0.0), 0.78, places=2)

    def test_attention_modulation_can_bias_next_tick_query_toward_new_input(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="three three three", tick_index=0)
        runtime.process_text_tick(text="three three", tick_index=1)
        handoff_tick = runtime.process_text_tick(text="eight", tick_index=2)
        effects = runtime.apply_selected_actions(
            [{"action_name": "continue_focus", "params": {}, "firmness_norm": 1.2}],
            runtime_tick=handoff_tick,
        )
        modulation = dict(effects.get("attention_modulation", {}) or {})
        self.assertTrue(bool(modulation.get("has_attention_action", False)))
        self.assertGreater(float(modulation.get("attention_lock", 0.0) or 0.0), 0.0)
        next_tick = runtime.process_text_tick(text="eight", tick_index=3)
        effective = dict(next_tick.get("effective_attention_controls", {}) or {})
        baseline = dict(next_tick.get("runtime_controls", {}) or {})
        self.assertGreater(
            float(effective.get("state.history_suppression_gain", 0.0) or 0.0),
            float(baseline.get("state.history_suppression_gain", 0.0) or 0.0),
        )
        preview = list(((next_tick.get("recall_query_preview", {}) or {}).get("preview", []) or []))
        self.assertGreaterEqual(len(preview), 1)
        top_labels = [str(item.get("sa_label", "") or "") for item in preview[:4]]
        self.assertTrue(any("eight" in label for label in top_labels))

    def test_non_visual_actions_push_vision_into_suppressed_mode(self) -> None:
        runtime = RuntimeV2(config=load_config())
        effects = runtime.apply_selected_actions(
            [{"action_name": "type_text", "params": {"text": "hello"}}],
            runtime_tick={"image_packet": {}},
        )
        self.assertFalse(effects["moved"])
        self.assertEqual(runtime.vision_sensor.attention_boost_snapshot().get("attention_mode"), "suppressed")

    def test_no_actions_restore_background_visual_mode(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.vision_sensor.set_attention_mode("suppressed")
        effects = runtime.apply_selected_actions([], runtime_tick={"image_packet": {}})
        self.assertFalse(effects["moved"])
        self.assertEqual(runtime.vision_sensor.attention_boost_snapshot().get("attention_mode"), "background")

    def test_multimodal_tick_uses_visual_raw_samples_for_state_pool_ingress(self) -> None:
        runtime = RuntimeV2(config=load_config())
        image_packet = {
            "patches": [
                {"sa_label": "vision::memory_0", "display_text": "视觉采样[0,0]", "energy": 0.6, "coords": {"cx": 0.2, "cy": 0.2}, "attributes": {"brightness": 0.2}, "channel": "vision"}
            ],
            "memory_write_samples": [
                {"sa_label": "vision::memory_0", "display_text": "视觉采样[0,0]", "energy": 0.6, "coords": {"cx": 0.2, "cy": 0.2}, "attributes": {"brightness": 0.2}, "channel": "vision"}
            ],
            "focus_priority_samples": [
                {"sa_label": "vision::focus_0", "display_text": "视觉采样[1,1]", "energy": 0.8, "coords": {"cx": 0.7, "cy": 0.4}, "attributes": {"brightness": 0.9}, "channel": "vision"}
            ],
            "raw_samples": [
                {"sa_label": "vision::raw_0", "display_text": "视觉采样[2,2]", "energy": 0.4, "coords": {"cx": 0.4, "cy": 0.4}, "attributes": {"brightness": 0.3}, "channel": "vision"},
                {"sa_label": "vision::raw_1", "display_text": "视觉采样[3,3]", "energy": 0.5, "coords": {"cx": 0.6, "cy": 0.6}, "attributes": {"brightness": 0.5}, "channel": "vision"},
            ],
            "global_structure_samples": [
                {
                    "sa_label": "vision_mem::global_shape::h1_c3_hs2_vs1",
                    "display_text": "视觉全局特征[global_shape::h1_c3_hs2_vs1]",
                    "energy": 0.7,
                    "coords": {"cx": 0.5, "cy": 0.5, "screen_x": 0.2, "screen_y": 0.2, "screen_w": 0.5, "screen_h": 0.5},
                    "attributes": {"sample_role": "global_structure", "global_feature_code": "global_shape::h1_c3_hs2_vs1"},
                    "channel": "vision",
                    "sa_kind": "visual_global_feature_unit",
                }
            ],
            "dynamic_motion_samples": [
                {
                    "sa_label": "vision_dyn::trk_0001",
                    "display_text": "动态对象[trk_0001]",
                    "energy": 0.66,
                    "coords": {"cx": 0.55, "cy": 0.52, "screen_x": 0.48, "screen_y": 0.44, "screen_w": 0.12, "screen_h": 0.16},
                    "attributes": {
                        "sample_role": "dynamic_motion_summary",
                        "track_id": "trk_0001",
                        "dynamic_objectness": 0.72,
                        "motion_speed": 0.18,
                    },
                    "channel": "vision",
                    "sa_kind": "visual_dynamic_track_unit",
                }
            ],
            "dynamic_track_summary": {"track_count": 1, "object_count": 1, "dynamic_salience_mean": 0.72},
            "budget_used": 1,
        }
        tick = runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("苹果", tick_index=0, source_type="multimodal_input"),
            image_packet=image_packet,
            source_type="multimodal_input",
        )
        pool_input_count = int((tick.get("pool_result_external", {}) or {}).get("pool_input_count", 0) or 0)
        self.assertGreaterEqual(pool_input_count, 3)
        competition_summary = dict(tick.get("competition_summary", {}) or {})
        channels = list(competition_summary.get("multimodal_channels", []) or [])
        self.assertTrue(
            any(
                int((row or {}).get("raw_count", 0) or 0) == 2
                and int((row or {}).get("memory_write_count", 0) or 0) == 1
                and int((row or {}).get("global_structure_count", 0) or 0) == 1
                and int((row or {}).get("dynamic_motion_count", 0) or 0) == 1
                and int((row or {}).get("count", 0) or 0) == 5
                for row in channels
                if str((row or {}).get("channel", "")) == "vision"
            )
        )
        metrics_snapshot = dict((tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertEqual(int(metrics_snapshot.get("metrics.vision_raw_sample_count", 0) or 0), 2)
        state_top_labels = [str(item.get("sa_label", "") or "") for item in ((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])]
        self.assertIn("vision_mem::global_shape::h1_c3_hs2_vs1", state_top_labels)

    def test_multimodal_exact_memory_includes_dynamic_visual_object_structure(self) -> None:
        runtime = RuntimeV2(config=load_config())
        image_packet = {
            "patches": [
                {"sa_label": "vision::memory_0", "display_text": "视觉采样[0,0]", "energy": 0.6, "coords": {"cx": 0.2, "cy": 0.2}, "attributes": {"brightness": 0.2}, "channel": "vision"}
            ],
            "memory_write_samples": [
                {
                    "sa_label": "vision_mem::feat_a",
                    "display_text": "视觉特征[a]",
                    "energy": 0.8,
                    "coords": {"cx": 0.52, "cy": 0.48, "screen_x": 0.46, "screen_y": 0.40, "screen_w": 0.12, "screen_h": 0.16},
                    "attributes": {"sample_role": "memory_feature", "memory_feature_code": "feat_a", "brightness": 0.6},
                    "channel": "vision",
                    "sa_kind": "visual_focus_feature_unit",
                }
            ],
            "global_structure_samples": [],
            "dynamic_motion_samples": [
                {
                    "sa_label": "vision_dyn::trk_0099",
                    "display_text": "动态对象[trk_0099]",
                    "energy": 0.72,
                    "coords": {"cx": 0.55, "cy": 0.52, "screen_x": 0.48, "screen_y": 0.44, "screen_w": 0.12, "screen_h": 0.16},
                    "attributes": {
                        "sample_role": "dynamic_motion_summary",
                        "track_id": "trk_0099",
                        "dynamic_objectness": 0.84,
                        "motion_speed": 0.18,
                        "motion_coherence": 0.73,
                        "boundary_motion_contrast": 0.34,
                        "shape_stability": 0.78,
                        "edge_strength": 0.77,
                        "stroke_likeness": 0.72,
                        "endpoint_likeness": 0.23,
                        "corner_likeness": 0.39,
                        "opening_likeness": 0.33,
                        "closure_likeness": 0.24,
                        "arc_balance": 0.81,
                        "structure_discriminability": 0.74,
                        "straight_likeness": 0.19,
                        "curvilinear_likeness": 0.85,
                        "angularity": 0.21,
                        "roundness": 0.80,
                        "local_symmetry": 0.44,
                        "horizontal_symmetry": 0.56,
                        "vertical_symmetry": 0.68,
                        "opening_dir_x": 0.0,
                        "opening_dir_y": 1.0,
                        "opening_direction_strength": 0.18,
                        "hole_like": 0.08,
                        "center_void": 0.16,
                        "proj_h_bin": "1231",
                        "proj_v_bin": "2122",
                        "orient_hist_bin": "1320",
                        "radial_hist_bin": "2210",
                        "radial_bin": "2210",
                        "quadrant_bin": "1122",
                        "foreground_polarity": "bright",
                        "local_patch_signature": "112233445",
                        "bbox_fill": 0.62,
                        "aspect_ratio": 0.72,
                        "area_ratio": 0.03,
                    },
                    "channel": "vision",
                    "sa_kind": "visual_dynamic_track_unit",
                }
            ],
            "dynamic_track_summary": {"track_count": 1, "object_count": 1, "dynamic_salience_mean": 0.84},
            "budget_used": 1,
        }
        tick = runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("three", tick_index=0, source_type="multimodal_input"),
            image_packet=image_packet,
            source_type="multimodal_input",
        )
        exact_memories = [
            row for row in (runtime.memory_store.export_payload().get("memories", []) or [])
            if str((row or {}).get("memory_kind", "") or "") == "exact_external"
        ]
        self.assertGreaterEqual(len(exact_memories), 1)
        items = list((exact_memories[-1] or {}).get("items", []) or [])
        item_labels = [str((row or {}).get("sa_label", "") or "") for row in items]
        self.assertIn("vision_dyn::trk_0099", item_labels)

    def test_multimodal_tick_uses_audio_summary_layers_for_state_pool_and_metrics(self) -> None:
        runtime = RuntimeV2(config=load_config())
        audio_packet = {
            "windows": [
                {
                    "sa_label": "audio::win_0",
                    "display_text": "听窗[0]",
                    "energy": 0.42,
                    "position": 0,
                    "coords": {"freq_center_hz": 440.0},
                    "attributes": {"sample_role": "audio_window", "raw_priority": 0.4},
                    "channel": "hearing",
                }
            ],
            "memory_write_samples": [
                {
                    "sa_label": "audio::mem_0",
                    "display_text": "听觉特征[mem_0]",
                    "energy": 0.78,
                    "position": 0,
                    "coords": {"freq_center_hz": 880.0},
                    "attributes": {"sample_role": "audio_memory", "sample_reason": "audio_focus"},
                    "channel": "hearing",
                    "sa_kind": "audio_window_unit",
                }
            ],
            "focus_priority_samples": [
                {
                    "sa_label": "audio::focus_0",
                    "display_text": "听觉焦点[0]",
                    "energy": 0.88,
                    "position": 0,
                    "coords": {"freq_center_hz": 880.0},
                    "attributes": {"focus_priority": 0.92, "sample_role": "audio_focus"},
                    "channel": "hearing",
                }
            ],
            "global_structure_samples": [
                {
                    "sa_label": "audio::global_band_2",
                    "display_text": "听觉全局特征[band::2]",
                    "energy": 0.67,
                    "position": 2,
                    "coords": {"freq_center_hz": 920.0},
                    "attributes": {"sample_role": "global_structure", "peak_band_index": 2},
                    "channel": "hearing",
                    "sa_kind": "audio_global_feature_unit",
                }
            ],
            "budget_used": 1,
        }
        tick = runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("声音", tick_index=0, source_type="multimodal_input"),
            audio_packet=audio_packet,
            source_type="multimodal_input",
        )
        competition_summary = dict(tick.get("competition_summary", {}) or {})
        channels = list(competition_summary.get("multimodal_channels", []) or [])
        hearing_row = next((row for row in channels if str((row or {}).get("channel", "")) == "hearing"), None)
        self.assertIsNotNone(hearing_row)
        self.assertEqual(int((hearing_row or {}).get("window_count", 0) or 0), 1)
        self.assertEqual(int((hearing_row or {}).get("memory_write_count", 0) or 0), 1)
        self.assertEqual(int((hearing_row or {}).get("global_structure_count", 0) or 0), 1)
        self.assertEqual(int((hearing_row or {}).get("count", 0) or 0), 3)
        metrics_snapshot = dict((tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertEqual(int(metrics_snapshot.get("metrics.audio_memory_write_count", 0) or 0), 1)
        state_top_labels = [str(item.get("sa_label", "") or "") for item in ((tick.get("state_pool_summary", {}) or {}).get("top", []) or [])]
        self.assertIn("audio::mem_0", state_top_labels)

    def test_audio_only_exact_memory_includes_audio_summary_layers(self) -> None:
        runtime = RuntimeV2(config=load_config())
        audio_packet = {
            "windows": [
                {
                    "sa_label": "audio::win_0",
                    "display_text": "听窗[0]",
                    "energy": 0.38,
                    "position": 0,
                    "coords": {"freq_center_hz": 440.0},
                    "attributes": {"sample_role": "audio_window"},
                    "channel": "hearing",
                }
            ],
            "memory_write_samples": [
                {
                    "sa_label": "audio::mem_voice",
                    "display_text": "听觉特征[voice]",
                    "energy": 0.81,
                    "position": 0,
                    "coords": {"freq_center_hz": 1100.0},
                    "attributes": {"sample_role": "audio_memory", "sample_reason": "audio_focus"},
                    "channel": "hearing",
                    "sa_kind": "audio_window_unit",
                }
            ],
            "global_structure_samples": [
                {
                    "sa_label": "audio::global_band_3",
                    "display_text": "听觉全局特征[band::3]",
                    "energy": 0.63,
                    "position": 3,
                    "coords": {"freq_center_hz": 1280.0},
                    "attributes": {"sample_role": "global_structure", "peak_band_index": 3},
                    "channel": "hearing",
                    "sa_kind": "audio_global_feature_unit",
                }
            ],
            "budget_used": 1,
        }
        runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("语音", tick_index=0, source_type="multimodal_input"),
            audio_packet=audio_packet,
            source_type="multimodal_input",
        )
        exact_memories = [
            row for row in (runtime.memory_store.export_payload().get("memories", []) or [])
            if str((row or {}).get("memory_kind", "") or "") == "exact_external"
        ]
        self.assertGreaterEqual(len(exact_memories), 1)
        items = list((exact_memories[-1] or {}).get("items", []) or [])
        labels = [str((row or {}).get("sa_label", "") or "") for row in items]
        self.assertIn("audio::mem_voice", labels)
        self.assertIn("audio::global_band_3", labels)

    def test_action_feedback_builds_long_term_action_bias(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        selected = [{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.6}]
        feedback = runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=selected,
            emotion_channels={"expectation": 0.7, "pressure": 0.1, "correctness": 0.4, "dissonance": 0.0},
            runtime_action_effects={"moved": True},
            external_feedback={"reward": 0.2},
        )
        self.assertGreater(feedback["feedback"], 0.0)
        summary = runtime.action_learning.bias_summary(limit=8)
        self.assertTrue(any(item.get("action_id") == "action::continue_focus" for item in summary))
        tick2 = runtime.process_text_tick(text="我 想 出门", tick_index=1)
        learned = tick2["rules_result"].get("action_learning_bias_summary", [])
        self.assertIsInstance(learned, list)
        self.assertIn("pending_feedback_metrics", tick2)
        self.assertEqual(float(tick2["pending_feedback_metrics"].get("reward", 0.0) or 0.0), 0.2)

    def test_pending_feedback_metrics_are_consumed_once(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="today weather nice", tick_index=0)
        runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.7}],
            emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
            runtime_action_effects={"moved": False},
            external_feedback={"reward": 0.4},
        )
        tick1 = runtime.process_text_tick(text="go outside", tick_index=1)
        tick2 = runtime.process_text_tick(text="come back", tick_index=2)
        metrics1 = dict((tick1.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        metrics2 = dict((tick2.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertEqual(float(metrics1.get("feedback.reward", 0.0) or 0.0), 0.4)
        self.assertEqual(float(metrics2.get("feedback.external_reward", 0.0) or 0.0), 0.0)

    def test_intrinsic_feedback_generates_reward_and_punishment_from_emotion_dynamics(self) -> None:
        runtime = RuntimeV2(config=load_config())
        first = runtime.build_intrinsic_feedback(
            emotion_channels={"expectation": 0.4, "pressure": 0.1, "correctness": 0.2, "dissonance": 0.0, "surprise": 0.0},
            balance_metrics={"alignment_score": 0.2, "grasp_score": 0.15, "overprediction_ratio": 0.2, "underprediction_ratio": 0.1},
        )
        second = runtime.build_intrinsic_feedback(
            emotion_channels={"expectation": 0.6, "pressure": 0.3, "correctness": 0.6, "dissonance": 0.5, "surprise": 0.4},
            balance_metrics={"alignment_score": 0.65, "grasp_score": 0.5, "overprediction_ratio": 0.4, "underprediction_ratio": 0.2},
        )
        self.assertTrue(first.get("enabled"))
        self.assertGreater(float(first.get("reward", 0.0) or 0.0), 0.0)
        self.assertGreater(float(second.get("reward", 0.0) or 0.0), 0.0)
        self.assertGreater(float(second.get("punishment", 0.0) or 0.0), 0.0)
        self.assertIn("intrinsic_correctness_delta_reward", list(second.get("notes", []) or []))
        self.assertIn("intrinsic_grasp_delta_reward", list(second.get("notes", []) or []))
        self.assertIn("intrinsic_dissonance_delta_punishment", list(second.get("notes", []) or []))
        self.assertIn("intrinsic_surprise_delta_punishment", list(second.get("notes", []) or []))

    def test_intrinsic_feedback_rewards_surprise_and_dissonance_recovery(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.build_intrinsic_feedback(
            emotion_channels={"expectation": 0.1, "pressure": 0.4, "correctness": 0.1, "dissonance": 0.8, "surprise": 0.7},
            balance_metrics={"alignment_score": 0.1, "grasp_score": 0.05, "overprediction_ratio": 0.8, "underprediction_ratio": 0.7},
        )
        recovered = runtime.build_intrinsic_feedback(
            emotion_channels={"expectation": 0.1, "pressure": 0.1, "correctness": 0.4, "dissonance": 0.2, "surprise": 0.1},
            balance_metrics={"alignment_score": 0.65, "grasp_score": 0.55, "overprediction_ratio": 0.2, "underprediction_ratio": 0.1},
        )
        notes = list(recovered.get("notes", []) or [])
        self.assertIn("intrinsic_surprise_recovery_reward", notes)
        self.assertIn("intrinsic_dissonance_recovery_reward", notes)
        self.assertGreater(float(recovered.get("reward", 0.0) or 0.0), 0.0)

    def test_first_novel_input_emits_surprise_and_repeated_input_builds_correctness(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick0 = runtime.process_text_tick(text="3", tick_index=0)
        emotion0 = dict((tick0.get("rules_result", {}) or {}).get("emotion_channels", {}) or {})
        metrics0 = dict((tick0.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertGreater(float(emotion0.get("surprise", 0.0) or 0.0), 0.0)
        self.assertGreater(float(metrics0.get("state.prediction_underprediction_mass", 0.0) or 0.0), 0.0)

        tick1 = runtime.process_text_tick(text="3", tick_index=1)
        emotion1 = dict((tick1.get("rules_result", {}) or {}).get("emotion_channels", {}) or {})
        metrics1 = dict((tick1.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        self.assertGreater(float(emotion1.get("correctness", 0.0) or 0.0), 0.0)
        self.assertGreater(float(metrics1.get("state.prediction_grasp_score", 0.0) or 0.0), 0.0)

    def test_queued_intrinsic_feedback_is_visible_on_next_tick_without_external_feedback(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick0 = runtime.process_text_tick(text="3", tick_index=0)
        queued0 = dict(tick0.get("queued_intrinsic_feedback_preview", {}) or {})
        self.assertTrue(bool(queued0.get("enabled", False)))
        tick1 = runtime.process_text_tick(text="3", tick_index=1)
        pending1 = dict(tick1.get("pending_feedback_metrics", {}) or {})
        breakdown1 = dict(tick1.get("pending_feedback_breakdown", {}) or {})
        sources1 = dict(breakdown1.get("sources", {}) or {})
        intrinsic1 = dict(sources1.get("intrinsic", {}) or {})
        self.assertGreater(float(pending1.get("punishment", 0.0) or 0.0), 0.0)
        self.assertGreater(float(intrinsic1.get("punishment", 0.0) or 0.0), 0.0)

        tick2 = runtime.process_text_tick(text="", tick_index=2)
        pending = dict(tick2.get("pending_feedback_metrics", {}) or {})
        breakdown = dict(tick2.get("pending_feedback_breakdown", {}) or {})
        sources = dict(breakdown.get("sources", {}) or {})
        intrinsic = dict(sources.get("intrinsic", {}) or {})
        self.assertGreater(float(pending.get("reward", 0.0) or 0.0), 0.0)
        self.assertGreater(float(intrinsic.get("reward", 0.0) or 0.0), 0.0)

    def test_repeated_new_input_eventually_builds_grasp_and_correctness(self) -> None:
        runtime = RuntimeV2(config=load_config())
        sequence = ["3", "3", ""] + ["8"] * 16
        best_grasp = 0.0
        best_committed_grasp = 0.0
        best_correctness = 0.0
        for tick_index, text in enumerate(sequence):
            tick = runtime.process_text_tick(text=text, tick_index=tick_index)
            metrics = dict((tick.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
            emotion = dict((tick.get("rules_result", {}) or {}).get("emotion_channels", {}) or {})
            if tick_index >= 3:
                best_grasp = max(best_grasp, float(metrics.get("state.prediction_grasp_score", 0.0) or 0.0))
                best_committed_grasp = max(best_committed_grasp, float(metrics.get("state.prediction_committed_grasp_score", 0.0) or 0.0))
                best_correctness = max(best_correctness, float(emotion.get("correctness", 0.0) or 0.0))
        self.assertGreater(best_grasp, 0.18)
        self.assertGreater(best_committed_grasp, 0.12)
        self.assertGreater(best_correctness, 0.12)

    def test_merged_feedback_breakdown_flows_into_next_tick_metrics(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="today weather nice", tick_index=0)
        intrinsic = runtime.build_intrinsic_feedback(
            emotion_channels={"expectation": 0.6, "pressure": 0.2, "correctness": 0.4, "dissonance": 0.1}
        )
        merged = runtime.merge_feedback_channels(
            external_feedback={"reward": 0.3, "notes": ["operator_reward"]},
            teacher_feedback={"punishment": 0.1, "notes": ["teacher_warn"]},
            intrinsic_feedback=intrinsic,
        )
        runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.7}],
            emotion_channels={"expectation": 0.6, "pressure": 0.2, "correctness": 0.4, "dissonance": 0.1},
            runtime_action_effects={"moved": True},
            external_feedback=merged,
        )
        tick1 = runtime.process_text_tick(text="go outside", tick_index=1)
        metrics = dict((tick1.get("rules_result", {}) or {}).get("metrics_snapshot", {}) or {})
        breakdown = dict(tick1.get("pending_feedback_breakdown", {}) or {})
        self.assertGreater(float(metrics.get("feedback.external_reward", 0.0) or 0.0), 0.0)
        self.assertGreaterEqual(float(metrics.get("feedback.teacher_punishment", 0.0) or 0.0), 0.1)
        self.assertGreater(float(metrics.get("feedback.intrinsic_reward", 0.0) or 0.0), 0.0)
        self.assertIn("sources", breakdown)
        self.assertIn("intrinsic", dict(breakdown.get("sources", {}) or {}))

    def test_context_bias_summary_is_visible_after_contextual_learning(self) -> None:
        runtime = RuntimeV2(config=load_config())
        for tick_index in range(4):
            runtime.action_learning.record_feedback(
                tick_index=tick_index,
                selected_actions=[{"action_id": "action::type_text", "action_name": "type_text", "drive": 0.7}],
                emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
                runtime_action_effects={"moved": False},
                external_feedback={"reward": 0.4},
                context_hints={"normalized_text": "打开 记事本", "context_keys": ["text::打开记事本"]},
            )
        summary = runtime.action_learning.context_bias_summary(limit=8)
        self.assertTrue(any(item.get("action_id") == "action::type_text" for item in summary))

    def test_tuner_learning_builds_long_term_control_bias_and_affects_next_tick(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.set_last_logic_ms(40.0)
        runtime.process_text_tick(text="some prompt for tuning", tick_index=0)
        feedback = runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.7}],
            emotion_channels={"expectation": 0.9, "pressure": 0.0, "correctness": 0.8, "dissonance": 0.0},
            runtime_action_effects={"moved": True},
            external_feedback={"reward": 0.4},
        )
        self.assertIn("tuner_learning_feedback", feedback)
        tick2 = runtime.process_text_tick(text="some followup for tuning", tick_index=1)
        tuner_learning_summary = tick2.get("tuner_learning_summary", {})
        self.assertIn("applied_offsets", tuner_learning_summary)
        self.assertGreaterEqual(len(tuner_learning_summary.get("target_bias_summary", [])), 1)
        self.assertGreater(tick2["runtime_controls"]["sampling.increment_budget"], 48.0)
        self.assertGreater(tick2["runtime_controls"]["attention.focus_gain"], 1.35)

    def test_action_planner_keeps_single_winner_per_actuator(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="打开 记事本", tick_index=0)
        planned = list((tick.get("rules_result", {}) or {}).get("planned_action_drives", []) or [])
        reports = list((tick.get("rules_result", {}) or {}).get("action_actuator_reports", []) or [])
        self.assertIsInstance(planned, list)
        self.assertIsInstance(reports, list)
        per_actuator: dict[str, int] = {}
        for item in (tick.get("rules_result", {}) or {}).get("planned_selected_actions_preview", []) or []:
            actuator_id = str(item.get("actuator_id", "") or "")
            per_actuator[actuator_id] = per_actuator.get(actuator_id, 0) + 1
        self.assertTrue(all(count <= 1 for count in per_actuator.values()))

    def test_action_planner_can_produce_hesitation_when_candidates_close(self) -> None:
        runtime = RuntimeV2(config=load_config())
        planner = runtime.action_planner
        view = planner.plan_actions(
            tick_index=0,
            raw_action_drives=[
                {"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.63, "params": {}},
                {"action_id": "action::inspect_residual", "action_name": "inspect_residual", "drive": 0.57, "params": {}},
            ],
            rules_result={"emotion_channels": {"expectation": 0.2, "pressure": 0.0, "correctness": 0.1, "dissonance": 0.1}},
            bn_list=[{"text": "今天 天气 有点 冷", "score": 0.8}],
            c_star={"items": [{"sa_label": "text::冷", "energy": 0.6}]},
            action_learning=runtime.action_learning,
            context_hints={"query_units": ["今天", "天气", "有点"], "focus_units": ["今天", "天气"], "context_keys": []},
            image_packet={"patches": []},
            pending_feedback={"reward": 0.0, "punishment": 0.0},
            recent_focus_units=["今天", "天气", "有点"],
        )
        reports = list(view.get("actuator_reports", []) or [])
        self.assertEqual(len(reports), 1)
        self.assertTrue(bool(reports[0].get("hesitation")))
        self.assertEqual(len(view.get("selected_actions_preview", []) or []), 0)

    def test_action_planner_keeps_single_winner_after_shared_inhibition(self) -> None:
        runtime = RuntimeV2(config=load_config())
        planner = runtime.action_planner
        view = planner.plan_actions(
            tick_index=0,
            raw_action_drives=[
                {"action_id": "action::continue_audio_focus", "action_name": "continue_audio_focus", "drive": 0.83, "params": {}},
                {"action_id": "action::inspect_audio_residual", "action_name": "inspect_audio_residual", "drive": 0.79, "params": {}},
            ],
            rules_result={"emotion_channels": {"surprise": 0.98, "dissonance": 1.0, "expectation": 0.0, "pressure": 1.0, "correctness": 0.0}},
            bn_list=[],
            c_star={"items": []},
            action_learning=runtime.action_learning,
            context_hints={"query_units": [], "focus_units": [], "context_keys": []},
            audio_packet={"windows": []},
            pending_feedback={"reward": 0.0, "punishment": 0.0},
            recent_focus_units=[],
        )
        selected = list(view.get("selected_actions_preview", []) or [])
        self.assertEqual(len(selected), 1)
        self.assertEqual(str(selected[0].get("action_name", "") or ""), "continue_audio_focus")

    def test_reset_transient_state_preserves_memory_store(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="today weather nice", tick_index=0)
        runtime.process_text_tick(text="today weather nice", tick_index=1)
        before = runtime.memory_store.count()
        self.assertGreater(before, 0)
        runtime.reset_transient_state()
        after = runtime.memory_store.count()
        self.assertEqual(after, before)
        self.assertEqual(len(runtime.short_term.snapshot()), 0)
        self.assertEqual(int(runtime.state_pool.snapshot_summary().get("state_pool_size", -1)), 0)

    def test_multimodal_tick_returns_latent_snapshot_and_latent_recall(self) -> None:
        runtime = RuntimeV2(config=load_config())
        first = runtime.process_text_tick(text="apple cold sweet", tick_index=0)
        self.assertIn("latent_snapshot_memory", first)
        self.assertEqual(str((first.get("latent_snapshot_memory", {}) or {}).get("memory_kind", "")), "latent_state_snapshot")

        second = runtime.process_text_tick(text="apple cold sweet", tick_index=1)
        latent_rows = list(second.get("latent_recall_list", []) or [])
        self.assertIsInstance(latent_rows, list)
        self.assertTrue(
            any(str((row or {}).get("memory_kind", "")) == "latent_state_snapshot" for row in latent_rows),
            msg=f"expected latent_state_snapshot in latent recall list, got: {latent_rows}",
        )

    def test_repeated_text_input_builds_surprise_habituation(self) -> None:
        runtime = RuntimeV2(config=load_config())
        first = runtime.process_text_tick(text="3", tick_index=0)
        second = runtime.process_text_tick(text="3", tick_index=1)
        third = runtime.process_text_tick(text="3", tick_index=2)

        raw_first = float(((first.get("rules_result", {}) or {}).get("raw_emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)
        eff_second = float(((second.get("rules_result", {}) or {}).get("emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)
        raw_second = float(((second.get("rules_result", {}) or {}).get("raw_emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)
        eff_third = float(((third.get("rules_result", {}) or {}).get("emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)
        hab_second = dict(((second.get("rules_result", {}) or {}).get("cognitive_feeling_habituation", {}) or {}))
        self.assertGreater(raw_first, 0.0)
        self.assertGreaterEqual(raw_second, 0.0)
        self.assertGreaterEqual(raw_second, eff_second)
        self.assertGreaterEqual(raw_second, eff_third)
        self.assertIn("state", hab_second)

    def test_dynamic_vision_keeps_motion_targeting_while_surprise_is_habituated(self) -> None:
        runtime = RuntimeV2(
            config=load_config(
                overrides={
                    "autonomous_teacher_enabled": False,
                    "autonomous_llm_gate_enabled": False,
                    "autonomous_external_teacher_enabled": False,
                    "vision_attention_boost_enabled": True,
                    "vision_patch_budget": 16,
                    "vision_focus_patch_budget": 8,
                    "vision_raw_state_budget": 64,
                    "vision_reconstruction_patch_budget": 1024,
                    "vision_attention_boost_max_extra_raw_budget": 192,
                    "vision_attention_boost_max_extra_focus_budget": 8,
                    "vision_dynamic_track_window": 6,
                    "vision_dynamic_candidate_limit_background": 12,
                    "vision_dynamic_candidate_limit_focus": 28,
                    "vision_dynamic_track_limit": 40,
                    "vision_dynamic_summary_limit": 4,
                    "vision_dynamic_match_threshold": 0.46,
                    "vision_dynamic_track_forget_ticks": 3,
                }
            )
        )
        runtime.vision_sensor.move_gaze(0.5, 0.5)
        static_bytes = self._vision_probe_png()
        motion_bytes = self._vision_probe_png(moving_rect=(146, 30, 170, 54))
        static_effective: list[float] = []
        static_raw: list[float] = []
        motion_tick = {}
        for tick_index in range(4):
            image_packet = runtime.vision_sensor.ingest_image_bytes(static_bytes, tick_index=tick_index, source_type="dyn_hab::static")
            tick = runtime.process_multimodal_tick(
                tick_index=tick_index,
                text_packet=runtime.text_sensor.ingest("", tick_index=tick_index, source_type="dyn_hab::static"),
                image_packet=image_packet,
                source_type="dyn_hab::static",
            )
            rules_result = dict(tick.get("rules_result", {}) or {})
            static_effective.append(float((rules_result.get("emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0))
            static_raw.append(float((rules_result.get("raw_emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0))

        motion_image_packet = runtime.vision_sensor.ingest_image_bytes(motion_bytes, tick_index=4, source_type="dyn_hab::motion")
        motion_tick = runtime.process_multimodal_tick(
            tick_index=4,
            text_packet=runtime.text_sensor.ingest("", tick_index=4, source_type="dyn_hab::motion"),
            image_packet=motion_image_packet,
            source_type="dyn_hab::motion",
        )
        motion_rules = dict((motion_tick.get("rules_result", {}) or {}))
        motion_effective = float((motion_rules.get("emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)
        motion_raw = float((motion_rules.get("raw_emotion_channels", {}) or {}).get("surprise", 0.0) or 0.0)
        motion_hab = dict((motion_rules.get("cognitive_feeling_habituation", {}) or {}))
        auto_reorient = dict(motion_rules.get("auto_visual_reorient", {}) or {})
        dynamic_count = int((((motion_tick.get("image_packet", {}) or {}).get("dynamic_track_summary", {}) or {}).get("object_count", 0) or 0))

        self.assertTrue(all(raw >= eff for raw, eff in zip(static_raw, static_effective)))
        self.assertLess(static_effective[-1], static_raw[-1])
        self.assertGreaterEqual(dynamic_count, 1)
        self.assertTrue(bool(auto_reorient))
        self.assertGreater(motion_raw, 0.0)
        self.assertLess(motion_effective, motion_raw)
        self.assertIn("gains", motion_hab)

    def test_dynamic_motion_can_disable_auto_surprise_reorient(self) -> None:
        runtime = RuntimeV2(
            config=load_config(
                overrides={
                    "autonomous_teacher_enabled": False,
                    "autonomous_llm_gate_enabled": False,
                    "autonomous_external_teacher_enabled": False,
                    "vision_attention_boost_enabled": True,
                    "vision_patch_budget": 16,
                    "vision_focus_patch_budget": 8,
                    "vision_raw_state_budget": 64,
                    "vision_reconstruction_patch_budget": 1024,
                    "vision_attention_boost_max_extra_raw_budget": 192,
                    "vision_attention_boost_max_extra_focus_budget": 8,
                    "vision_dynamic_track_window": 6,
                    "vision_dynamic_candidate_limit_background": 12,
                    "vision_dynamic_candidate_limit_focus": 28,
                    "vision_dynamic_track_limit": 40,
                    "vision_dynamic_summary_limit": 4,
                    "vision_dynamic_match_threshold": 0.46,
                    "vision_dynamic_track_forget_ticks": 3,
                    "vision_auto_surprise_reorient_enabled": False,
                }
            )
        )
        runtime.vision_sensor.move_gaze(0.5, 0.5)
        motion_bytes = self._vision_probe_png(moving_rect=(146, 30, 170, 54))
        motion_image_packet = runtime.vision_sensor.ingest_image_bytes(motion_bytes, tick_index=0, source_type="dyn_no_auto::motion")
        motion_tick = runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("", tick_index=0, source_type="dyn_no_auto::motion"),
            image_packet=motion_image_packet,
            source_type="dyn_no_auto::motion",
        )
        motion_rules = dict((motion_tick.get("rules_result", {}) or {}))
        self.assertFalse(bool(motion_rules.get("auto_visual_reorient")))

    def test_repeated_recall_can_build_time_feeling_and_export_query_spacetime(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="apple", tick_index=0)
        runtime.process_text_tick(text="", tick_index=1)
        runtime.process_text_tick(text="", tick_index=2)
        tick3 = runtime.process_text_tick(text="apple", tick_index=3)
        state_top_labels = [str(item.get("sa_label", "") or "") for item in ((tick3.get("state_pool_summary", {}) or {}).get("top", []) or [])]
        self.assertIn("timefelt::elapsed", state_top_labels)
        query_spacetime = dict(tick3.get("query_spacetime", {}) or {})
        self.assertGreater(float(query_spacetime.get("target_delta_t", 0.0) or 0.0), 0.0)
        self.assertGreater(float(query_spacetime.get("time_confidence", 0.0) or 0.0), 0.0)

    def test_dynamic_motion_generates_motion_feeling_item(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.vision_sensor.move_gaze(0.5, 0.5)
        static_bytes = self._vision_probe_png()
        motion_bytes = self._vision_probe_png(moving_rect=(146, 30, 170, 54))
        runtime.process_multimodal_tick(
            tick_index=0,
            text_packet=runtime.text_sensor.ingest("", tick_index=0, source_type="motionfelt::warmup"),
            image_packet=runtime.vision_sensor.ingest_image_bytes(static_bytes, tick_index=0, source_type="motionfelt::warmup"),
            source_type="motionfelt::warmup",
        )
        tick1 = runtime.process_multimodal_tick(
            tick_index=1,
            text_packet=runtime.text_sensor.ingest("", tick_index=1, source_type="motionfelt::probe"),
            image_packet=runtime.vision_sensor.ingest_image_bytes(motion_bytes, tick_index=1, source_type="motionfelt::probe"),
            source_type="motionfelt::probe",
        )
        feeling_labels = [str(item.get("sa_label", "") or "") for item in (tick1.get("channel_feeling_items", []) or [])]
        self.assertIn("motionfelt::trend", feeling_labels)
        trace = dict((tick1.get("channel_feeling_trace", {}) or {}).get("motion", {}) or {})
        self.assertGreater(float(trace.get("confidence", 0.0) or 0.0), 0.0)

    def test_feedback_signal_feeling_is_injected_from_pending_feedback(self) -> None:
        runtime = RuntimeV2(config=load_config())
        runtime.process_text_tick(text="hello", tick_index=0)
        runtime.apply_action_feedback(
            tick_index=0,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.7}],
            emotion_channels={"expectation": 0.0, "pressure": 0.0, "correctness": 0.0, "dissonance": 0.0},
            runtime_action_effects={"moved": False},
            external_feedback={"reward": 0.3},
        )
        tick1 = runtime.process_text_tick(text="world", tick_index=1)
        top_labels = [str(item.get("sa_label", "") or "") for item in ((tick1.get("state_pool_summary", {}) or {}).get("top", []) or [])]
        self.assertIn("attr::reward_signal", top_labels)

    def test_repeated_short_interval_input_can_build_rhythm_feelings(self) -> None:
        runtime = RuntimeV2(config=load_config())
        pulse_seen = False
        phase_seen = False
        for tick_index, text in enumerate(["beat", "", "beat", "", "beat", "", "beat"]):
            tick = runtime.process_text_tick(text=text, tick_index=tick_index)
            labels = [str(item.get("sa_label", "") or "") for item in (tick.get("channel_feeling_items", []) or [])]
            pulse_seen = pulse_seen or ("rhythmfelt::pulse" in labels)
            phase_seen = phase_seen or ("rhythmfelt::phase_expectation" in labels)
        self.assertTrue(pulse_seen)
        self.assertTrue(phase_seen)

    def test_rhythm_feeling_exports_query_spacetime_and_survives_checkpoint_payload(self) -> None:
        runtime = RuntimeV2(config=load_config())
        latest = {}
        for tick_index, text in enumerate(["tap", "", "tap", "", "tap", "", "tap"]):
            latest = runtime.process_text_tick(text=text, tick_index=tick_index)
        query_spacetime = dict(latest.get("query_spacetime", {}) or {})
        self.assertGreater(float(query_spacetime.get("rhythm_period_ticks", 0.0) or 0.0), 0.0)
        self.assertGreater(float(query_spacetime.get("rhythm_confidence", 0.0) or 0.0), 0.0)
        payload = runtime.export_payload()
        self.assertIn("rhythm_tracker", payload)
        restored = RuntimeV2(config=load_config())
        restored.import_payload(payload)
        restored_payload = restored.export_payload()
        self.assertIn("rhythm_tracker", restored_payload)
        self.assertGreaterEqual(len(dict(restored_payload.get("rhythm_tracker", {}) or {}).get("families", {})), 1)


if __name__ == "__main__":
    unittest.main()
