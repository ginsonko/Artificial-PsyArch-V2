# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory.memory_store_v2 import MemoryStoreV2


class MemoryStoreV2Tests(unittest.TestCase):
    def test_vector_backend_modes_are_visible_and_restoreable(self) -> None:
        store = MemoryStoreV2(vector_dim=64, vector_backend="numpy_flat", ann_enabled=True, ann_top_k=8)
        store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["apple"],
            items=[{"sa_label": "text::apple", "display_text": "apple", "energy": 1.0}],
            text="apple",
            reality_weight=1.0,
        )
        summary = store.index_summary()
        self.assertEqual(summary["vector"]["requested_backend"], "numpy_flat")
        self.assertEqual(summary["vector"]["effective_backend"], "numpy_flat")
        self.assertEqual(summary["vector"]["engine"], "numpy_flat")

        payload = store.export_payload()
        restored = MemoryStoreV2(vector_dim=64, vector_backend="auto", ann_enabled=True, ann_top_k=8)
        restored.import_payload(payload)
        restored_summary = restored.index_summary()
        self.assertEqual(restored_summary["vector"]["requested_backend"], "numpy_flat")
        self.assertEqual(restored_summary["vector"]["effective_backend"], "numpy_flat")

    def test_bundle_only_backend_still_supports_fallback_scan(self) -> None:
        store = MemoryStoreV2(vector_dim=64, vector_backend="bundle_only", ann_enabled=True, ann_top_k=8)
        store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["today", "weather"],
            items=[
                {"sa_label": "text::today", "display_text": "today", "energy": 1.0},
                {"sa_label": "text::weather", "display_text": "weather", "energy": 1.0},
            ],
            text="today weather",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["text::today"],
            query_weights={"text::today": 1.0},
            top_k=2,
            tick_index=1,
            query_units=["today"],
            recent_focus_units=["today"],
        )
        self.assertGreaterEqual(len(bn), 1)
        self.assertEqual(bn[0]["vector_engine"], "bundle_only_scan")
        self.assertIn("vector_ann", bn[0]["candidate_sources"])

    def test_vector_ann_and_posting_recall_are_visible(self) -> None:
        store = MemoryStoreV2(vector_dim=128, ann_enabled=True, ann_top_k=16, candidate_limit=32)
        store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["今天", "天气", "有点", "冷"],
            items=[
                {"sa_label": "text::今天", "display_text": "今天", "energy": 1.0},
                {"sa_label": "text::天气", "display_text": "天气", "energy": 1.0},
                {"sa_label": "text::有点", "display_text": "有点", "energy": 0.9},
                {"sa_label": "text::冷", "display_text": "冷", "energy": 1.0},
            ],
            text="今天 天气 有点 冷",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="focus_chain",
            units=["算了", "不说了"],
            items=[
                {"sa_label": "text::算了", "display_text": "算了", "energy": 1.0},
                {"sa_label": "text::不说了", "display_text": "不说了", "energy": 1.0},
            ],
            text="算了 不说了",
            reality_weight=0.6,
        )
        bn = store.recall_bn(
            query_labels=["text::今天", "text::天气", "text::有点"],
            query_weights={"text::今天": 1.0, "text::天气": 1.0, "text::有点": 0.9},
            top_k=4,
            tick_index=2,
            query_units=["今天", "天气", "有点"],
            recent_focus_units=["今天", "天气"],
        )
        self.assertGreaterEqual(len(bn), 1)
        top = bn[0]
        self.assertIn("vector_similarity", top["score_breakdown"])
        self.assertIn("vector_ann", top["candidate_sources"])

    def test_spacetime_prediction_branch_is_visible(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        first = store.write_memory(
            tick_index=10,
            memory_kind="visual",
            units=["苹果"],
            items=[
                {
                    "sa_label": "vision::apple",
                    "display_text": "苹果",
                    "energy": 1.0,
                    "coords": {"cx": 0.45, "cy": 0.52, "z": 0.0},
                    "channel": "vision",
                }
            ],
            text="苹果",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=11,
            memory_kind="visual",
            units=["香蕉"],
            items=[
                {
                    "sa_label": "vision::banana",
                    "display_text": "香蕉",
                    "energy": 1.0,
                    "coords": {"cx": 0.47, "cy": 0.5, "z": 0.0},
                    "channel": "vision",
                }
            ],
            text="香蕉",
            reality_weight=1.0,
        )
        c_i_list, c_star = store.build_prediction_branches(
            bn_list=[{"memory_id": first["memory_id"], "score": 0.8}],
            tick_index=12,
            recent_focus_units=["苹果"],
            max_neighbors=4,
        )
        self.assertGreaterEqual(len(c_i_list), 1)
        self.assertIn("summary", c_star)
        first_item = dict((c_star.get("items", []) or [])[0] or {})
        self.assertIn("commitment", first_item)
        self.assertIn("prediction_role", first_item)

    def test_branch_credibility_updates_and_exports(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["3"],
            items=[{"sa_label": "text::3", "display_text": "3", "energy": 1.0}],
            text="3",
            reality_weight=1.0,
        )
        c_i_list, _ = store.build_prediction_branches(
            bn_list=[{"memory_id": memory["memory_id"], "score": 0.9}],
            tick_index=1,
            recent_focus_units=["3"],
            max_neighbors=2,
        )
        update = store.update_branch_credibility(
            c_i_list=c_i_list,
            actual_items=[{"sa_label": "text::3", "display_text": "3", "energy": 1.0}],
            tick_index=1,
        )
        self.assertGreaterEqual(int(update.get("updated_count", 0) or 0), 1)
        payload = store.export_payload()
        self.assertIn("branch_credibility_state", payload)

    def test_query_spacetime_tokens_bias_visual_recall(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=True, ann_top_k=8, candidate_limit=16)
        store.write_memory(
            tick_index=0,
            memory_kind="visual",
            units=["苹果"],
            items=[
                {
                    "sa_label": "vision::apple_left",
                    "display_text": "苹果左",
                    "energy": 1.0,
                    "coords": {
                        "screen_x": 0.10,
                        "screen_y": 0.40,
                        "screen_w": 0.12,
                        "screen_h": 0.14,
                        "dx_from_gaze": -0.25,
                        "dy_from_gaze": 0.02,
                        "dr_from_gaze": 0.25,
                    },
                    "channel": "vision",
                }
            ],
            text="苹果左",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="visual",
            units=["苹果"],
            items=[
                {
                    "sa_label": "vision::apple_right",
                    "display_text": "苹果右",
                    "energy": 1.0,
                    "coords": {
                        "screen_x": 0.72,
                        "screen_y": 0.40,
                        "screen_w": 0.12,
                        "screen_h": 0.14,
                        "dx_from_gaze": 0.25,
                        "dy_from_gaze": 0.02,
                        "dr_from_gaze": 0.25,
                    },
                    "channel": "vision",
                }
            ],
            text="苹果右",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["vision::apple_left"],
            query_weights={"vision::apple_left": 1.0},
            top_k=2,
            tick_index=2,
            query_units=["苹果"],
            recent_focus_units=["苹果"],
            query_spacetime={
                "has_space": True,
                "x": 0.16,
                "y": 0.47,
                "z": 0.0,
                "screen_w": 0.12,
                "screen_h": 0.14,
                "has_relative_space": True,
                "rel_x": -0.25,
                "rel_y": 0.02,
                "rel_r": 0.25,
                "local_order_span": 0,
            },
        )
        self.assertGreaterEqual(len(bn), 1)
        self.assertIn("sp::rel_x::", " ".join(bn[0].get("query_vector_tokens", []) or []))

    def test_time_intent_can_bias_recall_toward_matching_elapsed_interval(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=2,
            memory_kind="exact_external",
            units=["eat", "apple"],
            items=[{"sa_label": "text::apple", "display_text": "apple", "energy": 1.0}],
            text="eat apple",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=12,
            memory_kind="exact_external",
            units=["eat", "banana"],
            items=[{"sa_label": "text::banana", "display_text": "banana", "energy": 1.0}],
            text="eat banana",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["text::eat"],
            query_weights={"text::eat": 1.0},
            top_k=2,
            tick_index=20,
            query_units=["eat"],
            recent_focus_units=["eat"],
            query_spacetime={
                "t": 20,
                "target_delta_t": 8.0,
                "time_sigma": 1.2,
                "time_confidence": 0.9,
                "time_recall_gain": 0.5,
            },
        )
        self.assertGreaterEqual(len(bn), 2)
        self.assertEqual(str(bn[0].get("text", "")), "eat banana")
        self.assertGreater(float((bn[0].get("score_breakdown", {}) or {}).get("time_intent_bonus", 0.0) or 0.0), 0.0)

    def test_rhythm_intent_can_bias_recall_toward_matching_period_and_family(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=2,
            memory_kind="exact_external",
            units=["beat"],
            items=[{"sa_label": "text::beat", "display_text": "beat", "energy": 1.0}],
            text="beat",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=8,
            memory_kind="exact_external",
            units=["beat"],
            items=[{"sa_label": "text::beat", "display_text": "beat", "energy": 1.0}],
            text="beat",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=11,
            memory_kind="exact_external",
            units=["offbeat"],
            items=[{"sa_label": "text::offbeat", "display_text": "offbeat", "energy": 1.0}],
            text="offbeat",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["text::beat"],
            query_weights={"text::beat": 1.0},
            top_k=3,
            tick_index=14,
            query_units=["beat"],
            recent_focus_units=["beat"],
            query_spacetime={
                "rhythm_period_ticks": 6.0,
                "rhythm_period_sigma": 1.2,
                "rhythm_confidence": 0.9,
                "rhythm_recall_gain": 0.5,
                "rhythm_family_key": "text::beat",
                "rhythm_time_to_next": 0.0,
            },
        )
        self.assertGreaterEqual(len(bn), 2)
        self.assertEqual(str(bn[0].get("text", "")), "beat")
        self.assertGreater(float((bn[0].get("score_breakdown", {}) or {}).get("rhythm_intent_bonus", 0.0) or 0.0), 0.0)

    def test_hearing_intent_can_bias_recall_toward_matching_audio_structure(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["tone"],
            items=[
                {
                    "sa_label": "audio::mem::tone_a",
                    "display_text": "听觉特征[tone_a]",
                    "energy": 1.0,
                    "channel": "hearing",
                    "sa_kind": "audio_memory_feature_unit",
                    "attributes": {
                        "sample_role": "memory_feature",
                        "memory_feature_code": "tone_a",
                        "tonal_clarity": 0.86,
                        "noisiness": 0.12,
                        "pitch_stability": 0.82,
                        "percussive_ratio": 0.08,
                        "harmonic_ratio": 0.78,
                        "voiced_probability": 0.74,
                        "spectral_contrast": 0.69,
                        "spectral_flatness": 0.18,
                        "spectral_bandwidth_ratio": 0.28,
                        "spectral_rolloff_ratio": 0.31,
                        "spectral_centroid_ratio": 0.24,
                        "dominant_hz": 880.0,
                        "dominant_band_index": 3,
                        "structure_profile": "tonal",
                    },
                }
            ],
            text="tone",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="exact_external",
            units=["noise"],
            items=[
                {
                    "sa_label": "audio::mem::noise_b",
                    "display_text": "听觉特征[noise_b]",
                    "energy": 1.0,
                    "channel": "hearing",
                    "sa_kind": "audio_memory_feature_unit",
                    "attributes": {
                        "sample_role": "memory_feature",
                        "memory_feature_code": "noise_b",
                        "tonal_clarity": 0.16,
                        "noisiness": 0.84,
                        "pitch_stability": 0.10,
                        "percussive_ratio": 0.24,
                        "harmonic_ratio": 0.12,
                        "voiced_probability": 0.08,
                        "spectral_contrast": 0.18,
                        "spectral_flatness": 0.82,
                        "spectral_bandwidth_ratio": 0.76,
                        "spectral_rolloff_ratio": 0.78,
                        "spectral_centroid_ratio": 0.72,
                        "dominant_hz": 4200.0,
                        "dominant_band_index": 9,
                        "structure_profile": "noisy",
                    },
                }
            ],
            text="noise",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["audio::mem::query"],
            query_weights={"audio::mem::query": 1.0},
            top_k=2,
            tick_index=2,
            query_units=["sound"],
            recent_focus_units=["sound"],
            query_spacetime={
                "hearing_confidence": 0.92,
                "hearing_timbre_center": 0.82,
                "hearing_timbre_sigma": 0.18,
                "hearing_timbre_recall_gain": 0.5,
                "hearing_pitch_stability_center": 0.78,
                "hearing_pitch_stability_sigma": 0.18,
                "hearing_pitch_recall_gain": 0.5,
                "hearing_dominant_hz": 880.0,
            },
        )
        self.assertGreaterEqual(len(bn), 2)
        self.assertEqual(str(bn[0].get("text", "")), "tone")
        self.assertGreater(float((bn[0].get("score_breakdown", {}) or {}).get("hearing_intent_bonus", 0.0) or 0.0), 0.0)

    def test_motion_intent_can_bias_recall_toward_matching_motion_speed(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        slow_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["move"],
            items=[
                {"sa_label": "text::move", "display_text": "move", "energy": 1.0},
                {
                    "sa_label": "vision_dyn::slow",
                    "display_text": "slow motion",
                    "energy": 0.8,
                    "attributes": {"motion_speed": 0.12},
                },
            ],
            text="move",
            reality_weight=1.0,
        )
        fast_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["move"],
            items=[
                {"sa_label": "text::move", "display_text": "move", "energy": 1.0},
                {
                    "sa_label": "vision_dyn::fast",
                    "display_text": "fast motion",
                    "energy": 0.8,
                    "attributes": {"motion_speed": 0.82},
                },
            ],
            text="move",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["text::move"],
            query_weights={"text::move": 1.0},
            top_k=2,
            tick_index=1,
            query_units=["move"],
            recent_focus_units=["move"],
            query_spacetime={
                "motion_center_speed": 0.78,
                "motion_sigma": 0.08,
                "motion_confidence": 0.9,
                "motion_recall_gain": 0.5,
            },
        )
        self.assertGreaterEqual(len(bn), 2)
        top = dict(bn[0] or {})
        self.assertEqual(str(top.get("memory_id", "")), str(fast_memory.get("memory_id", "")))
        self.assertGreater(float((top.get("score_breakdown", {}) or {}).get("motion_intent_bonus", 0.0) or 0.0), 0.0)
        self.assertNotEqual(str(top.get("memory_id", "")), str(slow_memory.get("memory_id", "")))

    def test_feedback_intent_can_bias_recall_toward_matching_valence(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        reward_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["feedback"],
            items=[
                {"sa_label": "text::feedback", "display_text": "feedback", "energy": 1.0},
                {"sa_label": "attr::reward_signal", "display_text": "reward", "energy": 0.42},
            ],
            text="feedback",
            reality_weight=1.0,
        )
        punishment_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["feedback"],
            items=[
                {"sa_label": "text::feedback", "display_text": "feedback", "energy": 1.0},
                {"sa_label": "attr::punishment_signal", "display_text": "punishment", "energy": 0.42},
            ],
            text="feedback",
            reality_weight=1.0,
        )
        bn_positive = store.recall_bn(
            query_labels=["text::feedback"],
            query_weights={"text::feedback": 1.0},
            top_k=2,
            tick_index=1,
            query_units=["feedback"],
            recent_focus_units=["feedback"],
            query_spacetime={
                "feedback_valence": 0.4,
                "feedback_sigma": 0.12,
                "feedback_confidence": 1.0,
                "feedback_recall_gain": 0.5,
            },
        )
        self.assertGreaterEqual(len(bn_positive), 2)
        self.assertEqual(str(bn_positive[0].get("memory_id", "")), str(reward_memory.get("memory_id", "")))
        self.assertGreater(float((bn_positive[0].get("score_breakdown", {}) or {}).get("feedback_intent_bonus", 0.0) or 0.0), 0.0)

        bn_negative = store.recall_bn(
            query_labels=["text::feedback"],
            query_weights={"text::feedback": 1.0},
            top_k=2,
            tick_index=1,
            query_units=["feedback"],
            recent_focus_units=["feedback"],
            query_spacetime={
                "feedback_valence": -0.4,
                "feedback_sigma": 0.12,
                "feedback_confidence": 1.0,
                "feedback_recall_gain": 0.5,
            },
        )
        self.assertGreaterEqual(len(bn_negative), 2)
        self.assertEqual(str(bn_negative[0].get("memory_id", "")), str(punishment_memory.get("memory_id", "")))
        self.assertGreater(float((bn_negative[0].get("score_breakdown", {}) or {}).get("feedback_intent_bonus", 0.0) or 0.0), 0.0)

    def test_visual_contour_similarity_prefers_matching_component_shape_and_color_signature(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        apple_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["apple"],
            items=[
                {
                    "sa_label": "vision_mem::global_contour::apple::0",
                    "display_text": "视觉轮廓[apple]",
                    "energy": 1.0,
                    "channel": "vision",
                    "sa_kind": "visual_global_feature_unit",
                    "attributes": {
                        "sample_role": "global_structure",
                        "hu_signature": "1334869",
                        "radial_signature": "22322222",
                        "proj_h_bin": "0332",
                        "proj_v_bin": "1331",
                        "radial_bin": "2232",
                        "quadrant_bin": "1233",
                        "edge_contact_bin": "0000",
                        "bbox_signature": "x1_y0_w1_h2",
                        "rgb_signature": "230",
                        "foreground_polarity": "bright",
                        "area_ratio": 0.1962,
                        "bbox_fill": 0.6522,
                        "solidity": 0.8921,
                        "roundness": 0.5428,
                        "aspect_ratio": 0.7799,
                        "hole_like": 0.0974,
                        "center_void": 0.3478,
                        "horizontal_symmetry": 0.65,
                        "vertical_symmetry": 0.9335,
                        "avg_r": 0.74,
                        "avg_g": 0.21,
                        "avg_b": 0.19,
                        "brightness": 0.36,
                    },
                }
            ],
            text="apple",
            reality_weight=1.0,
        )
        banana_memory = store.write_memory(
            tick_index=1,
            memory_kind="exact_external",
            units=["banana"],
            items=[
                {
                    "sa_label": "vision_mem::global_contour::banana::0",
                    "display_text": "视觉轮廓[banana]",
                    "energy": 1.0,
                    "channel": "vision",
                    "sa_kind": "visual_global_feature_unit",
                    "attributes": {
                        "sample_role": "global_structure",
                        "hu_signature": "0123646",
                        "radial_signature": "21122002",
                        "proj_h_bin": "1331",
                        "proj_v_bin": "2222",
                        "radial_bin": "2112",
                        "quadrant_bin": "2222",
                        "edge_contact_bin": "0100",
                        "bbox_signature": "x1_y1_w2_h0",
                        "rgb_signature": "220",
                        "foreground_polarity": "bright",
                        "area_ratio": 0.0586,
                        "bbox_fill": 0.5922,
                        "solidity": 0.7538,
                        "roundness": 0.3802,
                        "aspect_ratio": 3.2,
                        "hole_like": 0.1142,
                        "center_void": 0.4078,
                        "horizontal_symmetry": 0.5417,
                        "vertical_symmetry": 0.8901,
                        "avg_r": 0.77,
                        "avg_g": 0.69,
                        "avg_b": 0.20,
                        "brightness": 0.69,
                    },
                }
            ],
            text="banana",
            reality_weight=1.0,
        )
        bn = store.recall_bn(
            query_labels=["vision_mem::global_contour::query::0"],
            query_weights={"vision_mem::global_contour::query::0": 1.0},
            top_k=2,
            tick_index=2,
            query_units=["fruit"],
            recent_focus_units=["fruit"],
            query_items=[
                {
                    "sa_label": "vision_mem::global_contour::query::0",
                    "display_text": "视觉轮廓[query]",
                    "energy": 1.0,
                    "channel": "vision",
                    "sa_kind": "visual_global_feature_unit",
                    "attributes": {
                        "sample_role": "global_structure",
                        "hu_signature": "1334869",
                        "radial_signature": "22322222",
                        "proj_h_bin": "0332",
                        "proj_v_bin": "1331",
                        "radial_bin": "2232",
                        "quadrant_bin": "1233",
                        "edge_contact_bin": "0000",
                        "bbox_signature": "x1_y0_w1_h2",
                        "rgb_signature": "230",
                        "foreground_polarity": "bright",
                        "area_ratio": 0.19,
                        "bbox_fill": 0.65,
                        "solidity": 0.89,
                        "roundness": 0.54,
                        "aspect_ratio": 0.81,
                        "hole_like": 0.09,
                        "center_void": 0.35,
                        "horizontal_symmetry": 0.66,
                        "vertical_symmetry": 0.93,
                        "avg_r": 0.73,
                        "avg_g": 0.22,
                        "avg_b": 0.18,
                        "brightness": 0.36,
                    },
                }
            ],
        )
        self.assertGreaterEqual(len(bn), 2)
        self.assertEqual(str(bn[0].get("memory_id", "")), str(apple_memory.get("memory_id", "")))
        self.assertNotEqual(str(bn[0].get("memory_id", "")), str(banana_memory.get("memory_id", "")))
        self.assertGreater(float((bn[0].get("score_breakdown", {}) or {}).get("contour_similarity", 0.0) or 0.0), 0.7)

    def test_retrieval_aliases_enter_memory_vector_tokens_for_visual_and_audio_structure(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        visual_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["apple"],
            items=[
                {
                    "sa_label": "vision_mem::global_contour::apple::0",
                    "display_text": "visual contour apple",
                    "energy": 1.0,
                    "channel": "vision",
                    "sa_kind": "visual_global_feature_unit",
                    "attributes": {
                        "sample_role": "global_structure",
                        "hu_signature": "1334869",
                        "radial_signature": "22322222",
                        "proj_h_bin": "0332",
                        "proj_v_bin": "1331",
                        "radial_bin": "2232",
                        "quadrant_bin": "1233",
                        "edge_contact_bin": "0000",
                        "bbox_signature": "x1_y0_w1_h2",
                        "rgb_signature": "230",
                        "foreground_polarity": "bright",
                        "area_ratio": 0.19,
                        "bbox_fill": 0.65,
                        "solidity": 0.89,
                        "roundness": 0.54,
                        "aspect_ratio": 0.81,
                    },
                }
            ],
            text="apple",
            reality_weight=1.0,
        )
        audio_memory = store.write_memory(
            tick_index=1,
            memory_kind="exact_external",
            units=["banana"],
            items=[
                {
                    "sa_label": "audio::mem::banana::0",
                    "display_text": "audio banana",
                    "energy": 1.0,
                    "channel": "audio",
                    "sa_kind": "audio_memory_feature_unit",
                    "attributes": {
                        "sample_role": "memory_feature",
                        "structure_profile": "tonal",
                        "tonal_clarity": 0.84,
                        "noisiness": 0.12,
                        "pitch_stability": 0.76,
                        "harmonic_ratio": 0.73,
                        "percussive_ratio": 0.09,
                        "voiced_probability": 0.68,
                        "spectral_contrast": 0.54,
                        "spectral_flatness": 0.16,
                        "spectral_bandwidth_ratio": 0.31,
                        "spectral_rolloff_ratio": 0.28,
                        "spectral_centroid_ratio": 0.24,
                        "dominant_band_index": 1,
                        "dominant_hz": 320.0,
                    },
                }
            ],
            text="banana",
            reality_weight=1.0,
        )
        visual_tokens = " ".join(str(token or "") for token in (visual_memory.get("vector_tokens", []) or []))
        audio_tokens = " ".join(str(token or "") for token in (audio_memory.get("vector_tokens", []) or []))
        self.assertIn("vision_global_contour::", visual_tokens)
        self.assertTrue(
            "audio_core::" in audio_tokens or "audio_form::" in audio_tokens,
            msg=audio_tokens,
        )

    def test_memory_store_bundle_export_and_import_roundtrip(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["今天", "天气"],
            items=[
                {"sa_label": "text::今天", "display_text": "今天", "energy": 1.0},
                {"sa_label": "text::天气", "display_text": "天气", "energy": 1.0},
            ],
            text="今天 天气",
            reality_weight=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            result = store.save_deployment_bundle(Path(tmpdir))
            self.assertTrue(result["ok"])
            self.assertEqual(result["bundle_format"], "layered_v2")
            self.assertTrue((Path(tmpdir) / "bundle_meta.json").exists())
            self.assertTrue((Path(tmpdir) / "memories.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "posting_index.json").exists())
            self.assertTrue((Path(tmpdir) / "vector_index_meta.json").exists())
            self.assertTrue((Path(tmpdir) / "spacetime_index_meta.json").exists())
            restored = MemoryStoreV2(vector_dim=64, ann_enabled=False)
            loaded = restored.load_deployment_bundle(Path(tmpdir))
            self.assertTrue(loaded["ok"])
            self.assertEqual(loaded["loaded_via"], "layered_v2")
            self.assertEqual(restored.count(), 1)
            self.assertEqual(restored.index_summary()["vector"]["vector_count"], 1)

    def test_forget_score_prune_can_protect_kinds_and_energy(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=0,
            memory_kind="teacher_feedback",
            units=["warn"],
            items=[{"sa_label": "text::warn", "display_text": "warn", "energy": 2.4}],
            text="warn",
            reality_weight=0.9,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="focus_chain",
            units=["weak"],
            items=[{"sa_label": "text::weak", "display_text": "weak", "energy": 0.1}],
            text="weak",
            reality_weight=0.2,
        )
        store.write_memory(
            tick_index=2,
            memory_kind="exact_external",
            units=["latest"],
            items=[{"sa_label": "text::latest", "display_text": "latest", "energy": 0.6}],
            text="latest",
            reality_weight=0.3,
        )
        dry_run = store.forget_cold_memories(
            keep_latest=1,
            strategy="score_prune",
            protect_memory_kinds=["teacher_feedback"],
            min_total_item_energy=2.0,
            max_memory_count=2,
            dry_run=True,
        )
        self.assertTrue(dry_run["dry_run"])
        self.assertEqual(store.count(), 3)
        result = store.forget_cold_memories(
            keep_latest=1,
            strategy="score_prune",
            protect_memory_kinds=["teacher_feedback"],
            min_total_item_energy=2.0,
            max_memory_count=2,
        )
        self.assertEqual(result["memory_count"], 2)
        remaining_kinds = {memory.get("memory_kind", "") for memory in store._memories}
        self.assertIn("teacher_feedback", remaining_kinds)
        self.assertIn("exact_external", remaining_kinds)

    def test_repeated_recall_and_branch_build_hit_incremental_caches(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=True, ann_top_k=8, candidate_limit=16)
        first = store.write_memory(
            tick_index=0,
            memory_kind="visual",
            units=["apple", "left"],
            items=[
                {
                    "sa_label": "vision::apple_left",
                    "display_text": "apple_left",
                    "energy": 1.0,
                    "coords": {
                        "screen_x": 0.12,
                        "screen_y": 0.38,
                        "screen_w": 0.10,
                        "screen_h": 0.12,
                        "dx_from_gaze": -0.18,
                        "dy_from_gaze": 0.04,
                        "dr_from_gaze": 0.19,
                    },
                    "channel": "vision",
                }
            ],
            text="apple left",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="visual",
            units=["apple", "right"],
            items=[
                {
                    "sa_label": "vision::apple_right",
                    "display_text": "apple_right",
                    "energy": 1.0,
                    "coords": {
                        "screen_x": 0.70,
                        "screen_y": 0.40,
                        "screen_w": 0.10,
                        "screen_h": 0.12,
                        "dx_from_gaze": 0.18,
                        "dy_from_gaze": 0.04,
                        "dr_from_gaze": 0.19,
                    },
                    "channel": "vision",
                }
            ],
            text="apple right",
            reality_weight=1.0,
        )
        query_kwargs = {
            "query_labels": ["vision::apple_left"],
            "query_weights": {"vision::apple_left": 1.0},
            "top_k": 2,
            "tick_index": 2,
            "query_units": ["apple", "left"],
            "recent_focus_units": ["apple", "left"],
            "query_spacetime": {
                "has_space": True,
                "x": 0.17,
                "y": 0.44,
                "z": 0.0,
                "screen_w": 0.10,
                "screen_h": 0.12,
                "has_relative_space": True,
                "rel_x": -0.18,
                "rel_y": 0.04,
                "rel_r": 0.19,
                "local_order_span": 1,
            },
        }
        first_bn = store.recall_bn(**query_kwargs)
        second_bn = store.recall_bn(**query_kwargs)
        self.assertEqual(first_bn[0]["memory_id"], second_bn[0]["memory_id"])

        c_i_list_first, _ = store.build_prediction_branches(
            bn_list=[{"memory_id": first["memory_id"], "score": 0.8}],
            tick_index=3,
            recent_focus_units=["apple", "left"],
            max_neighbors=2,
        )
        c_i_list_second, _ = store.build_prediction_branches(
            bn_list=[{"memory_id": first["memory_id"], "score": 0.8}],
            tick_index=3,
            recent_focus_units=["apple", "left"],
            max_neighbors=2,
        )
        self.assertEqual(len(c_i_list_first), len(c_i_list_second))
        cache = store.cache_summary()
        self.assertGreaterEqual(int(cache["stats"].get("query_vector_hit", 0)), 1)
        self.assertGreaterEqual(int(cache["stats"].get("candidate_hit", 0)), 1)
        self.assertGreaterEqual(int(cache["stats"].get("neighbor_hit", 0)), 1)
        self.assertGreaterEqual(int(cache["stats"].get("pair_hit", 0)), 1)

    def test_write_memory_batch_commits_once_and_preserves_pair_relation_cache(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=True, ann_top_k=8, candidate_limit=16)
        rows = store.write_memory_batch(
            [
                {
                    "tick_index": 0,
                    "memory_kind": "visual",
                    "units": ["apple", "left"],
                    "items": [{"sa_label": "vision::apple_left", "display_text": "apple_left", "energy": 1.0, "channel": "vision"}],
                    "text": "apple left",
                    "reality_weight": 1.0,
                },
                {
                    "tick_index": 1,
                    "memory_kind": "visual",
                    "units": ["apple", "right"],
                    "items": [{"sa_label": "vision::apple_right", "display_text": "apple_right", "energy": 1.0, "channel": "vision"}],
                    "text": "apple right",
                    "reality_weight": 1.0,
                },
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(int(store.cache_summary()["memory_revision"]), 1)

        relation = store._pair_vector_related(rows[0]["memory_id"], rows[1]["memory_id"])
        self.assertGreaterEqual(relation, 0.0)
        cache_before = store.cache_summary()
        self.assertEqual(int(cache_before["pair_relation_cache_size"]), 1)

        store.write_memory_batch(
            [
                {
                    "tick_index": 2,
                    "memory_kind": "focus_chain",
                    "units": ["apple"],
                    "items": [{"sa_label": "text::apple", "display_text": "apple", "energy": 0.7, "channel": "text"}],
                    "text": "apple",
                    "reality_weight": 0.6,
                }
            ]
        )
        cache_after = store.cache_summary()
        self.assertEqual(int(cache_after["memory_revision"]), 2)
        self.assertEqual(int(cache_after["pair_relation_cache_size"]), 1)

    def test_recall_fatigue_lowers_immediate_repeat_and_recovers_later(self) -> None:
        store = MemoryStoreV2(
            vector_dim=64,
            ann_enabled=False,
            recall_fatigue_decay=0.8,
            recall_fatigue_gain=0.7,
            recall_fatigue_accumulate_scale=0.8,
            recall_fatigue_max=2.0,
            recall_fatigue_min_multiplier=0.2,
        )
        store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["winter", "cold"],
            items=[
                {"sa_label": "text::winter", "display_text": "winter", "energy": 1.0},
                {"sa_label": "text::cold", "display_text": "cold", "energy": 1.0},
            ],
            text="winter cold",
            reality_weight=1.0,
        )
        first = store.recall_bn(
            query_labels=["text::winter"],
            query_weights={"text::winter": 1.0},
            top_k=1,
            tick_index=1,
            query_units=["winter"],
            recent_focus_units=["winter"],
        )[0]
        second = store.recall_bn(
            query_labels=["text::winter"],
            query_weights={"text::winter": 1.0},
            top_k=1,
            tick_index=2,
            query_units=["winter"],
            recent_focus_units=["winter"],
        )[0]
        recovered = store.recall_bn(
            query_labels=["text::winter"],
            query_weights={"text::winter": 1.0},
            top_k=1,
            tick_index=10,
            query_units=["winter"],
            recent_focus_units=["winter"],
        )[0]
        self.assertGreater(float(first["score"]), float(second["score"]))
        self.assertLess(float(second["score_breakdown"]["recall_fatigue_multiplier"]), 1.0)
        self.assertGreater(float(recovered["score"]), float(second["score"]))

    def test_export_import_preserves_recall_fatigue_state(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["apple"],
            items=[{"sa_label": "text::apple", "display_text": "apple", "energy": 1.0}],
            text="apple",
            reality_weight=1.0,
        )
        store.recall_bn(
            query_labels=["text::apple"],
            query_weights={"text::apple": 1.0},
            top_k=1,
            tick_index=1,
            query_units=["apple"],
            recent_focus_units=["apple"],
        )
        payload = store.export_payload()
        restored = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        restored.import_payload(payload)
        self.assertIn(memory["memory_id"], restored._recall_fatigue)
        self.assertEqual(
            float(restored.index_summary()["recall_fatigue"]["gain"]),
            float(store.index_summary()["recall_fatigue"]["gain"]),
        )

    def test_latent_rows_contribute_to_c_star_without_entering_bn(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        explicit = store.write_memory(
            tick_index=0,
            memory_kind="focus_chain",
            units=["apple"],
            items=[{"sa_label": "text::apple", "display_text": "apple", "energy": 1.0}],
            text="apple",
            reality_weight=0.6,
        )
        latent = store.write_memory(
            tick_index=1,
            memory_kind="latent_state_snapshot",
            units=["apple"],
            items=[
                {"sa_label": "text::apple", "display_text": "apple", "energy": 1.0},
                {"sa_label": "vision::apple_shape", "display_text": "apple_shape", "energy": 0.8, "channel": "vision"},
            ],
            text="",
            reality_weight=0.95,
        )
        c_i_list, c_star = store.build_prediction_branches(
            bn_list=[{"memory_id": explicit["memory_id"], "score": 0.6}],
            tick_index=2,
            recent_focus_units=["apple"],
            max_neighbors=1,
            latent_candidates=[{"memory_id": latent["memory_id"], "memory_kind": "latent_state_snapshot", "score": 0.4}],
            latent_total_virtual_energy=0.4,
        )
        self.assertTrue(any(bool(item.get("is_latent_projection")) for item in c_i_list))
        labels = [str(item.get("sa_label", "") or "") for item in (c_star.get("items", []) or [])]
        self.assertIn("vision::apple_shape", labels)

    def test_prediction_branch_keeps_explicit_source_text_dominant_over_mixed_neighbor(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        mixed = store.write_memory(
            tick_index=0,
            memory_kind="focus_chain",
            units=["3", "8"],
            items=[
                {"sa_label": "text::3", "display_text": "3", "energy": 1.0},
                {"sa_label": "text::8", "display_text": "8", "energy": 0.35},
            ],
            text="3 8",
            reality_weight=0.6,
        )
        explicit = store.write_memory(
            tick_index=1,
            memory_kind="exact_external",
            units=["8"],
            items=[{"sa_label": "text::8", "display_text": "8", "energy": 1.0}],
            text="8",
            reality_weight=1.0,
        )
        _ = mixed
        c_i_list, c_star = store.build_prediction_branches(
            bn_list=[{"memory_id": explicit["memory_id"], "score": 0.7}],
            tick_index=2,
            recent_focus_units=["8"],
            max_neighbors=2,
        )
        self.assertGreaterEqual(len(c_i_list), 1)
        text_energy = {
            str(item.get("sa_label", "") or ""): float(item.get("energy", 0.0) or 0.0)
            for item in (c_star.get("items", []) or [])
            if str(item.get("sa_label", "") or "").startswith("text::")
        }
        self.assertGreater(text_energy.get("text::8", 0.0), text_energy.get("text::3", 0.0))

    def test_visual_memory_retrieval_aliases_support_shifted_structure_recall(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)

        def build_feature(label: str, *, opening_likeness: float, closure_likeness: float, hole_like: float, cx: float, cy: float) -> dict[str, object]:
            return {
                "sa_label": label,
                "display_text": label,
                "energy": 1.0,
                "channel": "vision",
                "sa_kind": "visual_focus_feature_unit",
                "coords": {
                    "cx": cx,
                    "cy": cy,
                    "screen_x": max(0.0, cx - 0.04),
                    "screen_y": max(0.0, cy - 0.05),
                    "screen_w": 0.08,
                    "screen_h": 0.10,
                    "dx_from_gaze": cx - 0.5,
                    "dy_from_gaze": cy - 0.5,
                    "dr_from_gaze": abs(cx - 0.5) + abs(cy - 0.5),
                },
                "attributes": {
                    "sample_role": "memory_feature",
                    "local_patch_signature": "112233445",
                    "edge_strength": 0.76,
                    "stroke_likeness": 0.68,
                    "endpoint_likeness": 0.24,
                    "corner_likeness": 0.38,
                    "opening_likeness": opening_likeness,
                    "closure_likeness": closure_likeness,
                    "arc_balance": 0.84,
                    "local_symmetry": 0.44,
                    "structure_discriminability": 0.73,
                    "opening_dir_x": 0.0,
                    "opening_dir_y": 1.0,
                    "opening_direction_strength": 0.18,
                    "straight_likeness": 0.18,
                    "curvilinear_likeness": 0.86,
                    "angularity": 0.21,
                    "roundness": 0.79,
                    "proj_h_bin": "1231",
                    "proj_v_bin": "2122",
                    "orient_hist_bin": "1320",
                    "radial_hist_bin": "2210",
                    "hole_like": hole_like,
                    "center_void": 0.72 if hole_like > 0.5 else 0.18,
                    "horizontal_symmetry": 0.58,
                    "vertical_symmetry": 0.66,
                },
            }

        three_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["three"],
            items=[
                {"sa_label": "text::three", "display_text": "three", "energy": 1.0, "channel": "text"},
                build_feature("vision_mem::three_pos_a", opening_likeness=0.34, closure_likeness=0.26, hole_like=0.10, cx=0.22, cy=0.46),
            ],
            text="three",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="exact_external",
            units=["eight"],
            items=[
                {"sa_label": "text::eight", "display_text": "eight", "energy": 1.0, "channel": "text"},
                build_feature("vision_mem::eight_pos_a", opening_likeness=0.06, closure_likeness=0.82, hole_like=0.88, cx=0.70, cy=0.48),
            ],
            text="eight",
            reality_weight=1.0,
        )

        shifted_query_feature = build_feature(
            "vision_mem::three_pos_b",
            opening_likeness=0.34,
            closure_likeness=0.26,
            hole_like=0.10,
            cx=0.74,
            cy=0.48,
        )
        bn = store.recall_bn(
            query_labels=[str(shifted_query_feature["sa_label"])],
            query_weights={str(shifted_query_feature["sa_label"]): 1.0},
            top_k=2,
            tick_index=2,
            query_items=[shifted_query_feature],
            query_units=[],
            recent_focus_units=[],
            query_spacetime={
                "has_space": True,
                "x": 0.74,
                "y": 0.48,
                "z": 0.0,
                "screen_w": 0.08,
                "screen_h": 0.10,
                "has_relative_space": True,
                "rel_x": 0.24,
                "rel_y": -0.02,
                "rel_r": 0.26,
                "local_order_span": 0,
            },
        )
        self.assertGreaterEqual(len(bn), 1)
        self.assertEqual(str(bn[0].get("memory_id", "") or ""), str(three_memory.get("memory_id", "") or ""))

    def test_dynamic_visual_retrieval_aliases_support_shifted_structure_recall(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)

        def build_dynamic(label: str, *, opening_likeness: float, closure_likeness: float, hole_like: float, cx: float, cy: float) -> dict[str, object]:
            return {
                "sa_label": label,
                "display_text": label,
                "energy": 0.9,
                "channel": "vision",
                "sa_kind": "visual_dynamic_track_unit",
                "coords": {
                    "cx": cx,
                    "cy": cy,
                    "screen_x": max(0.0, cx - 0.06),
                    "screen_y": max(0.0, cy - 0.08),
                    "screen_w": 0.12,
                    "screen_h": 0.16,
                    "dx_from_gaze": cx - 0.5,
                    "dy_from_gaze": cy - 0.5,
                    "dr_from_gaze": abs(cx - 0.5) + abs(cy - 0.5),
                },
                "attributes": {
                    "sample_role": "dynamic_motion_summary",
                    "track_id": label.split("::", 1)[-1],
                    "edge_strength": 0.78,
                    "stroke_likeness": 0.71,
                    "endpoint_likeness": 0.22,
                    "corner_likeness": 0.36,
                    "opening_likeness": opening_likeness,
                    "closure_likeness": closure_likeness,
                    "arc_balance": 0.83,
                    "structure_discriminability": 0.76,
                    "straight_likeness": 0.20,
                    "curvilinear_likeness": 0.84,
                    "angularity": 0.19,
                    "roundness": 0.81,
                    "local_symmetry": 0.46,
                    "horizontal_symmetry": 0.57,
                    "vertical_symmetry": 0.69,
                    "opening_dir_x": 0.0,
                    "opening_dir_y": 1.0,
                    "opening_direction_strength": 0.21,
                    "hole_like": hole_like,
                    "center_void": 0.74 if hole_like > 0.5 else 0.16,
                    "proj_h_bin": "1231",
                    "proj_v_bin": "2122",
                    "orient_hist_bin": "1320",
                    "radial_hist_bin": "2210",
                    "radial_bin": "2210",
                    "quadrant_bin": "1122",
                    "foreground_polarity": "bright",
                    "local_patch_signature": "112233445",
                    "dynamic_objectness": 0.82,
                    "motion_speed": 0.18,
                    "motion_coherence": 0.71,
                    "boundary_motion_contrast": 0.33,
                    "shape_stability": 0.77,
                    "bbox_fill": 0.62,
                    "aspect_ratio": 0.72,
                    "area_ratio": 0.03,
                },
            }

        three_memory = store.write_memory(
            tick_index=0,
            memory_kind="exact_external",
            units=["three"],
            items=[
                {"sa_label": "text::three", "display_text": "three", "energy": 1.0, "channel": "text"},
                build_dynamic("vision_dyn::trk_three_a", opening_likeness=0.35, closure_likeness=0.25, hole_like=0.08, cx=0.22, cy=0.46),
            ],
            text="three",
            reality_weight=1.0,
        )
        store.write_memory(
            tick_index=1,
            memory_kind="exact_external",
            units=["eight"],
            items=[
                {"sa_label": "text::eight", "display_text": "eight", "energy": 1.0, "channel": "text"},
                build_dynamic("vision_dyn::trk_eight_a", opening_likeness=0.05, closure_likeness=0.85, hole_like=0.92, cx=0.70, cy=0.48),
            ],
            text="eight",
            reality_weight=1.0,
        )

        shifted_query_dynamic = build_dynamic(
            "vision_dyn::trk_three_b",
            opening_likeness=0.35,
            closure_likeness=0.25,
            hole_like=0.08,
            cx=0.74,
            cy=0.48,
        )
        bn = store.recall_bn(
            query_labels=[str(shifted_query_dynamic["sa_label"])],
            query_weights={str(shifted_query_dynamic["sa_label"]): 1.0},
            top_k=2,
            tick_index=2,
            query_items=[shifted_query_dynamic],
            query_units=["three"],
            recent_focus_units=["three"],
        )
        self.assertGreaterEqual(len(bn), 1)
        self.assertEqual(str(bn[0].get("memory_id", "") or ""), str(three_memory.get("memory_id", "") or ""))
        overlap = set(str(item or "") for item in (bn[0].get("overlap_labels", []) or []) if str(item or ""))
        self.assertTrue(any(label.startswith(("vision_core::", "vision_form::")) for label in overlap))


if __name__ == "__main__":
    unittest.main()
