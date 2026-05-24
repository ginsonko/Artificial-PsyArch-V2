# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import struct
import unittest
import wave
from io import BytesIO
from unittest import mock

from PIL import Image

from sensors.hearing_sensor_v1 import HearingSensorV1
from sensors.vision_sensor_v1 import VisionSensorV1


class MultimodalSensorsPhase11To12Tests(unittest.TestCase):
    def test_vision_sensor_emits_bounded_patch_packet(self) -> None:
        image = Image.new("RGB", (64, 64), color=(20, 20, 20))
        for x in range(24, 40):
            for y in range(24, 40):
                image.putpixel((x, y), (255, 255, 255))
        buf = BytesIO()
        image.save(buf, format="PNG")
        packet = VisionSensorV1(patch_budget=8, focus_patch_budget=4, raw_state_budget=64, reconstruction_patch_budget=64).ingest_image_bytes(buf.getvalue(), tick_index=0)
        self.assertEqual(packet["schema_id"], "vision_sensor_packet/v1")
        self.assertLessEqual(packet["budget_used"], 8)
        self.assertGreater(len(packet["patches"]), 0)
        self.assertIn("stream_state", packet)
        self.assertIn("grid", packet)
        self.assertEqual(packet["cognitive_patch_budget"], 8)
        self.assertEqual(int(packet.get("raw_state_budget", 0) or 0), 64)
        self.assertGreaterEqual(int(packet.get("total_patch_count", 0) or 0), int(packet.get("budget_used", 0) or 0))
        self.assertIn("reconstruction_grid", packet)
        self.assertGreaterEqual(int(((packet.get("reconstruction_grid", {}) or {}).get("cell_count", 0) or 0)), 32)
        self.assertIn("raw_samples", packet)
        self.assertIn("memory_write_samples", packet)
        self.assertIn("focus_priority_samples", packet)
        self.assertIn("global_structure_samples", packet)
        self.assertGreater(len(packet.get("global_structure_samples", []) or []), 0)
        self.assertIn("shape_candidates", packet)
        self.assertIn("dynamic_tracks", packet)
        self.assertIn("dynamic_motion_samples", packet)
        self.assertIn("dynamic_track_summary", packet)
        self.assertGreaterEqual(int(((packet.get("global_structure_summary", {}) or {}).get("count", 0) or 0)), 1)
        self.assertIn("contour_reconstruction", packet)
        contour = dict(packet.get("contour_reconstruction", {}) or {})
        self.assertIn("composite_data_url", contour)
        self.assertIn("luma_edges_data_url", contour)
        self.assertIn("color_edges_data_url", contour)
        self.assertIn("motion_mask_data_url", contour)
        self.assertIn("motion_outline_data_url", contour)
        self.assertIn("motion_composite_data_url", contour)
        self.assertTrue(str(contour.get("composite_data_url", "") or "").startswith("data:image/png;base64,"))
        self.assertIn("fixation_buffer", packet)
        self.assertIn("preview_image", packet)
        self.assertGreater(len(packet.get("raw_samples", []) or []), 0)
        self.assertGreaterEqual(len(packet.get("raw_samples", []) or []), len(packet.get("patches", []) or []))
        self.assertLessEqual(len(packet.get("patches", []) or []), len(packet.get("raw_samples", []) or []))
        top = packet["patches"][0]
        coords = dict(top.get("coords", {}) or {})
        self.assertIn("screen_x", coords)
        self.assertIn("screen_y", coords)
        self.assertIn("screen_w", coords)
        self.assertIn("screen_h", coords)
        self.assertIn("dx_from_gaze", coords)
        self.assertIn("dy_from_gaze", coords)
        self.assertIn("dr_from_gaze", coords)
        attrs = dict(top.get("attributes", {}) or {})
        self.assertIn("endpoint_likeness", attrs)
        self.assertIn("corner_likeness", attrs)
        self.assertIn("opening_likeness", attrs)
        self.assertIn("closure_likeness", attrs)
        self.assertIn("arc_balance", attrs)
        self.assertIn("straight_likeness", attrs)
        self.assertIn("curvilinear_likeness", attrs)
        self.assertIn("angularity", attrs)
        self.assertIn("roundness", attrs)
        self.assertIn("local_symmetry", attrs)
        self.assertIn("opening_dir_x", attrs)
        self.assertIn("opening_dir_y", attrs)
        self.assertIn("opening_direction_strength", attrs)
        self.assertIn("structure_discriminability", attrs)
        self.assertIn("structure_priority", attrs)
        self.assertTrue(any(str(item.get("sa_label", "")).startswith("vision_mem::") for item in (packet.get("memory_write_samples", []) or [])))
        feature_item = next(
            (item for item in (packet.get("memory_write_samples", []) or []) if str(item.get("sa_label", "")).startswith("vision_mem::")),
            None,
        )
        self.assertIsNotNone(feature_item)
        feature_code = str(((feature_item or {}).get("attributes", {}) or {}).get("memory_feature_code", "") or "")
        self.assertIn("_f", feature_code)
        self.assertIn("_g", feature_code)
        self.assertIn("_y", feature_code)
        self.assertIn("_d", feature_code)
        global_feature = next(
            (item for item in (packet.get("global_structure_samples", []) or []) if str(item.get("sa_label", "")).startswith("vision_mem::global_")),
            None,
        )
        self.assertIsNotNone(global_feature)
        self.assertEqual(str((global_feature or {}).get("sa_kind", "") or ""), "visual_global_feature_unit")
        self.assertEqual(str((((global_feature or {}).get("attributes", {}) or {}).get("sample_role", "") or "")), "global_structure")
        dyn_summary = dict(packet.get("dynamic_track_summary", {}) or {})
        self.assertIn("track_count", dyn_summary)
        self.assertIn("dynamic_salience_mean", dyn_summary)

    def test_hearing_sensor_emits_bounded_window_packet(self) -> None:
        sample_rate = 8000
        duration_sec = 0.2
        frames = bytearray()
        for i in range(int(sample_rate * duration_sec)):
            sample = int(12000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames += struct.pack("<h", sample)
        buf = BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(frames))
        packet = HearingSensorV1(window_budget=6).ingest_wav_bytes(buf.getvalue(), tick_index=0)
        self.assertEqual(packet["schema_id"], "hearing_sensor_packet/v2")
        self.assertLessEqual(packet["budget_used"], 6)
        self.assertGreater(len(packet["windows"]), 0)
        self.assertTrue(str(packet.get("preview_wav_b64", "") or ""))
        self.assertGreater(int(packet.get("preview_audio_bytes_len", 0) or 0), 0)
        self.assertTrue(str(packet.get("proxy_preview_wav_b64", "") or ""))
        self.assertGreater(int(packet.get("proxy_preview_audio_bytes_len", 0) or 0), 0)
        self.assertGreater(float(packet.get("preview_duration_ms", 0.0) or 0.0), 0.0)
        self.assertIn("stream_state", packet)
        self.assertIn("audio_focus", packet)
        self.assertIn("attention_boost", packet)
        self.assertIn("focus_priority_samples", packet)
        self.assertIn("memory_write_samples", packet)
        self.assertIn("feature_summary", packet)
        self.assertIn("global_structure_samples", packet)
        top = packet["windows"][0]
        attrs = dict(top.get("attributes", {}) or {})
        self.assertIn("spectral_flatness", attrs)
        self.assertIn("spectral_contrast", attrs)
        self.assertIn("spectral_bandwidth_ratio", attrs)
        self.assertIn("spectral_rolloff_ratio", attrs)
        self.assertIn("tonal_clarity", attrs)
        self.assertIn("noisiness", attrs)
        self.assertIn("pitch_stability", attrs)
        self.assertIn("harmonic_ratio", attrs)
        self.assertIn("percussive_ratio", attrs)
        self.assertIn("voiced_probability", attrs)
        self.assertIn("signal_presence", attrs)
        self.assertIn("signal_gate", attrs)
        self.assertTrue(any(str(item.get("sa_label", "") or "").startswith("audio::mem::") for item in (packet.get("memory_write_samples", []) or [])))
        self.assertTrue(any(str(item.get("sa_label", "") or "").startswith("audio::global::") for item in (packet.get("global_structure_samples", []) or [])))
        self.assertGreater(float(attrs.get("signal_presence", 0.0) or 0.0), 0.0)
        self.assertGreater(float(((packet.get("feature_summary", {}) or {}).get("signal_presence", 0.0) or 0.0)), 0.0)

    def test_hearing_sensor_attention_boost_moves_focus_and_expands_sampling(self) -> None:
        sample_rate = 8000
        duration_sec = 0.2
        frames = bytearray()
        for i in range(int(sample_rate * duration_sec)):
            sample = int(12000 * math.sin(2 * math.pi * 880 * i / sample_rate))
            frames += struct.pack("<h", sample)
        buf = BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(frames))
        sensor = HearingSensorV1(
            window_budget=4,
            focus_band_count=6,
            attention_boost_enabled=True,
            attention_boost_max_extra_window_budget=8,
            attention_boost_max_extra_focus_budget=6,
            attention_boost_min_bandwidth_scale=0.35,
            attention_boost_focus_gain=1.4,
        )
        base = sensor.ingest_wav_bytes(buf.getvalue(), tick_index=0)
        sensor.move_audio_focus(880.0, bandwidth_octaves=0.8)
        sensor.apply_attention_boost(source_action="continue_audio_focus", firmness_norm=1.0, target_center_hz=880.0, target_bandwidth_octaves=0.8)
        boosted = sensor.ingest_wav_bytes(buf.getvalue(), tick_index=1)
        self.assertGreaterEqual(int((boosted.get("stream_state", {}) or {}).get("effective_window_budget", 0) or 0), int((base.get("stream_state", {}) or {}).get("effective_window_budget", 0) or 0))
        self.assertGreaterEqual(int((boosted.get("stream_state", {}) or {}).get("effective_focus_band_budget", 0) or 0), int((base.get("stream_state", {}) or {}).get("effective_focus_band_budget", 0) or 0))
        self.assertTrue(bool((boosted.get("attention_boost", {}) or {}).get("active", False)))

    def test_streaming_sensors_preserve_short_term_state(self) -> None:
        image1 = Image.new("RGB", (64, 64), color=(15, 15, 15))
        image2 = Image.new("RGB", (64, 64), color=(15, 15, 15))
        for x in range(20, 44):
            for y in range(20, 44):
                image2.putpixel((x, y), (240, 240, 240))
        buf1 = BytesIO()
        buf2 = BytesIO()
        image1.save(buf1, format="PNG")
        image2.save(buf2, format="PNG")
        vision = VisionSensorV1(patch_budget=8, focus_patch_budget=4, raw_state_budget=64, reconstruction_patch_budget=64)
        p1 = vision.ingest_image_bytes(buf1.getvalue(), tick_index=0)
        p2 = vision.ingest_image_bytes(buf2.getvalue(), tick_index=1)
        self.assertLessEqual(p2["budget_used"], 8)
        self.assertGreaterEqual(p2["stream_state"]["frame_index"], 1)
        self.assertTrue(any(float((item.get("attributes", {}) or {}).get("motion", 0.0) or 0.0) > 0.0 for item in p2["patches"]))
        self.assertGreater(int(((p2.get("reconstruction_grid", {}) or {}).get("cell_count", 0) or 0)), 0)
        fixation_count = int(((p2.get("fixation_buffer", {}) or {}).get("cell_count", 0) or 0))
        self.assertGreater(fixation_count, 0)
        self.assertGreaterEqual(len(p2.get("raw_samples", []) or []), 32)
        self.assertIn("dynamic_motion_samples", p2)
        self.assertIn("dynamic_track_summary", p2)
        self.assertGreaterEqual(int(((p2.get("dynamic_track_summary", {}) or {}).get("track_count", 0) or 0)), 0)

        sample_rate = 8000
        frames1 = bytearray()
        frames2 = bytearray()
        for i in range(int(sample_rate * 0.08)):
            frames1 += struct.pack("<h", int(3000 * math.sin(2 * math.pi * 220 * i / sample_rate)))
            frames2 += struct.pack("<h", int(16000 * math.sin(2 * math.pi * 660 * i / sample_rate)))
        audio_sensor = HearingSensorV1(window_budget=4)
        buf_a = BytesIO()
        buf_b = BytesIO()
        with wave.open(buf_a, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(frames1))
        with wave.open(buf_b, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(frames2))
        a1 = audio_sensor.ingest_wav_bytes(buf_a.getvalue(), tick_index=0)
        a2 = audio_sensor.ingest_wav_bytes(buf_b.getvalue(), tick_index=1)
        self.assertGreaterEqual(a2["stream_state"]["chunk_index"], 1)
        self.assertLessEqual(a2["budget_used"], 4)

    def test_vision_sensor_background_mode_caps_raw_budget_until_focus_boost(self) -> None:
        image = Image.new("RGB", (256, 256), color=(30, 30, 30))
        for x in range(48, 208):
            for y in range(80, 176):
                image.putpixel((x, y), (230, 230, 230))
        buf = BytesIO()
        image.save(buf, format="PNG")
        sensor = VisionSensorV1(
            patch_budget=48,
            focus_patch_budget=24,
            raw_state_budget=1024,
            reconstruction_patch_budget=2048,
            attention_boost_enabled=True,
            attention_boost_max_extra_raw_budget=256,
            attention_boost_max_extra_focus_budget=12,
        )
        base_packet = sensor.ingest_image_bytes(buf.getvalue(), tick_index=0)
        self.assertLessEqual(len(base_packet.get("raw_samples", []) or []), 64)
        self.assertLessEqual(len(base_packet.get("memory_write_samples", []) or []), 8)
        self.assertEqual(int(base_packet.get("raw_state_budget", 0) or 0), 64)

        sensor.apply_attention_boost(source_action="continue_focus", firmness_norm=1.0, target_gaze=(0.5, 0.5))
        boosted_packet = sensor.ingest_image_bytes(buf.getvalue(), tick_index=1)
        self.assertGreaterEqual(len(boosted_packet.get("raw_samples", []) or []), 200)
        self.assertLessEqual(len(boosted_packet.get("memory_write_samples", []) or []), 48)
        self.assertLessEqual(len(boosted_packet.get("focus_priority_samples", []) or []), 36)
        self.assertEqual(int(boosted_packet.get("raw_state_budget", 0) or 0), 256)

    def test_vision_sensor_edge_biased_sampling_prefers_high_contrast_band(self) -> None:
        image = Image.new("RGB", (96, 96), color=(20, 20, 20))
        for x in range(44, 52):
            for y in range(0, 96):
                image.putpixel((x, y), (250, 250, 250))
        buf = BytesIO()
        image.save(buf, format="PNG")
        sensor = VisionSensorV1(
            patch_budget=24,
            focus_patch_budget=12,
            raw_state_budget=256,
            reconstruction_patch_budget=512,
            edge_candidate_gain=2.4,
            edge_priority_gain=1.8,
        )
        packet = sensor.ingest_image_bytes(buf.getvalue(), tick_index=0)
        raw_samples = list(packet.get("raw_samples", []) or [])
        near_edge = [
            item
            for item in raw_samples
            if abs(int((item.get("coords", {}) or {}).get("pixel_x", -999) or -999) - 48) <= 8
        ]
        self.assertGreater(len(raw_samples), 0)
        self.assertGreaterEqual(len(near_edge), max(24, len(raw_samples) // 8))
        self.assertTrue(
            any(float((item.get("attributes", {}) or {}).get("edge_priority", 0.0) or 0.0) > 0.7 for item in near_edge)
        )

    def test_vision_sensor_reuses_cached_contour_bundle_for_identical_frame(self) -> None:
        image = Image.new("RGB", (96, 96), color=(18, 18, 18))
        for x in range(20, 76):
            for y in range(20, 76):
                image.putpixel((x, y), (220, 48, 48))
        buf = BytesIO()
        image.save(buf, format="PNG")
        raw = buf.getvalue()
        sensor = VisionSensorV1(
            patch_budget=16,
            focus_patch_budget=8,
            raw_state_budget=96,
            reconstruction_patch_budget=256,
        )
        with mock.patch.object(sensor, "_build_contour_bundle", wraps=sensor._build_contour_bundle) as wrapped_build:
            first = sensor.ingest_image_bytes(raw, tick_index=0)
            second = sensor.ingest_image_bytes(raw, tick_index=1)
        self.assertEqual(wrapped_build.call_count, 1)
        self.assertEqual(
            str(((first.get("contour_reconstruction", {}) or {}).get("composite_data_url", "") or "")),
            str(((second.get("contour_reconstruction", {}) or {}).get("composite_data_url", "") or "")),
        )
        self.assertEqual(
            str(((first.get("preview_image", {}) or {}).get("data_url", "") or "")),
            str(((second.get("preview_image", {}) or {}).get("data_url", "") or "")),
        )

    def test_attention_boost_increases_next_tick_sampling_budget(self) -> None:
        image = Image.new("RGB", (96, 96), color=(25, 25, 25))
        for x in range(36, 60):
            for y in range(32, 64):
                image.putpixel((x, y), (255, 255, 255))
        buf = BytesIO()
        image.save(buf, format="PNG")
        sensor = VisionSensorV1(
            patch_budget=24,
            focus_patch_budget=12,
            raw_state_budget=128,
            reconstruction_patch_budget=256,
            edge_candidate_gain=2.0,
            edge_priority_gain=1.4,
            attention_boost_enabled=True,
            attention_boost_max_extra_raw_budget=256,
            attention_boost_max_extra_focus_budget=12,
            attention_boost_min_radius_scale=0.25,
            attention_boost_edge_gain=1.5,
            attention_boost_gaze_sigma_scale=0.45,
        )
        base = sensor.ingest_image_bytes(buf.getvalue(), tick_index=0)
        sensor.apply_attention_boost(source_action="continue_focus", firmness_norm=1.0, target_gaze=(0.5, 0.5))
        boosted = sensor.ingest_image_bytes(buf.getvalue(), tick_index=1)
        self.assertGreater(int(boosted.get("raw_state_budget", 0) or 0), int(base.get("raw_state_budget", 0) or 0))
        self.assertGreater(
            int((boosted.get("stream_state", {}) or {}).get("effective_focus_priority_budget", 0) or 0),
            int((base.get("stream_state", {}) or {}).get("effective_focus_priority_budget", 0) or 0),
        )
        self.assertTrue(bool((boosted.get("attention_boost", {}) or {}).get("active", False)))

    def test_dynamic_motion_summary_emerges_for_shifted_object(self) -> None:
        image1 = Image.new("RGB", (96, 96), color=(18, 18, 18))
        image2 = Image.new("RGB", (96, 96), color=(18, 18, 18))
        for x in range(24, 40):
            for y in range(36, 52):
                image1.putpixel((x, y), (245, 245, 245))
        for x in range(32, 48):
            for y in range(36, 52):
                image2.putpixel((x, y), (245, 245, 245))
        buf1 = BytesIO()
        buf2 = BytesIO()
        image1.save(buf1, format="PNG")
        image2.save(buf2, format="PNG")
        sensor = VisionSensorV1(
            patch_budget=16,
            focus_patch_budget=8,
            raw_state_budget=128,
            reconstruction_patch_budget=256,
        )
        p1 = sensor.ingest_image_bytes(buf1.getvalue(), tick_index=0)
        p2 = sensor.ingest_image_bytes(buf2.getvalue(), tick_index=1)
        self.assertIn("dynamic_motion_samples", p2)
        dyn = list(p2.get("dynamic_motion_samples", []) or [])
        self.assertGreaterEqual(len(dyn), 1)
        self.assertTrue(
            any(
                float((((row.get("attributes", {}) or {}).get("dynamic_objectness", 0.0)) or 0.0)) > 0.0
                for row in dyn
            )
        )
        summary = dict(p2.get("dynamic_track_summary", {}) or {})
        self.assertGreaterEqual(int(summary.get("track_count", 0) or 0), 1)
        motion_bundle = dict(p2.get("contour_reconstruction", {}) or {})
        motion_summary = dict(motion_bundle.get("motion_summary", {}) or {})
        self.assertTrue(str(motion_bundle.get("motion_composite_data_url", "") or "").startswith("data:image/png;base64,"))
        self.assertGreaterEqual(int(motion_summary.get("component_count", 0) or 0), 1)
        self.assertGreater(float(motion_summary.get("motion_coverage", 0.0) or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
