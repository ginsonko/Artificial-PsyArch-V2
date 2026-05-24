# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
import wave
from io import BytesIO
from pathlib import Path

from PIL import Image

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_test_png_bytes() -> bytes:
    image = Image.new("RGB", (48, 48), color=(10, 10, 10))
    for x in range(16, 32):
        for y in range(16, 32):
            image.putpixel((x, y), (255, 255, 255))
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def build_test_wav_bytes() -> bytes:
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
    return buf.getvalue()


class MultimodalRuntimePhase15Tests(unittest.TestCase):
    def test_multimodal_run_writes_sidecar_with_image_and_audio(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_multimodal_run(
                items=[
                    {"text": "今天 天气 不错", "image_bytes": build_test_png_bytes(), "audio_bytes": build_test_wav_bytes(), "source_type": "multimodal_input"},
                    {"text": "我 想 出门", "source_type": "multimodal_input"},
                ],
                label="multimodal test",
                tick_interval_ms=0,
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            sidecar = app.get_tick_sidecar(result["run_id"], 0)
            self.assertIn("image_packet", sidecar)
            self.assertIn("audio_packet", sidecar)
            self.assertGreater(sidecar["image_packet"].get("budget_used", 0), 0)
            self.assertGreater(len(sidecar["image_packet"].get("raw_samples", []) or []), 0)
            self.assertGreaterEqual(len(sidecar["image_packet"].get("raw_samples", []) or []), len(sidecar["image_packet"].get("memory_write_samples", []) or []))
            self.assertGreater(sidecar["audio_packet"].get("budget_used", 0), 0)
            self.assertTrue(str(sidecar["audio_packet"].get("preview_wav_b64", "") or ""))
            self.assertGreater(int(sidecar["audio_packet"].get("preview_audio_bytes_len", 0) or 0), 0)
            self.assertIn("sandbox_result", sidecar)
            self.assertIn("runtime_controls", sidecar)
            self.assertIn("logic_feedback", sidecar)
            self.assertIn("runtime_action_effects", sidecar)
            self.assertIn("action_feedback", sidecar)
            self.assertIn("post_action_attention_modulation_state", sidecar)
            self.assertIn("post_action_effective_attention_controls", sidecar)

    def test_sidecar_externalizes_dense_vision_grid_but_restores_on_read(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_multimodal_run(
                items=[
                    {"text": "视觉重建测试", "image_bytes": build_test_png_bytes(), "source_type": "multimodal_input"},
                ],
                label="vision sidecar compact",
                tick_interval_ms=0,
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            run_dir = Path(tmpdir) / "runs" / str(result["run_id"])
            sidecar_path = next((run_dir / "chunks").glob("*.sidecar.jsonl"))
            raw_row = None
            for line in sidecar_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    raw_row = json.loads(line)
                    break
            self.assertIsNotNone(raw_row)
            raw_image_packet = dict((raw_row or {}).get("image_packet", {}) or {})
            self.assertTrue(raw_image_packet.get("externalized"))
            self.assertEqual(raw_image_packet.get("kind"), "vision")
            self.assertTrue(dict((raw_row or {}).get("competition_packet", {}) or {}).get("externalized"))
            self.assertTrue(dict((raw_row or {}).get("exact_memory", {}) or {}).get("externalized"))

            sidecar = app.get_tick_sidecar(str(result["run_id"]), 0)
            image_packet = dict(sidecar.get("image_packet", {}) or {})
            reconstruction_grid = dict(image_packet.get("reconstruction_grid", {}) or {})
            self.assertIn("cells", reconstruction_grid)
            self.assertGreater(len(reconstruction_grid.get("cells", []) or []), 0)
            self.assertIn("preview_image", image_packet)
            self.assertIn("fixation_buffer", image_packet)
            self.assertIn("competition_packet", sidecar)
            self.assertIn("exact_memory", sidecar)
            self.assertFalse(dict(sidecar.get("competition_packet", {}) or {}).get("externalized", False))


if __name__ == "__main__":
    unittest.main()
