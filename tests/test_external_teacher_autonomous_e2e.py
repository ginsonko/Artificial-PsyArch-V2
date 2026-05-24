# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from scripts.external_teacher_stub_server import make_handler

REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_grab() -> Image.Image:
    image = Image.new("RGB", (96, 64), color=(24, 24, 24))
    for x in range(36, 68):
        for y in range(14, 50):
            image.putpixel((x, y), (245, 245, 245))
    return image


class ExternalTeacherAutonomousE2ETests(unittest.TestCase):
    def test_autonomous_run_with_http_teacher_stub_generates_teacher_feedback(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(warn_drive=0.4, block_drive=0.75))
        host, port = server.server_address[:2]
        endpoint = f"http://{host}:{port}/teacher"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            config = load_config(
                overrides={
                    "executor_enabled": False,
                    "executor_screenshot_enabled": True,
                    "autonomous_external_teacher_enabled": True,
                    "autonomous_external_teacher_mode": "http_json",
                    "autonomous_external_teacher_http_endpoint": endpoint,
                }
            )
            with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
                app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
                result = app.start_autonomous_run(
                    ticks=2,
                    text_hint="teacher e2e",
                    tick_interval_ms=0,
                    reset_runtime=True,
                    external_teacher_enabled=True,
                    external_teacher_mode="http_json",
                    external_teacher_http_endpoint=endpoint,
                )
                self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
                sidecar = app.get_tick_sidecar(result["run_id"], 0)
                teacher_review = dict(sidecar.get("teacher_review", {}) or {})
                teacher_feedback = dict(sidecar.get("teacher_feedback", {}) or {})
                self.assertTrue((teacher_review.get("external_teacher_review", {}) or {}).get("applied"))
                self.assertEqual((teacher_review.get("external_teacher_review", {}) or {}).get("mode"), "http_json")
                self.assertIn("teacher_provenance", teacher_feedback)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
