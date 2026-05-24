# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.web import create_server

REPO_ROOT = Path(__file__).resolve().parents[1]


class WebApiTunerFeedbackTests(unittest.TestCase):
    def test_multimodal_api_forwards_external_feedback_and_exposes_tuner_learning(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                req_multi = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/multimodal/start",
                    data=json.dumps(
                        {
                            "items": [
                                {"text": "today weather nice", "external_feedback": {"reward": 0.35}},
                                {"text": "go outside", "external_feedback": {"punishment": 0.1}},
                            ],
                            "label": "web multimodal tuner feedback",
                            "tick_interval_ms": 0,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_multi, timeout=5) as resp:
                    multi_result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(multi_result["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=10.0))

                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{multi_result['run_id']}/ticks/0/sidecar", timeout=5) as resp:
                    sidecar = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(sidecar["input_item"]["external_feedback"]["reward"], 0.35)
                self.assertIn("tuner_learning_summary", sidecar)
                self.assertIn("action_learning_context_bias_summary", sidecar)

                with urllib.request.urlopen(f"http://{host}:{port}/api/runtime/export", timeout=5) as resp:
                    exported = json.loads(resp.read().decode("utf-8"))
                self.assertIn("tuner_learning", exported["runtime"])

                with urllib.request.urlopen(f"http://{host}:{port}/api/runtime/summary", timeout=5) as resp:
                    runtime_summary = json.loads(resp.read().decode("utf-8"))
                self.assertIn("tuner_learning_summary", runtime_summary)

                with urllib.request.urlopen(f"http://{host}:{port}/api/executor/status", timeout=5) as resp:
                    executor_status = json.loads(resp.read().decode("utf-8"))
                self.assertIn("tuner_learning_recent_feedback", executor_status)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
