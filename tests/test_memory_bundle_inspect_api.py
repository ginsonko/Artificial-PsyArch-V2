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


class MemoryBundleInspectApiTests(unittest.TestCase):
    def test_memory_bundle_inspect_api_returns_layered_bundle_details(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app.start_text_run(texts=["今天 天气 不错", "我 想 出门"], label="memory inspect api", tick_interval_ms=0)
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))

            bundle_dir = Path(tmpdir) / "memory_bundle"
            exported = app.export_memory_deployment_bundle(bundle_dir)
            self.assertTrue(exported["ok"])

            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                req = urllib.request.Request(
                    f"http://{host}:{port}/api/runtime/memory-bundle/inspect",
                    data=json.dumps({"directory": str(bundle_dir)}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertEqual(payload["bundle_format"], "layered_v2")
                self.assertIn("files", payload)
                self.assertIn("memory_count", payload)
                self.assertIn("index_summary", payload)
                self.assertIn("vector_bundle", payload)
                self.assertIn("spacetime_bundle", payload)
                self.assertIn("vector_meta", payload["files"])
                self.assertIn("spacetime_meta", payload["files"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
