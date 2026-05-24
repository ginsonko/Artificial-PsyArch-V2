# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

from scripts.external_teacher_stub_server import make_handler


class ExternalTeacherStubServerTests(unittest.TestCase):
    def test_stub_server_returns_schema_compatible_response(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(warn_drive=0.5, block_drive=0.9))
        host, port = server.server_address[:2]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request_payload = {
                "schema_id": "external_teacher_request/v1",
                "schema_version": "1.0",
                "tick_index": 3,
                "candidate_actions": [
                    {"action_id": "action::click", "action_name": "click", "drive": 0.95},
                    {"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.4},
                ],
                "focus_preview": ["today", "weather"],
                "bn_preview_ids": ["mem_000001"],
                "state_summary": {
                    "state_pool_size": 8,
                    "recent_external_count": 1,
                    "verbatim_chars": 12,
                    "anchor_count": 2,
                    "residual_count": 0,
                    "residual_total_unresolved_mass": 0.0,
                },
                "request_digest": {
                    "tick_index": 3,
                    "candidate_action_count": 2,
                    "top_actions": [{"action_id": "action::click", "action_name": "click", "drive": 0.95}],
                    "focus_preview": ["today", "weather"],
                    "bn_preview_ids": ["mem_000001"],
                },
            }
            req = urllib.request.Request(
                f"http://{host}:{port}/teacher",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(payload["schema_id"], "external_teacher_response/v1")
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["provider"], "local_stub_http")
            decisions = {item["target_action_name"]: item["decision"] for item in payload["decisions"]}
            self.assertEqual(decisions["click"], "block")
            self.assertEqual(decisions["continue_focus"], "allow")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
