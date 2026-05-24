# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from core.runtime_v2 import RuntimeV2
from observatory_v2.config import load_config


class TeacherLayerV1Tests(unittest.TestCase):
    def test_teacher_review_blocks_low_drive_risky_actions_but_keeps_ap_drives(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        review = runtime.teacher_layer.review_actions(
            tick_index=0,
            action_drives=[
                {"action_id": "action::click", "action_name": "click", "drive": 0.2},
                {"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.4},
            ],
            runtime_tick=tick,
            autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
            teacher_mode_override="heuristic",
            llm_gate_mode_override="heuristic",
        )
        blocked_names = {item.get("action_name", "") for item in review.get("blocked_actions", [])}
        kept_names = {item.get("action_name", "") for item in review.get("scored_action_drives", [])}
        self.assertIn("click", blocked_names)
        self.assertIn("continue_focus", kept_names)

    def test_teacher_reviews_planner_winners_without_reopening_candidate_selection(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="口令 甲", tick_index=0)
        review = runtime.teacher_layer.review_actions(
            tick_index=0,
            action_drives=[
                {
                    "action_id": "action::press_key",
                    "action_name": "press_key",
                    "drive": 0.72,
                    "firmness": 0.11,
                    "planner_selected": True,
                    "params": {"key": "left"},
                    "instance_id": "action::press_key::{\"key\":\"left\"}",
                },
                {
                    "action_id": "action::press_key",
                    "action_name": "press_key",
                    "drive": 0.65,
                    "firmness": 0.0,
                    "planner_selected": False,
                    "params": {"key": "right"},
                    "instance_id": "action::press_key::{\"key\":\"right\"}",
                },
            ],
            runtime_tick=tick,
            autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
            teacher_mode_override="heuristic",
            llm_gate_mode_override="heuristic",
        )
        selected = list(review.get("scored_action_drives", []) or [])
        candidate_pool = list(review.get("candidate_action_drives", []) or [])
        planner_pool = list(review.get("planner_selected_action_drives", []) or [])
        self.assertEqual(len(candidate_pool), 2)
        self.assertEqual(len(planner_pool), 1)
        self.assertEqual(len(selected), 1)
        self.assertEqual(str((selected[0].get("params", {}) or {}).get("key", "") or ""), "left")
        self.assertGreater(float(selected[0].get("teacher_effective_drive", 0.0) or 0.0), 0.78)

    def test_teacher_feedback_can_be_injected_into_state_pool_and_memory(self) -> None:
        runtime = RuntimeV2(config=load_config())
        tick = runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
        review = {
            "mode": "heuristic",
            "llm_gate_mode": "heuristic",
            "blocked_actions": [{"action_name": "click"}],
        }
        teacher_feedback = runtime.teacher_layer.build_teacher_feedback(
            tick_index=0,
            runtime_tick=tick,
            teacher_review=review,
            selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.5}],
            sandbox_result={"selected_actions": [{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.5}]},
            runtime_action_effects={"moved": True},
        )
        injected = runtime.inject_teacher_feedback(tick_index=0, teacher_feedback=teacher_feedback)
        self.assertIn("teacher_review", injected)
        labels = {item.get("sa_label", "") for item in injected.get("injected_items", [])}
        self.assertTrue("attr::reward_signal" in labels or "attr::punishment_signal" in labels)
        summary = runtime.state_pool.snapshot_summary()
        top_labels = {item.get("sa_label", "") for item in summary.get("top", [])}
        self.assertTrue("attr::reward_signal" in top_labels or "attr::punishment_signal" in top_labels)

    def test_external_teacher_stub_can_block_and_produce_provenance_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stub_path = Path(tmpdir) / "teacher_stub.json"
            stub_path.write_text(
                json.dumps(
                    {
                        "reviewer": "stub_file",
                        "decisions": [
                            {
                                "decision": "block",
                                "target_action_name": "click",
                                "confidence": 0.9,
                                "punishment": 0.2,
                                "warning_code": "unsafe_click",
                                "risk_tags": ["unsafe_action"],
                                "explanation": "block click in stub",
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            runtime = RuntimeV2(
                config=load_config(
                    overrides={
                        "autonomous_external_teacher_enabled": True,
                        "autonomous_external_teacher_mode": "stub_file",
                        "autonomous_external_teacher_stub_response_path": str(stub_path),
                    }
                )
            )
            tick = runtime.process_text_tick(text="今天 天气 不错", tick_index=0)
            review = runtime.teacher_layer.review_actions(
                tick_index=0,
                action_drives=[
                    {"action_id": "action::click", "action_name": "click", "drive": 0.95},
                    {"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.4},
                ],
                runtime_tick=tick,
                autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
                teacher_mode_override="heuristic",
                llm_gate_mode_override="heuristic",
            )
            blocked_names = {item.get("action_name", "") for item in review.get("blocked_actions", [])}
            self.assertIn("click", blocked_names)
            self.assertTrue(review.get("external_teacher_review", {}).get("applied"))

            feedback = runtime.teacher_layer.build_teacher_feedback(
                tick_index=0,
                runtime_tick=tick,
                teacher_review=review,
                selected_actions=[{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.5}],
                sandbox_result={"selected_actions": [{"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.5}]},
                runtime_action_effects={"moved": True},
            )
            injected = runtime.inject_teacher_feedback(
                tick_index=0,
                teacher_feedback=feedback,
                teacher_provenance={
                    "selected_action_ids": ["action::continue_focus"],
                    "bn_ids": ["mem_x"],
                    "focus_memory_id": "mem_focus",
                    "exact_memory_id": "mem_exact",
                },
            )
            self.assertIn("teacher_provenance", injected)
            latest = runtime.memory_store.get_memory(f"mem_{runtime.memory_store._counter:06d}")
            self.assertIsNotNone(latest)
            self.assertEqual(latest.get("memory_kind", ""), "teacher_feedback")
            self.assertIn("mem_focus", latest.get("source_refs", []))
            self.assertIn("mem_exact", latest.get("source_refs", []))
            self.assertIn("teacher_provenance", latest.get("meta", {}))

    def test_external_teacher_http_json_can_warn(self) -> None:
        captured: dict[str, object] = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length)
                captured["request"] = json.loads(body.decode("utf-8"))
                payload = {
                    "schema_id": "external_teacher_response/v1",
                    "schema_version": "1.0",
                    "ok": True,
                    "mode": "http_json",
                    "provider": "http_json",
                    "reviewer": "unit_http_teacher",
                    "path": "",
                    "request_digest": captured["request"]["request_digest"],
                    "decisions": [
                        {
                            "decision": "warn",
                            "target_action_name": "click",
                            "confidence": 0.88,
                            "reward": 0.0,
                            "punishment": 0.12,
                            "explanation": "warn click",
                            "warning_code": "unsafe_click",
                            "risk_tags": ["unsafe_action"],
                        }
                    ],
                    "error": "",
                }
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_address[1]}/teacher"
            runtime = RuntimeV2(
                config=load_config(
                    overrides={
                        "autonomous_external_teacher_enabled": True,
                        "autonomous_external_teacher_mode": "http_json",
                        "autonomous_external_teacher_http_endpoint": endpoint,
                        "autonomous_external_teacher_http_headers": {"X-Teacher": "ap"},
                    }
                )
            )
            tick = runtime.process_text_tick(text="今天天气不错", tick_index=0)
            review = runtime.teacher_layer.review_actions(
                tick_index=0,
                action_drives=[
                    {"action_id": "action::click", "action_name": "click", "drive": 0.95},
                    {"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.4},
                ],
                runtime_tick=tick,
                autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
                teacher_mode_override="heuristic",
                llm_gate_mode_override="heuristic",
            )
            self.assertTrue(review.get("external_teacher_review", {}).get("applied"))
            self.assertEqual(review.get("external_teacher_review", {}).get("mode"), "http_json")
            self.assertEqual(review.get("external_teacher_review", {}).get("reviewer"), "unit_http_teacher")
            self.assertEqual(review.get("external_teacher_review", {}).get("provider"), "http_json")
            self.assertEqual(review.get("external_teacher_review", {}).get("transport_audit", {}).get("status_code"), 200)
            self.assertEqual(review.get("external_teacher_review", {}).get("transport_audit", {}).get("attempt_count"), 1)
            self.assertEqual(captured["request"]["schema_id"], "external_teacher_request/v1")
            warning_codes = {item.get("code", "") for item in review.get("warnings", [])}
            self.assertIn("unsafe_click", warning_codes)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_external_teacher_http_json_retries_once_then_succeeds(self) -> None:
        state = {"attempts": 0}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                state["attempts"] += 1
                if state["attempts"] == 1:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(b'{"error":"try later"}')
                    return
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length)
                request_payload = json.loads(body.decode("utf-8"))
                payload = {
                    "schema_id": "external_teacher_response/v1",
                    "schema_version": "1.0",
                    "ok": True,
                    "mode": "http_json",
                    "provider": "http_json",
                    "reviewer": "retry_teacher",
                    "path": "",
                    "request_digest": request_payload["request_digest"],
                    "decisions": [
                        {
                            "decision": "warn",
                            "target_action_name": "click",
                            "confidence": 0.72,
                            "reward": 0.0,
                            "punishment": 0.08,
                            "explanation": "retry success",
                            "warning_code": "retry_warn",
                            "risk_tags": ["low_confidence"],
                        }
                    ],
                    "error": "",
                }
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_address[1]}/teacher"
            runtime = RuntimeV2(
                config=load_config(
                    overrides={
                        "autonomous_external_teacher_enabled": True,
                        "autonomous_external_teacher_mode": "http_json",
                        "autonomous_external_teacher_http_endpoint": endpoint,
                        "autonomous_external_teacher_max_retries": 2,
                        "autonomous_external_teacher_retry_backoff_ms": 0,
                    }
                )
            )
            tick = runtime.process_text_tick(text="浠婂ぉ澶╂皵涓嶉敊", tick_index=0)
            review = runtime.teacher_layer.review_actions(
                tick_index=0,
                action_drives=[{"action_id": "action::click", "action_name": "click", "drive": 0.95}],
                runtime_tick=tick,
                autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
                teacher_mode_override="heuristic",
                llm_gate_mode_override="heuristic",
            )
            external_review = dict(review.get("external_teacher_review", {}) or {})
            self.assertTrue(external_review.get("applied"))
            self.assertEqual(external_review.get("reviewer"), "retry_teacher")
            self.assertEqual((external_review.get("transport_audit", {}) or {}).get("attempt_count"), 2)
            self.assertEqual((external_review.get("transport_audit", {}) or {}).get("status_code"), 200)
            self.assertEqual(state["attempts"], 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_external_teacher_request_fills_missing_action_name_from_action_id(self) -> None:
        captured: dict[str, object] = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0") or 0)
                body = self.rfile.read(length)
                captured["request"] = json.loads(body.decode("utf-8"))
                payload = {
                    "schema_id": "external_teacher_response/v1",
                    "schema_version": "1.0",
                    "ok": True,
                    "mode": "http_json",
                    "provider": "http_json",
                    "reviewer": "name_fill_teacher",
                    "path": "",
                    "request_digest": captured["request"]["request_digest"],
                    "decisions": [],
                    "error": "",
                }
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_address[1]}/teacher"
            runtime = RuntimeV2(
                config=load_config(
                    overrides={
                        "autonomous_external_teacher_enabled": True,
                        "autonomous_external_teacher_mode": "http_json",
                        "autonomous_external_teacher_http_endpoint": endpoint,
                    }
                )
            )
            tick = runtime.process_text_tick(text="浠婂ぉ澶╂皵涓嶉敊", tick_index=0)
            runtime.teacher_layer.review_actions(
                tick_index=0,
                action_drives=[{"action_id": "action::continue_focus", "drive": 0.95}],
                runtime_tick=tick,
                autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
                teacher_mode_override="heuristic",
                llm_gate_mode_override="heuristic",
            )
            request_payload = dict(captured.get("request", {}) or {})
            self.assertEqual(request_payload.get("candidate_actions", [])[0].get("action_name"), "continue_focus")
            self.assertEqual(request_payload.get("request_digest", {}).get("top_actions", [])[0].get("action_name"), "continue_focus")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_external_teacher_fail_closed_blocks_risky_action_when_provider_unavailable(self) -> None:
        runtime = RuntimeV2(
            config=load_config(
                overrides={
                    "autonomous_external_teacher_enabled": True,
                    "autonomous_external_teacher_fail_open": False,
                    "autonomous_external_teacher_mode": "http_json",
                    "autonomous_external_teacher_http_endpoint": "http://127.0.0.1:9/unavailable",
                }
            )
        )
        tick = runtime.process_text_tick(text="今天天气不错", tick_index=0)
        review = runtime.teacher_layer.review_actions(
            tick_index=0,
            action_drives=[
                {"action_id": "action::click", "action_name": "click", "drive": 0.95},
                {"action_id": "action::continue_focus", "action_name": "continue_focus", "drive": 0.4},
            ],
            runtime_tick=tick,
            autonomous_state={"idle_ticks": 0, "capture_failures": 0, "action_errors": 0},
            teacher_mode_override="heuristic",
            llm_gate_mode_override="heuristic",
        )
        blocked = {item.get("blocked_reason", "") for item in review.get("blocked_actions", [])}
        self.assertIn("external_teacher_unavailable_fail_closed", blocked)
        self.assertFalse(review.get("external_teacher_review", {}).get("applied"))
        self.assertTrue(review.get("external_teacher_review", {}).get("fail_closed"))
        self.assertFalse(review.get("external_teacher_review", {}).get("fail_open"))
        self.assertGreaterEqual(review.get("external_teacher_review", {}).get("transport_audit", {}).get("attempt_count", 0), 1)


if __name__ == "__main__":
    unittest.main()
