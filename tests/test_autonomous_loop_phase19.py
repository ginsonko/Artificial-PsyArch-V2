# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
import wave
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from observatory_v2 import __main__ as cli_main
from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.web import create_server

REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_grab() -> Image.Image:
    image = Image.new("RGB", (96, 64), color=(18, 18, 18))
    for x in range(32, 64):
        for y in range(16, 48):
            image.putpixel((x, y), (240, 240, 240))
    return image


class AutonomousLoopPhase19Tests(unittest.TestCase):
    def _build_test_wav_bytes(self, *, ms: int = 120, sample_rate: int = 16000) -> bytes:
        frame_count = max(1, int(sample_rate * (ms / 1000.0)))
        payload = BytesIO()
        with wave.open(payload, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2)
            wav_out.setframerate(sample_rate)
            wav_out.writeframes(b"\x00\x00" * frame_count)
        return payload.getvalue()

    def test_autonomous_run_captures_screen_and_records_feedback(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_autonomous_run(
                ticks=3,
                text_hint="observe desktop",
                tick_interval_ms=0,
                reset_runtime=True,
                reward_schedule=[{"tick_index": 1, "reward": 0.4}],
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["tick_done"], 3)
            summary = app.get_tick_summary(result["run_id"], 1)
            sidecar = app.get_tick_sidecar(result["run_id"], 1)
            self.assertEqual(sidecar["input_item"]["source_type"], "autonomous_loop")
            self.assertIn("screen_capture", summary["multimodal_summary"])
            self.assertGreater(sidecar["image_packet"].get("budget_used", 0), 0)
            self.assertEqual(sidecar["input_item"]["external_feedback"]["reward"], 0.4)
            self.assertIn("autonomous_summary", summary)
            self.assertIn("autonomous_sidecar", sidecar)
            self.assertIn("feedback_used", sidecar["autonomous_sidecar"])
            self.assertIn("teacher_review", sidecar["autonomous_sidecar"])
            self.assertIn("teacher_feedback", sidecar["autonomous_sidecar"])
            self.assertIn("teacher_provenance", sidecar["autonomous_sidecar"]["teacher_feedback"])

    def test_autonomous_web_and_cli_entrypoints(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                req = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/autonomous/start",
                    data=json.dumps(
                        {
                            "ticks": 2,
                            "text_hint": "web autonomous",
                            "tick_interval_ms": 0,
                            "reward_schedule": [{"tick_index": 0, "reward": 0.2}],
                            "teacher_mode": "heuristic",
                            "llm_gate_mode": "heuristic",
                            "external_teacher_enabled": False,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

            cli_app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            with patch.object(cli_main, "load_config", return_value=config), patch.object(cli_main, "ObservatoryV2App", return_value=cli_app):
                with patch("sys.argv", ["observatory_v2", "run-autonomous", "--ticks", "2", "--text-hint", "cli autonomous"]):
                    cli_main.main()

    def test_autonomous_external_teacher_parameters_flow_into_tick_meta(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_autonomous_run(
                ticks=1,
                text_hint="external teacher meta",
                tick_interval_ms=0,
                reset_runtime=True,
                external_teacher_enabled=True,
                external_teacher_mode="stub_file",
                external_teacher_stub_response_path="C:/tmp/teacher_stub.json",
                external_teacher_http_endpoint="http://127.0.0.1:8765/teacher",
                external_teacher_http_headers={"X-Teacher": "ap"},
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            sidecar = app.get_tick_sidecar(result["run_id"], 0)
            tick_meta = dict((sidecar.get("autonomous_sidecar", {}) or {}).get("tick_meta", {}) or {})
            self.assertTrue(tick_meta.get("external_teacher_enabled"))
            self.assertEqual(tick_meta.get("external_teacher_mode"), "stub_file")
            self.assertEqual(tick_meta.get("external_teacher_stub_response_path"), "C:/tmp/teacher_stub.json")
            self.assertTrue(tick_meta.get("external_teacher_fail_open"))
            self.assertEqual(tick_meta.get("external_teacher_max_retries"), 1)
            self.assertEqual(tick_meta.get("external_teacher_retry_backoff_ms"), 25)
            self.assertEqual(tick_meta.get("external_teacher_http_endpoint"), "http://127.0.0.1:8765/teacher")
            self.assertEqual(tick_meta.get("external_teacher_http_headers", {}).get("X-Teacher"), "ap")

    def test_autonomous_run_stops_gracefully_after_capture_failures(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=RuntimeError("grab failed")):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_autonomous_run(
                ticks=5,
                text_hint="observe desktop",
                tick_interval_ms=0,
                reset_runtime=True,
                stop_on_capture_failures=1,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "stopped")
            self.assertIn("completion_details", manifest)
            self.assertGreaterEqual(int(manifest.get("latest_tick_index", -1) or -1), -1)

    def test_autonomous_session_lifecycle(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            started = app.start_autonomous_session(
                text_hint="session autonomous",
                tick_interval_ms=10,
                reset_runtime=True,
                max_ticks=4,
            )
            self.assertTrue(started["session_id"])
            time.sleep(0.08)
            status_running = app.get_autonomous_session_status()
            self.assertTrue(status_running.get("active"))
            pause_result = app.pause_autonomous_session()
            self.assertTrue(pause_result.get("ok"))
            time.sleep(0.08)
            status_paused = app.get_autonomous_session_status()
            self.assertTrue(bool(status_paused.get("paused")) or str(status_paused.get("status", "")) in {"paused", "pausing"})
            resume_result = app.resume_autonomous_session()
            self.assertTrue(resume_result.get("ok"))
            stop_result = app.stop_autonomous_session()
            self.assertTrue(stop_result.get("ok"))
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            final_status = app.get_autonomous_session_status()
            self.assertEqual(final_status.get("status"), "stopped")
            session_file = Path(started["run_dir"]) / "live" / "autonomous_session_status.json"
            self.assertTrue(session_file.exists())
            manifest = app.get_manifest(started["run_id"])
            self.assertIn(manifest.get("status"), {"completed", "stopped"})

    def test_autonomous_session_web_entrypoints(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                req = urllib.request.Request(
                    f"http://{host}:{port}/api/autonomous-session/start",
                    data=json.dumps({"text_hint": "web session", "tick_interval_ms": 10, "max_ticks": 3}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                with urllib.request.urlopen(f"http://{host}:{port}/api/autonomous-session/status", timeout=5) as resp:
                    status_payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(status_payload.get("active"))

                pause_req = urllib.request.Request(
                    f"http://{host}:{port}/api/autonomous-session/pause",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(pause_req, timeout=5) as resp:
                    pause_payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(pause_payload.get("ok"))

                resume_req = urllib.request.Request(
                    f"http://{host}:{port}/api/autonomous-session/resume",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(resume_req, timeout=5) as resp:
                    resume_payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(resume_payload.get("ok"))

                stop_req = urllib.request.Request(
                    f"http://{host}:{port}/api/autonomous-session/stop",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(stop_req, timeout=5) as resp:
                    stop_payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(stop_payload.get("ok"))
                self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_autonomous_session_cli_waits_until_max_ticks(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            cli_app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            with patch.object(cli_main, "load_config", return_value=config), patch.object(cli_main, "ObservatoryV2App", return_value=cli_app):
                with patch(
                    "sys.argv",
                    [
                        "observatory_v2",
                        "run-autonomous-session",
                        "--max-ticks",
                        "2",
                        "--wait",
                        "--timeout-sec",
                        "20",
                        "--text-hint",
                        "cli session wait",
                    ],
                ):
                    cli_main.main()
            status = cli_app.get_autonomous_session_status()
            self.assertEqual(status.get("status"), "completed")
            self.assertEqual(int(status.get("tick_done", 0) or 0), 2)
            manifest = cli_app.get_manifest(status["run_id"])
            self.assertEqual(manifest.get("status"), "completed")
            self.assertEqual(int(manifest.get("tick_done", 0) or 0), 2)

    def test_autonomous_session_cli_can_control_running_web_session(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            server_url = f"http://{host}:{port}"
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.object(cli_main, "load_config", return_value=config):
                    with patch("sys.argv", ["observatory_v2", "run-autonomous-session", "--server-url", server_url, "--max-ticks", "4", "--interval-ms", "20"]):
                        cli_main.main()
                    time.sleep(0.05)
                    with patch("sys.argv", ["observatory_v2", "pause-autonomous-session", "--server-url", server_url]):
                        cli_main.main()
                    self.assertIn(app.get_autonomous_session_status().get("status"), {"pausing", "paused"})
                    with patch("sys.argv", ["observatory_v2", "resume-autonomous-session", "--server-url", server_url]):
                        cli_main.main()
                    self.assertEqual(app.get_autonomous_session_status().get("status"), "running")
                    with patch("sys.argv", ["observatory_v2", "stop-autonomous-session", "--server-url", server_url]):
                        cli_main.main()
                self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
                self.assertIn(app.get_autonomous_session_status().get("status"), {"stopped", "completed"})
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_autonomous_session_can_recover_from_checkpoint(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            started = app.start_autonomous_session(
                text_hint="recover session",
                tick_interval_ms=5,
                reset_runtime=True,
                max_ticks=4,
            )
            time.sleep(0.08)
            stop_result = app.stop_autonomous_session()
            self.assertTrue(stop_result.get("ok"))
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            stopped_status = app.get_autonomous_session_status()
            self.assertEqual(stopped_status.get("status"), "stopped")
            recovered = app.recover_autonomous_session(run_id=started["run_id"])
            self.assertTrue(recovered.get("ok"))
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            final_status = app.get_autonomous_session_status()
            self.assertEqual(final_status.get("status"), "completed")
            self.assertGreaterEqual(int(final_status.get("tick_done", 0) or 0), 4)
            tick_zero = app.get_tick_summary(started["run_id"], 0)
            tick_one = app.get_tick_summary(started["run_id"], 1)
            tick_two = app.get_tick_summary(started["run_id"], 2)
            tick_three = app.get_tick_summary(started["run_id"], 3)
            self.assertTrue(tick_zero)
            self.assertTrue(tick_one)
            self.assertTrue(tick_two)
            self.assertTrue(tick_three)
            self.assertEqual(int(tick_two.get("tick_index", -1) or -1), 2)
            self.assertEqual(int(tick_three.get("tick_index", -1) or -1), 3)
            lifecycle = dict(final_status.get("lifecycle", {}) or {})
            goal = dict(final_status.get("session_goal", {}) or {})
            self.assertGreaterEqual(int(lifecycle.get("recover_count", 0) or 0), 1)
            self.assertGreaterEqual(int(lifecycle.get("stop_request_count", 0) or 0), 1)
            self.assertGreaterEqual(int(lifecycle.get("completion_count", 0) or 0), 1)
            self.assertEqual(str(goal.get("phase_status", "")), "completed")
            self.assertGreaterEqual(float(goal.get("completion_ratio", 0.0) or 0.0), 1.0)

    def test_autonomous_session_status_tracks_goal_and_lifecycle_counts(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            started = app.start_autonomous_session(
                text_hint="goal tracking",
                tick_interval_ms=10,
                reset_runtime=True,
                max_ticks=5,
                label="Goal Tracking Session",
            )
            time.sleep(0.08)
            pause_result = app.pause_autonomous_session()
            self.assertTrue(pause_result.get("ok"))
            time.sleep(0.08)
            resume_result = app.resume_autonomous_session()
            self.assertTrue(resume_result.get("ok"))
            stop_result = app.stop_autonomous_session()
            self.assertTrue(stop_result.get("ok"))
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

            final_status = app.get_autonomous_session_status()
            lifecycle = dict(final_status.get("lifecycle", {}) or {})
            goal = dict(final_status.get("session_goal", {}) or {})

            self.assertEqual(str(goal.get("label", "")), "Goal Tracking Session")
            self.assertEqual(str(goal.get("goal_text", "")), "goal tracking")
            self.assertEqual(str(goal.get("phase_status", "")), "stopped")
            self.assertGreaterEqual(int(lifecycle.get("pause_request_count", 0) or 0), 1)
            self.assertGreaterEqual(int(lifecycle.get("resume_count", 0) or 0), 1)
            self.assertGreaterEqual(int(lifecycle.get("stop_request_count", 0) or 0), 1)
            self.assertEqual(str(lifecycle.get("last_status", "")), "stopped")
            self.assertGreaterEqual(int(goal.get("ticks_completed", 0) or 0), 1)
            self.assertGreaterEqual(float(goal.get("completion_ratio", 0.0) or 0.0), 0.0)

            session_file = Path(started["run_dir"]) / "live" / "autonomous_session_status.json"
            payload = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertIn("session_goal", payload)
            self.assertIn("lifecycle", payload)
            self.assertEqual(str((payload.get("session_goal", {}) or {}).get("phase_status", "")), "stopped")
            self.assertIn("session_health", payload)
            self.assertIn("session_context", payload)
            self.assertTrue(str((payload.get("session_health", {}) or {}).get("health_status", "")))

    def test_autonomous_session_runtime_snapshot_tracks_health_and_context(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            started = app.start_autonomous_session(
                text_hint="health snapshot",
                tick_interval_ms=5,
                reset_runtime=True,
                max_ticks=2,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            status = app.get_autonomous_session_status()
            self.assertEqual(str(status.get("status", "")), "completed")
            goal = dict(status.get("session_goal", {}) or {})
            health = dict(status.get("session_health", {}) or {})
            context = dict(status.get("session_context", {}) or {})
            self.assertTrue(str(goal.get("phase_label", "")))
            self.assertGreaterEqual(int(goal.get("phase_index", 0) or 0), 0)
            self.assertEqual(str(health.get("health_status", "")), "completed")
            self.assertTrue("target_completed" in str(health.get("health_reason", "") or ""))
            self.assertGreaterEqual(float(health.get("last_logic_ms", 0.0) or 0.0), 0.0)
            self.assertGreaterEqual(int(health.get("last_checkpoint_tick_done", 0) or 0), 1)
            self.assertTrue(bool(context.get("last_tick_id", "")))
            self.assertIsInstance(context.get("last_focus_preview", []), list)
            session_file = Path(started["run_dir"]) / "live" / "autonomous_session_status.json"
            payload = json.loads(session_file.read_text(encoding="utf-8"))
            self.assertIn("session_health", payload)
            self.assertIn("session_context", payload)
            self.assertEqual(str((payload.get("session_health", {}) or {}).get("health_status", "")), "completed")

    def test_autonomous_session_resume_clears_pause_request_even_if_status_still_running(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            run_dir = Path(tmpdir) / "runs" / "resume-race"
            (run_dir / "live").mkdir(parents=True, exist_ok=True)
            pause_event = threading.Event()
            pause_event.set()
            app._autonomous_session_pause_event = pause_event
            app._autonomous_session_status = app._ensure_autonomous_session_status_defaults(
                {
                    "session_id": "session::resume-race",
                    "run_id": "resume-race",
                    "run_dir": str(run_dir),
                    "status": "running",
                    "active": True,
                    "paused": False,
                    "stopping": False,
                }
            )
            result = app.resume_autonomous_session()
            self.assertTrue(result.get("ok"))
            self.assertFalse(pause_event.is_set())
            self.assertFalse(bool(app._autonomous_session_status.get("paused", False)))
            self.assertEqual(str(app._autonomous_session_status.get("status", "")), "running")
            lifecycle = dict((app._autonomous_session_status.get("lifecycle", {}) or {}))
            self.assertGreaterEqual(int(lifecycle.get("resume_count", 0) or 0), 1)

    def test_realtime_source_run_tick_index_is_not_polluted_by_autonomous_session_status(self) -> None:
        config = load_config(overrides={"executor_enabled": False, "executor_screenshot_enabled": True})
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app._autonomous_session_status = {"tick_done": 5, "session_id": "fake-session", "status": "stopped"}
            result = app.start_audio_stream_run(
                audio_bytes=self._build_test_wav_bytes(ms=140),
                tick_window_ms=50,
                text_prefix="audio stream",
                reset_runtime=True,
            )
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest.get("status"), "completed")
            self.assertGreaterEqual(int(manifest.get("tick_done", 0) or 0), 1)
            ticks = app.list_tick_summaries(result["run_id"])
            self.assertTrue(ticks)
            self.assertEqual(int((ticks[0] or {}).get("tick_index", -1)), 0)
            tick_zero = app.get_tick_summary(result["run_id"], 0)
            self.assertTrue(tick_zero)
            self.assertEqual(int(tick_zero.get("tick_index", -1)), 0)


if __name__ == "__main__":
    unittest.main()
