# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.io_utils import read_json, write_json
from observatory_v2.storage import build_storage_layout, chunk_bounds, list_runs, safe_slug

REPO_ROOT = Path(__file__).resolve().parents[1]


class StorageAndRunTests(unittest.TestCase):
    def test_safe_slug(self) -> None:
        self.assertEqual(safe_slug("hello world"), "hello_world")
        self.assertEqual(safe_slug("中文 标题", fallback="item"), "item")

    def test_chunk_bounds(self) -> None:
        self.assertEqual(chunk_bounds(0, 1000), (0, 999))
        self.assertEqual(chunk_bounds(1001, 1000), (1000, 1999))

    def test_demo_run_writes_manifest_and_tick(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_demo_run(tick_count=3, tick_interval_ms=5, label="test run")
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
            manifest = app.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["tick_done"], 3)
            tick = app.get_tick_summary(result["run_id"], 2)
            self.assertEqual(tick["tick_index"], 2)
            self.assertEqual(tick["run_id"], result["run_id"])

    def test_live_ring_bootstrap_from_disk(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = app.start_demo_run(tick_count=2, tick_interval_ms=5, label="bootstrap test")
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            live = restored.get_live_snapshot()
            self.assertEqual(live["latest_run_id"], result["run_id"])
            self.assertEqual(len(live["recent_ticks"]), 2)
            self.assertEqual(live["recent_ticks"][-1]["tick_index"], 1)

    def test_service_runtime_state_bootstrap_restores_last_forget_summary(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_root = Path(tmpdir)
            write_json(
                outputs_root / "live" / "service_runtime_state.json",
                {
                    "schema_id": "observatory_service_runtime_state/v1",
                    "schema_version": "1.0",
                    "updated_at_ms": 1234,
                    "latest_run_id": "",
                    "last_forget_summary": {
                        "strategy": "score_prune",
                        "removed_count": 3,
                        "generated_at_ms": 9876,
                    },
                    "last_forget_preview_summary": {
                        "strategy": "latest_only",
                        "dry_run": True,
                        "removed_count": 1,
                        "generated_at_ms": 9999,
                    },
                },
            )
            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=outputs_root)
            summary = restored.export_runtime_summary()
            self.assertEqual(summary["export_meta"]["last_forget_summary"].get("generated_at_ms"), 9876)
            self.assertEqual(summary["export_meta"]["last_forget_summary"].get("removed_count"), 3)
            self.assertEqual(summary["export_meta"]["last_forget_preview_summary"].get("generated_at_ms"), 9999)
            self.assertTrue(bool(summary["export_meta"]["last_forget_preview_summary"].get("dry_run")))

    def test_write_json_is_atomic_and_cleans_temp_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "live" / "state.json"
            write_json(target, {"a": 1, "nested": {"b": 2}})
            payload = read_json(target, default={})
            self.assertEqual(payload.get("a"), 1)
            self.assertEqual((payload.get("nested") or {}).get("b"), 2)
            leftovers = list(target.parent.glob(f"{target.name}.tmp.*"))
            self.assertEqual(leftovers, [])

    def test_list_runs_prefers_manifest_time_over_directory_mtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = build_storage_layout(REPO_ROOT, tmpdir)
            older_run = layout.runs_root / "older_run"
            newer_run = layout.runs_root / "newer_run"
            older_run.mkdir(parents=True, exist_ok=True)
            newer_run.mkdir(parents=True, exist_ok=True)

            write_json(
                older_run / "manifest.json",
                {
                    "status": "completed",
                    "label": "older by manifest",
                    "created_at_ms": 1000,
                    "updated_at_ms": 1000,
                    "finished_at_ms": 1000,
                },
            )
            write_json(
                newer_run / "manifest.json",
                {
                    "status": "completed",
                    "label": "newer by manifest",
                    "created_at_ms": 2000,
                    "updated_at_ms": 2000,
                    "finished_at_ms": 2000,
                },
            )

            os.utime(older_run, (9999999999, 9999999999))
            os.utime(newer_run, (1, 1))

            rows = list_runs(layout, limit=8)
            self.assertEqual([row["run_id"] for row in rows[:2]], ["newer_run", "older_run"])
            self.assertEqual(int(rows[0]["run_timestamp_ms"]), 2000)
            self.assertEqual(int(rows[1]["run_timestamp_ms"]), 1000)

    def test_latest_run_id_uses_manifest_order_when_bootstrapping(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = build_storage_layout(REPO_ROOT, tmpdir)
            older_run = layout.runs_root / "older_run"
            newer_run = layout.runs_root / "newer_run"
            for run_dir in (older_run, newer_run):
                (run_dir / "live").mkdir(parents=True, exist_ok=True)

            write_json(
                older_run / "manifest.json",
                {
                    "status": "completed",
                    "label": "older by manifest",
                    "created_at_ms": 1000,
                    "updated_at_ms": 1000,
                    "finished_at_ms": 1000,
                },
            )
            write_json(
                newer_run / "manifest.json",
                {
                    "status": "completed",
                    "label": "newer by manifest",
                    "created_at_ms": 2000,
                    "updated_at_ms": 2000,
                    "finished_at_ms": 2000,
                },
            )
            write_json(
                older_run / "live" / "latest.json",
                {
                    "schema_id": "live_snapshot/v1",
                    "status": "completed",
                    "active_run_id": "",
                    "latest_run_id": "older_run",
                    "recent_ticks": [{"tick_index": 0, "tick_id": "old-0", "input_preview": "old", "a_focus_preview": []}],
                    "server_time_ms": 1000,
                },
            )
            write_json(
                newer_run / "live" / "latest.json",
                {
                    "schema_id": "live_snapshot/v1",
                    "status": "completed",
                    "active_run_id": "",
                    "latest_run_id": "newer_run",
                    "recent_ticks": [{"tick_index": 0, "tick_id": "new-0", "input_preview": "new", "a_focus_preview": []}],
                    "server_time_ms": 2000,
                },
            )

            os.utime(older_run, (9999999999, 9999999999))
            os.utime(newer_run, (1, 1))

            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            self.assertEqual(restored.latest_run_id(), "newer_run")
            live = restored.get_live_snapshot()
            self.assertEqual(live["latest_run_id"], "newer_run")
            self.assertEqual(live["recent_ticks"][-1]["input_preview"], "new")

    def test_bootstrap_restores_interrupted_autonomous_session_status_from_disk(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = build_storage_layout(REPO_ROOT, tmpdir)
            run_dir = layout.runs_root / "session_run"
            (run_dir / "live").mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "manifest.json",
                {
                    "schema_id": "run_manifest/v1",
                    "schema_version": "1.0",
                    "run_id": "session_run",
                    "run_kind": "phase20_autonomous_session_run",
                    "status": "running",
                    "label": "session run",
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 0,
                    "tick_planned": 5,
                    "tick_done": 2,
                    "latest_tick_index": 1,
                    "paths": {
                        "run_dir": str(run_dir),
                        "live_dir": str(run_dir / "live"),
                        "chunks_dir": str(run_dir / "chunks"),
                        "system_dir": str(run_dir / "system"),
                    },
                    "config_snapshot": {
                        "host": "127.0.0.1",
                        "port": 8766,
                        "live_ring_limit": 32,
                        "run_chunk_size": 1000,
                        "text_sensor_budget": 12,
                        "r_state_head_limit": 4,
                    },
                },
            )
            write_json(
                run_dir / "live" / "autonomous_session_status.json",
                {
                    "active": True,
                    "session_id": "session::session_run",
                    "run_id": "session_run",
                    "run_dir": str(run_dir),
                    "status": "running",
                    "paused": False,
                    "stopping": False,
                    "tick_done": 2,
                    "max_ticks": 5,
                    "tick_interval_ms": 10,
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 0,
                    "recoverable": True,
                },
            )

            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            status = restored.get_autonomous_session_status()
            self.assertEqual(status["status"], "interrupted")
            self.assertFalse(bool(status["active"]))
            self.assertTrue(bool(status["recoverable"]))
            self.assertIn("session_goal", status)
            self.assertIn("lifecycle", status)
            self.assertEqual(str((status.get("session_goal", {}) or {}).get("phase_status", "")), "interrupted")
            self.assertGreaterEqual(int((status.get("lifecycle", {}) or {}).get("interrupt_count", 0) or 0), 1)
            live = restored.get_live_snapshot()
            self.assertEqual((live.get("autonomous_session") or {}).get("status"), "interrupted")
            manifest = restored.get_manifest("session_run")
            self.assertEqual(manifest["status"], "interrupted")
            summary = dict(manifest.get("autonomous_session_status_summary", {}) or {})
            self.assertEqual(summary.get("status"), "interrupted")
            self.assertIn("goal", summary)
            self.assertIn("health", summary)
            self.assertIn("context", summary)
            self.assertEqual(str((summary.get("health", {}) or {}).get("health_status", "")), "interrupted")

            status_file = run_dir / "live" / "autonomous_session_status.json"
            repaired = read_json(status_file, default={})
            self.assertEqual(str(repaired.get("status", "")), "interrupted")
            self.assertGreaterEqual(int(((repaired.get("lifecycle", {}) or {}).get("interrupt_count", 0) or 0)), 1)

    def test_get_manifest_keeps_live_autonomous_session_status_without_bootstrap_interrupting_it(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = build_storage_layout(REPO_ROOT, tmpdir)
            run_dir = layout.runs_root / "live_session_run"
            (run_dir / "live").mkdir(parents=True, exist_ok=True)
            write_json(
                run_dir / "manifest.json",
                {
                    "schema_id": "run_manifest/v1",
                    "schema_version": "1.0",
                    "run_id": "live_session_run",
                    "run_kind": "phase20_autonomous_session_run",
                    "status": "running",
                    "label": "live session run",
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 0,
                    "tick_planned": 5,
                    "tick_done": 1,
                    "latest_tick_index": 0,
                },
            )
            write_json(
                run_dir / "live" / "autonomous_session_status.json",
                {
                    "active": True,
                    "session_id": "session::live_session_run",
                    "run_id": "live_session_run",
                    "run_dir": str(run_dir),
                    "status": "paused",
                    "paused": True,
                    "stopping": False,
                    "tick_done": 1,
                    "max_ticks": 5,
                    "tick_interval_ms": 10,
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 0,
                    "recoverable": True,
                },
            )

            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app._active_run_id = "live_session_run"
            app._autonomous_session_status = app._ensure_autonomous_session_status_defaults(
                {
                    "active": True,
                    "session_id": "session::live_session_run",
                    "run_id": "live_session_run",
                    "run_dir": str(run_dir),
                    "status": "paused",
                    "paused": True,
                    "stopping": False,
                    "tick_done": 1,
                    "max_ticks": 5,
                    "tick_interval_ms": 10,
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 0,
                    "recoverable": True,
                }
            )
            app._active_thread = threading.Thread(target=lambda: time.sleep(0.2), daemon=True)
            app._active_thread.start()
            try:
                manifest = app.get_manifest("live_session_run")
                self.assertEqual(str(manifest.get("status", "")), "paused")
                summary = dict(manifest.get("autonomous_session_status_summary", {}) or {})
                self.assertEqual(str(summary.get("status", "")), "paused")
                self.assertTrue(bool(summary.get("active", False)))
                self.assertIn("goal", summary)
                self.assertIn("health", summary)
                self.assertIn("context", summary)
            finally:
                app._active_thread.join(timeout=1.0)

    def test_recoverable_lookup_skips_session_without_checkpoint(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            layout = build_storage_layout(REPO_ROOT, tmpdir)
            bad_run = layout.runs_root / "recoverable_without_checkpoint"
            (bad_run / "live").mkdir(parents=True, exist_ok=True)
            write_json(
                bad_run / "manifest.json",
                {
                    "schema_id": "run_manifest/v1",
                    "schema_version": "1.0",
                    "run_id": "recoverable_without_checkpoint",
                    "run_kind": "phase20_autonomous_session_run",
                    "status": "stopped",
                    "label": "recoverable without checkpoint",
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 1300,
                    "tick_planned": 5,
                    "tick_done": 2,
                    "latest_tick_index": 1,
                    "paths": {
                        "run_dir": str(bad_run),
                        "live_dir": str(bad_run / "live"),
                        "chunks_dir": str(bad_run / "chunks"),
                        "system_dir": str(bad_run / "system"),
                    },
                    "config_snapshot": {
                        "host": "127.0.0.1",
                        "port": 8766,
                        "live_ring_limit": 32,
                        "run_chunk_size": 1000,
                        "text_sensor_budget": 12,
                        "r_state_head_limit": 4,
                    },
                },
            )
            write_json(
                bad_run / "live" / "autonomous_session_status.json",
                {
                    "active": False,
                    "session_id": "session::recoverable_without_checkpoint",
                    "run_id": "recoverable_without_checkpoint",
                    "run_dir": str(bad_run),
                    "status": "stopped",
                    "paused": False,
                    "stopping": False,
                    "tick_done": 2,
                    "max_ticks": 5,
                    "tick_interval_ms": 10,
                    "created_at_ms": 1000,
                    "started_at_ms": 1100,
                    "updated_at_ms": 1200,
                    "finished_at_ms": 1300,
                    "recoverable": True,
                },
            )

            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            self.assertEqual(app._find_recoverable_autonomous_session_run_id(), "")

    def test_live_snapshot_syncs_latest_run_from_disk_when_idle(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            first = app.start_demo_run(tick_count=1, tick_interval_ms=1, label="first run")
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
            live_first = app.get_live_snapshot()
            self.assertEqual(live_first["latest_run_id"], first["run_id"])

            second = app.start_demo_run(tick_count=1, tick_interval_ms=1, label="second run")
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
            latest_before = dict(app._latest_live)
            app._latest_live = {
                **latest_before,
                "latest_run_id": first["run_id"],
                "status": "completed",
                "active_run_id": "",
            }

            live_second = app.get_live_snapshot()
            self.assertEqual(live_second["latest_run_id"], second["run_id"])
            self.assertEqual(app.latest_run_id(), second["run_id"])


if __name__ == "__main__":
    unittest.main()
