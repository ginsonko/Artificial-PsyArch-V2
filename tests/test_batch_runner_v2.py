# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
from observatory_v2 import __main__ as cli_main
from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config


def fake_grab():
    from PIL import Image

    image = Image.new("RGB", (96, 64), color=(18, 18, 18))
    for x in range(32, 64):
        for y in range(16, 48):
            image.putpixel((x, y), (240, 240, 240))
    return image


class BatchRunnerV2Tests(unittest.TestCase):
    def _run_batch(self, dataset_path: Path, outputs_root: Path) -> dict:
        command = [
            sys.executable,
            "scripts/batch_runner_v2.py",
            "--dataset",
            str(dataset_path),
            "--label",
            "test batch",
            "--outputs-root",
            str(outputs_root),
        ]
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            env={**os.environ, **{"PYTHONIOENCODING": "utf-8"}},
            check=True,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["dataset"], str(dataset_path.resolve()))
        self.assertGreaterEqual(int(payload.get("run_count", 0) or 0), 1)
        return payload

    def test_batch_runner_supports_text_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset_text.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "label": "text dataset test",
                        "texts": ["今天 天气 不错", "我 想 出门", "如果 下雨 就 带 伞"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload = self._run_batch(dataset_path, Path(tmpdir))
            run = payload["runs"][0]
            self.assertEqual(run["mode"], "text")
            self.assertEqual(run["manifest"]["status"], "completed")
            self.assertEqual(int(run["manifest"].get("tick_done", 0) or 0), 3)

    def test_batch_runner_supports_multimodal_items_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = REPO_ROOT / "config" / "sample_dataset_multimodal.json"
            payload = self._run_batch(dataset_path, Path(tmpdir))
            run = payload["runs"][0]
            self.assertEqual(run["mode"], "multimodal")
            self.assertEqual(run["manifest"]["status"], "completed")
            self.assertEqual(int(run["manifest"].get("tick_done", 0) or 0), 3)

    def test_batch_runner_supports_multiple_runs_in_one_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset_runs.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "label": "batch run #1",
                                "texts": ["今天 天气 不错", "我 想 出门"],
                            },
                            {
                                "label": "batch run #2",
                                "mode": "multimodal",
                                "items": [
                                    {"text": "今天 天气 不错"},
                                    {"text": "算了 不说了"},
                                ],
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload = self._run_batch(dataset_path, Path(tmpdir))
            self.assertEqual(int(payload.get("run_count", 0) or 0), 2)
            self.assertEqual(payload["runs"][0]["manifest"]["status"], "completed")
            self.assertEqual(payload["runs"][1]["manifest"]["status"], "completed")

    def test_cli_parser_supports_run_dataset(self) -> None:
        parser = cli_main.build_parser()
        args = parser.parse_args(["run-dataset", "--dataset", "config/sample_dataset_text.json"])
        self.assertEqual(args.command, "run-dataset")
        self.assertEqual(args.dataset, "config/sample_dataset_text.json")

    def test_dataset_runner_supports_checkpoint_bundle_and_continue_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset_pipeline.json"
            outputs_root = root / "outputs"
            bundle_dir = outputs_root / "bundle_pipe"
            checkpoint_path = outputs_root / "runtime_pipe.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "label": "pipe #1",
                                "texts": ["今天 天气 不错", "我 想 出门"],
                                "after": {
                                    "save_checkpoint_path": "outputs/runtime_pipe.json",
                                    "export_memory_bundle_dir": "outputs/bundle_pipe",
                                    "inspect_memory_bundle": True,
                                },
                            },
                            {
                                "label": "pipe #2",
                                "mode": "continue_from_checkpoint",
                                "checkpoint_path": "outputs/runtime_pipe.json",
                                "texts": ["今天 天气 有点 冷", "算了 不说了"],
                                "after": {
                                    "forget_keep_latest": 4,
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload = self._run_batch(dataset_path, outputs_root)
            self.assertEqual(payload["run_count"], 2)
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(bundle_dir.exists())
            first_after = payload["runs"][0]["artifacts"]["after"]
            self.assertTrue(first_after["save_checkpoint"]["ok"])
            self.assertTrue(first_after["export_memory_bundle"]["ok"])
            self.assertTrue(first_after["inspect_memory_bundle"]["ok"])
            second_after = payload["runs"][1]["artifacts"]["after"]
            self.assertIn("forget", second_after)
            self.assertEqual(payload["runs"][1]["manifest"]["status"], "completed")

    def test_dataset_runner_supports_richer_forget_hook(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset_richer_forget.json"
            outputs_root = root / "outputs"
            checkpoint_path = outputs_root / "runtime_pipe.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "label": "forget pipe #1",
                                "texts": ["今天 天气 不错", "我 想 出门"],
                                "after": {
                                    "save_checkpoint_path": "outputs/runtime_pipe.json"
                                },
                            },
                            {
                                "label": "forget pipe #2",
                                "mode": "continue_from_checkpoint",
                                "checkpoint_path": "outputs/runtime_pipe.json",
                                "texts": ["今天 天气 有点 冷", "算了 不说了"],
                                "after": {
                                    "forget": {
                                        "keep_latest": 4,
                                        "strategy": "score_prune",
                                        "min_reality_weight": 0.4,
                                        "min_total_item_energy": 0.2,
                                        "protect_memory_kinds": ["teacher_feedback"],
                                        "max_memory_count": 8,
                                        "dry_run": True
                                    }
                                },
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload = self._run_batch(dataset_path, outputs_root)
            self.assertTrue(checkpoint_path.exists())
            second_after = payload["runs"][1]["artifacts"]["after"]
            self.assertIn("forget", second_after)
            self.assertTrue(second_after["forget"]["dry_run"])
            self.assertEqual(second_after["forget"]["strategy"], "score_prune")
            self.assertIn("kind_histogram_before", second_after["forget"])

    def test_dataset_runner_supports_autonomous_session_pipeline_and_hooks(self) -> None:
        from observatory_v2.dataset_runner import run_dataset_file

        config = load_config(
            overrides={
                "executor_enabled": False,
                "executor_screenshot_enabled": True,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            root = Path(tmpdir)
            dataset_path = root / "dataset_autonomous_pipeline.json"
            status_file = root / "session_status_snapshot.json"
            checkpoint_path = root / "auto_session_checkpoint.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "mode": "autonomous_session",
                                "label": "autonomous session pipeline",
                                "tick_interval_ms": 5,
                                "max_ticks": 4,
                                "text_hint": "dataset autonomous session",
                                "after": {
                                    "pause_session": {"delay_ms": 30, "timeout_sec": 10},
                                    "resume_session": {"delay_ms": 20, "timeout_sec": 10},
                                    "wait_for_session": {"timeout_sec": 20},
                                    "status_snapshot": {"path": "session_status_snapshot.json"},
                                    "save_checkpoint_path": "auto_session_checkpoint.json",
                                },
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            payload = run_dataset_file(dataset_path, app=app, timeout_sec=30.0)
            self.assertEqual(payload["run_count"], 1)
            run = payload["runs"][0]
            self.assertEqual(run["mode"], "autonomous_session")
            self.assertEqual(run["manifest"]["status"], "completed")
            self.assertEqual(int(run["manifest"].get("tick_done", 0) or 0), 4)
            after = run["artifacts"]["after"]
            self.assertTrue(after["pause_session"]["request"]["ok"])
            self.assertTrue(after["pause_session"]["wait"]["ok"])
            self.assertTrue(after["resume_session"]["request"]["ok"])
            self.assertTrue(after["resume_session"]["wait"]["ok"])
            self.assertTrue(after["wait_for_session"]["ok"])
            self.assertTrue(after["save_checkpoint"]["ok"])
            self.assertTrue(status_file.exists())
            self.assertTrue(checkpoint_path.exists())
            snapshot = json.loads(status_file.read_text(encoding="utf-8"))
            self.assertEqual(str((snapshot.get("status", {}) or {}).get("status", "")), "completed")

    def test_dataset_runner_propagates_external_teacher_retry_controls(self) -> None:
        from observatory_v2.dataset_runner import run_dataset_file

        config = load_config(
            overrides={
                "executor_enabled": False,
                "executor_screenshot_enabled": True,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch("observatory_v2.computer_executor.ImageGrab.grab", side_effect=fake_grab):
            root = Path(tmpdir)
            dataset_path = root / "dataset_external_teacher_controls.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "mode": "autonomous_run",
                                "label": "external teacher controls",
                                "ticks": 1,
                                "text_hint": "dataset teacher controls",
                                "external_teacher_enabled": True,
                                "external_teacher_mode": "stub_file",
                                "external_teacher_stub_response_path": "C:/tmp/teacher_stub.json",
                                "external_teacher_fail_open": False,
                                "external_teacher_max_retries": 3,
                                "external_teacher_retry_backoff_ms": 0,
                                "external_teacher_http_endpoint": "http://127.0.0.1:8877/teacher",
                                "external_teacher_http_headers": {"X-Teacher": "dataset"},
                            }
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            payload = run_dataset_file(dataset_path, app=app, timeout_sec=20.0)
            self.assertEqual(payload["run_count"], 1)
            run = payload["runs"][0]
            sidecar = app.get_tick_sidecar(str(run["result"]["run_id"]), 0)
            tick_meta = dict((sidecar.get("autonomous_sidecar", {}) or {}).get("tick_meta", {}) or {})
            self.assertTrue(tick_meta.get("external_teacher_enabled"))
            self.assertEqual(tick_meta.get("external_teacher_mode"), "stub_file")
            self.assertEqual(tick_meta.get("external_teacher_stub_response_path"), "C:/tmp/teacher_stub.json")
            self.assertFalse(tick_meta.get("external_teacher_fail_open"))
            self.assertEqual(tick_meta.get("external_teacher_max_retries"), 3)
            self.assertEqual(tick_meta.get("external_teacher_retry_backoff_ms"), 0)
            self.assertEqual(tick_meta.get("external_teacher_http_endpoint"), "http://127.0.0.1:8877/teacher")
            self.assertEqual(tick_meta.get("external_teacher_http_headers", {}).get("X-Teacher"), "dataset")

    def test_dataset_runner_supports_recover_autonomous_session_and_status_controls(self) -> None:
        from observatory_v2.dataset_runner import run_dataset_file

        config = load_config(
            overrides={
                "executor_enabled": False,
                "executor_screenshot_enabled": False,
                "autonomous_capture_required": False,
            }
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset_recover_pipeline.json"
            before_stop_snapshot = root / "before_stop_status.json"
            final_snapshot = root / "final_status.json"
            checkpoint_path = root / "recover_checkpoint.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "runs": [
                            {
                                "mode": "autonomous_session",
                                "label": "recoverable session start",
                                "tick_interval_ms": 5,
                                "max_ticks": 6,
                                "text_hint": "recover start",
                                "after": {
                                    "status_snapshot": {"path": "before_stop_status.json"},
                                    "stop_session": {"delay_ms": 30, "timeout_sec": 20},
                                    "save_checkpoint_path": "recover_checkpoint.json",
                                },
                            },
                            {
                                "mode": "recover_autonomous_session",
                                "label": "recover session",
                                "after": {
                                    "wait_for_session": {"timeout_sec": 20},
                                    "status_snapshot": {"path": "final_status.json"},
                                },
                            },
                            {
                                "mode": "autonomous_session_status",
                                "label": "final status row",
                            },
                        ]
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            payload = run_dataset_file(dataset_path, app=app, timeout_sec=30.0)
            self.assertEqual(payload["run_count"], 3)
            first_run = payload["runs"][0]
            self.assertEqual(first_run["manifest"]["status"], "stopped")
            self.assertTrue(first_run["artifacts"]["after"]["stop_session"]["request"]["ok"])
            self.assertTrue(checkpoint_path.exists())
            self.assertTrue(before_stop_snapshot.exists())

            second_run = payload["runs"][1]
            self.assertEqual(second_run["mode"], "recover_autonomous_session")
            self.assertEqual(second_run["manifest"]["status"], "completed")
            self.assertTrue(second_run["artifacts"]["after"]["wait_for_session"]["ok"])
            self.assertTrue(final_snapshot.exists())

            third_run = payload["runs"][2]
            self.assertEqual(third_run["mode"], "autonomous_session_status")
            self.assertEqual(str((third_run["result"].get("status", {}) or {}).get("status", "")), "completed")

    def test_dataset_runner_supports_config_overrides_for_ap_agent_coupling_smoke(self) -> None:
        from observatory_v2.dataset_runner import run_dataset_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "dataset_ap_agent.json"
            dataset_path.write_text(
                json.dumps(
                    {
                        "config_overrides": {
                            "executor_enabled": True,
                            "executor_dry_run": True,
                            "executor_screenshot_enabled": False,
                            "autonomous_external_teacher_enabled": False,
                        },
                        "runs": [
                            {
                                "label": "ap-agent dataset smoke",
                                "texts": ["今天 天气 不错", "今天 天气 不错", "我 想 出门"],
                            }
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            payload = run_dataset_file(dataset_path, timeout_sec=20.0, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            self.assertEqual(payload["run_count"], 1)
            self.assertEqual(payload["config_overrides"]["executor_enabled"], True)
            run = payload["runs"][0]
            self.assertEqual(run["manifest"]["status"], "completed")
            run_id = str(run["result"]["run_id"])
            app = ObservatoryV2App(config=load_config(overrides={"executor_enabled": True, "executor_dry_run": True}), repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            sidecar1 = app.get_tick_sidecar(run_id, 1)
            sidecar2 = app.get_tick_sidecar(run_id, 2)
            rules_result_1 = dict(sidecar1.get("rules_result", {}) or {})
            teacher_review_1 = dict(sidecar1.get("teacher_review", {}) or {})
            sandbox_result_1 = dict(sidecar1.get("sandbox_result", {}) or {})
            teacher_feedback_1 = dict(sidecar1.get("teacher_feedback", {}) or {})
            self.assertIn(
                "action::continue_focus",
                [str(item.get("action_id", "") or "") for item in (rules_result_1.get("action_drives", []) or [])],
            )
            self.assertIn(
                "action::continue_focus",
                [str(item.get("action_id", "") or "") for item in (teacher_review_1.get("scored_action_drives", []) or [])],
            )
            self.assertEqual(
                len(teacher_review_1.get("planner_selected_action_drives", []) or []),
                len(rules_result_1.get("planned_selected_actions_preview", []) or []),
            )
            self.assertIn(
                "action::continue_focus",
                [str(item.get("action_id", "") or "") for item in (sandbox_result_1.get("selected_actions", []) or [])],
            )
            self.assertGreater(float(teacher_feedback_1.get("reward", 0.0) or 0.0), 0.0)
            action_drives_2 = list((dict(sidecar2.get("rules_result", {}) or {})).get("action_drives", []) or [])
            continue_focus_row = next(
                (item for item in action_drives_2 if str(item.get("action_id", "") or "") == "action::continue_focus"),
                {},
            )
            self.assertGreater(float(continue_focus_row.get("learned_bias", 0.0) or 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
