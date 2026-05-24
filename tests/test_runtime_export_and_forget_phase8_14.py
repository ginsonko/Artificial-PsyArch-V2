# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from observatory_v2.app import ObservatoryV2App
from observatory_v2.app import AppError
from observatory_v2.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class RuntimeExportAndForgetPhase8To14Tests(unittest.TestCase):
    def test_export_import_and_forget_runtime(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app.start_text_run(texts=["今天 天气 不错", "我 想 出门", "如果 下雨 就 带 伞"], label="runtime test", tick_interval_ms=0)
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

            exported = app.export_runtime()
            self.assertIn("runtime", exported)
            self.assertIn("sandbox", exported)
            self.assertGreater(exported["runtime"]["memory_store"]["memory_count"], 0)
            self.assertIn("runtime_controls", exported["runtime"])
            self.assertIn("last_logic_ms", exported["runtime"])
            self.assertIn("action_learning", exported["runtime"])
            self.assertIn("tuner_learning", exported["runtime"])
            self.assertIn("last_control_feedback_context", exported["runtime"])

            checkpoint = Path(tmpdir) / "runtime_checkpoint.json"
            saved = app.save_checkpoint(checkpoint)
            self.assertTrue(saved["ok"])
            self.assertTrue(checkpoint.exists())

            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            loaded = restored.load_checkpoint(checkpoint)
            self.assertTrue(loaded["ok"])
            restored_export = restored.export_runtime()
            self.assertIn("runtime_controls", restored_export["runtime"])
            self.assertIn("tuner_learning", restored_export["runtime"])

            forget = restored.forget_cold_memories(keep_latest=2)
            self.assertIn("memory_count", forget)

            payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertIn("runtime", payload)
            bundle_dir = Path(tmpdir) / "layered_bundle"
            bundle_result = restored.export_memory_deployment_bundle(bundle_dir)
            self.assertTrue(bundle_result["ok"])
            self.assertEqual(bundle_result["bundle_format"], "layered_v2")
            imported_bundle = restored.import_memory_deployment_bundle(bundle_dir)
            self.assertTrue(imported_bundle["ok"])
            self.assertEqual(imported_bundle["loaded_via"], "layered_v2")

    def test_continue_from_checkpoint_keeps_memory_and_can_run(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app.start_text_run(texts=["今天 天气 不错", "我 想 出门"], label="before checkpoint", tick_interval_ms=0)
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))
            checkpoint = Path(tmpdir) / "continue_checkpoint.json"
            app.save_checkpoint(checkpoint)

            resumed = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            result = resumed.continue_from_checkpoint(
                checkpoint_path=checkpoint,
                texts=["今天 天气 有点 冷", "算了 不说了"],
                label="after checkpoint",
                tick_interval_ms=0,
            )
            self.assertTrue(resumed.wait_for_idle(timeout_sec=20.0))
            manifest = resumed.get_manifest(result["run_id"])
            self.assertEqual(manifest["status"], "completed")
            exported = resumed.export_runtime()
            self.assertGreater(exported["runtime"]["memory_store"]["memory_count"], 2)

    def test_forget_is_blocked_while_run_is_active(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            blocker = threading.Event()
            app._active_thread = threading.Thread(target=blocker.wait, daemon=True)
            app._active_thread.start()
            try:
                with self.assertRaises(AppError):
                    app.forget_cold_memories(keep_latest=2)
            finally:
                blocker.set()
                self.assertTrue(app.wait_for_idle(timeout_sec=2.0))

    def test_forget_summary_persists_across_service_restart_and_checkpoint_import(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app.start_text_run(texts=["今天 天气 不错", "我 想 出门", "如果 下雨 就 带 伞"], label="forget persistence", tick_interval_ms=0)
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

            forget = app.forget_cold_memories(
                keep_latest=2,
                strategy="score_prune",
                min_reality_weight=0.1,
                min_total_item_energy=0.0,
            )
            self.assertIn("generated_at_ms", forget)

            restored = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            restored_summary = restored.export_runtime_summary()
            self.assertEqual(
                restored_summary["export_meta"]["last_forget_summary"].get("generated_at_ms"),
                forget.get("generated_at_ms"),
            )

            checkpoint = Path(tmpdir) / "forget_persistence_checkpoint.json"
            saved = app.save_checkpoint(checkpoint)
            self.assertTrue(saved["ok"])

            imported = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=Path(tmpdir) / "imported")
            loaded = imported.load_checkpoint(checkpoint)
            self.assertTrue(loaded["ok"])
            imported_summary = imported.export_runtime_summary()
            self.assertEqual(
                imported_summary["export_meta"]["last_forget_summary"].get("generated_at_ms"),
                forget.get("generated_at_ms"),
            )

    def test_forget_dry_run_updates_preview_without_overwriting_applied_summary(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            app.start_text_run(texts=["a b c", "d e f", "g h i"], label="forget dry run", tick_interval_ms=0)
            self.assertTrue(app.wait_for_idle(timeout_sec=20.0))

            applied = app.forget_cold_memories(keep_latest=2, strategy="latest_only", dry_run=False)
            preview = app.forget_cold_memories(keep_latest=1, strategy="score_prune", dry_run=True)
            summary = app.export_runtime_summary()
            export_meta = summary.get("export_meta", {})

            self.assertEqual(export_meta["last_forget_summary"].get("generated_at_ms"), applied.get("generated_at_ms"))
            self.assertEqual(export_meta["last_forget_summary"].get("strategy"), "latest_only")
            self.assertEqual(export_meta["last_forget_preview_summary"].get("generated_at_ms"), preview.get("generated_at_ms"))
            self.assertTrue(bool(export_meta["last_forget_preview_summary"].get("dry_run", False)))


if __name__ == "__main__":
    unittest.main()
