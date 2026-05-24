# -*- coding: utf-8 -*-
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from observatory_v2 import __main__ as cli_main
from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


class CliMemoryBundleCommandsTests(unittest.TestCase):
    def test_parser_supports_memory_bundle_commands(self) -> None:
        parser = cli_main.build_parser()
        export_args = parser.parse_args(["export-memory-bundle", "--out-dir", "X:/bundle"])
        self.assertEqual(export_args.command, "export-memory-bundle")
        self.assertEqual(export_args.out_dir, "X:/bundle")

        inspect_args = parser.parse_args(["inspect-memory-bundle", "--dir", "X:/bundle"])
        self.assertEqual(inspect_args.command, "inspect-memory-bundle")
        self.assertEqual(inspect_args.directory, "X:/bundle")

        import_args = parser.parse_args(["import-memory-bundle", "--dir", "X:/bundle"])
        self.assertEqual(import_args.command, "import-memory-bundle")
        self.assertEqual(import_args.directory, "X:/bundle")

        forget_args = parser.parse_args(
            [
                "forget",
                "--keep-latest",
                "16",
                "--strategy",
                "score_prune",
                "--min-reality-weight",
                "0.8",
                "--min-total-item-energy",
                "1.5",
                "--protect-memory-kind",
                "teacher_feedback",
                "--max-memory-count",
                "64",
                "--dry-run",
            ]
        )
        self.assertEqual(forget_args.command, "forget")
        self.assertEqual(forget_args.strategy, "score_prune")
        self.assertEqual(forget_args.keep_latest, 16)
        self.assertTrue(forget_args.dry_run)

    def test_main_can_export_inspect_and_import_memory_bundle(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_root = Path(tmpdir) / "outputs"
            bundle_dir = Path(tmpdir) / "bundle"
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=str(outputs_root))
            app.start_text_run(texts=["今天 天气 不错", "我 想 出门"], label="cli memory bundle", tick_interval_ms=0)
            self.assertTrue(app.wait_for_idle(timeout_sec=10.0))

            with patch.object(cli_main, "load_config", return_value=config), patch.object(
                cli_main, "ObservatoryV2App", return_value=app
            ):
                with patch("sys.argv", ["observatory_v2", "export-memory-bundle", "--out-dir", str(bundle_dir)]):
                    cli_main.main()
                self.assertTrue((bundle_dir / "bundle_meta.json").exists())

                with patch("sys.argv", ["observatory_v2", "inspect-memory-bundle", "--dir", str(bundle_dir)]):
                    cli_main.main()

                with patch("sys.argv", ["observatory_v2", "import-memory-bundle", "--dir", str(bundle_dir)]):
                    cli_main.main()


if __name__ == "__main__":
    unittest.main()
