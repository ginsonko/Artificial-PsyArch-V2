# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory.memory_store_v2 import MemoryStoreV2


class DeploymentBundleLayeredV2Tests(unittest.TestCase):
    def test_layered_bundle_contains_meta_and_component_files(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=3,
            memory_kind="visual",
            units=["apple"],
            items=[
                {
                    "sa_label": "vision::apple",
                    "display_text": "apple",
                    "energy": 1.0,
                    "coords": {"cx": 0.4, "cy": 0.6, "z": 0.0},
                    "channel": "vision",
                }
            ],
            text="apple",
            reality_weight=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            result = store.save_deployment_bundle(bundle_dir)
            self.assertTrue(result["ok"])
            meta = json.loads((bundle_dir / "bundle_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["schema_id"], "memory_store_bundle/v1")
            self.assertEqual(meta["schema_version"], "2.0")
            self.assertIn("files", meta)
            self.assertTrue((bundle_dir / meta["files"]["memories_jsonl"]).exists())
            self.assertTrue((bundle_dir / meta["files"]["posting_json"]).exists())
            self.assertTrue((bundle_dir / meta["files"]["vector_meta"]).exists())
            self.assertTrue((bundle_dir / meta["files"]["spacetime_meta"]).exists())

    def test_layered_bundle_can_restore_without_legacy_json_path(self) -> None:
        store = MemoryStoreV2(vector_dim=64, ann_enabled=False)
        store.write_memory(
            tick_index=5,
            memory_kind="exact_external",
            units=["today", "weather"],
            items=[
                {"sa_label": "text::today", "display_text": "today", "energy": 1.0},
                {"sa_label": "text::weather", "display_text": "weather", "energy": 1.0},
            ],
            text="today weather",
            reality_weight=1.0,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir)
            store.save_deployment_bundle(bundle_dir)
            legacy_path = bundle_dir / "memory_store_v2.json"
            if legacy_path.exists():
                legacy_path.unlink()
            restored = MemoryStoreV2(vector_dim=64, ann_enabled=False)
            loaded = restored.load_deployment_bundle(bundle_dir)
            self.assertTrue(loaded["ok"])
            self.assertEqual(loaded["loaded_via"], "layered_v2")
            self.assertEqual(restored.count(), 1)


if __name__ == "__main__":
    unittest.main()
