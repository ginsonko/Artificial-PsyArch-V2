"""Strict acceptance test for ShortTermSlotHistoryReload-1."""

from __future__ import annotations

import importlib.util
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_short_term_slot_history_reload_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_short_term_slot_history_reload_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _probe():
    module = _load_module()
    with tempfile.TemporaryDirectory() as tmp:
        return module._probe(Path(tmp))


def test_slot_history_reload_all_checks_pass():
    probe = _probe()
    assert probe["passed"] is True, probe["checks"]


def test_items_identical_after_reload():
    probe = _probe()
    assert probe["checks"]["packet_count_match"] is True
    assert probe["checks"]["items_identical_after_reload"] is True


def test_recall_identical_after_reload():
    probe = _probe()
    assert probe["recall_original"] == probe["recall_reloaded"]
    assert len(probe["recall_original"]) > 0


def test_real_file_boundary_is_hashable():
    probe = _probe()
    assert len(probe["jsonl_sha256"]) == 64
