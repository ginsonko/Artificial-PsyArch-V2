"""Strict acceptance test for ResidualBRecall-OrderedCompletion-1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_residual_b_recall_ordered_completion_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_residual_b_recall_ordered_completion_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_residual_recall_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_winners_unique_one_per_round():
    module = _load_module()
    p = module._probe()
    winners = [r["winner"] for r in p["rounds"]]
    assert len(winners) >= 4
    assert len(winners) == len(set(winners))
    assert all(winners)


def test_residual_mass_monotonic_decreasing():
    module = _load_module()
    p = module._probe()
    before = p["mass_curve"]["before"]
    assert all(before[i + 1] <= before[i] + 1e-9 for i in range(len(before) - 1))
    # Overall absorption: last after well below first before.
    assert p["mass_curve"]["after"][-1] < p["mass_curve"]["before"][0]


def test_each_round_absorbs_real_query_labels():
    module = _load_module()
    p = module._probe()
    for r in p["rounds"]:
        assert r["mass_after"] < r["mass_before"]
        assert r["matched_labels"]
