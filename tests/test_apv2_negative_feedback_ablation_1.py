"""Strict acceptance test for NegativeFeedback-Ablation-1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_negative_feedback_ablation_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_negative_feedback_ablation_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_negative_feedback_ablation_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_negative_feedback_suppresses_error():
    module = _load_module()
    p = module._probe()
    assert p["summary_finals"]["full"] < 0.0


def test_ablation_removes_suppression():
    module = _load_module()
    p = module._probe()
    full = p["summary_finals"]["full"]
    no_neg = p["summary_finals"]["no_negative"]
    # Without the negative signal the error action is not suppressed.
    assert (no_neg - full) > 0.2
    assert abs(no_neg) < 0.05


def test_outcome_memory_is_not_dead():
    module = _load_module()
    p = module._probe()
    # Positive control moves the same machinery upward, proving ablation is causal.
    assert p["summary_finals"]["positive_control"] > 0.0
