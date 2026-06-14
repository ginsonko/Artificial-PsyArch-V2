"""Strict acceptance test for FeedbackOverride-1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_feedback_override_mechanism_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_feedback_override_mechanism_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_feedback_override_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_wrong_action_driven_negative_and_right_positive():
    module = _load_module()
    p = module._probe()
    assert p["final"]["wrong_drive_bias"] < 0.0
    assert p["final"]["right_drive_bias"] > 0.0


def test_override_flip_is_significant():
    module = _load_module()
    p = module._probe()
    assert p["final"]["gap"] > 0.5


def test_counts_are_auditable():
    module = _load_module()
    p = module._probe()
    assert p["final"]["wrong_failure_count"] > 0
    assert p["final"]["wrong_success_count"] == 0
    assert p["final"]["right_success_count"] > 0


def test_wrong_bias_trajectory_is_monotone_down():
    module = _load_module()
    p = module._probe()
    traj = p["wrong_drive_bias_trajectory"]
    # Each step should not increase (monotone non-increasing under repeated punishment).
    assert all(traj[i + 1] <= traj[i] + 1e-9 for i in range(len(traj) - 1))
    assert traj[-1] < traj[0]
