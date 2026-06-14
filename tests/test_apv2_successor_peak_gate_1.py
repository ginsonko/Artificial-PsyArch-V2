"""Strict acceptance test for SuccessorPeakGate-1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_successor_peak_gate_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_successor_peak_gate_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_successor_peak_gate_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_sharp_peak_drives_stronger_bias_than_blur():
    module = _load_module()
    gate = module._gate_probe()
    assert gate["sharp"]["top_bias"] > gate["blur"]["top_bias"]
    # Sharp context should be near full gain; blurry far below it.
    assert gate["sharp"]["top_bias"] >= 4 * gate["blur"]["top_bias"]


def test_gating_is_entropy_driven():
    module = _load_module()
    gate = module._gate_probe()
    assert gate["sharp"]["entropy"] < gate["blur"]["entropy"]
    assert abs(gate["blur"]["damping"] - gate["entropy_floor"]) < 0.02


def test_gate_is_soft_not_hard():
    module = _load_module()
    gate = module._gate_probe()
    # Blurry context is suppressed but not hard-zeroed.
    assert gate["blur"]["top_bias"] > 0.0


def test_lag_kernel_has_next_beat_peak_and_tail():
    module = _load_module()
    lag = module._lag_kernel_probe()["lag_kernel"]
    assert lag["lag1"] > lag["lag2"]
    assert (lag["lag1"] / lag["lag2"]) > 2.0
    assert lag["lag2"] > lag["lag3"] > lag["lag4"] > 0.0
