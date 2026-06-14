"""Strict acceptance test for AttentionBand-OnlineVectorIntegration-1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_attention_band_online_vector_integration_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_attention_band_online_vector_integration_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_attention_band_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_main_vector_channel_unchanged_across_learned_weight():
    module = _load_module()
    ct = module._probe()["channel_table"]
    assert ct["vector_score"]["off"] == ct["vector_score"]["default"]


def test_learned_channel_measurements_are_stable():
    module = _load_module()
    ct = module._probe()["channel_table"]
    assert ct["learned_score"]["off"] == ct["learned_score"]["default"]
    assert ct["learned_vector_score"]["off"] == ct["learned_vector_score"]["default"]


def test_learned_vector_coefficient_is_bounded():
    module = _load_module()
    p = module._probe()
    assert p["effective_learned_vector_coeff"]["high"] == p["learned_vector_cap"]
    assert p["effective_learned_vector_coeff"]["default"] <= p["learned_vector_cap"]
