"""Strict acceptance test for CognitiveFeeling-CausalAblation-2."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_cognitive_feeling_causal_ablation_2.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_cognitive_feeling_causal_ablation_2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_causal_ablation_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_relation_feelings_target_relation_head():
    module = _load_module()
    rc = module._probe()["matrix"]["relation_case"]["ablate"]
    assert rc["teacher_context"]["drops"]["relation_trigger"] > 0.0
    assert rc["teacher_context"]["drops"]["local_repair"] == 0.0


def test_repair_feelings_target_repair_head():
    module = _load_module()
    pc = module._probe()["matrix"]["repair_case"]["ablate"]
    assert pc["mismatch"]["drops"]["local_repair"] > 0.0
    assert pc["mismatch"]["drops"]["relation_trigger"] == 0.0


def test_matrix_is_diagonal_and_all_feelings_contribute():
    module = _load_module()
    checks = module._probe()["checks"]
    assert checks["causal_matrix_is_diagonal"] is True
    assert checks["every_feeling_has_nonzero_contribution"] is True
