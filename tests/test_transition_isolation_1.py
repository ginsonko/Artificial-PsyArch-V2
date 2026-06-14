"""Strict acceptance test for TransitionIsolation-1.

Locks the bottom-loop claim that training A->B successor strengthens the directed
transition score while leaving the symmetric concept similarity (learned vector
and learned similarity) untouched, and keeps the successor directed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "transition_isolation_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("transition_isolation_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_transition_isolation_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_transition_rises_but_concept_similarity_unchanged():
    module = _load_module()
    direct = module._direct_isolation_probe()
    s = direct["scores"]
    # Successor strengthens.
    assert s["transition_after"] > s["transition_before"]
    assert s["transition_after"] > 0.5
    # Concept similarity does not move because of transition.
    assert abs(s["vec_sim_after"] - s["vec_sim_before"]) < module.UNCHANGED_TOLERANCE
    assert abs(s["sim_after"] - s["sim_before"]) < module.UNCHANGED_TOLERANCE


def test_transition_is_directed():
    module = _load_module()
    direct = module._direct_isolation_probe()
    s = direct["scores"]
    assert s["transition_after"] > s["transition_reverse"]


def test_pair_evidence_is_pure_transition():
    module = _load_module()
    direct = module._direct_isolation_probe()
    ev = direct["pair_evidence"]
    assert ev["transition_raw"] > 0.0
    assert ev["positive_raw"] == 0.0


def test_successor_recall_exposes_positive_transition():
    module = _load_module()
    recall = module._recall_successor_probe()
    assert recall["top_learned_transition_score"] > 0.0
