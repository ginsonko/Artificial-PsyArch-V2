"""Strict acceptance test for OnlineVector-NegativePressure-1.

Locks the bottom-loop claim that repeated negative cognitive pressure prunes a
wrong over-predicted subject away from its real context, without damaging the
correct association, and that the correct subject wins audit-path recall.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "online_vector_negative_pressure_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("online_vector_negative_pressure_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_negative_pressure_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_wrong_residue_pushed_to_non_positive():
    module = _load_module()
    direct = module._direct_pressure_probe()
    scores = direct["scores"]
    # The wrong residue starts positive and is pushed to non-positive similarity.
    assert scores["wrong_before"] > 0.0
    assert scores["wrong_after"] <= 0.0
    assert scores["wrong_after"] < scores["wrong_before"]


def test_correct_association_preserved():
    module = _load_module()
    direct = module._direct_pressure_probe()
    scores = direct["scores"]
    # Targeted pressure must not damage the correct association beyond tolerance.
    assert scores["good_after"] >= scores["good_before"] - module.GOOD_PRESERVE_TOLERANCE


def test_negative_evidence_is_white_box_visible():
    module = _load_module()
    direct = module._direct_pressure_probe()
    ev = direct["pair_evidence"]
    assert ev["negative_raw"] > 0.0
    assert ev["source_negative_support"] > 0.0


def test_correct_subject_wins_audit_recall():
    module = _load_module()
    recall = module._recall_competition_probe()
    assert recall["good_snapshot_learned_vector_score"] > recall["wrong_snapshot_learned_vector_score"]
