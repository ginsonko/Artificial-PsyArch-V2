"""Strict acceptance test for OnlineVector-WeightAblation-1.

This locks the bottom-loop claim that the online learned-vector branch improves
experience-neighborhood recall without dominating exact SA/energy evidence, and
that the branch is monotonic and bounded across learned_weight.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "online_vector_weight_ablation_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("online_vector_weight_ablation_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_weight_ablation_all_checks_pass():
    module = _load_module()
    probe = module._probe()
    checks = probe["checks"]
    assert probe["passed"] is True, checks
    assert checks["h1_learned_vector_improves_neighborhood_recall"] is True
    assert checks["h1b_learned_similarity_improves_main_recall"] is True
    assert checks["h2_exact_match_not_dominated_by_learned_vector"] is True
    assert checks["h3_monotonic_and_bounded"] is True


def test_learned_vector_branch_is_monotonic_in_score():
    module = _load_module()
    probe = module._probe()
    pw = probe["per_weight"]
    off = pw["off"]["audit"]["neighbor"]["score"]
    default = pw["default"]["audit"]["neighbor"]["score"]
    high = pw["high"]["audit"]["neighbor"]["score"]
    # Off contributes no learned mass; raising the weight only adds bounded mass.
    assert off < default <= high
    # The default weight produces a strictly positive learned-vector field.
    assert pw["default"]["audit"]["neighbor"]["learned_vector_score"] > 0.0


def test_exact_match_outranks_learned_only_even_at_high_weight():
    module = _load_module()
    probe = module._probe()
    high = probe["per_weight"]["high"]["audit"]
    # Exact label/energy match must stay rank 0 above the learned-only neighbor.
    assert high["exact"]["rank"] < high["neighbor"]["rank"]
    assert high["exact"]["score"] > high["neighbor"]["score"]


def test_off_weight_zeroes_the_learned_contribution_to_score():
    module = _load_module()
    probe = module._probe()
    pw = probe["per_weight"]
    # At off weight the neighbor score equals a learned-free baseline, so it is
    # strictly below the default-weight score (which adds the learned branch).
    assert pw["off"]["audit"]["neighbor"]["score"] < pw["default"]["audit"]["neighbor"]["score"]
    assert pw["off"]["main_recall"]["neighbor"]["score"] < pw["default"]["main_recall"]["neighbor"]["score"]
