"""Strict acceptance test for ActionFeedbackWriteback-TraceAudit-1."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "apv2_action_feedback_writeback_trace_audit_1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("apv2_action_feedback_writeback_trace_audit_1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_action_feedback_writeback_all_checks_pass():
    module = _load_module()
    payload = module._build_payload()
    assert payload["summary"]["passed"] is True, payload["summary"]["checks"]


def test_feedback_sa_built_and_punishment_is_virtual():
    module = _load_module()
    p = module._probe()
    assert p["feedback_sa_label"].startswith("action_feedback::")
    assert p["real_energy"] == 0.0
    assert p["virtual_energy"] > 0.0


def test_negative_drive_bias_and_provenance_complete():
    module = _load_module()
    p = module._probe()
    assert p["outcome_drive_bias"] < 0.0
    for key in ["action_id", "observed_feedback", "outcome_memory_estimate", "feedback_energy_semantics", "causal_window"]:
        assert key in p["provenance_present"]


def test_feedback_written_and_recallable():
    module = _load_module()
    p = module._probe()
    assert p["recall_top_score"] > 0.0
