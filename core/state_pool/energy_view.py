from __future__ import annotations

from collections import Counter, defaultdict


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clean_label(value) -> str:
    return str(value or "").strip()


def _mass_template() -> dict:
    return {
        "item_count": 0,
        "real_energy": 0.0,
        "virtual_energy": 0.0,
        "net_pressure": 0.0,
        "abs_pressure": 0.0,
        "positive_pressure": 0.0,
        "negative_pressure": 0.0,
    }


def _add_mass(bucket: dict, row: dict) -> None:
    real = float((row or {}).get("real_energy", 0.0) or 0.0)
    virtual = float((row or {}).get("virtual_energy", 0.0) or 0.0)
    pressure = float((row or {}).get("cognitive_pressure", real - virtual) or 0.0)
    bucket["item_count"] = int(bucket.get("item_count", 0) or 0) + 1
    bucket["real_energy"] = float(bucket.get("real_energy", 0.0) or 0.0) + real
    bucket["virtual_energy"] = float(bucket.get("virtual_energy", 0.0) or 0.0) + virtual
    bucket["net_pressure"] = float(bucket.get("net_pressure", 0.0) or 0.0) + pressure
    bucket["abs_pressure"] = float(bucket.get("abs_pressure", 0.0) or 0.0) + abs(pressure)
    bucket["positive_pressure"] = float(bucket.get("positive_pressure", 0.0) or 0.0) + max(0.0, pressure)
    bucket["negative_pressure"] = float(bucket.get("negative_pressure", 0.0) or 0.0) + max(0.0, -pressure)


def _round_mass(bucket: dict) -> dict:
    return {
        "item_count": int(bucket.get("item_count", 0) or 0),
        "real_energy": _round4(float(bucket.get("real_energy", 0.0) or 0.0)),
        "virtual_energy": _round4(float(bucket.get("virtual_energy", 0.0) or 0.0)),
        "net_pressure": _round4(float(bucket.get("net_pressure", 0.0) or 0.0)),
        "abs_pressure": _round4(float(bucket.get("abs_pressure", 0.0) or 0.0)),
        "positive_pressure": _round4(float(bucket.get("positive_pressure", 0.0) or 0.0)),
        "negative_pressure": _round4(float(bucket.get("negative_pressure", 0.0) or 0.0)),
    }


def _top_mass_groups(groups: dict[str, dict], *, limit: int = 12) -> dict:
    rows = sorted(
        groups.items(),
        key=lambda pair: (
            -float(pair[1].get("abs_pressure", 0.0) or 0.0),
            -float(pair[1].get("real_energy", 0.0) or 0.0),
            str(pair[0]),
        ),
    )
    return {key: _round_mass(value) for key, value in rows[: max(1, int(limit))]}


def _pressure_peaks(items: list[dict], *, limit: int = 8) -> dict:
    positive = []
    negative = []
    for row in items:
        label = _clean_label((row or {}).get("sa_label", ""))
        if not label:
            continue
        real = float((row or {}).get("real_energy", 0.0) or 0.0)
        virtual = float((row or {}).get("virtual_energy", 0.0) or 0.0)
        pressure = float((row or {}).get("cognitive_pressure", real - virtual) or 0.0)
        public = {
            "sa_label": label,
            "family": _clean_label((row or {}).get("family", "")),
            "source_type": _clean_label((row or {}).get("source_type", "")),
            "real_energy": _round4(real),
            "virtual_energy": _round4(virtual),
            "cognitive_pressure": _round4(pressure),
        }
        if pressure > 0:
            positive.append(public)
        elif pressure < 0:
            negative.append(public)
    positive.sort(key=lambda item: (-float(item["cognitive_pressure"]), str(item["sa_label"])))
    negative.sort(key=lambda item: (float(item["cognitive_pressure"]), str(item["sa_label"])))
    return {
        "positive": positive[: max(1, int(limit))],
        "negative": negative[: max(1, int(limit))],
    }


def _external_count(items: list[dict]) -> int:
    count = 0
    for row in items:
        source = _clean_label((row or {}).get("source_type", ""))
        family = _clean_label((row or {}).get("family", ""))
        label = _clean_label((row or {}).get("sa_label", ""))
        if source in {"external_text", "vision_numeric", "audio_numeric", "action_feedback"}:
            count += 1
        elif source.startswith(("vision_bridge", "audio_bridge")):
            count += 1
        elif family.startswith(("vision", "audio")) or label.startswith(("vision::", "vision_obj::", "audio::", "audio_event::")):
            count += 1
    return count


def _prediction_validation(prediction_trace: dict) -> dict:
    trace = dict(prediction_trace or {})
    return {
        "schema_id": "prediction_validation_compact/v1",
        "alignment_score": _round4(float(trace.get("alignment_score", 0.0) or 0.0)),
        "mismatch_ratio": _round4(float(trace.get("mismatch_ratio", 0.0) or 0.0)),
        "predicted_mass": _round4(float(trace.get("predicted_mass", 0.0) or 0.0)),
        "actual_mass": _round4(float(trace.get("actual_mass", 0.0) or 0.0)),
        "match_mass": _round4(float(trace.get("match_mass", 0.0) or 0.0)),
        "missed_expected_mass": _round4(float(trace.get("missed_expected_mass", 0.0) or 0.0)),
        "unexpected_novelty_mass": _round4(float(trace.get("unexpected_novelty_mass", 0.0) or 0.0)),
        "match_count": int(trace.get("match_count", 0) or 0),
        "missed_count": int(trace.get("missed_count", 0) or 0),
        "unexpected_count": int(trace.get("unexpected_count", 0) or 0),
        "matched_labels": list(trace.get("matched_labels", []) or [])[:8],
        "missed_predicted_labels": list(trace.get("missed_predicted_labels", []) or [])[:8],
        "unexpected_labels": list(trace.get("unexpected_labels", []) or [])[:8],
    }


def _residual_semantics(residual_summary: dict) -> dict:
    summary = dict(residual_summary or {})
    reason_counts: Counter[str] = Counter()
    top = []
    for row in list(summary.get("top", []) or [])[:8]:
        public = {
            "sa_label": _clean_label((row or {}).get("sa_label", "")),
            "unresolved_mass": _round4(float((row or {}).get("unresolved_mass", 0.0) or 0.0)),
            "last_reason": _clean_label((row or {}).get("last_reason", "")),
            "hit_count": int((row or {}).get("hit_count", 0) or 0),
        }
        if public["last_reason"]:
            reason_counts[public["last_reason"]] += 1
        for reason, value in dict((row or {}).get("reason_counts", {}) or {}).items():
            reason_counts[str(reason)] += int(value or 0)
        top.append(public)
    total = float(summary.get("total_unresolved_mass", 0.0) or 0.0)
    if total <= 0.0:
        interpretation = "stable_or_no_recent_residual"
    elif reason_counts.get("prediction_miss", 0) and reason_counts.get("prediction_unexpected", 0):
        interpretation = "mixed_mismatch_expected_and_actual_compete"
    elif reason_counts.get("prediction_miss", 0):
        interpretation = "over_prediction_virtual_without_actual"
    elif reason_counts.get("prediction_unexpected", 0):
        interpretation = "under_prediction_actual_without_virtual"
    else:
        interpretation = "unresolved_attention_residual"
    return {
        "schema_id": "state_pool_residual_semantics/v1",
        "count": int(summary.get("count", 0) or 0),
        "total_unresolved_mass": _round4(total),
        "reason_counts": {key: int(value) for key, value in sorted(reason_counts.items())},
        "top": top,
        "interpretation": interpretation,
        "meaning": "residual_is_not_new_energy;it_is_unresolved_prediction_actual_mismatch_for_attention_and_slow_recall",
    }


def _r_state_contract(r_state: dict | None) -> dict:
    if not isinstance(r_state, dict):
        return {"available": False}
    heads = list(r_state.get("heads", []) or [])
    head_ids = [_clean_label((head or {}).get("head_id", "")) for head in heads]
    item_count = sum(len(list((head or {}).get("items", []) or [])) for head in heads)
    return {
        "available": True,
        "schema_id": _clean_label(r_state.get("schema_id", "r_state_snapshot/v1")),
        "head_count": int(r_state.get("head_count", len(heads)) or len(heads)),
        "items_per_head": int(r_state.get("items_per_head", 0) or 0),
        "head_ids": head_ids,
        "item_count": item_count,
        "has_recent_head": "head_recent" in head_ids,
        "has_prediction_head": "head_prediction" in head_ids,
        "has_residual_head": "head_residual" in head_ids,
        "contract": "fixed_budget_multi_head_readout_for_fast_attention_query",
    }


def build_energy_flow_trace(
    *,
    items: list[dict],
    tick_index: int,
    prediction_trace: dict | None = None,
    residual_summary: dict | None = None,
    r_state: dict | None = None,
    memory_write_items: list[dict] | None = None,
    limit: int = 8,
) -> dict:
    """
    Build a read-only map of AP's dual-energy cognitive field.

    This is deliberately a view layer. It explains how real energy, virtual
    energy, pressure, residuals, R_state, and memory-write rows relate, but it
    never changes learning or attention. That keeps AP's development open while
    making the "minimum prediction error" loop auditable.
    """

    clean_items = [dict(row) for row in list(items or []) if isinstance(row, dict)]
    mass = _mass_template()
    family_groups: dict[str, dict] = defaultdict(_mass_template)
    source_groups: dict[str, dict] = defaultdict(_mass_template)
    for row in clean_items:
        _add_mass(mass, row)
        _add_mass(family_groups[_clean_label(row.get("family", "")) or "unknown"], row)
        _add_mass(source_groups[_clean_label(row.get("source_type", "")) or "unknown"], row)
    memory_rows = [dict(row) for row in list(memory_write_items or []) if isinstance(row, dict)]
    return {
        "schema_id": "state_pool_energy_flow/v1",
        "tick_index": int(tick_index),
        "mass": _round_mass(mass),
        "family_mass": _top_mass_groups(family_groups, limit=12),
        "source_mass": _top_mass_groups(source_groups, limit=12),
        "pressure_peaks": _pressure_peaks(clean_items, limit=limit),
        "prediction_validation": _prediction_validation(prediction_trace or {}),
        "residual_semantics": _residual_semantics(residual_summary or {}),
        "view_contracts": {
            "snapshot": {
                "item_count": len(clean_items),
                "external_evidence_count": _external_count(clean_items),
                "contract": "bounded_whitebox_cognitive_field_view",
            },
            "memory_write": {
                "available": bool(memory_rows),
                "item_count": len(memory_rows),
                "external_evidence_count": _external_count(memory_rows),
                "contract": "preserve_current_external_evidence_for_successor_learning",
            },
            "r_state": _r_state_contract(r_state),
        },
        "theory_roles": {
            "real_energy": "actual_or_executed_evidence_that_has_happened",
            "virtual_energy": "predicted_or_recalled_evidence_that_should_happen",
            "cognitive_pressure": "real_minus_virtual;positive_is_surprise_negative_is_dissonance",
            "residual": "unresolved_prediction_error_kept_available_for_attention_and_slow_system",
            "energy_flow_trace": "whitebox_observation_only_not_a_new_rule_or_learning_writer",
        },
    }
