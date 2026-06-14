from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _round4(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return round(float(default), 4)


def _clamp(value: Any, low: float = 0.0, high: float = 1.0, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return max(float(low), min(float(high), numeric))


def _clean_notes(notes: Any, *, extra: list[str] | None = None) -> list[str]:
    rows = []
    for note in list(notes or []):
        clean = str(note or "").strip()
        if clean and clean not in rows:
            rows.append(clean)
    for note in list(extra or []):
        clean = str(note or "").strip()
        if clean and clean not in rows:
            rows.append(clean)
    return rows


def _normalize_state_item(item: dict, *, source: str, teacher_kind: str, index: int) -> dict | None:
    label = str((item or {}).get("sa_label", "") or "").strip()
    if not label:
        hint = str((item or {}).get("hint", "") or (item or {}).get("display_text", "") or "hint").strip()
        safe_hint = hint.replace(" ", "_")[:48] or f"item_{index}"
        label = f"education_hint::{safe_hint}"
    meta = dict((item or {}).get("anchor_meta", {}) or {})
    meta.update(
        {
            "schema_id": str(meta.get("schema_id", "") or "education_state_item/v1"),
            "source": source,
            "teacher_kind": teacher_kind,
            "meaning": str(meta.get("meaning", "") or "external_teacher_hint_first_class_state_item"),
        }
    )
    normalized = {
        "sa_label": label,
        "display_text": str((item or {}).get("display_text", "") or label),
        "family": str((item or {}).get("family", "") or "education_intervention"),
        "source_type": str((item or {}).get("source_type", "") or "external_teacher"),
        "real_energy": _round4(_clamp((item or {}).get("real_energy", 0.16), 0.0, 3.0, default=0.16)),
        "virtual_energy": _round4(_clamp((item or {}).get("virtual_energy", 0.0), 0.0, 3.0)),
        "cognitive_pressure": _round4(_clamp((item or {}).get("cognitive_pressure", 0.06), 0.0, 3.0, default=0.06)),
        "attention_gain": _round4(_clamp((item or {}).get("attention_gain", 0.0), 0.0, 3.0)),
        "anchor_meta": meta,
    }
    if "query_weight" in (item or {}):
        normalized["query_weight"] = _round4(_clamp((item or {}).get("query_weight", 0.0), 0.0, 3.0))
    for key in ("numeric_features", "reconstruction_payload", "position"):
        if isinstance((item or {}).get(key), dict):
            normalized[key] = dict((item or {}).get(key, {}) or {})
        elif key == "position" and key in (item or {}):
            normalized[key] = (item or {}).get(key)
    return normalized


def _normalize_action_bias(bias: dict, *, source: str, teacher_kind: str) -> dict | None:
    action_id = str((bias or {}).get("action_id", "") or "").strip()
    if not action_id:
        return None
    drive_delta = _round4(float((bias or {}).get("drive_delta", (bias or {}).get("drive", 0.0)) or 0.0))
    if abs(drive_delta) <= 0.00001:
        return None
    notes = _clean_notes(
        (bias or {}).get("notes", []),
        extra=[
            "education_intervention_bias",
            "soft_drive_bias_only",
            f"teacher_kind={teacher_kind}",
            f"source={source}",
        ],
    )
    normalized = {
        "schema_id": str((bias or {}).get("schema_id", "") or "education_action_bias/v1"),
        "action_id": action_id,
        "drive_delta": drive_delta,
        "params": dict((bias or {}).get("params", {}) or {}),
        "notes": notes,
        "source": source,
        "teacher_kind": teacher_kind,
    }
    if str((bias or {}).get("actuator_id", "") or ""):
        normalized["actuator_id"] = str((bias or {}).get("actuator_id", "") or "")
    if str((bias or {}).get("skill_id", "") or ""):
        normalized["skill_id"] = str((bias or {}).get("skill_id", "") or "")
    if str((bias or {}).get("step_id", "") or ""):
        normalized["step_id"] = str((bias or {}).get("step_id", "") or "")
    return normalized


def _normalize_feedback(feedback: dict, *, source: str, teacher_kind: str) -> dict:
    if not isinstance(feedback, dict) or not feedback:
        return {}
    reward = _round4(_clamp(feedback.get("reward", 0.0), 0.0, 1.0))
    punishment = _round4(_clamp(feedback.get("punishment", 0.0), 0.0, 1.0))
    correctness = _round4(_clamp(feedback.get("correctness", 0.0), 0.0, 1.0))
    confidence = _round4(_clamp(feedback.get("confidence", 1.0), 0.0, 1.0, default=1.0))
    if max(reward, punishment, correctness) <= 0.0:
        return {}
    return {
        "schema_id": "education_feedback/v1",
        "reward": reward,
        "punishment": punishment,
        "correctness": correctness,
        "confidence": confidence,
        "source": str(feedback.get("source", "") or f"education::{source}"),
        "teacher_kind": teacher_kind,
        "notes": _clean_notes(feedback.get("notes", []), extra=["education_feedback", f"teacher_kind={teacher_kind}"]),
    }


def normalize_education_intervention(intervention: dict | None, *, tick_index: int | None = None) -> dict:
    raw = dict(intervention or {})
    source = str(raw.get("source", "") or "external_teacher")
    teacher_kind = str(raw.get("teacher_kind", "") or "unknown_teacher")
    state_items = [
        item
        for index, item in enumerate(
            _normalize_state_item(dict(row), source=source, teacher_kind=teacher_kind, index=index)
            for index, row in enumerate(list(raw.get("state_items", []) or []))
            if isinstance(row, dict)
        )
        if item is not None
    ]
    action_biases = [
        bias
        for bias in (
            _normalize_action_bias(dict(row), source=source, teacher_kind=teacher_kind)
            for row in list(raw.get("action_biases", []) or [])
            if isinstance(row, dict)
        )
        if bias is not None
    ]
    feedback = _normalize_feedback(dict(raw.get("feedback", {}) or {}), source=source, teacher_kind=teacher_kind)
    normalized = {
        "schema_id": "education_intervention/v1",
        "source": source,
        "teacher_kind": teacher_kind,
        "goal": str(raw.get("goal", "") or ""),
        "tick_index": int(tick_index) if tick_index is not None else raw.get("tick_index", None),
        "state_items": state_items,
        "action_biases": action_biases,
        "feedback": feedback,
        "notes": _clean_notes(raw.get("notes", []), extra=["external_education_intervention"]),
        "raw_summary": {
            "state_item_count": len(list(raw.get("state_items", []) or [])),
            "action_bias_count": len(list(raw.get("action_biases", []) or [])),
            "has_feedback": bool(raw.get("feedback", {})),
        },
    }
    return normalized


@dataclass
class EducationInterventionBuffer:
    """One-tick external teaching queue.

    The buffer has no skill logic. It only stores normalized interventions until
    the next AP tick consumes them. This keeps AP core generic: a human teacher,
    a deterministic scaffold, and an LLM teacher all share the same doorway.
    """

    _pending: list[dict] = field(default_factory=list)

    def queue(self, intervention: dict, *, tick_index: int | None = None) -> dict:
        normalized = normalize_education_intervention(intervention, tick_index=tick_index)
        self._pending.append(normalized)
        return {
            "schema_id": "education_intervention_queue_trace/v1",
            "queued": True,
            "pending_count": len(self._pending),
            "intervention": normalized,
        }

    def consume(self, *, tick_index: int) -> dict:
        rows = [normalize_education_intervention(row, tick_index=tick_index) for row in self._pending]
        self._pending = []
        state_items = [item for row in rows for item in list(row.get("state_items", []) or [])]
        action_biases = [bias for row in rows for bias in list(row.get("action_biases", []) or [])]
        feedbacks = [dict(row.get("feedback", {}) or {}) for row in rows if dict(row.get("feedback", {}) or {})]
        merged_feedback = self._merge_feedbacks(feedbacks)
        return {
            "schema_id": "education_intervention_consumption/v1",
            "tick_index": int(tick_index),
            "applied": bool(rows),
            "interventions": rows,
            "state_items": state_items,
            "action_biases": action_biases,
            "feedback": merged_feedback,
            "intervention_count": len(rows),
        }

    def _merge_feedbacks(self, feedbacks: list[dict]) -> dict:
        if not feedbacks:
            return {}
        reward = min(1.0, sum(max(0.0, float(row.get("reward", 0.0) or 0.0)) for row in feedbacks))
        punishment = min(1.0, sum(max(0.0, float(row.get("punishment", 0.0) or 0.0)) for row in feedbacks))
        correctness = min(1.0, sum(max(0.0, float(row.get("correctness", 0.0) or 0.0)) for row in feedbacks))
        confidence = max(max(0.0, min(1.0, float(row.get("confidence", 0.0) or 0.0))) for row in feedbacks)
        notes = []
        for row in feedbacks:
            notes.extend(list(row.get("notes", []) or []))
        return {
            "schema_id": "education_feedback_merged/v1",
            "reward": _round4(reward),
            "punishment": _round4(punishment),
            "correctness": _round4(correctness),
            "confidence": _round4(confidence),
            "source": "education_intervention_buffer",
            "notes": _clean_notes(notes, extra=["merged_education_feedback"]),
        }
