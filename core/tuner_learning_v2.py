# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class TunerLearningV2:
    TARGET_SPECS: dict[str, dict[str, float]] = {
        "attention.focus_gain": {
            "min": 0.25,
            "max": 4.0,
            "max_offset": 0.85,
            "step": 0.6,
            "learning_rate": 0.12,
            "cost_sensitivity": 0.22,
            "growth_sensitivity": 0.10,
            "profile_weight": 0.6,
        },
        "sampling.increment_budget": {
            "min": 4.0,
            "max": 256.0,
            "max_offset": 24.0,
            "step": 10.0,
            "learning_rate": 0.11,
            "cost_sensitivity": 0.34,
            "growth_sensitivity": 0.18,
            "profile_weight": 0.7,
        },
        "prediction.successor_bias_gain": {
            "min": 0.0,
            "max": 4.0,
            "max_offset": 0.85,
            "step": 0.55,
            "learning_rate": 0.11,
            "cost_sensitivity": 0.12,
            "growth_sensitivity": 0.11,
            "profile_weight": 0.6,
        },
        "state.anchor_bias_gain": {
            "min": 0.0,
            "max": 4.0,
            "max_offset": 0.8,
            "step": 0.5,
            "learning_rate": 0.10,
            "cost_sensitivity": 0.08,
            "growth_sensitivity": 0.10,
            "profile_weight": 0.55,
        },
        "rules.dissonance_gain": {
            "min": 0.0,
            "max": 4.0,
            "max_offset": 0.75,
            "step": 0.45,
            "learning_rate": 0.08,
            "cost_sensitivity": 0.05,
            "growth_sensitivity": 0.04,
            "profile_weight": 0.45,
        },
    }

    def __init__(
        self,
        *,
        max_target_entries: int = 64,
        max_profile_entries: int = 128,
        max_feedback_events: int = 512,
        decay: float = 0.988,
        target_logic_ms: float = 120.0,
    ) -> None:
        self.max_target_entries = max(8, int(max_target_entries))
        self.max_profile_entries = max(16, int(max_profile_entries))
        self.max_feedback_events = max(32, int(max_feedback_events))
        self.decay = max(0.9, min(0.9999, float(decay)))
        self.target_logic_ms = max(10.0, float(target_logic_ms))
        self._target_bias: dict[str, dict[str, Any]] = {}
        self._profile_target_bias: dict[str, dict[str, dict[str, Any]]] = {}
        self._recent_feedback: deque[dict[str, Any]] = deque(maxlen=self.max_feedback_events)

    def apply_to_controls(
        self,
        *,
        controls: dict[str, float],
        matched_profiles: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        next_controls = {str(key): float(value) for key, value in dict(controls or {}).items()}
        matched_profiles = [dict(item) for item in (matched_profiles or []) if isinstance(item, dict)]
        matched_profile_ids = [str(item.get("profile_id", "") or "") for item in matched_profiles if str(item.get("profile_id", "") or "")]
        applied_offsets: list[dict[str, Any]] = []

        for target, spec in self.TARGET_SPECS.items():
            base_value = float(next_controls.get(target, spec["min"]) or spec["min"])
            global_entry = self._target_bias.get(target, {})
            global_offset = float(global_entry.get("offset", 0.0) or 0.0) * max(0.1, float(global_entry.get("confidence", 0.0) or 0.0))
            profile_offset = 0.0
            profile_parts: list[dict[str, Any]] = []
            for profile_id in matched_profile_ids:
                profile_entry = ((self._profile_target_bias.get(profile_id) or {}).get(target) or {})
                if not profile_entry:
                    continue
                delta = float(profile_entry.get("offset", 0.0) or 0.0) * max(0.1, float(profile_entry.get("confidence", 0.0) or 0.0)) * float(spec.get("profile_weight", 0.5) or 0.5)
                if abs(delta) <= 1e-9:
                    continue
                profile_offset += delta
                profile_parts.append(
                    {
                        "profile_id": profile_id,
                        "offset": _round4(delta),
                        "confidence": _round4(profile_entry.get("confidence", 0.0) or 0.0),
                    }
                )
            total_offset = global_offset + profile_offset
            if abs(total_offset) <= 1e-9:
                continue
            adjusted = _clamp(base_value + total_offset, spec["min"], spec["max"])
            applied_delta = adjusted - base_value
            if abs(applied_delta) <= 1e-9:
                continue
            next_controls[target] = adjusted
            applied_offsets.append(
                {
                    "target": target,
                    "base_value": _round4(base_value),
                    "adjusted_value": _round4(adjusted),
                    "applied_offset": _round4(applied_delta),
                    "global_offset": _round4(global_offset),
                    "profile_offset": _round4(profile_offset),
                    "matched_profiles": profile_parts,
                }
            )

        return {
            "controls": {key: _round4(value) for key, value in next_controls.items()},
            "applied_offsets": applied_offsets,
            "matched_profile_ids": matched_profile_ids,
            "target_bias_summary": self.target_bias_summary(limit=12),
            "profile_bias_summary": self.profile_bias_summary(limit=12),
        }

    def record_feedback(
        self,
        *,
        tick_index: int,
        control_feedback_context: dict[str, Any],
        emotion_channels: dict[str, Any],
        action_feedback: dict[str, Any] | None = None,
        logic_ms: float = 0.0,
    ) -> dict[str, Any]:
        control_feedback_context = dict(control_feedback_context or {})
        action_feedback = dict(action_feedback or {})
        emotion_channels = dict(emotion_channels or {})
        controls = {
            str(key): float(value)
            for key, value in dict(control_feedback_context.get("runtime_controls", {}) or {}).items()
            if str(key)
        }
        matched_profiles = [dict(item) for item in (control_feedback_context.get("matched_profiles", []) or []) if isinstance(item, dict)]
        matched_profile_ids = [str(item.get("profile_id", "") or "") for item in matched_profiles if str(item.get("profile_id", "") or "")]
        if not controls:
            controls = {target: spec["min"] for target, spec in self.TARGET_SPECS.items()}

        expectation = float(emotion_channels.get("expectation", 0.0) or 0.0)
        pressure = float(emotion_channels.get("pressure", 0.0) or 0.0)
        correctness = float(emotion_channels.get("correctness", 0.0) or 0.0)
        dissonance = float(emotion_channels.get("dissonance", 0.0) or 0.0)
        base_valence = float(action_feedback.get("feedback", 0.0) or 0.0)
        if abs(base_valence) <= 1e-9:
            base_valence = (expectation + correctness) - (pressure + dissonance)
            base_valence = _clamp(base_valence, -1.0, 1.0)

        logic_ms = max(0.0, float(logic_ms))
        overload = _clamp((logic_ms - self.target_logic_ms) / max(1.0, self.target_logic_ms), 0.0, 1.0)
        slack = _clamp((self.target_logic_ms - logic_ms) / max(1.0, self.target_logic_ms), 0.0, 1.0)

        applied_targets: list[dict[str, Any]] = []
        for target, spec in self.TARGET_SPECS.items():
            if target not in controls:
                continue
            signal = base_valence
            signal -= overload * float(spec.get("cost_sensitivity", 0.0) or 0.0)
            signal += slack * max(0.0, base_valence) * float(spec.get("growth_sensitivity", 0.0) or 0.0)
            if target == "rules.dissonance_gain":
                signal += dissonance * 0.16
                signal -= correctness * 0.08
            signal = _clamp(signal, -1.0, 1.0)
            target_entry = self._ensure_target_entry(target=target, tick_index=tick_index)
            next_offset = self._next_offset(target_entry, signal=signal, spec=spec)
            applied_targets.append(
                self._apply_entry_update(
                    entry=target_entry,
                    target=target,
                    tick_index=tick_index,
                    signal=signal,
                    next_offset=next_offset,
                    context_label="global",
                )
            )
            for profile_id in matched_profile_ids:
                profile_entry = self._ensure_profile_entry(profile_id=profile_id, target=target, tick_index=tick_index)
                profile_next_offset = self._next_offset(
                    profile_entry,
                    signal=signal * float(spec.get("profile_weight", 0.5) or 0.5),
                    spec=spec,
                    profile_scale=0.75,
                )
                self._apply_entry_update(
                    entry=profile_entry,
                    target=target,
                    tick_index=tick_index,
                    signal=signal,
                    next_offset=profile_next_offset,
                    context_label=profile_id,
                    collect=False,
                )

        self._trim_profile_entries()
        event = {
            "tick_index": int(tick_index),
            "logic_ms": _round4(logic_ms),
            "target_logic_ms": _round4(self.target_logic_ms),
            "overload": _round4(overload),
            "slack": _round4(slack),
            "feedback": _round4(base_valence),
            "matched_profile_ids": matched_profile_ids,
            "control_targets": sorted(controls.keys()),
            "applied_targets": applied_targets,
        }
        self._recent_feedback.append(event)
        return event

    def target_bias_summary(self, *, limit: int = 12) -> list[dict[str, Any]]:
        rows = sorted(
            self._target_bias.values(),
            key=lambda item: (
                -abs(float(item.get("offset", 0.0) or 0.0) * float(item.get("confidence", 0.0) or 0.0)),
                -int(item.get("sample_count", 0) or 0),
                str(item.get("target", "") or ""),
            ),
        )
        return [dict(row) for row in rows[: max(1, int(limit))]]

    def profile_bias_summary(self, *, limit: int = 12) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for profile_id, target_rows in self._profile_target_bias.items():
            for target, entry in target_rows.items():
                rows.append({"profile_id": profile_id, "target": target, **dict(entry)})
        rows.sort(
            key=lambda item: (
                -abs(float(item.get("offset", 0.0) or 0.0) * float(item.get("confidence", 0.0) or 0.0)),
                -int(item.get("sample_count", 0) or 0),
                str(item.get("profile_id", "") or ""),
                str(item.get("target", "") or ""),
            )
        )
        return rows[: max(1, int(limit))]

    def recent_feedback(self, *, limit: int = 16) -> list[dict[str, Any]]:
        return list(self._recent_feedback)[-max(1, int(limit)) :]

    def export_payload(self) -> dict[str, Any]:
        return {
            "target_bias": self._target_bias,
            "profile_target_bias": self._profile_target_bias,
            "recent_feedback": list(self._recent_feedback),
            "decay": self.decay,
            "target_logic_ms": self.target_logic_ms,
            "max_target_entries": self.max_target_entries,
            "max_profile_entries": self.max_profile_entries,
            "max_feedback_events": self.max_feedback_events,
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        target_bias = payload.get("target_bias", {}) or {}
        self._target_bias = {
            str(key or ""): dict(value)
            for key, value in target_bias.items()
            if str(key or "") and isinstance(value, dict)
        }
        profile_payload = payload.get("profile_target_bias", {}) or {}
        profile_rows: dict[str, dict[str, dict[str, Any]]] = {}
        for profile_id, target_rows in dict(profile_payload).items():
            clean_profile_id = str(profile_id or "")
            if not clean_profile_id or not isinstance(target_rows, dict):
                continue
            clean_targets: dict[str, dict[str, Any]] = {}
            for target, entry in target_rows.items():
                clean_target = str(target or "")
                if not clean_target or not isinstance(entry, dict):
                    continue
                clean_targets[clean_target] = dict(entry)
            if clean_targets:
                profile_rows[clean_profile_id] = clean_targets
        self._profile_target_bias = profile_rows
        self._recent_feedback = deque(list(payload.get("recent_feedback", []) or []), maxlen=self.max_feedback_events)
        self.target_logic_ms = max(10.0, float(payload.get("target_logic_ms", self.target_logic_ms) or self.target_logic_ms))

    def _ensure_target_entry(self, *, target: str, tick_index: int) -> dict[str, Any]:
        entry = self._target_bias.get(target)
        if entry is None:
            entry = {
                "target": target,
                "offset": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
                "last_tick": int(tick_index),
                "last_signal": 0.0,
            }
            self._target_bias[target] = entry
        return entry

    def _ensure_profile_entry(self, *, profile_id: str, target: str, tick_index: int) -> dict[str, Any]:
        profile_rows = self._profile_target_bias.setdefault(profile_id, {})
        entry = profile_rows.get(target)
        if entry is None:
            entry = {
                "target": target,
                "offset": 0.0,
                "confidence": 0.0,
                "sample_count": 0,
                "last_tick": int(tick_index),
                "last_signal": 0.0,
            }
            profile_rows[target] = entry
        return entry

    def _next_offset(
        self,
        entry: dict[str, Any],
        *,
        signal: float,
        spec: dict[str, float],
        profile_scale: float = 1.0,
    ) -> float:
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        offset = float(entry.get("offset", 0.0) or 0.0)
        learning_rate = float(spec.get("learning_rate", 0.1) or 0.1) * max(0.25, 1.0 - min(0.88, confidence)) * profile_scale
        step = float(spec.get("step", 0.5) or 0.5)
        max_offset = abs(float(spec.get("max_offset", step) or step))
        next_offset = offset * self.decay + float(signal) * learning_rate * step
        return _clamp(next_offset, -max_offset, max_offset)

    def _apply_entry_update(
        self,
        *,
        entry: dict[str, Any],
        target: str,
        tick_index: int,
        signal: float,
        next_offset: float,
        context_label: str,
        collect: bool = True,
    ) -> dict[str, Any]:
        confidence = float(entry.get("confidence", 0.0) or 0.0)
        entry["offset"] = _round4(next_offset)
        entry["confidence"] = _round4(min(1.0, confidence * self.decay + 0.10 + abs(signal) * 0.08))
        entry["sample_count"] = int(entry.get("sample_count", 0) or 0) + 1
        entry["last_tick"] = int(tick_index)
        entry["last_signal"] = _round4(signal)
        if not collect:
            return {}
        return {
            "target": target,
            "context": context_label,
            "signal": _round4(signal),
            "next_offset": entry["offset"],
            "confidence": entry["confidence"],
        }

    def _trim_profile_entries(self) -> None:
        if len(self._target_bias) > self.max_target_entries:
            rows = sorted(
                self._target_bias.items(),
                key=lambda item: (
                    -abs(float((item[1] or {}).get("offset", 0.0) or 0.0) * float((item[1] or {}).get("confidence", 0.0) or 0.0)),
                    -int((item[1] or {}).get("sample_count", 0) or 0),
                    item[0],
                ),
            )
            self._target_bias = {key: dict(value) for key, value in rows[: self.max_target_entries]}

        flat_rows: list[tuple[str, str, dict[str, Any]]] = []
        for profile_id, target_rows in self._profile_target_bias.items():
            for target, entry in target_rows.items():
                flat_rows.append((profile_id, target, dict(entry)))
        if len(flat_rows) <= self.max_profile_entries:
            return
        flat_rows.sort(
            key=lambda item: (
                -abs(float((item[2] or {}).get("offset", 0.0) or 0.0) * float((item[2] or {}).get("confidence", 0.0) or 0.0)),
                -int((item[2] or {}).get("sample_count", 0) or 0),
                item[0],
                item[1],
            )
        )
        kept: dict[str, dict[str, dict[str, Any]]] = {}
        for profile_id, target, entry in flat_rows[: self.max_profile_entries]:
            kept.setdefault(profile_id, {})[target] = dict(entry)
        self._profile_target_bias = kept
