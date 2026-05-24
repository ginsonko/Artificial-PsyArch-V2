# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


class ActionLearningV2:
    def __init__(
        self,
        *,
        max_bias_entries: int = 128,
        max_feedback_events: int = 512,
        decay: float = 0.985,
        context_bias_gain: float = 0.82,
    ) -> None:
        self.max_bias_entries = max(16, int(max_bias_entries))
        self.max_feedback_events = max(32, int(max_feedback_events))
        self.decay = max(0.9, min(0.9999, float(decay)))
        self.context_bias_gain = max(0.0, min(2.0, float(context_bias_gain)))
        self._action_bias: dict[str, dict[str, Any]] = {}
        self._action_instance_bias: dict[str, dict[str, Any]] = {}
        self._context_action_bias: dict[str, dict[str, Any]] = {}
        self._context_instance_bias: dict[str, dict[str, Any]] = {}
        self._recent_feedback: deque[dict[str, Any]] = deque(maxlen=self.max_feedback_events)

    def score_action_drives(
        self,
        action_drives: list[dict[str, Any]],
        *,
        context_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context_keys = self._extract_context_keys(context_hints)
        scored: list[dict[str, Any]] = []
        for raw in action_drives:
            if not isinstance(raw, dict):
                continue
            action_id = str(raw.get("action_id", "") or "")
            if not action_id:
                continue
            base_drive = float(raw.get("drive", 0.0) or 0.0)
            modulation = self.modulation_snapshot(
                action_id=action_id,
                instance_id=str(raw.get("instance_id", "") or ""),
                context_hints={"context_keys": context_keys},
            )
            learned_bias = float(modulation.get("learned_bias", 0.0) or 0.0)
            confidence = float(modulation.get("bias_confidence", 0.0) or 0.0)
            context_bias = float(modulation.get("context_bias", 0.0) or 0.0)
            context_confidence = float(modulation.get("context_bias_confidence", 0.0) or 0.0)
            habit_modulation = float(modulation.get("habit_modulation", 1.0) or 1.0)
            context_modulation = float(modulation.get("context_modulation", 1.0) or 1.0)
            final_drive = max(0.0, min(1.5, base_drive * habit_modulation * context_modulation))
            scored.append(
                {
                    **raw,
                    "base_drive": _round4(base_drive),
                    "learned_bias": _round4(learned_bias),
                    "bias_confidence": _round4(confidence),
                    "context_bias": _round4(context_bias),
                    "context_bias_confidence": _round4(context_confidence),
                    "context_bias_keys": list(modulation.get("context_bias_keys", []) or []),
                    "habit_modulation": _round4(habit_modulation),
                    "context_modulation": _round4(context_modulation),
                    "drive": _round4(final_drive),
                }
            )
        scored.sort(key=lambda item: (-float(item.get("drive", 0.0) or 0.0), str(item.get("action_id", "") or "")))
        return {
            "scored_action_drives": scored,
            "bias_summary": self.bias_summary(limit=12),
            "context_bias_summary": self.context_bias_summary(limit=12),
            "context_keys": context_keys,
        }

    def modulation_snapshot(
        self,
        *,
        action_id: str,
        instance_id: str | None = None,
        context_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_action_id = str(action_id or "").strip()
        clean_instance_id = str(instance_id or "").strip()
        context_keys = self._extract_context_keys(context_hints)
        bias_entry = self._action_bias.get(clean_action_id, {})
        instance_entry = self._action_instance_bias.get(clean_instance_id, {}) if clean_instance_id else {}
        learned_bias = float(bias_entry.get("bias", 0.0) or 0.0)
        confidence = float(bias_entry.get("confidence", 0.0) or 0.0)
        instance_bias = float(instance_entry.get("bias", 0.0) or 0.0)
        instance_confidence = float(instance_entry.get("confidence", 0.0) or 0.0)
        context_bias_row = self._aggregate_context_bias(action_id=clean_action_id, context_keys=context_keys)
        context_bias = float(context_bias_row.get("bias", 0.0) or 0.0)
        context_confidence = float(context_bias_row.get("confidence", 0.0) or 0.0)
        context_instance_row = self._aggregate_context_instance_bias(instance_id=clean_instance_id, context_keys=context_keys)
        context_instance_bias = float(context_instance_row.get("bias", 0.0) or 0.0)
        context_instance_confidence = float(context_instance_row.get("confidence", 0.0) or 0.0)

        instance_global_weight = 0.08 if clean_instance_id else 0.14
        global_signal = learned_bias * max(0.05, confidence) * 0.18 + instance_bias * max(0.05, instance_confidence) * instance_global_weight
        context_action_signal = context_bias * max(0.05, context_confidence) * self.context_bias_gain * 0.78
        context_instance_signal = context_instance_bias * max(0.05, context_instance_confidence) * (self.context_bias_gain + 0.75)
        habit_modulation = max(0.78, min(1.85, pow(2.718281828, 0.32 * global_signal + 0.42 * context_instance_signal)))
        context_modulation = max(0.62, min(1.7, pow(2.718281828, 0.48 * context_action_signal)))
        return {
            "action_id": clean_action_id,
            "learned_bias": _round4(learned_bias),
            "bias_confidence": _round4(confidence),
            "instance_bias": _round4(instance_bias),
            "instance_confidence": _round4(instance_confidence),
            "context_bias": _round4(context_bias),
            "context_bias_confidence": _round4(context_confidence),
            "context_instance_bias": _round4(context_instance_bias),
            "context_instance_confidence": _round4(context_instance_confidence),
            "context_bias_keys": list(context_bias_row.get("matched_keys", []) or []),
            "context_instance_bias_keys": list(context_instance_row.get("matched_keys", []) or []),
            "habit_modulation": _round4(habit_modulation),
            "context_modulation": _round4(context_modulation),
        }

    def record_feedback(
        self,
        *,
        tick_index: int,
        selected_actions: list[dict[str, Any]],
        emotion_channels: dict[str, Any],
        runtime_action_effects: dict[str, Any] | None = None,
        external_feedback: dict[str, Any] | None = None,
        context_hints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime_action_effects = dict(runtime_action_effects or {})
        external_feedback = dict(external_feedback or {})
        context_keys = self._extract_context_keys(context_hints)
        expectation = float(emotion_channels.get("expectation", 0.0) or 0.0)
        pressure = float(emotion_channels.get("pressure", 0.0) or 0.0)
        correctness = float(emotion_channels.get("correctness", 0.0) or 0.0)
        dissonance = float(emotion_channels.get("dissonance", 0.0) or 0.0)
        moved_bonus = 0.08 if bool(runtime_action_effects.get("moved", False)) else 0.0
        ext_reward = float(external_feedback.get("reward", 0.0) or 0.0)
        ext_punish = float(external_feedback.get("punishment", 0.0) or 0.0)
        intrinsic_valence = (expectation + correctness + moved_bonus) - (pressure + dissonance)
        teacher_valence = ext_reward - ext_punish
        if abs(teacher_valence) > 0.0001:
            valence = teacher_valence * 1.6 + intrinsic_valence * 0.35
        else:
            valence = intrinsic_valence
        normalized = max(-1.0, min(1.0, valence))

        applied: list[dict[str, Any]] = []
        for action in selected_actions:
            if not isinstance(action, dict):
                continue
            action_id = str(action.get("action_id", "") or "")
            if not action_id:
                continue
            instance_id = str(action.get("instance_id", "") or "")
            entry = self._action_bias.get(action_id)
            if entry is None:
                entry = {
                    "action_id": action_id,
                    "bias": 0.0,
                    "confidence": 0.0,
                    "sample_count": 0,
                    "last_tick": int(tick_index),
                    "last_feedback": 0.0,
                }
                self._action_bias[action_id] = entry
            bias = float(entry.get("bias", 0.0) or 0.0)
            confidence = float(entry.get("confidence", 0.0) or 0.0)
            learning_rate = 0.18 * max(0.25, 1.0 - min(0.85, confidence))
            next_bias = bias * self.decay + normalized * learning_rate
            next_confidence = min(1.0, confidence * self.decay + 0.12 + abs(normalized) * 0.08)
            entry["bias"] = _round4(max(-1.0, min(1.0, next_bias)))
            entry["confidence"] = _round4(next_confidence)
            entry["sample_count"] = int(entry.get("sample_count", 0) or 0) + 1
            entry["last_tick"] = int(tick_index)
            entry["last_feedback"] = _round4(normalized)
            instance_updates = []
            if instance_id:
                instance_entry = self._action_instance_bias.get(instance_id)
                if instance_entry is None:
                    instance_entry = {
                        "instance_id": instance_id,
                        "action_id": action_id,
                        "bias": 0.0,
                        "confidence": 0.0,
                        "sample_count": 0,
                        "last_tick": int(tick_index),
                        "last_feedback": 0.0,
                    }
                    self._action_instance_bias[instance_id] = instance_entry
                inst_bias = float(instance_entry.get("bias", 0.0) or 0.0)
                inst_conf = float(instance_entry.get("confidence", 0.0) or 0.0)
                inst_lr = 0.28 * max(0.2, 1.0 - min(0.9, inst_conf))
                next_inst_bias = inst_bias * self.decay + normalized * inst_lr
                next_inst_conf = min(1.0, inst_conf * self.decay + 0.12 + abs(normalized) * 0.10)
                instance_entry["bias"] = _round4(max(-1.0, min(1.0, next_inst_bias)))
                instance_entry["confidence"] = _round4(next_inst_conf)
                instance_entry["sample_count"] = int(instance_entry.get("sample_count", 0) or 0) + 1
                instance_entry["last_tick"] = int(tick_index)
                instance_entry["last_feedback"] = _round4(normalized)
                instance_updates.append(
                    {
                        "instance_id": instance_id,
                        "next_bias": instance_entry["bias"],
                        "confidence": instance_entry["confidence"],
                    }
                )
            context_instance_updates = self._update_context_instance_bias_entries(
                tick_index=tick_index,
                instance_id=instance_id,
                feedback=normalized,
                context_keys=context_keys,
            ) if instance_id else []
            context_updates = self._update_context_bias_entries(
                tick_index=tick_index,
                action_id=action_id,
                feedback=normalized,
                context_keys=context_keys,
            )
            applied.append(
                {
                    "action_id": action_id,
                    "instance_id": instance_id,
                    "feedback": _round4(normalized),
                    "next_bias": entry["bias"],
                    "confidence": entry["confidence"],
                    "instance_updates": instance_updates,
                    "context_instance_updates": context_instance_updates,
                    "context_updates": context_updates,
                }
            )
        self._trim_bias_entries()
        event = {
            "tick_index": int(tick_index),
            "feedback": _round4(normalized),
            "selected_action_count": len(selected_actions),
            "expectation": _round4(expectation),
            "pressure": _round4(pressure),
            "correctness": _round4(correctness),
            "dissonance": _round4(dissonance),
            "external_feedback": {
                "reward": _round4(ext_reward),
                "punishment": _round4(ext_punish),
            },
            "intrinsic_valence": _round4(intrinsic_valence),
            "teacher_valence": _round4(teacher_valence),
            "moved": bool(runtime_action_effects.get("moved", False)),
            "context_keys": context_keys,
            "applied": applied,
        }
        self._recent_feedback.append(event)
        return event

    def bias_summary(self, *, limit: int = 12) -> list[dict[str, Any]]:
        rows = sorted(
            self._action_bias.values(),
            key=lambda item: (
                -abs(float(item.get("bias", 0.0) or 0.0) * float(item.get("confidence", 0.0) or 0.0)),
                -int(item.get("sample_count", 0) or 0),
                str(item.get("action_id", "") or ""),
            ),
        )
        return [dict(row) for row in rows[: max(1, int(limit))]]

    def context_bias_summary(self, *, limit: int = 12) -> list[dict[str, Any]]:
        rows = sorted(
            self._context_action_bias.values(),
            key=lambda item: (
                -abs(float(item.get("bias", 0.0) or 0.0) * float(item.get("confidence", 0.0) or 0.0)),
                -int(item.get("sample_count", 0) or 0),
                str(item.get("context_key", "") or ""),
                str(item.get("action_id", "") or ""),
            ),
        )
        return [dict(row) for row in rows[: max(1, int(limit))]]

    def recent_feedback(self, *, limit: int = 16) -> list[dict[str, Any]]:
        return list(self._recent_feedback)[-max(1, int(limit)) :]

    def export_payload(self) -> dict[str, Any]:
        return {
            "action_bias": self._action_bias,
            "action_instance_bias": self._action_instance_bias,
            "context_action_bias": self._context_action_bias,
            "context_instance_bias": self._context_instance_bias,
            "recent_feedback": list(self._recent_feedback),
            "decay": self.decay,
            "context_bias_gain": self.context_bias_gain,
            "max_bias_entries": self.max_bias_entries,
            "max_feedback_events": self.max_feedback_events,
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        action_bias = payload.get("action_bias", {}) or {}
        self._action_bias = {
            str(key or ""): dict(value)
            for key, value in action_bias.items()
            if str(key or "") and isinstance(value, dict)
        }
        action_instance_bias = payload.get("action_instance_bias", {}) or {}
        self._action_instance_bias = {
            str(key or ""): dict(value)
            for key, value in action_instance_bias.items()
            if str(key or "") and isinstance(value, dict)
        }
        context_action_bias = payload.get("context_action_bias", {}) or {}
        self._context_action_bias = {
            str(key or ""): dict(value)
            for key, value in context_action_bias.items()
            if str(key or "") and isinstance(value, dict)
        }
        context_instance_bias = payload.get("context_instance_bias", {}) or {}
        self._context_instance_bias = {
            str(key or ""): dict(value)
            for key, value in context_instance_bias.items()
            if str(key or "") and isinstance(value, dict)
        }
        self.context_bias_gain = max(0.0, min(2.0, float(payload.get("context_bias_gain", self.context_bias_gain) or self.context_bias_gain)))
        self._recent_feedback = deque(list(payload.get("recent_feedback", []) or []), maxlen=self.max_feedback_events)

    def _trim_bias_entries(self) -> None:
        if len(self._action_bias) <= self.max_bias_entries:
            pass
        else:
            rows = sorted(
                self._action_bias.items(),
                key=lambda item: (
                    -abs(float((item[1] or {}).get("bias", 0.0) or 0.0) * float((item[1] or {}).get("confidence", 0.0) or 0.0)),
                    -int((item[1] or {}).get("sample_count", 0) or 0),
                    item[0],
                ),
            )
            self._action_bias = {key: dict(value) for key, value in rows[: self.max_bias_entries]}
        instance_limit = max(self.max_bias_entries * 4, self.max_bias_entries)
        if len(self._action_instance_bias) > instance_limit:
            rows = sorted(
                self._action_instance_bias.items(),
                key=lambda item: (
                    -abs(float((item[1] or {}).get("bias", 0.0) or 0.0) * float((item[1] or {}).get("confidence", 0.0) or 0.0)),
                    -int((item[1] or {}).get("sample_count", 0) or 0),
                    item[0],
                ),
            )
            self._action_instance_bias = {key: dict(value) for key, value in rows[:instance_limit]}
        context_limit = max(self.max_bias_entries * 8, self.max_bias_entries)
        if len(self._context_action_bias) <= context_limit:
            pass
        else:
            rows = sorted(
                self._context_action_bias.items(),
                key=lambda item: (
                    -abs(float((item[1] or {}).get("bias", 0.0) or 0.0) * float((item[1] or {}).get("confidence", 0.0) or 0.0)),
                    -int((item[1] or {}).get("sample_count", 0) or 0),
                    item[0],
                ),
            )
            self._context_action_bias = {key: dict(value) for key, value in rows[:context_limit]}
        context_instance_limit = max(self.max_bias_entries * 8, self.max_bias_entries)
        if len(self._context_instance_bias) > context_instance_limit:
            rows = sorted(
                self._context_instance_bias.items(),
                key=lambda item: (
                    -abs(float((item[1] or {}).get("bias", 0.0) or 0.0) * float((item[1] or {}).get("confidence", 0.0) or 0.0)),
                    -int((item[1] or {}).get("sample_count", 0) or 0),
                    item[0],
                ),
            )
            self._context_instance_bias = {key: dict(value) for key, value in rows[:context_instance_limit]}

    def _extract_context_keys(self, context_hints: dict[str, Any] | None) -> list[str]:
        payload = dict(context_hints or {})
        keys: list[str] = []
        for raw in payload.get("context_keys", []) or []:
            clean = str(raw or "").strip()
            if clean and clean not in keys:
                keys.append(clean[:160])
        normalized_text = str(payload.get("normalized_text", "") or "").strip()
        if normalized_text:
            joined = "".join(part for part in normalized_text.split(" ") if part)
            if joined:
                key = f"text::{joined[:96]}"
                if key not in keys:
                    keys.append(key)
        for raw in payload.get("focus_units", []) or []:
            clean = str(raw or "").strip()
            if len(clean) >= 2:
                key = f"focus::{clean[:48]}"
                if key not in keys:
                    keys.append(key)
        for raw in payload.get("query_units", []) or []:
            clean = str(raw or "").strip()
            if len(clean) >= 2:
                key = f"unit::{clean[:48]}"
                if key not in keys:
                    keys.append(key)
        return keys[:8]

    def _aggregate_context_bias(self, *, action_id: str, context_keys: list[str]) -> dict[str, Any]:
        matched: list[dict[str, Any]] = []
        for key in context_keys:
            entry = self._context_action_bias.get(self._context_entry_id(key, action_id))
            if isinstance(entry, dict):
                matched.append(entry)
        if not matched:
            return {"bias": 0.0, "confidence": 0.0, "matched_keys": []}
        matched.sort(
            key=lambda item: (
                -abs(float(item.get("bias", 0.0) or 0.0) * float(item.get("confidence", 0.0) or 0.0)),
                -int(item.get("sample_count", 0) or 0),
                str(item.get("context_key", "") or ""),
            )
        )
        top = matched[:3]
        total_weight = 0.0
        weighted_bias = 0.0
        weighted_conf = 0.0
        for row in top:
            confidence = max(0.05, float(row.get("confidence", 0.0) or 0.0))
            key = str(row.get("context_key", "") or "")
            specificity = self._context_specificity_weight(key) / max(1.0, float(self._context_key_action_count(key)))
            weight = confidence * specificity
            total_weight += weight
            weighted_bias += float(row.get("bias", 0.0) or 0.0) * weight
            weighted_conf += weight
        return {
            "bias": _round4(weighted_bias / max(0.0001, total_weight)),
            "confidence": _round4(min(1.0, weighted_conf / max(1.0, float(len(top))))),
            "matched_keys": [str(row.get("context_key", "") or "") for row in top if str(row.get("context_key", "") or "")],
        }

    def _aggregate_context_instance_bias(self, *, instance_id: str, context_keys: list[str]) -> dict[str, Any]:
        clean_instance_id = str(instance_id or "").strip()
        if not clean_instance_id:
            return {"bias": 0.0, "confidence": 0.0, "matched_keys": []}
        matched: list[dict[str, Any]] = []
        for key in context_keys:
            entry = self._context_instance_bias.get(self._context_instance_entry_id(key, clean_instance_id))
            if isinstance(entry, dict):
                matched.append(entry)
        if not matched:
            return {"bias": 0.0, "confidence": 0.0, "matched_keys": []}
        matched.sort(
            key=lambda item: (
                -abs(float(item.get("bias", 0.0) or 0.0) * float(item.get("confidence", 0.0) or 0.0)),
                -int(item.get("sample_count", 0) or 0),
                str(item.get("context_key", "") or ""),
            )
        )
        top = matched[:3]
        total_weight = 0.0
        weighted_bias = 0.0
        weighted_conf = 0.0
        for row in top:
            confidence = max(0.05, float(row.get("confidence", 0.0) or 0.0))
            key = str(row.get("context_key", "") or "")
            specificity = self._context_specificity_weight(key) / max(1.0, float(self._context_key_instance_count(key)))
            weight = confidence * specificity
            total_weight += weight
            weighted_bias += float(row.get("bias", 0.0) or 0.0) * weight
            weighted_conf += weight
        return {
            "bias": _round4(weighted_bias / max(0.0001, total_weight)),
            "confidence": _round4(min(1.0, weighted_conf / max(1.0, float(len(top))))),
            "matched_keys": [str(row.get("context_key", "") or "") for row in top if str(row.get("context_key", "") or "")],
        }

    def _update_context_bias_entries(
        self,
        *,
        tick_index: int,
        action_id: str,
        feedback: float,
        context_keys: list[str],
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        for context_key in context_keys:
            entry_id = self._context_entry_id(context_key, action_id)
            entry = self._context_action_bias.get(entry_id)
            if entry is None:
                entry = {
                    "entry_id": entry_id,
                    "context_key": context_key,
                    "action_id": action_id,
                    "bias": 0.0,
                    "confidence": 0.0,
                    "sample_count": 0,
                    "last_tick": int(tick_index),
                    "last_feedback": 0.0,
                }
                self._context_action_bias[entry_id] = entry
            bias = float(entry.get("bias", 0.0) or 0.0)
            confidence = float(entry.get("confidence", 0.0) or 0.0)
            learning_rate = 0.26 * max(0.2, 1.0 - min(0.9, confidence))
            next_bias = bias * self.decay + float(feedback) * learning_rate
            next_confidence = min(1.0, confidence * self.decay + 0.10 + abs(float(feedback)) * 0.10)
            entry["bias"] = _round4(max(-1.0, min(1.0, next_bias)))
            entry["confidence"] = _round4(next_confidence)
            entry["sample_count"] = int(entry.get("sample_count", 0) or 0) + 1
            entry["last_tick"] = int(tick_index)
            entry["last_feedback"] = _round4(feedback)
            updates.append(
                {
                    "context_key": context_key,
                    "next_bias": entry["bias"],
                    "confidence": entry["confidence"],
                }
            )
        return updates

    def _update_context_instance_bias_entries(
        self,
        *,
        tick_index: int,
        instance_id: str,
        feedback: float,
        context_keys: list[str],
    ) -> list[dict[str, Any]]:
        updates: list[dict[str, Any]] = []
        clean_instance_id = str(instance_id or "").strip()
        if not clean_instance_id:
            return updates
        for context_key in context_keys:
            entry_id = self._context_instance_entry_id(context_key, clean_instance_id)
            entry = self._context_instance_bias.get(entry_id)
            if entry is None:
                entry = {
                    "entry_id": entry_id,
                    "context_key": context_key,
                    "instance_id": clean_instance_id,
                    "bias": 0.0,
                    "confidence": 0.0,
                    "sample_count": 0,
                    "last_tick": int(tick_index),
                    "last_feedback": 0.0,
                }
                self._context_instance_bias[entry_id] = entry
            bias = float(entry.get("bias", 0.0) or 0.0)
            confidence = float(entry.get("confidence", 0.0) or 0.0)
            learning_rate = 0.32 * max(0.2, 1.0 - min(0.9, confidence))
            next_bias = bias * self.decay + float(feedback) * learning_rate
            next_confidence = min(1.0, confidence * self.decay + 0.12 + abs(float(feedback)) * 0.10)
            entry["bias"] = _round4(max(-1.0, min(1.0, next_bias)))
            entry["confidence"] = _round4(next_confidence)
            entry["sample_count"] = int(entry.get("sample_count", 0) or 0) + 1
            entry["last_tick"] = int(tick_index)
            entry["last_feedback"] = _round4(feedback)
            updates.append(
                {
                    "context_key": context_key,
                    "instance_id": clean_instance_id,
                    "next_bias": entry["bias"],
                    "confidence": entry["confidence"],
                }
            )
        return updates

    def _context_entry_id(self, context_key: str, action_id: str) -> str:
        return f"{str(context_key or '').strip()}||{str(action_id or '').strip()}"

    def _context_instance_entry_id(self, context_key: str, instance_id: str) -> str:
        return f"{str(context_key or '').strip()}||{str(instance_id or '').strip()}"

    def _context_key_action_count(self, context_key: str) -> int:
        clean = str(context_key or "").strip()
        if not clean:
            return 1
        count = 0
        suffix = "||"
        for entry_id in self._context_action_bias.keys():
            if entry_id.startswith(clean + suffix):
                count += 1
        return max(1, count)

    def _context_key_instance_count(self, context_key: str) -> int:
        clean = str(context_key or "").strip()
        if not clean:
            return 1
        count = 0
        suffix = "||"
        for entry_id in self._context_instance_bias.keys():
            if entry_id.startswith(clean + suffix):
                count += 1
        return max(1, count)

    def _context_specificity_weight(self, context_key: str) -> float:
        clean = str(context_key or "").strip()
        if not clean:
            return 1.0
        if clean.startswith("text::"):
            return 2.2
        if clean.startswith("focus_text::"):
            return 1.8
        if clean.startswith("short_term::"):
            return 1.35
        if clean.startswith("focus::"):
            return 0.85
        if clean.startswith("unit::"):
            return 0.6
        return 1.0
