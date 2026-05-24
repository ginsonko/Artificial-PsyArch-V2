# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
import math
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ActionPlannerV2:
    def __init__(self) -> None:
        self._actuator_state: dict[str, dict[str, Any]] = {}
        self._recent_action_feedback: dict[str, dict[str, Any]] = {}
        self._last_tick = -1

    def plan_actions(
        self,
        *,
        tick_index: int,
        raw_action_drives: list[dict[str, Any]],
        rules_result: dict[str, Any],
        bn_list: list[dict[str, Any]],
        c_star: dict[str, Any],
        action_learning: Any,
        context_hints: dict[str, Any] | None = None,
        image_packet: dict[str, Any] | None = None,
        audio_packet: dict[str, Any] | None = None,
        pending_feedback: dict[str, float] | None = None,
        recent_focus_units: list[str] | None = None,
    ) -> dict[str, Any]:
        self._advance_tick(tick_index)
        pending_feedback = dict(pending_feedback or {})
        emotion_channels = dict((rules_result or {}).get("emotion_channels", {}) or {})
        metrics_snapshot = dict((rules_result or {}).get("metrics_snapshot", {}) or {})
        recent_focus_units = [str(item or "") for item in (recent_focus_units or []) if str(item or "")]
        query_units = [str(item or "") for item in ((context_hints or {}).get("query_units", []) or []) if str(item or "")]
        focus_units = [str(item or "") for item in ((context_hints or {}).get("focus_units", []) or []) if str(item or "")]
        context_signature = self._context_signature(context_hints=context_hints, query_units=query_units, focus_units=focus_units)
        top_prediction_items = [dict(item) for item in ((c_star or {}).get("items", []) or [])[:12] if isinstance(item, dict)]
        top_prediction_labels = {str(item.get("sa_label", "") or "") for item in top_prediction_items if str(item.get("sa_label", "") or "")}
        memory_texts = [str(item.get("text", "") or "") for item in (bn_list or [])[:6] if str(item.get("text", "") or "")]

        candidates: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_action_drives or []):
            if not isinstance(raw, dict):
                continue
            action_id = str(raw.get("action_id", "") or "").strip()
            if not action_id:
                continue
            params = dict(raw.get("params", {}) or {}) if isinstance(raw.get("params"), dict) else {}
            action_name = str(raw.get("action_name", "") or action_id.replace("action::", "")).strip()
            actuator_id = self._infer_actuator_id(action_name=action_name, params=params)
            instance_id = self._build_instance_id(action_id=action_id, params=params)
            base_drive = self._compute_base_drive(float(raw.get("drive", 0.0) or 0.0))
            modulation = action_learning.modulation_snapshot(action_id=action_id, instance_id=instance_id, context_hints=context_hints or {})
            outcome = self._predict_outcome(
                action_name=action_name,
                action_id=action_id,
                params=params,
                bn_list=bn_list,
                c_star_items=top_prediction_items,
                prediction_labels=top_prediction_labels,
                emotion_channels=emotion_channels,
                image_packet=image_packet or {},
                audio_packet=audio_packet or {},
                recent_focus_units=recent_focus_units,
                query_units=query_units,
                focus_units=focus_units,
                memory_texts=memory_texts,
                pending_feedback=pending_feedback,
                metrics_snapshot=metrics_snapshot,
            )
            pred_modulation = self._pred_modulation(outcome)
            validity_modulation = self._validity_modulation(
                action_name=action_name,
                params=params,
                image_packet=image_packet or {},
                audio_packet=audio_packet or {},
                recent_focus_units=recent_focus_units,
            )
            satisfied_modulation = self._satisfied_modulation(
                action_name=action_name,
                params=params,
                outcome=outcome,
                focus_units=focus_units,
                query_units=query_units,
            )
            fatigue_modulation = self._fatigue_modulation(actuator_id=actuator_id)
            recent_suppression = self._recent_failure_modulation(instance_id=instance_id, context_signature=context_signature)
            final_drive = base_drive
            final_drive *= float(modulation.get("habit_modulation", 1.0) or 1.0)
            final_drive *= float(modulation.get("context_modulation", 1.0) or 1.0)
            final_drive *= pred_modulation
            final_drive *= validity_modulation
            final_drive *= satisfied_modulation
            final_drive *= fatigue_modulation
            final_drive *= recent_suppression
            final_drive = _clamp(final_drive, 0.0, 2.2)
            candidates.append(
                {
                    **copy.deepcopy(raw),
                    "candidate_index": index,
                    "instance_id": instance_id,
                    "action_name": action_name,
                    "action_family": action_name,
                    "actuator_id": actuator_id,
                    "raw_drive": _round4(base_drive),
                    "base_drive": _round4(base_drive),
                    "pred_modulation": _round4(pred_modulation),
                    "learned_bias": _round4(float(modulation.get("learned_bias", 0.0) or 0.0)),
                    "bias_confidence": _round4(float(modulation.get("bias_confidence", 0.0) or 0.0)),
                    "instance_bias": _round4(float(modulation.get("instance_bias", 0.0) or 0.0)),
                    "instance_confidence": _round4(float(modulation.get("instance_confidence", 0.0) or 0.0)),
                    "context_bias": _round4(float(modulation.get("context_bias", 0.0) or 0.0)),
                    "context_bias_confidence": _round4(float(modulation.get("context_bias_confidence", 0.0) or 0.0)),
                    "context_instance_bias": _round4(float(modulation.get("context_instance_bias", 0.0) or 0.0)),
                    "context_instance_confidence": _round4(float(modulation.get("context_instance_confidence", 0.0) or 0.0)),
                    "context_bias_keys": list(modulation.get("context_bias_keys", []) or []),
                    "context_instance_bias_keys": list(modulation.get("context_instance_bias_keys", []) or []),
                    "habit_modulation": _round4(float(modulation.get("habit_modulation", 1.0) or 1.0)),
                    "context_modulation": _round4(float(modulation.get("context_modulation", 1.0) or 1.0)),
                    "validity_modulation": _round4(validity_modulation),
                    "satisfied_modulation": _round4(satisfied_modulation),
                    "fatigue_modulation": _round4(fatigue_modulation),
                    "recent_failure_modulation": _round4(recent_suppression),
                    "outcome_prediction": outcome,
                    "drive": _round4(final_drive),
                    "goal_ids": self._goal_ids(action_name=action_name, params=params),
                    "planner_context_signature": context_signature,
                    "planner_selected": False,
                    "firmness": 0.0,
                    "firmness_norm": 0.0,
                }
            )

        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in candidates:
            grouped.setdefault(str(row.get("actuator_id", "") or "actuator::generic"), []).append(row)

        planned_rows: list[dict[str, Any]] = []
        selected_actions: list[dict[str, Any]] = []
        actuator_reports: list[dict[str, Any]] = []
        for actuator_id, rows in grouped.items():
            report = self._resolve_actuator(
                tick_index=tick_index,
                actuator_id=actuator_id,
                candidates=rows,
            )
            actuator_reports.append(report)
            planned_rows.extend(report["planned_candidates"])
            selected_actions.extend(report["selected_actions"])

        planned_rows.sort(
            key=lambda item: (
                -float(item.get("planned_drive", item.get("drive", 0.0)) or item.get("drive", 0.0) or 0.0),
                str(item.get("action_id", "") or ""),
                str(item.get("instance_id", "") or ""),
            )
        )
        selected_actions.sort(
            key=lambda item: (
                -float(item.get("drive", 0.0) or 0.0),
                str(item.get("action_id", "") or ""),
                str(item.get("instance_id", "") or ""),
            )
        )
        return {
            "candidates": candidates,
            "planned_action_drives": planned_rows,
            "selected_actions_preview": selected_actions,
            "actuator_reports": actuator_reports,
            "actuator_state": self.snapshot_actuator_state(),
            "context_signature": context_signature,
        }

    def record_execution_feedback(
        self,
        *,
        tick_index: int,
        selected_actions: list[dict[str, Any]],
        external_feedback: dict[str, Any] | None = None,
        runtime_action_effects: dict[str, Any] | None = None,
    ) -> None:
        external_feedback = dict(external_feedback or {})
        runtime_action_effects = dict(runtime_action_effects or {})
        reward = float(external_feedback.get("reward", 0.0) or 0.0)
        punishment = float(external_feedback.get("punishment", 0.0) or 0.0)
        moved = bool(runtime_action_effects.get("moved", False))
        for row in selected_actions or []:
            if not isinstance(row, dict):
                continue
            instance_id = str(row.get("instance_id", "") or self._build_instance_id(action_id=str(row.get("action_id", "") or ""), params=dict(row.get("params", {}) or {})))
            action_name = str(row.get("action_name", "") or row.get("action_id", "") or "")
            actuator_id = str(row.get("actuator_id", "") or self._infer_actuator_id(action_name=action_name, params=dict(row.get("params", {}) or {})))
            state = self._ensure_actuator_state(actuator_id)
            state["last_selected_tick"] = int(tick_index)
            state["last_selected_instance_id"] = instance_id
            state["fatigue"] = _clamp(float(state.get("fatigue", 0.0) or 0.0) + 0.18, 0.0, 1.0)
            suppress = 0.0
            suppress_ticks = 0
            if punishment > reward or (action_name in {"continue_focus", "inspect_residual", "move_gaze"} and not moved):
                suppress = _clamp(0.55 + punishment * 0.35, 0.2, 0.95)
                suppress_ticks = 2
            elif reward > punishment:
                suppress = 1.0
                suppress_ticks = 0
            feedback_key = self._feedback_key(instance_id=instance_id, context_signature=str(row.get("planner_context_signature", "") or ""))
            self._recent_action_feedback[feedback_key] = {
                "last_tick": int(tick_index),
                "reward": _round4(reward),
                "punishment": _round4(punishment),
                "suppress_modulation": _round4(suppress),
                "suppress_ticks": int(suppress_ticks),
            }

    def snapshot_actuator_state(self) -> dict[str, Any]:
        return {key: dict(value) for key, value in self._actuator_state.items()}

    def export_payload(self) -> dict[str, Any]:
        return {
            "actuator_state": self.snapshot_actuator_state(),
            "recent_action_feedback": {key: dict(value) for key, value in self._recent_action_feedback.items()},
            "last_tick": int(self._last_tick),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        actuator_state = payload.get("actuator_state", {}) or {}
        self._actuator_state = {
            str(key or ""): dict(value)
            for key, value in actuator_state.items()
            if str(key or "") and isinstance(value, dict)
        }
        recent = payload.get("recent_action_feedback", {}) or {}
        self._recent_action_feedback = {
            str(key or ""): dict(value)
            for key, value in recent.items()
            if str(key or "") and isinstance(value, dict)
        }
        self._last_tick = int(payload.get("last_tick", -1) or -1)

    def _advance_tick(self, tick_index: int) -> None:
        tick_index = int(tick_index)
        delta = max(1, tick_index - self._last_tick) if self._last_tick >= 0 else 1
        for state in self._actuator_state.values():
            fatigue = float(state.get("fatigue", 0.0) or 0.0)
            state["fatigue"] = _round4(max(0.0, fatigue * pow(0.58, delta)))
            threshold = float(state.get("threshold", 0.45) or 0.45)
            target_threshold = self._baseline_threshold(str(state.get("actuator_id", "") or ""))
            state["threshold"] = _round4(target_threshold + (threshold - target_threshold) * pow(0.55, delta))
        for entry in self._recent_action_feedback.values():
            suppress_ticks = int(entry.get("suppress_ticks", 0) or 0)
            if suppress_ticks > 0:
                entry["suppress_ticks"] = max(0, suppress_ticks - delta)
        self._last_tick = tick_index

    def _ensure_actuator_state(self, actuator_id: str) -> dict[str, Any]:
        clean = str(actuator_id or "actuator::generic")
        state = self._actuator_state.get(clean)
        if state is None:
            state = {
                "actuator_id": clean,
                "threshold": self._baseline_threshold(clean),
                "fatigue": 0.0,
                "last_selected_tick": -1,
                "last_selected_instance_id": "",
                "last_hesitation_tick": -1,
            }
            self._actuator_state[clean] = state
        return state

    def _compute_base_drive(self, rule_drive: float) -> float:
        rule_drive = _clamp(rule_drive, 0.0, 1.5)
        return _clamp(1.0 - max(0.0, 1.0 - rule_drive * 0.92), 0.0, 1.5)

    def _build_instance_id(self, *, action_id: str, params: dict[str, Any]) -> str:
        clean_params = json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return f"{str(action_id or '').strip()}::{clean_params}"

    def _infer_actuator_id(self, *, action_name: str, params: dict[str, Any]) -> str:
        clean = str(action_name or "").strip()
        if clean in {"move_gaze", "continue_focus", "inspect_residual"}:
            return "actuator::vision_gaze"
        if clean in {"move_audio_focus", "continue_audio_focus", "inspect_audio_residual"}:
            return "actuator::hearing_focus"
        if clean in {"move_mouse", "click", "double_click", "scroll"}:
            return "actuator::computer_pointer"
        if clean in {"type_text", "press_key"}:
            return "actuator::computer_keyboard"
        if clean in {"wait", "noop"}:
            return "actuator::timing"
        return f"actuator::{clean or 'generic'}"

    def _goal_ids(self, *, action_name: str, params: dict[str, Any]) -> list[str]:
        if action_name in {"move_gaze", "continue_focus", "inspect_residual"}:
            x = params.get("x")
            y = params.get("y")
            if x is not None and y is not None:
                return [f"goal::gaze::{_round4(float(x))}::{_round4(float(y))}"]
            return [f"goal::{action_name}"]
        if action_name in {"move_audio_focus", "continue_audio_focus", "inspect_audio_residual"}:
            center_hz = params.get("center_hz")
            bandwidth = params.get("bandwidth_octaves")
            if center_hz is not None:
                if bandwidth is not None:
                    return [f"goal::audio_focus::{_round4(float(center_hz))}::{_round4(float(bandwidth))}"]
                return [f"goal::audio_focus::{_round4(float(center_hz))}"]
            return [f"goal::{action_name}"]
        if action_name in {"type_text", "press_key"}:
            return [f"goal::{action_name}::{json.dumps(params, ensure_ascii=False, sort_keys=True)}"]
        return [f"goal::{action_name}"]

    def _predict_outcome(
        self,
        *,
        action_name: str,
        action_id: str,
        params: dict[str, Any],
        bn_list: list[dict[str, Any]],
        c_star_items: list[dict[str, Any]],
        prediction_labels: set[str],
        emotion_channels: dict[str, Any],
        image_packet: dict[str, Any],
        audio_packet: dict[str, Any],
        recent_focus_units: list[str],
        query_units: list[str],
        focus_units: list[str],
        memory_texts: list[str],
        pending_feedback: dict[str, float],
        metrics_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        predicted_reward = 0.0
        predicted_punishment = 0.0
        predicted_expectation = float(emotion_channels.get("expectation", 0.0) or 0.0) * 0.2
        predicted_pressure = float(emotion_channels.get("pressure", 0.0) or 0.0) * 0.2
        predicted_correctness = 0.0
        predicted_dissonance = float(emotion_channels.get("dissonance", 0.0) or 0.0) * 0.2
        confidence = 0.18
        notes: list[str] = []
        grasp_score = _clamp(float(metrics_snapshot.get("state.prediction_grasp_score", 0.0) or 0.0), 0.0, 1.0)
        committed_grasp = _clamp(float(metrics_snapshot.get("state.prediction_committed_grasp_score", 0.0) or 0.0), 0.0, 1.0)
        committed_alignment = _clamp(float(metrics_snapshot.get("state.prediction_committed_alignment_score", 0.0) or 0.0), 0.0, 1.0)
        active_commitment = max(grasp_score, committed_grasp * 1.08)

        if action_name == "continue_focus":
            continuity = self._tail_continuity_score(recent_focus_units=recent_focus_units, focus_units=focus_units, bn_list=bn_list, c_star_items=c_star_items)
            predicted_reward += 0.08 + continuity * 0.28
            predicted_expectation += continuity * 0.34
            predicted_correctness += continuity * 0.18
            predicted_punishment += max(0.0, 0.08 - continuity * 0.05)
            predicted_dissonance += max(0.0, 0.12 - continuity * 0.10)
            confidence += 0.30
            notes.append("continuity_bias")
        elif action_name == "inspect_residual":
            unresolved = max(0.0, float(emotion_channels.get("dissonance", 0.0) or 0.0))
            predicted_reward += unresolved * 0.16
            predicted_expectation += unresolved * 0.14
            predicted_correctness += unresolved * 0.08
            predicted_punishment += max(0.0, 0.05 - unresolved * 0.03)
            predicted_dissonance += max(0.0, 0.03, unresolved * 0.10)
            confidence += 0.24
            notes.append("residual_probe")
        elif action_name == "move_gaze":
            target = self._gaze_target_match_score(params=params, image_packet=image_packet)
            predicted_reward += 0.04 + target * 0.18
            predicted_expectation += target * 0.10
            predicted_correctness += target * 0.06
            predicted_punishment += max(0.0, 0.07 - target * 0.04)
            predicted_dissonance += max(0.0, 0.08 - target * 0.05)
            confidence += 0.18
            notes.append("gaze_targeting")
        elif action_name == "move_audio_focus":
            target = self._audio_target_match_score(params=params, audio_packet=audio_packet)
            predicted_reward += 0.04 + target * 0.16
            predicted_expectation += target * 0.12
            predicted_correctness += target * 0.08
            predicted_punishment += max(0.0, 0.07 - target * 0.05)
            predicted_dissonance += max(0.0, 0.08 - target * 0.06)
            confidence += 0.18
            notes.append("audio_focus_targeting")
        elif action_name == "continue_audio_focus":
            continuity = self._audio_focus_continuity_score(audio_packet=audio_packet)
            predicted_reward += 0.08 + continuity * 0.24
            predicted_expectation += continuity * 0.26
            predicted_correctness += continuity * 0.12
            predicted_punishment += max(0.0, 0.08 - continuity * 0.05)
            predicted_dissonance += max(0.0, 0.10 - continuity * 0.08)
            confidence += 0.24
            notes.append("audio_focus_continuity")
        elif action_name == "inspect_audio_residual":
            residual = self._audio_residual_score(audio_packet=audio_packet)
            predicted_reward += residual * 0.15
            predicted_expectation += residual * 0.12
            predicted_correctness += residual * 0.08
            predicted_punishment += max(0.0, 0.05 - residual * 0.03)
            predicted_dissonance += max(0.0, 0.04, residual * 0.10)
            confidence += 0.22
            notes.append("audio_residual_probe")
        elif action_name == "type_text":
            lexical_fit = self._symbolic_action_fit(params=params, query_units=query_units, memory_texts=memory_texts, action_name=action_name)
            predicted_reward += lexical_fit * 0.22
            predicted_expectation += lexical_fit * 0.16
            predicted_correctness += lexical_fit * 0.14
            predicted_punishment += max(0.0, 0.22 - lexical_fit * 0.14)
            predicted_pressure += max(0.0, 0.08 - lexical_fit * 0.05)
            predicted_dissonance += max(0.0, 0.16 - lexical_fit * 0.10)
            confidence += 0.32
            notes.append("symbolic_affordance")
        elif action_name == "press_key":
            symbolic_fit = self._symbolic_action_fit(params=params, query_units=query_units, memory_texts=memory_texts, action_name=action_name)
            predicted_reward += symbolic_fit * 0.24
            predicted_expectation += symbolic_fit * 0.12
            predicted_correctness += symbolic_fit * 0.16
            predicted_punishment += max(0.0, 0.18 - symbolic_fit * 0.16)
            predicted_pressure += max(0.0, 0.10 - symbolic_fit * 0.06)
            predicted_dissonance += max(0.0, 0.12 - symbolic_fit * 0.10)
            confidence += 0.32
            notes.append("symbolic_affordance")

        if prediction_labels:
            overlap = sum(1 for unit in query_units[-3:] if f"text::{unit}" in prediction_labels)
            predicted_correctness += overlap * 0.03
            confidence += min(0.12, overlap * 0.04)

        if active_commitment > 0.0:
            predicted_correctness += active_commitment * 0.18
            predicted_expectation += committed_alignment * 0.12
            predicted_reward += committed_grasp * 0.08
            predicted_dissonance = max(0.0, predicted_dissonance - committed_grasp * 0.10)
            predicted_pressure = max(0.0, predicted_pressure - committed_grasp * 0.06)
            confidence += 0.18 * active_commitment
            notes.append("commitment_supported")
        else:
            predicted_punishment += 0.04
            predicted_pressure += 0.05
            notes.append("low_grasp_caution")

        predicted_reward += float(pending_feedback.get("reward", 0.0) or 0.0) * 0.08
        predicted_punishment += float(pending_feedback.get("punishment", 0.0) or 0.0) * 0.08
        confidence = _clamp(confidence, 0.0, 1.0)
        return {
            "predicted_reward": _round4(_clamp(predicted_reward, 0.0, 1.0)),
            "predicted_punishment": _round4(_clamp(predicted_punishment, 0.0, 1.0)),
            "predicted_expectation": _round4(_clamp(predicted_expectation, 0.0, 1.0)),
            "predicted_pressure": _round4(_clamp(predicted_pressure, 0.0, 1.0)),
            "predicted_correctness": _round4(_clamp(predicted_correctness, 0.0, 1.0)),
            "predicted_dissonance": _round4(_clamp(predicted_dissonance, 0.0, 1.0)),
            "confidence": _round4(confidence),
            "notes": notes,
        }

    def _pred_modulation(self, outcome: dict[str, Any]) -> float:
        confidence = float(outcome.get("confidence", 0.0) or 0.0)
        grasp_hint = max(
            0.0,
            float(outcome.get("predicted_correctness", 0.0) or 0.0)
            - 0.55 * float(outcome.get("predicted_dissonance", 0.0) or 0.0),
        )
        utility = confidence * (
            1.00 * float(outcome.get("predicted_reward", 0.0) or 0.0)
            + 0.45 * float(outcome.get("predicted_expectation", 0.0) or 0.0)
            + 0.35 * float(outcome.get("predicted_correctness", 0.0) or 0.0)
            - 1.15 * float(outcome.get("predicted_punishment", 0.0) or 0.0)
            - 0.55 * float(outcome.get("predicted_pressure", 0.0) or 0.0)
            - 0.60 * float(outcome.get("predicted_dissonance", 0.0) or 0.0)
        )
        modulation = math.exp(0.9 * utility)
        modulation *= 0.92 + 0.28 * _clamp(grasp_hint, 0.0, 1.0)
        return _clamp(modulation, 0.25, 2.2)

    def _validity_modulation(
        self,
        *,
        action_name: str,
        params: dict[str, Any],
        image_packet: dict[str, Any],
        audio_packet: dict[str, Any],
        recent_focus_units: list[str],
    ) -> float:
        if action_name == "move_gaze":
            target_score = self._gaze_target_match_score(params=params, image_packet=image_packet)
            return _clamp(0.35 + target_score * 0.95, 0.2, 1.2)
        if action_name == "move_audio_focus":
            target_score = self._audio_target_match_score(params=params, audio_packet=audio_packet)
            return _clamp(0.35 + target_score * 0.95, 0.2, 1.2)
        if action_name in {"continue_focus", "inspect_residual"} and not recent_focus_units:
            return 0.55
        return 1.0

    def _satisfied_modulation(
        self,
        *,
        action_name: str,
        params: dict[str, Any],
        outcome: dict[str, Any],
        focus_units: list[str],
        query_units: list[str],
    ) -> float:
        if action_name == "continue_focus":
            tail = "".join(focus_units[-2:] or [])
            if tail and tail in "".join(query_units[-4:] or []):
                return 0.92
        if action_name == "move_gaze":
            return 1.0
        if action_name == "move_audio_focus":
            return 1.0
        if action_name == "type_text":
            text = str(params.get("text", "") or "")
            if text and text in "".join(focus_units):
                return 0.18
        return _clamp(1.05 - float(outcome.get("predicted_reward", 0.0) or 0.0) * 0.12, 0.1, 1.0)

    def _audio_focus_rows(self, *, audio_packet: dict[str, Any]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in ("focus_priority_samples", "memory_write_samples", "windows", "global_structure_samples"):
            rows.extend([dict(item) for item in (audio_packet.get(key, []) or []) if isinstance(item, dict)])
        return rows

    def _audio_target_match_score(self, *, params: dict[str, Any], audio_packet: dict[str, Any]) -> float:
        target_center = float(params.get("center_hz", 0.0) or 0.0)
        if target_center <= 0.0:
            return 0.0
        rows = self._audio_focus_rows(audio_packet=audio_packet)
        if not rows:
            return 0.0
        best = 0.0
        for item in rows:
            coords = dict(item.get("coords", {}) or {})
            attrs = dict(item.get("attributes", {}) or {})
            center_hz = float(coords.get("freq_center_hz", attrs.get("dominant_hz", 0.0)) or 0.0)
            if center_hz <= 0.0:
                continue
            octave_distance = abs(math.log(max(1e-6, center_hz), 2.0) - math.log(max(1e-6, target_center), 2.0))
            proximity = math.exp(-(octave_distance * octave_distance) / max(1e-6, 2.0 * 0.55 * 0.55))
            energy = float(item.get("energy", 0.0) or 0.0)
            focus_priority = float(attrs.get("focus_priority", 0.0) or 0.0)
            score = proximity * (0.55 + 0.25 * min(1.0, energy) + 0.20 * min(1.0, focus_priority))
            best = max(best, score)
        return _clamp(best, 0.0, 1.0)

    def _audio_focus_continuity_score(self, *, audio_packet: dict[str, Any]) -> float:
        rows = self._audio_focus_rows(audio_packet=audio_packet)
        if not rows:
            return 0.0
        score = 0.0
        count = 0
        for item in rows[:8]:
            attrs = dict(item.get("attributes", {}) or {})
            score += (
                0.35 * min(1.0, float(attrs.get("focus_bonus", 0.0) or 0.0))
                + 0.25 * min(1.0, float(attrs.get("novelty", 0.0) or 0.0))
                + 0.20 * min(1.0, float(attrs.get("onset_strength", 0.0) or 0.0))
                + 0.20 * min(1.0, float(item.get("energy", 0.0) or 0.0))
            )
            count += 1
        return _clamp(score / max(1, count), 0.0, 1.0)

    def _audio_residual_score(self, *, audio_packet: dict[str, Any]) -> float:
        rows = self._audio_focus_rows(audio_packet=audio_packet)
        if not rows:
            return 0.0
        best = 0.0
        for item in rows[:12]:
            attrs = dict(item.get("attributes", {}) or {})
            novelty = float(attrs.get("novelty", 0.0) or 0.0)
            onset = float(attrs.get("onset_strength", 0.0) or 0.0)
            fatigue_penalty = float(attrs.get("fatigue_penalty", 0.0) or 0.0)
            best = max(best, novelty * 0.55 + onset * 0.35 + max(0.0, 0.1 - fatigue_penalty))
        return _clamp(best, 0.0, 1.0)

    def _fatigue_modulation(self, *, actuator_id: str) -> float:
        state = self._ensure_actuator_state(actuator_id)
        fatigue = float(state.get("fatigue", 0.0) or 0.0)
        return _clamp(1.0 - fatigue * 0.28, 0.72, 1.0)

    def _recent_failure_modulation(self, *, instance_id: str, context_signature: str) -> float:
        row = self._recent_action_feedback.get(self._feedback_key(instance_id=instance_id, context_signature=context_signature))
        if not row:
            row = self._recent_action_feedback.get(self._feedback_key(instance_id=instance_id, context_signature=""))
        if not row:
            return 1.0
        if int(row.get("suppress_ticks", 0) or 0) <= 0:
            return 1.0
        return _clamp(float(row.get("suppress_modulation", 1.0) or 1.0), 0.2, 1.0)

    def _resolve_actuator(
        self,
        *,
        tick_index: int,
        actuator_id: str,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        state = self._ensure_actuator_state(actuator_id)
        threshold = float(state.get("threshold", 0.45) or 0.45)
        rows = sorted(
            [dict(item) for item in candidates],
            key=lambda item: (-float(item.get("drive", 0.0) or 0.0), str(item.get("action_id", "") or ""), str(item.get("instance_id", "") or "")),
        )
        supra = [row for row in rows if float(row.get("drive", 0.0) or 0.0) > threshold]
        inhibition = 0.0
        hesitation = False
        selected: list[dict[str, Any]] = []
        if len(supra) >= 2:
            second = float(supra[1].get("drive", 0.0) or 0.0)
            inhibition = max(0.0, second - (threshold - 0.01))
        planned_rows: list[dict[str, Any]] = []
        for row in rows:
            planned_drive = max(0.0, float(row.get("drive", 0.0) or 0.0) - (inhibition if float(row.get("drive", 0.0) or 0.0) > threshold else 0.0))
            next_row = {
                **row,
                "actuator_threshold": _round4(threshold),
                "shared_inhibition": _round4(inhibition),
                "planned_drive": _round4(planned_drive),
            }
            planned_rows.append(next_row)

        if planned_rows:
            top = planned_rows[0]
            margin = float(top.get("planned_drive", 0.0) or 0.0) - threshold
            near_conflict = False
            if len(planned_rows) >= 2:
                second_drive = float(planned_rows[1].get("planned_drive", 0.0) or 0.0)
                near_conflict = second_drive >= (threshold - 0.015) and abs(float(top.get("planned_drive", 0.0) or 0.0) - second_drive) <= 0.05
            top_drive = float(top.get("planned_drive", 0.0) or 0.0)
            second_gap = top_drive - (float(planned_rows[1].get("planned_drive", 0.0) or 0.0) if len(planned_rows) >= 2 else 0.0)
            hesitation = (
                len(supra) >= 2
                and top_drive <= threshold
                and (
                    margin <= 0.04
                    or (margin <= 0.025 and near_conflict)
                    or second_gap <= 0.012
                )
            )
            if float(top.get("planned_drive", 0.0) or 0.0) > threshold and not hesitation:
                firmness = max(0.0, float(top.get("planned_drive", 0.0) or 0.0) - threshold)
                selected_row = {
                    **top,
                    "drive_before_planner": _round4(float(top.get("drive", 0.0) or 0.0)),
                    "drive": _round4(float(top.get("planned_drive", 0.0) or 0.0)),
                    "firmness": _round4(firmness),
                    "firmness_norm": _round4(_clamp(firmness / 0.25, 0.0, 1.5)),
                    "planner_context_signature": str(top.get("planner_context_signature", "") or ""),
                    "planner_selected": True,
                }
                selected.append(selected_row)
                state["threshold"] = _round4(_clamp(0.45 + min(0.22, firmness * 0.75), 0.35, 0.85))
                state["fatigue"] = _round4(_clamp(float(state.get("fatigue", 0.0) or 0.0) + min(0.18, firmness * 0.28), 0.0, 1.0))
                state["last_selected_tick"] = int(tick_index)
                state["last_selected_instance_id"] = str(selected_row.get("instance_id", "") or "")
            elif hesitation:
                state["last_hesitation_tick"] = int(tick_index)
                state["threshold"] = _round4(_clamp(threshold + 0.04, 0.35, 0.85))

        return {
            "actuator_id": actuator_id,
            "threshold": _round4(threshold),
            "shared_inhibition": _round4(inhibition),
            "hesitation": hesitation,
            "planned_candidates": planned_rows,
            "selected_actions": selected,
        }

    def _tail_continuity_score(
        self,
        *,
        recent_focus_units: list[str],
        focus_units: list[str],
        bn_list: list[dict[str, Any]],
        c_star_items: list[dict[str, Any]],
    ) -> float:
        tail = "".join((focus_units or recent_focus_units)[-3:])
        if not tail:
            return 0.35 if bn_list else 0.1
        score = 0.0
        for row in bn_list[:4]:
            text = str(row.get("text", "") or "")
            if tail and tail in text:
                score += 0.25
        for item in c_star_items[:6]:
            label = str(item.get("sa_label", "") or "")
            if any(unit and unit in label for unit in recent_focus_units[-2:]):
                score += 0.08
        return _clamp(score, 0.0, 1.0)

    def _gaze_target_match_score(self, *, params: dict[str, Any], image_packet: dict[str, Any]) -> float:
        if "x" not in params or "y" not in params:
            return 0.0
        x = float(params.get("x", 0.5) or 0.5)
        y = float(params.get("y", 0.5) or 0.5)
        best = 0.0
        candidate_items = (
            image_packet.get("focus_priority_samples", [])
            or image_packet.get("memory_write_samples", [])
            or image_packet.get("patches", [])
            or []
        )
        for item in candidate_items:
            if not isinstance(item, dict):
                continue
            coords = dict(item.get("coords", {}) or {})
            cx = None
            cy = None
            if "screen_x" in coords and "screen_w" in coords:
                cx = float(coords.get("screen_x", 0.0) or 0.0) + float(coords.get("screen_w", 0.0) or 0.0) * 0.5
                cy = float(coords.get("screen_y", 0.0) or 0.0) + float(coords.get("screen_h", 0.0) or 0.0) * 0.5
            elif "cx" in coords and "cy" in coords:
                cx = float(coords.get("cx", 0.0) or 0.0)
                cy = float(coords.get("cy", 0.0) or 0.0)
            if cx is None or cy is None:
                continue
            dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            energy = float(item.get("energy", 0.0) or 0.0)
            score = max(0.0, 1.0 - dist * 1.6) * min(1.0, max(0.05, energy))
            best = max(best, score)
        return _clamp(best, 0.0, 1.0)

    def _symbolic_action_fit(self, *, params: dict[str, Any], query_units: list[str], memory_texts: list[str], action_name: str) -> float:
        query_text = "".join(query_units)
        memory_text = " ".join(memory_texts)
        if action_name == "type_text":
            text = str(params.get("text", "") or "")
            score = 0.18
            if "记事本" in query_text or "notepad" in query_text.lower():
                score += 0.62
            if text and text in memory_text:
                score += 0.12
            return _clamp(score, 0.0, 1.0)
        if action_name == "press_key":
            key = str(params.get("key", "") or "")
            score = 0.16
            if "计算器" in query_text or "calc" in query_text.lower():
                score += 0.68
            if key and key in memory_text:
                score += 0.08
            return _clamp(score, 0.0, 1.0)
        return 0.2

    def _context_signature(self, *, context_hints: dict[str, Any] | None, query_units: list[str], focus_units: list[str]) -> str:
        parts = []
        for raw in (context_hints or {}).get("context_keys", []) or []:
            clean = str(raw or "").strip()
            if clean:
                parts.append(clean)
        if not parts and query_units:
            parts.append("query::" + "".join(query_units[:8]))
        if focus_units:
            parts.append("focus::" + "".join(focus_units[:8]))
        return "||".join(parts[:4])

    def _feedback_key(self, *, instance_id: str, context_signature: str) -> str:
        clean_instance_id = str(instance_id or "").strip()
        clean_context = str(context_signature or "").strip()
        if clean_context:
            return f"{clean_context}||{clean_instance_id}"
        return clean_instance_id

    def _baseline_threshold(self, actuator_id: str) -> float:
        clean = str(actuator_id or "")
        if clean == "actuator::computer_keyboard":
            return 0.30
        if clean == "actuator::computer_pointer":
            return 0.34
        if clean == "actuator::timing":
            return 0.28
        return 0.45
