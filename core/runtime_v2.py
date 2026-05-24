# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import math
import time

from typing import Any

from core.action_learning_v2 import ActionLearningV2
from core.action_planner_v2 import ActionPlannerV2
from core.sa_registry_v2 import SARegistryV2
from core.state_pool_v2 import StatePoolV2
from core.teacher_layer_v1 import TeacherLayerV1
from core.tuner_learning_v2 import TunerLearningV2
from iesm.rules_engine_v2 import RulesEngineV2
from memory.memory_store_v2 import MemoryStoreV2
from memory.short_term_memory_v2 import ShortTermMemoryV2
from sensors.hearing_sensor_v1 import HearingSensorV1
from sensors.text_sensor_v2 import TextSensorV2, join_text_units
from sensors.vision_sensor_v1 import VisionSensorV1


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _stage_ms(started_at: float) -> float:
    return round((time.perf_counter() - float(started_at)) * 1000.0, 4)


def _clone_item_shallow(item: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(item)
    if "coords" in cloned:
        cloned["coords"] = dict(item.get("coords", {}) or {})
    if "attributes" in cloned:
        cloned["attributes"] = dict(item.get("attributes", {}) or {})
    if "support" in cloned:
        cloned["support"] = dict(item.get("support", {}) or {})
    return cloned


class RuntimeV2:
    def __init__(self, *, config: Any, repo_root: Any | None = None) -> None:
        self.config = config
        self._last_logic_ms = 0.0
        self._runtime_controls = self._default_runtime_controls()
        self._last_emotion_channels: dict[str, float] = {
            "dissonance": 0.0,
            "surprise": 0.0,
            "correctness": 0.0,
            "expectation": 0.0,
            "pressure": 0.0,
            "grasp": 0.0,
        }
        self._last_cognitive_balance: dict[str, float] = {
            "alignment_score": 0.0,
            "grasp_score": 0.0,
            "overprediction_ratio": 0.0,
            "underprediction_ratio": 0.0,
            "committed_alignment_score": 0.0,
            "committed_grasp_score": 0.0,
            "committed_overprediction_ratio": 0.0,
        }
        self.text_sensor = TextSensorV2(
            budget_limit=self.config.text_sensor_budget,
            fatigue_window=self.config.text_sensor_fatigue_window,
            fatigue_threshold=self.config.text_sensor_fatigue_threshold,
            max_suppression=self.config.text_sensor_max_suppression,
        )
        self.vision_sensor = VisionSensorV1(
            patch_budget=self.config.vision_patch_budget,
            focus_patch_budget=self.config.vision_focus_patch_budget,
            raw_state_budget=self.config.vision_raw_state_budget,
            reconstruction_patch_budget=self.config.vision_reconstruction_patch_budget,
            edge_candidate_gain=self.config.vision_edge_candidate_gain,
            edge_priority_gain=self.config.vision_edge_priority_gain,
            attention_boost_enabled=self.config.vision_attention_boost_enabled,
            attention_boost_decay=self.config.vision_attention_boost_decay,
            attention_boost_max_extra_raw_budget=self.config.vision_attention_boost_max_extra_raw_budget,
            attention_boost_max_extra_focus_budget=self.config.vision_attention_boost_max_extra_focus_budget,
            attention_boost_min_radius_scale=self.config.vision_attention_boost_min_radius_scale,
            attention_boost_edge_gain=self.config.vision_attention_boost_edge_gain,
            attention_boost_gaze_sigma_scale=self.config.vision_attention_boost_gaze_sigma_scale,
            dynamic_track_window=self.config.vision_dynamic_track_window,
            dynamic_candidate_limit_background=self.config.vision_dynamic_candidate_limit_background,
            dynamic_candidate_limit_focus=self.config.vision_dynamic_candidate_limit_focus,
            dynamic_track_limit=self.config.vision_dynamic_track_limit,
            dynamic_summary_limit=self.config.vision_dynamic_summary_limit,
            dynamic_match_threshold=self.config.vision_dynamic_match_threshold,
            dynamic_track_forget_ticks=self.config.vision_dynamic_track_forget_ticks,
        )
        self.hearing_sensor = HearingSensorV1(
            window_budget=self.config.hearing_window_budget,
            window_ms=self.config.hearing_window_ms,
            focus_band_count=self.config.hearing_focus_band_count,
            focus_bandwidth_octaves=self.config.hearing_focus_bandwidth_octaves,
            attention_boost_enabled=self.config.hearing_attention_boost_enabled,
            attention_boost_decay=self.config.hearing_attention_boost_decay,
            attention_boost_max_extra_window_budget=self.config.hearing_attention_boost_max_extra_window_budget,
            attention_boost_max_extra_focus_budget=self.config.hearing_attention_boost_max_extra_focus_budget,
            attention_boost_min_bandwidth_scale=self.config.hearing_attention_boost_min_bandwidth_scale,
            attention_boost_focus_gain=self.config.hearing_attention_boost_focus_gain,
            static_dedup_delta_threshold=self.config.hearing_static_dedup_delta_threshold,
            static_dedup_band_similarity_threshold=self.config.hearing_static_dedup_band_similarity_threshold,
            static_dedup_max_suppression=self.config.hearing_static_dedup_max_suppression,
            auditory_fatigue_decay=self.config.hearing_auditory_fatigue_decay,
            auditory_fatigue_step=self.config.hearing_auditory_fatigue_step,
            auditory_fatigue_max=self.config.hearing_auditory_fatigue_max,
        )
        self.state_pool = StatePoolV2(
            decay=self.config.state_pool_decay,
            prune_threshold=self.config.state_pool_prune_threshold,
            recent_queue_limit=self.config.state_pool_recent_queue_limit,
            verbatim_window_chars=self.config.text_sensor_verbatim_window_chars,
            head_limit=self.config.r_state_head_limit,
            items_per_head=self.config.r_state_items_per_head,
            anchor_cache_limit=self.config.state_pool_anchor_cache_limit,
            residual_limit=self.config.state_pool_residual_limit,
            handle_limit=self.config.state_pool_handle_limit,
            residual_unit_limit=self.config.state_pool_residual_unit_limit,
            attention_object_fatigue_decay=self.config.state_pool_attention_object_fatigue_decay,
            attention_object_fatigue_step=self.config.state_pool_attention_object_fatigue_step,
            attention_object_fatigue_gain=self.config.state_pool_attention_object_fatigue_gain,
            attention_object_fatigue_max=self.config.state_pool_attention_object_fatigue_max,
            attention_object_min_multiplier=self.config.state_pool_attention_object_min_multiplier,
        )
        self.sa_registry = SARegistryV2()
        self.memory_store = MemoryStoreV2(
            max_recent=self.config.memory_store_recent_limit,
            vector_dim=self.config.memory_vector_dim,
            vector_backend=self.config.memory_vector_backend,
            ann_enabled=self.config.memory_ann_enabled,
            ann_top_k=self.config.memory_ann_top_k,
            candidate_limit=self.config.memory_candidate_limit,
            spacetime_backend=self.config.memory_spacetime_backend,
            time_bucket_size=self.config.memory_spacetime_time_bucket_size,
            space_bucket_size=self.config.memory_spacetime_space_bucket_size,
            spacetime_time_radius=self.config.memory_spacetime_time_radius,
            spacetime_space_radius=self.config.memory_spacetime_space_radius,
            recall_fatigue_decay=self.config.memory_recall_fatigue_decay,
            recall_fatigue_gain=self.config.memory_recall_fatigue_gain,
            recall_fatigue_accumulate_scale=self.config.memory_recall_fatigue_accumulate_scale,
            recall_fatigue_max=self.config.memory_recall_fatigue_max,
            recall_fatigue_min_multiplier=self.config.memory_recall_fatigue_min_multiplier,
        )
        self.short_term = ShortTermMemoryV2(
            max_items=self.config.short_term_memory_limit,
            successor_tail_limit=self.config.short_term_successor_tail_limit,
        )
        self.action_learning = ActionLearningV2()
        self.action_planner = ActionPlannerV2()
        self.tuner_learning = TunerLearningV2()
        self.teacher_layer = TeacherLayerV1(
            enabled=self.config.autonomous_teacher_enabled,
            mode=self.config.autonomous_teacher_mode,
            llm_gate_enabled=self.config.autonomous_llm_gate_enabled,
            llm_gate_mode=self.config.autonomous_llm_gate_mode,
            llm_gate_fail_open=self.config.autonomous_llm_gate_fail_open,
            reward_scale=self.config.autonomous_teacher_reward_scale,
            punishment_scale=self.config.autonomous_teacher_punishment_scale,
            repeat_window=self.config.autonomous_teacher_repeat_window,
            repeat_penalty=self.config.autonomous_teacher_repeat_penalty,
            risky_action_min_drive=self.config.autonomous_teacher_risky_action_min_drive,
            external_teacher_enabled=self.config.autonomous_external_teacher_enabled,
            external_teacher_mode=self.config.autonomous_external_teacher_mode,
            external_teacher_stub_response_path=self.config.autonomous_external_teacher_stub_response_path,
            external_teacher_fail_open=self.config.autonomous_external_teacher_fail_open,
            external_teacher_timeout_ms=self.config.autonomous_external_teacher_timeout_ms,
            external_teacher_max_retries=self.config.autonomous_external_teacher_max_retries,
            external_teacher_retry_backoff_ms=self.config.autonomous_external_teacher_retry_backoff_ms,
            external_teacher_http_endpoint=self.config.autonomous_external_teacher_http_endpoint,
            external_teacher_http_headers=self.config.autonomous_external_teacher_http_headers,
        )
        self.rules_engine = RulesEngineV2(repo_root=repo_root)
        self._last_control_feedback_context: dict[str, Any] = {
            "runtime_controls": self.runtime_controls_snapshot(),
            "matched_profiles": [],
            "applied_tuner_adjustments": [],
            "learned_tuner_offsets": [],
        }
        self._attention_modulation_state: dict[str, Any] = self._blank_attention_modulation_state()
        self._pending_feedback_metrics: dict[str, float] = {
            "reward": 0.0,
            "punishment": 0.0,
        }
        self._pending_feedback_breakdown: dict[str, Any] = self._blank_feedback_breakdown()
        self._queued_intrinsic_feedback: dict[str, Any] = {}
        self._cognitive_feeling_habituation: dict[str, Any] = self._blank_cognitive_feeling_habituation()
        self._channel_feeling_fatigue: dict[str, dict[str, float | int]] = {}
        self._rhythm_tracker: dict[str, Any] = {"families": {}, "last_tick": -1}

    def _blank_cognitive_feeling_habituation(self) -> dict[str, Any]:
        return {
            "state": {
                "surprise": 0.0,
                "dissonance": 0.0,
            },
            "last_signature": "",
            "last_tick": -1,
            "last_metrics": {},
        }

    def _blank_feedback_breakdown(self) -> dict[str, Any]:
        return {
            "reward": 0.0,
            "punishment": 0.0,
            "notes": [],
            "sources": {
                "external": {"reward": 0.0, "punishment": 0.0, "notes": []},
                "teacher": {"reward": 0.0, "punishment": 0.0, "notes": []},
                "intrinsic": {"reward": 0.0, "punishment": 0.0, "notes": []},
            },
            "intrinsic_detail": {
                "enabled": False,
                "current_emotion": dict(self._last_emotion_channels),
                "previous_emotion": dict(self._last_emotion_channels),
                "delta_emotion": {
                    "dissonance": 0.0,
                    "surprise": 0.0,
                    "correctness": 0.0,
                    "expectation": 0.0,
                    "pressure": 0.0,
                },
                "components": {
                    "correctness_delta_reward": 0.0,
                    "grasp_delta_reward": 0.0,
                    "committed_grasp_delta_reward": 0.0,
                    "surprise_recovery_reward": 0.0,
                    "dissonance_recovery_reward": 0.0,
                    "dissonance_delta_punishment": 0.0,
                    "surprise_delta_punishment": 0.0,
                    "expectation_tonic_reward": 0.0,
                    "pressure_tonic_punishment": 0.0,
                },
                "current_balance": dict(self._last_cognitive_balance),
                "previous_balance": dict(self._last_cognitive_balance),
                "delta_balance": {
                    "alignment_score": 0.0,
                    "grasp_score": 0.0,
                    "overprediction_ratio": 0.0,
                    "underprediction_ratio": 0.0,
                    "committed_alignment_score": 0.0,
                    "committed_grasp_score": 0.0,
                    "committed_overprediction_ratio": 0.0,
                },
            },
        }

    def _normalize_rhythm_tracker(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(payload or {})
        families_payload = dict(data.get("families", {}) or {})
        families: dict[str, Any] = {}
        def _int_field(value: Any, default: int = -1) -> int:
            try:
                if value is None:
                    return int(default)
                return int(value)
            except Exception:
                return int(default)
        for family_key, family_payload in families_payload.items():
            clean_key = str(family_key or "")
            if not clean_key or not isinstance(family_payload, dict):
                continue
            hits_payload = list(family_payload.get("hits", []) or [])
            hits: list[dict[str, Any]] = []
            for row in hits_payload[-64:]:
                if not isinstance(row, dict):
                    continue
                hits.append(
                    {
                        "tick": _int_field(row.get("tick", 0), 0),
                        "strength": _round4(max(0.0, float(row.get("strength", 0.0) or 0.0))),
                        "commitment": _round4(_clamp(float(row.get("commitment", 0.0) or 0.0), 0.0, 1.0)),
                        "reality": _round4(_clamp(float(row.get("reality", 0.0) or 0.0), 0.0, 1.5)),
                        "salience": _round4(max(0.0, float(row.get("salience", 0.0) or 0.0))),
                    }
                )
            families[clean_key] = {
                "hits": hits,
                "fatigue": _round4(_clamp(float(family_payload.get("fatigue", 0.0) or 0.0), 0.0, float(getattr(self.config, "rhythm_fatigue_max", 1.0)))),
                "phase_fatigue": _round4(_clamp(float(family_payload.get("phase_fatigue", 0.0) or 0.0), 0.0, float(getattr(self.config, "rhythm_fatigue_max", 1.0)))),
                "last_tick": _int_field(family_payload.get("last_tick", -1), -1),
            }
        return {
            "families": families,
            "last_tick": _int_field(data.get("last_tick", -1), -1),
        }

    def _channel_feeling_fatigue_value(
        self,
        *,
        channel_key: str,
        signal_key: str,
        tick_index: int,
        decay: float,
    ) -> float:
        channel_map = dict(self._channel_feeling_fatigue.get(str(channel_key or ""), {}) or {})
        entry = dict(channel_map.get(str(signal_key or ""), {}) or {})
        value = max(0.0, float(entry.get("value", 0.0) or 0.0))
        last_tick = int(entry.get("tick_index", tick_index) or tick_index)
        steps = max(0, int(tick_index) - last_tick)
        if steps > 0:
            value *= max(0.0, min(1.0, float(decay))) ** steps
        value = _round4(value)
        if value <= 0.0001:
            channel_map.pop(str(signal_key or ""), None)
            if channel_map:
                self._channel_feeling_fatigue[str(channel_key or "")] = channel_map
            else:
                self._channel_feeling_fatigue.pop(str(channel_key or ""), None)
            return 0.0
        channel_map[str(signal_key or "")] = {"value": value, "tick_index": int(tick_index)}
        self._channel_feeling_fatigue[str(channel_key or "")] = channel_map
        return value

    def _channel_feeling_commit_fatigue(
        self,
        *,
        channel_key: str,
        signal_key: str,
        tick_index: int,
        value: float,
    ) -> None:
        clean_channel = str(channel_key or "")
        clean_signal = str(signal_key or "")
        if not clean_channel or not clean_signal:
            return
        channel_map = dict(self._channel_feeling_fatigue.get(clean_channel, {}) or {})
        channel_map[clean_signal] = {"value": _round4(max(0.0, float(value))), "tick_index": int(tick_index)}
        self._channel_feeling_fatigue[clean_channel] = channel_map

    def _make_channel_feeling_item(
        self,
        *,
        channel_key: str,
        signal_key: str,
        tick_index: int,
        sa_label: str,
        display_text: str,
        source_strength: float,
        confidence: float,
        threshold: float,
        gain: float,
        fatigue_decay: float,
        fatigue_step: float,
        fatigue_gain: float,
        fatigue_max: float,
        channel: str,
        sa_kind: str,
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        clean_label = str(sa_label or "")
        if not clean_label:
            return None
        strength = max(0.0, float(source_strength))
        conf = _clamp(float(confidence), 0.0, 1.0)
        if strength <= float(threshold) or conf <= 0.0:
            return None
        fatigue = self._channel_feeling_fatigue_value(
            channel_key=channel_key,
            signal_key=signal_key,
            tick_index=tick_index,
            decay=fatigue_decay,
        )
        fatigue_multiplier = max(0.0, 1.0 - fatigue * max(0.0, float(fatigue_gain)))
        energy = max(0.0, float(gain)) * max(0.0, strength - float(threshold)) * conf * fatigue_multiplier
        energy = _clamp(energy, 0.0, 1.5)
        if energy <= 0.0:
            return None
        next_fatigue = min(max(0.0, float(fatigue_max)), fatigue + max(0.0, float(fatigue_step)) * energy)
        self._channel_feeling_commit_fatigue(
            channel_key=channel_key,
            signal_key=signal_key,
            tick_index=tick_index,
            value=next_fatigue,
        )
        payload_attrs = dict(attributes or {})
        payload_attrs["feeling_source_strength"] = _round4(strength)
        payload_attrs["feeling_confidence"] = _round4(conf)
        payload_attrs["feeling_fatigue"] = _round4(fatigue)
        payload_attrs["feeling_signal_key"] = clean_label
        return {
            "sa_label": clean_label,
            "display_text": str(display_text or clean_label),
            "energy": _round4(energy),
            "channel": str(channel or "attr"),
            "sa_kind": str(sa_kind or "channel_feeling_unit"),
            "attributes": payload_attrs,
        }

    def _feedback_has_signal(self, payload: dict[str, Any] | None) -> bool:
        data = dict(payload or {})
        if float(data.get("reward", 0.0) or 0.0) > 0.0:
            return True
        if float(data.get("punishment", 0.0) or 0.0) > 0.0:
            return True
        if any(str(item or "") for item in (data.get("notes", []) or [])):
            return True
        return False

    def _normalize_emotion_channels(self, emotion_channels: dict[str, Any] | None) -> dict[str, float]:
        payload = dict(emotion_channels or {})
        return {
            "dissonance": _round4(max(0.0, float(payload.get("dissonance", 0.0) or 0.0))),
            "surprise": _round4(max(0.0, float(payload.get("surprise", 0.0) or 0.0))),
            "correctness": _round4(max(0.0, float(payload.get("correctness", 0.0) or 0.0))),
            "expectation": _round4(max(0.0, float(payload.get("expectation", 0.0) or 0.0))),
            "pressure": _round4(max(0.0, float(payload.get("pressure", 0.0) or 0.0))),
            "grasp": _round4(max(0.0, float(payload.get("grasp", 0.0) or 0.0))),
        }

    def _normalize_habituation_payload(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = dict(payload or {})
        state = dict(data.get("state", {}) or {})
        return {
            "state": {
                "surprise": _round4(_clamp(float(state.get("surprise", 0.0) or 0.0), 0.0, 1.0)),
                "dissonance": _round4(_clamp(float(state.get("dissonance", 0.0) or 0.0), 0.0, 1.0)),
            },
            "last_signature": str(data.get("last_signature", "") or ""),
            "last_tick": int(data.get("last_tick", -1) or -1),
            "last_metrics": dict(data.get("last_metrics", {}) or {}),
        }

    def _build_cognitive_feeling_signature(
        self,
        *,
        text_packet: dict[str, Any] | None,
        image_packet: dict[str, Any] | None,
        rules_result: dict[str, Any] | None,
    ) -> str:
        text_payload = dict(text_packet or {})
        image_payload = dict(image_packet or {})
        parts: list[str] = []
        normalized_text = str(text_payload.get("normalized_text", "") or "").strip()
        if normalized_text:
            parts.append("txt:" + normalized_text[:96])
        for item in (image_payload.get("global_structure_samples", []) or [])[:6]:
            if not isinstance(item, dict):
                continue
            attrs = dict(item.get("attributes", {}) or {})
            feature_code = str(attrs.get("global_feature_code", "") or item.get("sa_label", "") or "")
            feature_group = str(attrs.get("global_feature_group", "") or "")
            if feature_code:
                parts.append((feature_group + ":" if feature_group else "") + feature_code)
        if not parts:
            return "empty"
        return "|".join(parts[:16])

    def _compute_cognitive_feeling_habituation(
        self,
        *,
        tick_index: int,
        text_packet: dict[str, Any] | None,
        image_packet: dict[str, Any] | None,
        rules_result: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = bool(getattr(self.config, "cognitive_feeling_habituation_enabled", False))
        previous = self._normalize_habituation_payload(self._cognitive_feeling_habituation)
        metrics = dict((rules_result.get("metrics_snapshot", {}) or {}))
        raw_emotion = self._normalize_emotion_channels((rules_result.get("emotion_channels", {}) or {}))
        signature = self._build_cognitive_feeling_signature(
            text_packet=text_packet,
            image_packet=image_packet,
            rules_result=rules_result,
        )
        same_signature = bool(signature) and signature == str(previous.get("last_signature", "") or "")
        decay = float(getattr(self.config, "cognitive_feeling_habituation_decay", 0.72))
        same_gain = float(getattr(self.config, "cognitive_feeling_habituation_same_signature_gain", 0.26))
        cross_gain = float(getattr(self.config, "cognitive_feeling_habituation_cross_signature_gain", 0.08))
        signature_change_retention = float(getattr(self.config, "cognitive_feeling_habituation_signature_change_retention", 0.35))
        surprise_gain = float(getattr(self.config, "cognitive_feeling_habituation_surprise_gain", 0.9))
        dissonance_gain = float(getattr(self.config, "cognitive_feeling_habituation_dissonance_gain", 0.58))
        grasp_release = float(getattr(self.config, "cognitive_feeling_habituation_release_on_grasp_gain", 0.65))
        min_multiplier = float(getattr(self.config, "cognitive_feeling_habituation_min_multiplier", 0.22))
        previous_state = dict(previous.get("state", {}) or {})
        surprise_prev = _clamp(float(previous_state.get("surprise", 0.0) or 0.0), 0.0, 1.0)
        dissonance_prev = _clamp(float(previous_state.get("dissonance", 0.0) or 0.0), 0.0, 1.0)
        raw_surprise = float(raw_emotion.get("surprise", 0.0) or 0.0)
        raw_dissonance = float(raw_emotion.get("dissonance", 0.0) or 0.0)
        grasp = float(raw_emotion.get("grasp", 0.0) or 0.0)
        alignment = float(metrics.get("state.prediction_alignment_score", 0.0) or 0.0)
        release = _clamp(grasp * grasp_release + max(0.0, alignment - 0.3) * 0.25, 0.0, 1.0)
        surprise_drive = raw_surprise * (same_gain if same_signature else cross_gain)
        dissonance_drive = raw_dissonance * (same_gain * 0.85 if same_signature else cross_gain * 0.75)
        carry = 1.0 if same_signature else _clamp(signature_change_retention, 0.0, 1.0)
        effective_surprise_state = surprise_prev * decay * carry
        effective_dissonance_state = dissonance_prev * decay * carry
        surprise_state = _clamp(effective_surprise_state + surprise_drive - release, 0.0, 1.0)
        dissonance_state = _clamp(effective_dissonance_state + dissonance_drive - release * 0.85, 0.0, 1.0)
        if not enabled:
            effective_surprise_state = 0.0
            effective_dissonance_state = 0.0
            surprise_state = 0.0
            dissonance_state = 0.0
        surprise_multiplier = 1.0 - surprise_gain * effective_surprise_state
        dissonance_multiplier = 1.0 - dissonance_gain * effective_dissonance_state
        gains = {
            "surprise": _round4(_clamp(surprise_multiplier, min_multiplier, 1.0)),
            "dissonance": _round4(_clamp(dissonance_multiplier, min_multiplier, 1.0)),
        }
        next_payload = {
            "state": {
                "surprise": _round4(surprise_state),
                "dissonance": _round4(dissonance_state),
            },
            "last_signature": signature,
            "last_tick": int(tick_index),
            "last_metrics": {
                "same_signature": bool(same_signature),
                "release": _round4(release),
                "raw_surprise": _round4(raw_surprise),
                "raw_dissonance": _round4(raw_dissonance),
                "raw_grasp": _round4(grasp),
                "alignment": _round4(alignment),
                "surprise_gain": gains["surprise"],
                "dissonance_gain": gains["dissonance"],
            },
        }
        return {
            "enabled": enabled,
            "signature": signature,
            "same_signature": bool(same_signature),
            "previous_state": {"surprise": _round4(surprise_prev), "dissonance": _round4(dissonance_prev)},
            "state": dict(next_payload["state"]),
            "release": _round4(release),
            "gains": gains,
            "raw_emotion": {"surprise": _round4(raw_surprise), "dissonance": _round4(raw_dissonance), "grasp": _round4(grasp)},
            "next_payload": next_payload,
        }

    def _build_time_feeling(
        self,
        *,
        tick_index: int,
        bn_list: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        trace: dict[str, Any] = {"enabled": bool(getattr(self.config, "time_feeling_enabled", False)), "source_count": 0}
        if not trace["enabled"]:
            return None, trace
        candidates: list[dict[str, Any]] = []
        for row in (bn_list or []):
            if not isinstance(row, dict):
                continue
            memory_tick = int(row.get("tick_index", -1) or -1)
            if memory_tick < 0:
                continue
            score = max(0.0, float(row.get("score", 0.0) or 0.0))
            if score <= 0.0:
                continue
            delta_t = max(0.0, float(int(tick_index) - memory_tick))
            reality_weight = 1.0
            memory = self.memory_store.get_memory(str(row.get("memory_id", "") or ""))
            if isinstance(memory, dict):
                reality_weight = max(0.05, float(memory.get("reality_weight", 1.0) or 1.0))
            commitment = max(0.2, min(1.0, score))
            weight = score * commitment * min(1.0, reality_weight)
            candidates.append(
                {
                    "memory_id": str(row.get("memory_id", "") or ""),
                    "delta_t": delta_t,
                    "weight": weight,
                    "score": score,
                }
            )
        trace["source_count"] = len(candidates)
        if not candidates:
            return None, trace
        radius = max(0.5, float(getattr(self.config, "time_feeling_default_radius_ticks", 4.0)))
        clusters: list[dict[str, Any]] = []
        for seed in candidates:
            seed_delta = float(seed["delta_t"])
            cluster_mass = 0.0
            support_count = 0
            weighted_center = 0.0
            for other in candidates:
                other_delta = float(other["delta_t"])
                closeness = max(0.0, 1.0 - abs(other_delta - seed_delta) / radius)
                if closeness <= 0.0:
                    continue
                mass = float(other["weight"]) * closeness
                cluster_mass += mass
                weighted_center += other_delta * mass
                support_count += 1
            if cluster_mass <= 0.0:
                continue
            center = weighted_center / max(1e-6, cluster_mass)
            clusters.append(
                {
                    "seed_delta": seed_delta,
                    "center": center,
                    "mass": cluster_mass,
                    "support_count": support_count,
                }
            )
        if not clusters:
            return None, trace
        clusters.sort(key=lambda row: (-float(row["mass"]), float(row["center"])))
        best = clusters[0]
        second_mass = float(clusters[1]["mass"]) if len(clusters) > 1 else 0.0
        total_mass = sum(float(row["mass"]) for row in clusters)
        dominance = float(best["mass"]) / max(1e-6, float(best["mass"]) + second_mass)
        confidence = _clamp(
            0.55 * dominance + 0.45 * min(1.0, int(best["support_count"]) / max(1, len(candidates))),
            0.0,
            1.0,
        )
        signal_strength = float(best["mass"]) / max(1e-6, total_mass)
        trace.update(
            {
                "cluster_count": len(clusters),
                "best_center": _round4(float(best["center"])),
                "best_mass": _round4(float(best["mass"])),
                "dominance": _round4(dominance),
                "confidence": _round4(confidence),
                "signal_strength": _round4(signal_strength),
            }
        )
        if confidence < float(getattr(self.config, "time_feeling_min_confidence", 0.24)):
            return None, trace
        item = self._make_channel_feeling_item(
            channel_key="time",
            signal_key=f"{int(round(float(best['center'])))}",
            tick_index=tick_index,
            sa_label="timefelt::elapsed",
            display_text="时间间隔感",
            source_strength=signal_strength,
            confidence=confidence,
            threshold=float(getattr(self.config, "time_feeling_threshold", 0.22)),
            gain=float(getattr(self.config, "time_feeling_gain", 0.95)),
            fatigue_decay=float(getattr(self.config, "time_feeling_fatigue_decay", 0.82)),
            fatigue_step=float(getattr(self.config, "time_feeling_fatigue_step", 0.18)),
            fatigue_gain=float(getattr(self.config, "time_feeling_fatigue_gain", 0.55)),
            fatigue_max=float(getattr(self.config, "time_feeling_fatigue_max", 1.0)),
            channel="time",
            sa_kind="temporal_feeling_unit",
            attributes={
                "delta_t_norm": _round4(float(best["center"]) / max(1.0, float(radius) * 4.0)),
                "delta_sigma_norm": _round4(float(radius) / max(1.0, float(radius) * 4.0)),
                "confidence": _round4(confidence),
                "cluster_mass": _round4(float(best["mass"])),
                "dominance": _round4(dominance),
                "source_count": int(best["support_count"]),
            },
        )
        return item, trace

    def _build_motion_feeling(
        self,
        *,
        tick_index: int,
        dynamic_motion_samples: list[dict[str, Any]],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        trace: dict[str, Any] = {"enabled": bool(getattr(self.config, "motion_feeling_enabled", False)), "source_count": 0}
        if not trace["enabled"]:
            return None, trace
        rows: list[dict[str, Any]] = []
        for item in (dynamic_motion_samples or []):
            if not isinstance(item, dict):
                continue
            attrs = dict(item.get("attributes", {}) or {})
            speed = max(0.0, float(attrs.get("motion_speed", 0.0) or 0.0))
            coherence = max(0.0, float(attrs.get("motion_coherence", 0.0) or 0.0))
            boundary = max(0.0, float(attrs.get("boundary_motion_contrast", 0.0) or 0.0))
            surprise = max(0.0, float(attrs.get("motion_surprise", 0.0) or 0.0))
            persistence = max(0.0, float(attrs.get("temporal_persistence", 0.0) or 0.0))
            dynamic_objectness = max(0.0, float(attrs.get("dynamic_objectness", 0.0) or 0.0))
            signal_strength = (
                speed * 0.34
                + coherence * 0.22
                + boundary * 0.22
                + surprise * 0.12
                + persistence * 0.10
                + dynamic_objectness * 0.18
            )
            if signal_strength <= 0.0:
                continue
            rows.append(
                {
                    "speed": speed,
                    "coherence": coherence,
                    "boundary": boundary,
                    "surprise": surprise,
                    "persistence": persistence,
                    "dynamic_objectness": dynamic_objectness,
                    "signal_strength": signal_strength,
                    "track_id": str(attrs.get("track_id", item.get("sa_label", "")) or ""),
                }
            )
        trace["source_count"] = len(rows)
        if not rows:
            return None, trace
        rows.sort(key=lambda row: (-float(row["signal_strength"]), -float(row["speed"])))
        top = rows[0]
        second = rows[1] if len(rows) > 1 else {"signal_strength": 0.0}
        dominance = float(top["signal_strength"]) / max(1e-6, float(top["signal_strength"]) + float(second["signal_strength"]))
        confidence = _clamp(
            0.45 * dominance + 0.30 * float(top["coherence"]) + 0.25 * float(top["boundary"]),
            0.0,
            1.0,
        )
        trace.update(
            {
                "best_track_id": str(top["track_id"]),
                "best_speed": _round4(float(top["speed"])),
                "best_signal_strength": _round4(float(top["signal_strength"])),
                "dominance": _round4(dominance),
                "confidence": _round4(confidence),
            }
        )
        if confidence < float(getattr(self.config, "motion_feeling_min_confidence", 0.2)):
            return None, trace
        item = self._make_channel_feeling_item(
            channel_key="motion",
            signal_key=str(top["track_id"]),
            tick_index=tick_index,
            sa_label="motionfelt::trend",
            display_text="运动趋势感",
            source_strength=float(top["signal_strength"]),
            confidence=confidence,
            threshold=float(getattr(self.config, "motion_feeling_threshold", 0.18)),
            gain=float(getattr(self.config, "motion_feeling_gain", 0.92)),
            fatigue_decay=float(getattr(self.config, "motion_feeling_fatigue_decay", 0.8)),
            fatigue_step=float(getattr(self.config, "motion_feeling_fatigue_step", 0.16)),
            fatigue_gain=float(getattr(self.config, "motion_feeling_fatigue_gain", 0.52)),
            fatigue_max=float(getattr(self.config, "motion_feeling_fatigue_max", 1.0)),
            channel="motion",
            sa_kind="motion_feeling_unit",
            attributes={
                "motion_center_speed": _round4(float(top["speed"])),
                "motion_sigma": 0.18,
                "confidence": _round4(confidence),
                "dynamic_objectness": _round4(float(top["dynamic_objectness"])),
                "motion_coherence": _round4(float(top["coherence"])),
                "boundary_motion_contrast": _round4(float(top["boundary"])),
                "track_id": str(top["track_id"]),
            },
        )
        return item, trace

    def _rhythm_event_candidates(
        self,
        *,
        external_items: list[dict[str, Any]],
        bn_list: list[dict[str, Any]],
        c_star: dict[str, Any],
        dynamic_motion_samples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        seen_keys: set[str] = set()

        def add_candidate(
            *,
            family_key: str,
            strength: float,
            commitment: float,
            reality: float,
            salience: float,
            label: str,
            source: str,
        ) -> None:
            clean_family = str(family_key or "")
            clean_label = str(label or "")
            if not clean_family or not clean_label:
                return
            dedupe_key = f"{source}|{clean_family}|{clean_label}"
            if dedupe_key in seen_keys:
                return
            seen_keys.add(dedupe_key)
            candidates.append(
                {
                    "family_key": clean_family,
                    "strength": _round4(max(0.0, float(strength))),
                    "commitment": _round4(_clamp(float(commitment), 0.0, 1.0)),
                    "reality": _round4(max(0.0, float(reality))),
                    "salience": _round4(max(0.0, float(salience))),
                    "label": clean_label,
                    "source": str(source or ""),
                }
            )

        for item in (external_items or [])[:32]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            attrs = dict(item.get("attributes", {}) or {})
            energy = max(0.0, float(item.get("energy", 0.0) or 0.0))
            salience = energy
            if label.startswith(("text::", "phrase::")):
                add_candidate(
                    family_key=label,
                    strength=energy,
                    commitment=min(1.0, 0.55 + energy * 0.25),
                    reality=1.0,
                    salience=salience,
                    label=label,
                    source="external",
                )
            else:
                aliases = self.memory_store._item_retrieval_label_rows(item)
                for alias_label, alias_scale in aliases[:4]:
                    clean_alias = str(alias_label or "")
                    if not clean_alias.startswith(("vision_core::", "vision_form::", "vision_dyn_core::", "vision_dyn_form::", "vision_global::")):
                        continue
                    add_candidate(
                        family_key=clean_alias,
                        strength=energy * max(0.2, float(alias_scale or 0.0)),
                        commitment=min(1.0, 0.38 + energy * 0.32),
                        reality=0.95,
                        salience=salience,
                        label=label,
                        source="external",
                    )

        for item in (dynamic_motion_samples or [])[:8]:
            if not isinstance(item, dict):
                continue
            attrs = dict(item.get("attributes", {}) or {})
            core_label, form_label = self.memory_store._vision_memory_alias_labels(attrs)
            dynamic_objectness = max(0.0, float(attrs.get("dynamic_objectness", 0.0) or 0.0))
            motion_speed = max(0.0, float(attrs.get("motion_speed", 0.0) or 0.0))
            coherence = max(0.0, float(attrs.get("motion_coherence", 0.0) or 0.0))
            strength = max(0.0, float(item.get("energy", 0.0) or 0.0)) + dynamic_objectness * 0.4 + motion_speed * 0.25
            salience = dynamic_objectness * 0.55 + coherence * 0.25 + motion_speed * 0.20
            for clean_alias in [label for label in [core_label.replace("vision_core::", "vision_dyn_core::", 1) if core_label else "", form_label.replace("vision_form::", "vision_dyn_form::", 1) if form_label else "", core_label, form_label] if label]:
                add_candidate(
                    family_key=clean_alias,
                    strength=strength,
                    commitment=min(1.0, 0.4 + dynamic_objectness * 0.4),
                    reality=0.95,
                    salience=salience,
                    label=str(item.get("sa_label", "") or clean_alias),
                    source="dynamic",
                )
        candidates.sort(
            key=lambda row: (
                -(float(row.get("strength", 0.0) or 0.0) * (0.55 + 0.45 * float(row.get("commitment", 0.0) or 0.0))),
                str(row.get("family_key", "") or ""),
            )
        )
        return candidates[:32]

    def _build_rhythm_feelings(
        self,
        *,
        tick_index: int,
        external_items: list[dict[str, Any]],
        bn_list: list[dict[str, Any]],
        c_star: dict[str, Any],
        dynamic_motion_samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trace: dict[str, Any] = {
            "enabled": bool(getattr(self.config, "rhythm_feeling_enabled", False)),
            "candidate_count": 0,
            "family_count": 0,
        }
        if not trace["enabled"]:
            return [], trace
        tracker = self._normalize_rhythm_tracker(self._rhythm_tracker)
        families = dict(tracker.get("families", {}) or {})
        last_tick = int(tracker.get("last_tick", -1)) if tracker.get("last_tick", -1) is not None else -1
        fatigue_decay = max(0.0, min(1.0, float(getattr(self.config, "rhythm_fatigue_decay", 0.82))))
        window_ticks = max(4, int(getattr(self.config, "rhythm_window_ticks", 12)))
        min_hits = max(2, int(getattr(self.config, "rhythm_min_hits", 3)))
        min_period = max(0.5, float(getattr(self.config, "rhythm_min_period_ticks", 2.0)))
        max_period = max(min_period, float(getattr(self.config, "rhythm_max_period_ticks", 12.0)))
        period_sigma_ratio = max(0.01, float(getattr(self.config, "rhythm_period_sigma_ratio", 0.18)))
        phase_sigma_ratio = max(0.01, float(getattr(self.config, "rhythm_phase_sigma_ratio", 0.22)))
        recovery_center = max(0.5, float(getattr(self.config, "rhythm_recovery_center_ticks", 4.0)))
        recovery_sigma = max(0.05, float(getattr(self.config, "rhythm_recovery_sigma_ticks", 2.0)))
        min_confidence = _clamp(float(getattr(self.config, "rhythm_min_confidence", 0.24)), 0.0, 1.0)
        fatigue_step = max(0.0, float(getattr(self.config, "rhythm_fatigue_step", 0.12)))
        fatigue_gain = max(0.0, float(getattr(self.config, "rhythm_fatigue_gain", 0.55)))
        fatigue_max = max(0.0, float(getattr(self.config, "rhythm_fatigue_max", 1.0)))

        step_gap = max(0, int(tick_index) - last_tick) if last_tick >= 0 else 0
        if step_gap > 0:
            for family in families.values():
                if not isinstance(family, dict):
                    continue
                family["fatigue"] = _round4(max(0.0, float(family.get("fatigue", 0.0) or 0.0) * (fatigue_decay ** step_gap)))
                family["phase_fatigue"] = _round4(max(0.0, float(family.get("phase_fatigue", 0.0) or 0.0) * (fatigue_decay ** step_gap)))
                family["hits"] = [
                    hit for hit in list(family.get("hits", []) or [])
                    if isinstance(hit, dict) and int(hit.get("tick", -999999)) >= int(tick_index) - window_ticks
                ]

        candidates = self._rhythm_event_candidates(
            external_items=external_items,
            bn_list=bn_list,
            c_star=c_star,
            dynamic_motion_samples=dynamic_motion_samples,
        )
        trace["candidate_count"] = len(candidates)
        candidate_map: dict[str, dict[str, Any]] = {}
        for row in candidates:
            family_key = str(row.get("family_key", "") or "")
            if not family_key:
                continue
            current = candidate_map.get(family_key)
            if current is None or float(row.get("strength", 0.0) or 0.0) > float(current.get("strength", 0.0) or 0.0):
                candidate_map[family_key] = dict(row)

        for family_key, row in candidate_map.items():
            family = dict(families.get(family_key, {}) or {})
            hits = [dict(hit) for hit in (family.get("hits", []) or []) if isinstance(hit, dict)]
            strength = max(0.0, float(row.get("strength", 0.0) or 0.0))
            commitment = _clamp(float(row.get("commitment", 0.0) or 0.0), 0.0, 1.0)
            reality = max(0.0, float(row.get("reality", 0.0) or 0.0))
            salience = max(0.0, float(row.get("salience", 0.0) or 0.0))
            if hits and int(hits[-1].get("tick", -1)) == int(tick_index):
                hits[-1] = {
                    "tick": int(tick_index),
                    "strength": _round4(max(float(hits[-1].get("strength", 0.0) or 0.0), strength)),
                    "commitment": _round4(max(float(hits[-1].get("commitment", 0.0) or 0.0), commitment)),
                    "reality": _round4(max(float(hits[-1].get("reality", 0.0) or 0.0), reality)),
                    "salience": _round4(max(float(hits[-1].get("salience", 0.0) or 0.0), salience)),
                }
            else:
                hits.append(
                    {
                        "tick": int(tick_index),
                        "strength": _round4(strength),
                        "commitment": _round4(commitment),
                        "reality": _round4(reality),
                        "salience": _round4(salience),
                    }
                )
            families[family_key] = {
                "hits": hits[-window_ticks:],
                "fatigue": _round4(max(0.0, float(family.get("fatigue", 0.0) or 0.0))),
                "phase_fatigue": _round4(max(0.0, float(family.get("phase_fatigue", 0.0) or 0.0))),
                "last_tick": int(tick_index),
            }

        items: list[dict[str, Any]] = []
        pulse_rows: list[dict[str, Any]] = []
        phase_rows: list[dict[str, Any]] = []

        for family_key, family in list(families.items()):
            hits = [dict(hit) for hit in (family.get("hits", []) or []) if isinstance(hit, dict)]
            if len(hits) < min_hits:
                continue
            intervals: list[dict[str, float]] = []
            for idx in range(1, len(hits)):
                prev_hit = hits[idx - 1]
                cur_hit = hits[idx]
                delta = float(int(cur_hit.get("tick", 0)) - int(prev_hit.get("tick", 0)))
                if delta < min_period or delta > max_period:
                    continue
                weight = (
                    max(0.05, float(cur_hit.get("strength", 0.0) or 0.0))
                    * (0.55 + 0.45 * _clamp(float(cur_hit.get("commitment", 0.0) or 0.0), 0.0, 1.0))
                    * (0.55 + 0.45 * max(0.0, float(cur_hit.get("reality", 0.0) or 0.0)))
                )
                intervals.append({"delta": delta, "weight": weight})
            if len(intervals) < max(2, min_hits - 1):
                continue

            clusters: list[dict[str, float]] = []
            for seed in intervals:
                seed_delta = float(seed["delta"])
                sigma = max(0.15, seed_delta * period_sigma_ratio)
                cluster_mass = 0.0
                support_count = 0
                weighted_center = 0.0
                for row in intervals:
                    delta = float(row["delta"])
                    weight = float(row["weight"])
                    kernel = math.exp(-((delta - seed_delta) ** 2) / max(1e-6, 2.0 * sigma * sigma))
                    contribution = weight * kernel
                    cluster_mass += contribution
                    weighted_center += delta * contribution
                    if kernel >= 0.62:
                        support_count += 1
                center = weighted_center / max(1e-6, cluster_mass)
                clusters.append({"center": center, "mass": cluster_mass, "support_count": float(support_count)})
            if not clusters:
                continue
            merged_clusters: list[dict[str, float]] = []
            for cluster in sorted(clusters, key=lambda row: float(row["center"])):
                if merged_clusters and abs(float(cluster["center"]) - float(merged_clusters[-1]["center"])) <= max(0.15, float(cluster["center"]) * period_sigma_ratio * 0.85):
                    prev = merged_clusters[-1]
                    combined_mass = float(prev["mass"]) + float(cluster["mass"])
                    merged_clusters[-1] = {
                        "center": (
                            float(prev["center"]) * float(prev["mass"]) + float(cluster["center"]) * float(cluster["mass"])
                        ) / max(1e-6, combined_mass),
                        "mass": combined_mass,
                        "support_count": max(float(prev["support_count"]), float(cluster["support_count"])),
                    }
                else:
                    merged_clusters.append(dict(cluster))
            clusters = merged_clusters
            clusters.sort(key=lambda row: (-float(row["mass"]), float(row["center"])))
            best = clusters[0]
            second_mass = float(clusters[1]["mass"]) if len(clusters) > 1 else 0.0
            total_mass = sum(float(row["mass"]) for row in clusters)
            tau = max(min_period, min(max_period, float(best["center"])))
            sigma_tau = max(0.15, tau * period_sigma_ratio)
            phase_sigma = max(0.75, tau * phase_sigma_ratio)
            dominance = float(best["mass"]) / max(1e-6, float(best["mass"]) + second_mass)
            recurrence = min(1.0, len(intervals) / max(1.0, float(max(1, min_hits - 1))))
            support_ratio = min(1.0, float(best["support_count"]) / max(1.0, float(len(intervals))))
            regularity = min(1.0, float(best["mass"]) / max(1e-6, total_mass))
            salience_support = _clamp(sum(float(hit.get("salience", 0.0) or 0.0) for hit in hits[-min(len(hits), 4):]) / max(1.0, min(len(hits), 4)), 0.0, 1.5)
            salience_norm = _clamp(salience_support, 0.0, 1.0)
            confidence = _clamp(0.38 * dominance + 0.28 * support_ratio + 0.20 * regularity + 0.14 * salience_norm, 0.0, 1.0)
            if confidence < min_confidence:
                continue
            rhythmicity = _clamp(regularity * recurrence * max(0.1, salience_norm), 0.0, 1.0)
            recovery_match = math.exp(-((tau - recovery_center) ** 2) / max(1e-6, 2.0 * recovery_sigma * recovery_sigma))
            fatigue = _clamp(float(family.get("fatigue", 0.0) or 0.0), 0.0, fatigue_max)
            freshness = max(0.0, 1.0 - fatigue * fatigue_gain)
            last_hit_tick = int(hits[-1].get("tick", tick_index))
            time_since_last = max(0.0, float(int(tick_index) - last_hit_tick))
            phase_error = abs(time_since_last - tau)
            phase_expectation = regularity * math.exp(-(phase_error * phase_error) / max(1e-6, 2.0 * phase_sigma * phase_sigma))
            phase_fatigue = _clamp(float(family.get("phase_fatigue", 0.0) or 0.0), 0.0, fatigue_max)
            anticipation_success = math.exp(-(phase_error * phase_error) / max(1e-6, 2.0 * sigma_tau * sigma_tau))
            groove = _clamp(rhythmicity * recovery_match * freshness, 0.0, 1.0)

            pulse_item = self._make_channel_feeling_item(
                channel_key="rhythm",
                signal_key=f"pulse::{family_key}",
                tick_index=tick_index,
                sa_label="rhythmfelt::pulse",
                display_text="节拍感",
                source_strength=groove,
                confidence=confidence,
                threshold=float(getattr(self.config, "rhythm_pulse_threshold", 0.18)),
                gain=float(getattr(self.config, "rhythm_pulse_gain", 0.92)),
                fatigue_decay=fatigue_decay,
                fatigue_step=fatigue_step,
                fatigue_gain=fatigue_gain,
                fatigue_max=fatigue_max,
                channel="rhythm",
                sa_kind="rhythm_feeling_unit",
                attributes={
                    "period_ticks": _round4(tau),
                    "period_sigma_ticks": _round4(sigma_tau),
                    "phase_sigma_ticks": _round4(phase_sigma),
                    "regularity": _round4(regularity),
                    "recurrence": _round4(recurrence),
                    "salience_support": _round4(salience_norm),
                    "recovery_match": _round4(recovery_match),
                    "freshness": _round4(freshness),
                    "confidence": _round4(confidence),
                    "family_key": family_key,
                    "repeat_count": len(hits),
                },
            )
            if pulse_item:
                items.append(pulse_item)
                pulse_rows.append(
                    {
                        "family_key": family_key,
                        "period_ticks": _round4(tau),
                        "regularity": _round4(regularity),
                        "confidence": _round4(confidence),
                        "groove": _round4(groove),
                    }
                )
                family["fatigue"] = _round4(min(fatigue_max, fatigue + fatigue_step * float(pulse_item.get("energy", 0.0) or 0.0)))

            phase_item = self._make_channel_feeling_item(
                channel_key="rhythm_phase",
                signal_key=f"phase::{family_key}",
                tick_index=tick_index,
                sa_label="rhythmfelt::phase_expectation",
                display_text="节奏期待感",
                source_strength=phase_expectation * freshness,
                confidence=confidence,
                threshold=float(getattr(self.config, "rhythm_phase_threshold", 0.14)),
                gain=float(getattr(self.config, "rhythm_phase_gain", 0.86)),
                fatigue_decay=fatigue_decay,
                fatigue_step=fatigue_step * 0.8,
                fatigue_gain=fatigue_gain,
                fatigue_max=fatigue_max,
                channel="rhythm",
                sa_kind="rhythm_phase_feeling_unit",
                attributes={
                    "period_ticks": _round4(tau),
                    "period_sigma_ticks": _round4(sigma_tau),
                    "phase_sigma_ticks": _round4(phase_sigma),
                    "time_to_next": _round4(max(0.0, tau - time_since_last)),
                    "phase_error": _round4(phase_error),
                    "regularity": _round4(regularity),
                    "confidence": _round4(confidence),
                    "family_key": family_key,
                    "repeat_count": len(hits),
                },
            )
            if phase_item:
                items.append(phase_item)
                phase_rows.append(
                    {
                        "family_key": family_key,
                        "phase_error": _round4(phase_error),
                        "phase_expectation": _round4(phase_expectation),
                        "confidence": _round4(confidence),
                    }
                )
                family["phase_fatigue"] = _round4(min(fatigue_max, phase_fatigue + fatigue_step * float(phase_item.get("energy", 0.0) or 0.0)))

            families[family_key] = {
                "hits": hits[-window_ticks:],
                "fatigue": _round4(max(0.0, float(family.get("fatigue", 0.0) or 0.0))),
                "phase_fatigue": _round4(max(0.0, float(family.get("phase_fatigue", 0.0) or 0.0))),
                "last_tick": int(tick_index),
            }

        trimmed_families: dict[str, Any] = {}
        for family_key, family in sorted(
            families.items(),
            key=lambda row: (
                -max((float(hit.get("strength", 0.0) or 0.0) for hit in (row[1].get("hits", []) or [])), default=0.0),
                row[0],
            ),
        )[:64]:
            hits = [dict(hit) for hit in (family.get("hits", []) or []) if isinstance(hit, dict)]
            if hits:
                trimmed_families[family_key] = {
                    "hits": hits[-window_ticks:],
                    "fatigue": _round4(max(0.0, float(family.get("fatigue", 0.0) or 0.0))),
                    "phase_fatigue": _round4(max(0.0, float(family.get("phase_fatigue", 0.0) or 0.0))),
                    "last_tick": int(family.get("last_tick", tick_index)),
                }
        self._rhythm_tracker = {"families": trimmed_families, "last_tick": int(tick_index)}
        trace["family_count"] = len(trimmed_families)
        trace["pulse_preview"] = pulse_rows[:8]
        trace["phase_preview"] = phase_rows[:8]
        if pulse_rows:
            trace["best_pulse"] = dict(pulse_rows[0])
        if phase_rows:
            trace["best_phase"] = dict(phase_rows[0])
        return items[:4], trace

    def _build_feedback_signal_feeling(
        self,
        *,
        tick_index: int,
        pending_feedback: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trace: dict[str, Any] = {"enabled": bool(getattr(self.config, "feedback_signal_feeling_enabled", False))}
        if not trace["enabled"]:
            return [], trace
        reward = max(0.0, float((pending_feedback or {}).get("reward", 0.0) or 0.0))
        punishment = max(0.0, float((pending_feedback or {}).get("punishment", 0.0) or 0.0))
        net = reward - punishment
        confidence = 1.0 if (reward > 0.0 or punishment > 0.0) else 0.0
        trace.update({"reward": _round4(reward), "punishment": _round4(punishment), "net": _round4(net)})
        if confidence <= 0.0:
            return [], trace
        item = self._make_channel_feeling_item(
            channel_key="feedback",
            signal_key="reward" if net >= 0.0 else "punishment",
            tick_index=tick_index,
            sa_label="attr::reward_signal" if net >= 0.0 else "attr::punishment_signal",
            display_text="奖励信号" if net >= 0.0 else "惩罚信号",
            source_strength=abs(net) if abs(net) > 0.0 else max(reward, punishment),
            confidence=confidence,
            threshold=float(getattr(self.config, "feedback_signal_feeling_threshold", 0.08)),
            gain=float(getattr(self.config, "feedback_signal_feeling_gain", 1.0)),
            fatigue_decay=float(getattr(self.config, "feedback_signal_fatigue_decay", 0.84)),
            fatigue_step=float(getattr(self.config, "feedback_signal_fatigue_step", 0.12)),
            fatigue_gain=float(getattr(self.config, "feedback_signal_fatigue_gain", 0.48)),
            fatigue_max=float(getattr(self.config, "feedback_signal_fatigue_max", 1.0)),
            channel="attr",
            sa_kind="feedback_signal_unit",
            attributes={
                "feedback_valence": _round4(net),
                "feedback_sigma": 0.18,
                "confidence": _round4(confidence),
            },
        )
        return ([item] if item else []), trace

    def _build_hearing_feelings(
        self,
        *,
        tick_index: int,
        audio_packet: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        trace: dict[str, Any] = {
            "enabled": bool(getattr(self.config, "hearing_feeling_enabled", True)),
            "source_count": 0,
            "summary": {},
        }
        if not trace["enabled"]:
            return [], trace
        packet = dict(audio_packet or {})
        windows = [item for item in (packet.get("windows", []) or []) if isinstance(item, dict)]
        feature_summary = dict(packet.get("feature_summary", {}) or {})
        if not feature_summary and windows:
            weighted: dict[str, float] = {}
            total_weight = 0.0
            for item in windows:
                attrs = dict(item.get("attributes", {}) or {})
                weight = max(0.01, float(item.get("energy", 0.0) or 0.0))
                total_weight += weight
                for key in (
                    "tonal_clarity",
                    "noisiness",
                    "pitch_stability",
                    "percussive_ratio",
                    "harmonic_ratio",
                    "voiced_probability",
                    "spectral_contrast",
                ):
                    weighted[key] = float(weighted.get(key, 0.0) or 0.0) + weight * float(attrs.get(key, 0.0) or 0.0)
            if total_weight > 0.0:
                feature_summary = {key: _round4(value / total_weight) for key, value in weighted.items()}
        trace["source_count"] = len(windows)
        trace["summary"] = dict(feature_summary)
        if not feature_summary:
            return [], trace

        profile = str(feature_summary.get("dominant_profile", "") or "")
        confidence = _clamp(
            0.42 * float(feature_summary.get("spectral_contrast", 0.0) or 0.0)
            + 0.24 * max(
                float(feature_summary.get("tonal_clarity", 0.0) or 0.0),
                float(feature_summary.get("noisiness", 0.0) or 0.0),
                float(feature_summary.get("percussive_ratio", 0.0) or 0.0),
            )
            + 0.18 * min(1.0, float(feature_summary.get("window_count", len(windows)) or len(windows)) / 8.0)
            + 0.16 * max(0.0, 1.0 - float(feature_summary.get("novelty", 0.0) or 0.0) * 0.25),
            0.0,
            1.0,
        )
        items: list[dict[str, Any]] = []
        min_confidence = float(getattr(self.config, "hearing_feeling_min_confidence", 0.18))

        def add_item(
            *,
            signal_key: str,
            sa_label: str,
            display_text: str,
            source_strength: float,
            attributes: dict[str, Any],
        ) -> None:
            if confidence < min_confidence:
                return
            item = self._make_channel_feeling_item(
                channel_key="hearing",
                signal_key=signal_key,
                tick_index=tick_index,
                sa_label=sa_label,
                display_text=display_text,
                source_strength=source_strength,
                confidence=confidence,
                threshold=float(getattr(self.config, "hearing_feeling_threshold", 0.18)),
                gain=float(getattr(self.config, "hearing_feeling_gain", 0.9)),
                fatigue_decay=float(getattr(self.config, "hearing_feeling_fatigue_decay", 0.82)),
                fatigue_step=float(getattr(self.config, "hearing_feeling_fatigue_step", 0.14)),
                fatigue_gain=float(getattr(self.config, "hearing_feeling_fatigue_gain", 0.52)),
                fatigue_max=float(getattr(self.config, "hearing_feeling_fatigue_max", 1.0)),
                channel="hearing",
                sa_kind="hearing_feeling_unit",
                attributes=attributes,
            )
            if item:
                items.append(item)

        tonal = max(0.0, float(feature_summary.get("tonal_clarity", 0.0) or 0.0))
        noisy = max(0.0, float(feature_summary.get("noisiness", 0.0) or 0.0))
        pitch_stability = max(0.0, float(feature_summary.get("pitch_stability", 0.0) or 0.0))
        percussive = max(0.0, float(feature_summary.get("percussive_ratio", 0.0) or 0.0))
        harmonic = max(0.0, float(feature_summary.get("harmonic_ratio", 0.0) or 0.0))
        voiced_probability = max(0.0, float(feature_summary.get("voiced_probability", 0.0) or 0.0))
        contrast = max(0.0, float(feature_summary.get("spectral_contrast", 0.0) or 0.0))
        dominant_hz = float(feature_summary.get("dominant_hz", 0.0) or 0.0)

        add_item(
            signal_key=f"tonal::{profile or 'tonal'}",
            sa_label="hearingfelt::timbre_clarity",
            display_text="音色清晰感",
            source_strength=tonal,
            attributes={
                "confidence": _round4(confidence),
                "hearing_profile": profile or "tonal",
                "hearing_timbre_center": _round4(tonal),
                "hearing_timbre_sigma": 0.18,
                "hearing_timbre_recall_gain": _round4(float(getattr(self.config, "hearing_timbre_recall_gain", 0.16))),
                "harmonic_ratio": _round4(harmonic),
                "spectral_contrast": _round4(contrast),
            },
        )
        add_item(
            signal_key=f"noise::{profile or 'noise'}",
            sa_label="hearingfelt::noisiness",
            display_text="噪声感",
            source_strength=noisy,
            attributes={
                "confidence": _round4(confidence),
                "hearing_profile": profile or "noisy",
                "hearing_noise_center": _round4(noisy),
                "hearing_noise_sigma": 0.18,
                "hearing_noise_recall_gain": _round4(float(getattr(self.config, "hearing_noise_recall_gain", 0.16))),
                "spectral_flatness": _round4(float(feature_summary.get("spectral_flatness", 0.0) or 0.0)),
                "spectral_bandwidth_ratio": _round4(float(feature_summary.get("spectral_bandwidth_ratio", 0.0) or 0.0)),
            },
        )
        add_item(
            signal_key=f"pitch::{int(round(dominant_hz))}",
            sa_label="hearingfelt::pitch_stability",
            display_text="音高稳定感",
            source_strength=pitch_stability,
            attributes={
                "confidence": _round4(confidence),
                "hearing_profile": profile or "pitch",
                "hearing_pitch_stability_center": _round4(pitch_stability),
                "hearing_pitch_stability_sigma": 0.16,
                "hearing_pitch_recall_gain": _round4(float(getattr(self.config, "hearing_pitch_recall_gain", 0.18))),
                "dominant_hz": _round4(dominant_hz),
                "voiced_probability": _round4(voiced_probability),
            },
        )
        add_item(
            signal_key=f"perc::{profile or 'perc'}",
            sa_label="hearingfelt::percussive_burst",
            display_text="击打突发感",
            source_strength=percussive,
            attributes={
                "confidence": _round4(confidence),
                "hearing_profile": profile or "percussive",
                "hearing_percussive_center": _round4(percussive),
                "hearing_percussive_sigma": 0.16,
                "hearing_percussive_recall_gain": _round4(float(getattr(self.config, "hearing_percussive_recall_gain", 0.16))),
                "onset_strength": _round4(float(feature_summary.get("onset_strength", 0.0) or 0.0)),
                "novelty": _round4(float(feature_summary.get("novelty", 0.0) or 0.0)),
            },
        )
        trace["confidence"] = _round4(confidence)
        trace["profile"] = profile
        trace["generated_count"] = len(items)
        return items[:4], trace

    def _feedback_source_metric(self, breakdown: dict[str, Any] | None, source_name: str, metric: str) -> float:
        payload = dict(breakdown or {})
        sources = dict(payload.get("sources", {}) or {})
        source = dict(sources.get(str(source_name or ""), {}) or {})
        return float(source.get(str(metric or ""), 0.0) or 0.0)

    def _feedback_breakdown_has_signal(self, breakdown: dict[str, Any] | None) -> bool:
        payload = dict(breakdown or {})
        if not payload:
            return False
        if self._feedback_has_signal({"reward": payload.get("reward", 0.0), "punishment": payload.get("punishment", 0.0)}):
            return True
        if any(str(item or "") for item in (payload.get("notes", []) or [])):
            return True
        sources = dict(payload.get("sources", {}) or {})
        for source_name in ("external", "teacher", "intrinsic"):
            source_payload = dict(sources.get(source_name, {}) or {})
            if self._feedback_has_signal(source_payload):
                return True
            if any(str(item or "") for item in (source_payload.get("notes", []) or [])):
                return True
        intrinsic_detail = dict(payload.get("intrinsic_detail", {}) or {})
        return bool(intrinsic_detail.get("enabled", False))

    def _normalize_cognitive_balance(self, balance_metrics: dict[str, Any] | None) -> dict[str, float]:
        payload = dict(balance_metrics or {})
        return {
            "alignment_score": _round4(_clamp(float(payload.get("alignment_score", 0.0) or 0.0), 0.0, 1.0)),
            "grasp_score": _round4(_clamp(float(payload.get("grasp_score", 0.0) or 0.0), 0.0, 1.0)),
            "overprediction_ratio": _round4(max(0.0, float(payload.get("overprediction_ratio", 0.0) or 0.0))),
            "underprediction_ratio": _round4(max(0.0, float(payload.get("underprediction_ratio", 0.0) or 0.0))),
            "committed_alignment_score": _round4(_clamp(float(payload.get("committed_alignment_score", 0.0) or 0.0), 0.0, 1.0)),
            "committed_grasp_score": _round4(_clamp(float(payload.get("committed_grasp_score", 0.0) or 0.0), 0.0, 1.0)),
            "committed_overprediction_ratio": _round4(max(0.0, float(payload.get("committed_overprediction_ratio", 0.0) or 0.0))),
        }

    def _build_intrinsic_feedback(
        self,
        *,
        emotion_channels: dict[str, Any] | None,
        balance_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        current = self._normalize_emotion_channels(emotion_channels)
        previous = dict(self._last_emotion_channels or {})
        self._last_emotion_channels = dict(current)
        current_balance = self._normalize_cognitive_balance(balance_metrics)
        previous_balance = dict(self._last_cognitive_balance or {})
        self._last_cognitive_balance = dict(current_balance)
        delta = {
            key: _round4(float(current.get(key, 0.0) or 0.0) - float(previous.get(key, 0.0) or 0.0))
            for key in current.keys()
        }
        delta_balance = {
            key: _round4(float(current_balance.get(key, 0.0) or 0.0) - float(previous_balance.get(key, 0.0) or 0.0))
            for key in current_balance.keys()
        }
        enabled = bool(getattr(self.config, "intrinsic_feedback_enabled", False))
        components = {
            "correctness_delta_reward": 0.0,
            "grasp_delta_reward": 0.0,
            "committed_grasp_delta_reward": 0.0,
            "surprise_recovery_reward": 0.0,
            "dissonance_recovery_reward": 0.0,
            "dissonance_delta_punishment": 0.0,
            "surprise_delta_punishment": 0.0,
            "expectation_tonic_reward": 0.0,
            "pressure_tonic_punishment": 0.0,
        }
        payload: dict[str, Any] = {
            "reward": 0.0,
            "punishment": 0.0,
            "notes": [],
            "enabled": enabled,
            "current_emotion": current,
            "previous_emotion": previous,
            "delta_emotion": delta,
            "current_balance": current_balance,
            "previous_balance": previous_balance,
            "delta_balance": delta_balance,
            "components": components,
        }
        if not enabled:
            return payload

        if delta["correctness"] > 0.0:
            components["correctness_delta_reward"] = _round4(delta["correctness"] * float(self.config.intrinsic_correctness_reward_gain))
        if delta_balance["grasp_score"] > 0.0:
            components["grasp_delta_reward"] = _round4(delta_balance["grasp_score"] * float(self.config.intrinsic_correctness_reward_gain))
        if delta_balance["committed_grasp_score"] > 0.0:
            components["committed_grasp_delta_reward"] = _round4(delta_balance["committed_grasp_score"] * float(self.config.intrinsic_correctness_reward_gain))
        if delta["dissonance"] > 0.0:
            components["dissonance_delta_punishment"] = _round4(delta["dissonance"] * float(self.config.intrinsic_dissonance_punishment_gain))
        elif delta["dissonance"] < 0.0:
            components["dissonance_recovery_reward"] = _round4(abs(delta["dissonance"]) * float(self.config.intrinsic_correctness_reward_gain))
        if delta["surprise"] > 0.0:
            components["surprise_delta_punishment"] = _round4(delta["surprise"] * float(self.config.intrinsic_surprise_punishment_gain))
        elif delta["surprise"] < 0.0:
            components["surprise_recovery_reward"] = _round4(abs(delta["surprise"]) * float(self.config.intrinsic_correctness_reward_gain))
        if current["expectation"] > 0.0:
            components["expectation_tonic_reward"] = _round4(current["expectation"] * float(self.config.intrinsic_expectation_tonic_reward_gain))
        if current["pressure"] > 0.0:
            components["pressure_tonic_punishment"] = _round4(current["pressure"] * float(self.config.intrinsic_pressure_tonic_punishment_gain))

        reward = min(
            float(self.config.intrinsic_feedback_max_reward_per_tick),
            max(
                0.0,
                components["correctness_delta_reward"]
                + components["grasp_delta_reward"]
                + components["committed_grasp_delta_reward"]
                + components["surprise_recovery_reward"]
                + components["dissonance_recovery_reward"]
                + components["expectation_tonic_reward"],
            ),
        )
        punishment = min(
            float(self.config.intrinsic_feedback_max_punishment_per_tick),
            max(0.0, components["dissonance_delta_punishment"] + components["surprise_delta_punishment"] + components["pressure_tonic_punishment"]),
        )
        notes: list[str] = []
        if components["correctness_delta_reward"] > 0.0:
            notes.append("intrinsic_correctness_delta_reward")
        if components["grasp_delta_reward"] > 0.0:
            notes.append("intrinsic_grasp_delta_reward")
        if components["committed_grasp_delta_reward"] > 0.0:
            notes.append("intrinsic_committed_grasp_delta_reward")
        if components["surprise_recovery_reward"] > 0.0:
            notes.append("intrinsic_surprise_recovery_reward")
        if components["dissonance_recovery_reward"] > 0.0:
            notes.append("intrinsic_dissonance_recovery_reward")
        if components["dissonance_delta_punishment"] > 0.0:
            notes.append("intrinsic_dissonance_delta_punishment")
        if components["surprise_delta_punishment"] > 0.0:
            notes.append("intrinsic_surprise_delta_punishment")
        if components["expectation_tonic_reward"] > 0.0:
            notes.append("intrinsic_expectation_tonic_reward")
        if components["pressure_tonic_punishment"] > 0.0:
            notes.append("intrinsic_pressure_tonic_punishment")
        payload["reward"] = _round4(reward)
        payload["punishment"] = _round4(punishment)
        payload["notes"] = notes
        return payload

    def build_intrinsic_feedback(
        self,
        *,
        emotion_channels: dict[str, Any] | None,
        balance_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._build_intrinsic_feedback(emotion_channels=emotion_channels, balance_metrics=balance_metrics)

    def reset_transient_state(self, *, keep_runtime_controls: bool = True) -> None:
        current_controls = self.runtime_controls_snapshot()
        self.text_sensor = TextSensorV2(
            budget_limit=self.config.text_sensor_budget,
            fatigue_window=self.config.text_sensor_fatigue_window,
            fatigue_threshold=self.config.text_sensor_fatigue_threshold,
            max_suppression=self.config.text_sensor_max_suppression,
        )
        self.vision_sensor = VisionSensorV1(
            patch_budget=self.config.vision_patch_budget,
            focus_patch_budget=self.config.vision_focus_patch_budget,
            raw_state_budget=self.config.vision_raw_state_budget,
            reconstruction_patch_budget=self.config.vision_reconstruction_patch_budget,
            edge_candidate_gain=self.config.vision_edge_candidate_gain,
            edge_priority_gain=self.config.vision_edge_priority_gain,
            attention_boost_enabled=self.config.vision_attention_boost_enabled,
            attention_boost_decay=self.config.vision_attention_boost_decay,
            attention_boost_max_extra_raw_budget=self.config.vision_attention_boost_max_extra_raw_budget,
            attention_boost_max_extra_focus_budget=self.config.vision_attention_boost_max_extra_focus_budget,
            attention_boost_min_radius_scale=self.config.vision_attention_boost_min_radius_scale,
            attention_boost_edge_gain=self.config.vision_attention_boost_edge_gain,
            attention_boost_gaze_sigma_scale=self.config.vision_attention_boost_gaze_sigma_scale,
            dynamic_track_window=self.config.vision_dynamic_track_window,
            dynamic_candidate_limit_background=self.config.vision_dynamic_candidate_limit_background,
            dynamic_candidate_limit_focus=self.config.vision_dynamic_candidate_limit_focus,
            dynamic_track_limit=self.config.vision_dynamic_track_limit,
            dynamic_summary_limit=self.config.vision_dynamic_summary_limit,
            dynamic_match_threshold=self.config.vision_dynamic_match_threshold,
            dynamic_track_forget_ticks=self.config.vision_dynamic_track_forget_ticks,
        )
        self.hearing_sensor = HearingSensorV1(
            window_budget=self.config.hearing_window_budget,
            window_ms=self.config.hearing_window_ms,
            focus_band_count=self.config.hearing_focus_band_count,
            focus_bandwidth_octaves=self.config.hearing_focus_bandwidth_octaves,
            attention_boost_enabled=self.config.hearing_attention_boost_enabled,
            attention_boost_decay=self.config.hearing_attention_boost_decay,
            attention_boost_max_extra_window_budget=self.config.hearing_attention_boost_max_extra_window_budget,
            attention_boost_max_extra_focus_budget=self.config.hearing_attention_boost_max_extra_focus_budget,
            attention_boost_min_bandwidth_scale=self.config.hearing_attention_boost_min_bandwidth_scale,
            attention_boost_focus_gain=self.config.hearing_attention_boost_focus_gain,
            static_dedup_delta_threshold=self.config.hearing_static_dedup_delta_threshold,
            static_dedup_band_similarity_threshold=self.config.hearing_static_dedup_band_similarity_threshold,
            static_dedup_max_suppression=self.config.hearing_static_dedup_max_suppression,
            auditory_fatigue_decay=self.config.hearing_auditory_fatigue_decay,
            auditory_fatigue_step=self.config.hearing_auditory_fatigue_step,
            auditory_fatigue_max=self.config.hearing_auditory_fatigue_max,
        )
        self.state_pool = StatePoolV2(
            decay=self.config.state_pool_decay,
            prune_threshold=self.config.state_pool_prune_threshold,
            recent_queue_limit=self.config.state_pool_recent_queue_limit,
            verbatim_window_chars=self.config.text_sensor_verbatim_window_chars,
            head_limit=self.config.r_state_head_limit,
            items_per_head=self.config.r_state_items_per_head,
            anchor_cache_limit=self.config.state_pool_anchor_cache_limit,
            residual_limit=self.config.state_pool_residual_limit,
            handle_limit=self.config.state_pool_handle_limit,
            residual_unit_limit=self.config.state_pool_residual_unit_limit,
            attention_object_fatigue_decay=self.config.state_pool_attention_object_fatigue_decay,
            attention_object_fatigue_step=self.config.state_pool_attention_object_fatigue_step,
            attention_object_fatigue_gain=self.config.state_pool_attention_object_fatigue_gain,
            attention_object_fatigue_max=self.config.state_pool_attention_object_fatigue_max,
            attention_object_min_multiplier=self.config.state_pool_attention_object_min_multiplier,
        )
        self.short_term = ShortTermMemoryV2(
            max_items=self.config.short_term_memory_limit,
            successor_tail_limit=self.config.short_term_successor_tail_limit,
        )
        self.action_planner = ActionPlannerV2()
        self._pending_feedback_metrics = {"reward": 0.0, "punishment": 0.0}
        self._pending_feedback_breakdown = self._blank_feedback_breakdown()
        self._queued_intrinsic_feedback = {}
        self._channel_feeling_fatigue = {}
        self._rhythm_tracker = {"families": {}, "last_tick": -1}
        self._last_emotion_channels = self._normalize_emotion_channels({})
        self._last_cognitive_balance = self._normalize_cognitive_balance({})
        self._cognitive_feeling_habituation = self._blank_cognitive_feeling_habituation()
        self._last_control_feedback_context = {
            "runtime_controls": current_controls if keep_runtime_controls else self._default_runtime_controls(),
            "matched_profiles": [],
            "applied_tuner_adjustments": [],
            "learned_tuner_offsets": [],
        }
        self._attention_modulation_state = self._blank_attention_modulation_state()
        if keep_runtime_controls:
            self._runtime_controls = {key: float(value) for key, value in current_controls.items()}
        else:
            self._runtime_controls = self._default_runtime_controls()

    def merge_feedback_channels(
        self,
        *,
        external_feedback: dict[str, Any] | None = None,
        teacher_feedback: dict[str, Any] | None = None,
        intrinsic_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sources = {
            "external": dict(external_feedback or {}),
            "teacher": dict(teacher_feedback or {}),
            "intrinsic": dict(intrinsic_feedback or {}),
        }
        merged = self._blank_feedback_breakdown()
        merged["intrinsic_detail"] = {
            "enabled": bool(dict(intrinsic_feedback or {}).get("enabled", False)),
            "current_emotion": dict(dict(intrinsic_feedback or {}).get("current_emotion", {}) or {}),
            "previous_emotion": dict(dict(intrinsic_feedback or {}).get("previous_emotion", {}) or {}),
            "delta_emotion": dict(dict(intrinsic_feedback or {}).get("delta_emotion", {}) or {}),
            "components": dict(dict(intrinsic_feedback or {}).get("components", {}) or {}),
        }
        notes: list[str] = []
        for source_name, payload in sources.items():
            reward = _round4(float(payload.get("reward", 0.0) or 0.0))
            punishment = _round4(float(payload.get("punishment", 0.0) or 0.0))
            source_notes = [str(item or "") for item in (payload.get("notes", []) or []) if str(item or "")]
            merged["sources"][source_name] = {
                "reward": reward,
                "punishment": punishment,
                "notes": source_notes,
            }
            merged["reward"] = _round4(float(merged["reward"]) + reward)
            merged["punishment"] = _round4(float(merged["punishment"]) + punishment)
            notes.extend(source_notes)
        merged["notes"] = notes
        return merged

    def _default_runtime_controls(self) -> dict[str, float]:
        return {
            "attention.focus_gain": 1.35,
            "sampling.increment_budget": float(max(24, min(256, self.config.r_state_items_per_head * 6))),
            "prediction.successor_bias_gain": 1.18,
            "state.anchor_bias_gain": 0.9,
            "rules.dissonance_gain": 1.0,
            "state.current_input_gain": 1.0,
            "state.history_suppression_gain": 1.0,
            "state.prediction_suppression_gain": 1.0,
            "state.surprise_focus_gain": 1.0,
        }

    def _blank_attention_modulation_state(self) -> dict[str, Any]:
        return {
            "tick_index": -1,
            "attention_lock": 0.0,
            "firmness_norm": 0.0,
            "surprise_pull": 0.0,
            "dissonance_pull": 0.0,
            "current_pull": 0.0,
            "selected_action_names": [],
            "has_attention_action": False,
            "modulated_controls": {},
        }

    def _attention_action_names(self) -> set[str]:
        return {
            "move_gaze",
            "continue_focus",
            "inspect_residual",
            "move_audio_focus",
            "continue_audio_focus",
            "inspect_audio_residual",
        }

    def _derive_attention_modulated_controls(
        self,
        *,
        selected_actions: list[dict[str, Any]] | None,
        tick_index: int,
    ) -> dict[str, Any]:
        rows = [dict(row) for row in (selected_actions or []) if isinstance(row, dict)]
        action_names = [
            str(row.get("action_name", "") or row.get("effect", "") or "").strip()
            for row in rows
            if str(row.get("action_name", "") or row.get("effect", "") or "").strip()
        ]
        attention_rows = [row for row in rows if str(row.get("action_name", "") or row.get("effect", "") or "").strip() in self._attention_action_names()]
        ctx = dict(self.state_pool._attention_context() or {})
        surprise_pull = _clamp(float(ctx.get("surprise_pull", 0.0) or 0.0), 0.0, 1.0)
        dissonance_pull = _clamp(float(ctx.get("dissonance_pull", 0.0) or 0.0), 0.0, 1.0)
        current_pull = _clamp(float(ctx.get("current_pull", 0.0) or 0.0), 0.0, 1.0)
        firmness_norm = 0.0
        for row in attention_rows:
            value = row.get("firmness_norm")
            if value is None:
                firmness = float(row.get("firmness", 0.0) or 0.0)
                value = _clamp(firmness / 0.25, 0.0, 1.5) if firmness > 0.0 else 0.0
            firmness_norm = max(firmness_norm, _clamp(float(value or 0.0), 0.0, 1.5))
        attention_lock = _clamp(
            firmness_norm * (0.45 + 0.40 * surprise_pull + 0.15 * dissonance_pull),
            0.0,
            1.5,
        )
        base_focus_gain = self._runtime_control("attention.focus_gain")
        base_current_gain = self._runtime_control("state.current_input_gain")
        base_history_gain = self._runtime_control("state.history_suppression_gain")
        base_prediction_gain = self._runtime_control("state.prediction_suppression_gain")
        base_surprise_gain = self._runtime_control("state.surprise_focus_gain")
        if attention_rows:
            focus_gain = base_focus_gain * (1.0 + 0.42 * attention_lock)
            current_input_gain = base_current_gain * (1.0 + 0.48 * attention_lock)
            history_suppression_gain = base_history_gain * (1.0 + 0.95 * attention_lock)
            prediction_suppression_gain = base_prediction_gain * (1.0 + 1.20 * attention_lock)
            surprise_focus_gain = base_surprise_gain * (1.0 + 0.82 * attention_lock)
        else:
            focus_gain = base_focus_gain
            current_input_gain = base_current_gain
            history_suppression_gain = base_history_gain
            prediction_suppression_gain = base_prediction_gain
            surprise_focus_gain = base_surprise_gain
        return {
            "tick_index": int(tick_index),
            "attention_lock": _round4(attention_lock),
            "firmness_norm": _round4(firmness_norm),
            "surprise_pull": _round4(surprise_pull),
            "dissonance_pull": _round4(dissonance_pull),
            "current_pull": _round4(current_pull),
            "selected_action_names": action_names,
            "has_attention_action": bool(attention_rows),
            "modulated_controls": {
                "attention.focus_gain": _round4(max(0.25, min(4.0, focus_gain))),
                "state.current_input_gain": _round4(max(0.0, min(4.0, current_input_gain))),
                "state.history_suppression_gain": _round4(max(0.0, min(4.0, history_suppression_gain))),
                "state.prediction_suppression_gain": _round4(max(0.0, min(4.0, prediction_suppression_gain))),
                "state.surprise_focus_gain": _round4(max(0.0, min(4.0, surprise_focus_gain))),
            },
        }

    def _attention_modulation_decay(self) -> float:
        decay = max(
            float(getattr(self.config, "vision_attention_boost_decay", 0.78) or 0.78),
            float(getattr(self.config, "hearing_attention_boost_decay", 0.78) or 0.78),
        )
        return _clamp(decay, 0.5, 0.98)

    def _attention_modulation_snapshot_for_tick(self, tick_index: int) -> dict[str, Any]:
        state = dict(self._attention_modulation_state or {})
        tick_mark = int(state.get("tick_index", -1) or -1)
        if tick_mark < 0:
            snapshot = self._blank_attention_modulation_state()
            snapshot["tick_index"] = int(tick_index)
            snapshot["decay_factor"] = 0.0
            return snapshot
        age = max(0, int(tick_index) - tick_mark - 1)
        decay_factor = self._attention_modulation_decay() ** age
        snapshot = dict(state)
        snapshot["decay_factor"] = _round4(decay_factor)
        return snapshot

    def _effective_attention_controls(self, tick_index: int | None = None) -> dict[str, float]:
        controls = self.runtime_controls_snapshot()
        state = self._attention_modulation_snapshot_for_tick(
            int(self.state_pool._tick_index if tick_index is None else tick_index)
        )
        modulated = dict(state.get("modulated_controls", {}) or {})
        decay_factor = float(state.get("decay_factor", 0.0) or 0.0)
        for key, value in modulated.items():
            base = float(controls.get(key, self._runtime_control(key)))
            target = float(value)
            controls[key] = _round4(base + (target - base) * decay_factor)
        return controls

    def _merge_attention_control_sets(self, *control_sets: dict[str, Any]) -> dict[str, float]:
        merged = self.runtime_controls_snapshot()
        for control_set in control_sets:
            for key, value in dict(control_set or {}).items():
                if key not in merged:
                    continue
                try:
                    merged[key] = _round4(max(float(merged[key]), float(value)))
                except Exception:
                    continue
        return merged

    def get_last_logic_ms(self) -> float:
        return float(self._last_logic_ms)

    def set_last_logic_ms(self, value: float) -> None:
        self._last_logic_ms = max(0.0, float(value))

    def runtime_controls_snapshot(self) -> dict[str, float]:
        return {key: _round4(value) for key, value in self._runtime_controls.items()}

    def vision_gaze_snapshot(self) -> dict[str, float]:
        return {
            "x": _round4(self.vision_sensor.gaze_center[0]),
            "y": _round4(self.vision_sensor.gaze_center[1]),
        }

    def _auto_visual_reorient_from_surprise(
        self,
        *,
        rules_result: dict[str, Any],
        image_packet: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not bool(getattr(self.config, "vision_auto_surprise_reorient_enabled", True)):
            return None
        if not image_packet:
            return None
        emotion = dict((rules_result or {}).get("emotion_channels", {}) or {})
        surprise = float(emotion.get("surprise", 0.0) or 0.0)
        dissonance = float(emotion.get("dissonance", 0.0) or 0.0)
        trigger = max(surprise, surprise + dissonance * 0.35)
        if trigger < 0.18:
            return None
        target = self._pick_gaze_target_from_image_packet(dict(image_packet or {}), mode="residual")
        if not target:
            return None
        x = float(target.get("x", self.vision_sensor.gaze_center[0]) or self.vision_sensor.gaze_center[0])
        y = float(target.get("y", self.vision_sensor.gaze_center[1]) or self.vision_sensor.gaze_center[1])
        self.vision_sensor.move_gaze(x, y)
        boost = self.vision_sensor.apply_attention_boost(
            source_action="auto_surprise_reorient",
            firmness_norm=_clamp(0.42 + trigger * 0.9, 0.0, 1.0),
            target_gaze=(x, y),
        )
        return {
            "trigger_surprise": _round4(surprise),
            "trigger_dissonance": _round4(dissonance),
            "target": {"x": _round4(x), "y": _round4(y)},
            "attention_boost": boost,
        }

    def _runtime_control(self, key: str) -> float:
        return float(self._runtime_controls.get(key, self._default_runtime_controls().get(key, 1.0)))

    def _apply_tuner_adjustments(self, tuner_result: dict[str, Any]) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        if not isinstance(tuner_result, dict):
            return applied
        defaults = self._default_runtime_controls()
        self._runtime_controls = dict(defaults)
        for row in tuner_result.get("adjustments", []) or []:
            if not isinstance(row, dict):
                continue
            target = str(row.get("target", "") or "")
            if target not in defaults:
                continue
            value = float(row.get("value", defaults[target]) or defaults[target])
            if target == "attention.focus_gain":
                value = max(0.25, min(4.0, value))
            elif target == "sampling.increment_budget":
                value = max(8.0, min(1024.0, value))
            elif target == "prediction.successor_bias_gain":
                value = max(0.0, min(4.0, value))
            elif target == "state.anchor_bias_gain":
                value = max(0.0, min(4.0, value))
            elif target == "rules.dissonance_gain":
                value = max(0.0, min(4.0, value))
            elif target in {
                "state.current_input_gain",
                "state.history_suppression_gain",
                "state.prediction_suppression_gain",
                "state.surprise_focus_gain",
            }:
                value = max(0.0, min(4.0, value))
            self._runtime_controls[target] = value
            applied.append(
                {
                    "target": target,
                    "value": _round4(value),
                    "profile_id": str(row.get("profile_id", "") or ""),
                }
            )
        return applied

    def _apply_tuner_learning_offsets(self, *, matched_profiles: list[dict[str, Any]]) -> dict[str, Any]:
        learning_result = self.tuner_learning.apply_to_controls(
            controls=self._runtime_controls,
            matched_profiles=matched_profiles,
        )
        learned_controls = dict(learning_result.get("controls", {}) or {})
        if learned_controls:
            self._runtime_controls = {key: float(value) for key, value in learned_controls.items()}
        return learning_result

    def apply_selected_actions(self, selected_actions: list[dict[str, Any]], *, runtime_tick: dict[str, Any] | None = None) -> dict[str, Any]:
        before = self.vision_gaze_snapshot()
        before_audio = dict(self.hearing_sensor.audio_focus_snapshot() or {})
        applied: list[dict[str, Any]] = []
        image_packet = dict((runtime_tick or {}).get("image_packet", {}) or {})
        audio_packet = dict((runtime_tick or {}).get("audio_packet", {}) or {})
        has_visual_action = False
        has_audio_action = False
        has_any_action = False
        for raw in selected_actions:
            if not isinstance(raw, dict):
                continue
            action_name = str(raw.get("action_name", "") or raw.get("effect", "") or "").strip()
            if not action_name:
                continue
            has_any_action = True
            params = dict(raw.get("params", {}) or {})
            applied_params = dict(raw.get("applied_params", {}) or {})
            raw_firmness_norm = raw.get("firmness_norm")
            if raw_firmness_norm is None:
                firmness = float(raw.get("firmness", 0.0) or 0.0)
                if firmness > 0.0:
                    firmness_norm = _clamp(firmness / 0.25, 0.0, 1.5)
                elif action_name in {"move_gaze", "continue_focus", "inspect_residual", "move_audio_focus", "continue_audio_focus", "inspect_audio_residual"}:
                    firmness_norm = 0.6
                else:
                    firmness_norm = 0.0
            else:
                firmness_norm = _clamp(float(raw_firmness_norm or 0.0), 0.0, 1.5)
            if action_name == "move_gaze":
                has_visual_action = True
                self.vision_sensor.set_attention_mode("visual_focus")
                x = float(params.get("x", applied_params.get("x", before["x"])) or before["x"])
                y = float(params.get("y", applied_params.get("y", before["y"])) or before["y"])
                self.vision_sensor.move_gaze(x, y)
                boost = self.vision_sensor.apply_attention_boost(
                    source_action=action_name,
                    firmness_norm=firmness_norm,
                    target_gaze=(x, y),
                )
                applied.append(
                    {
                        "action_name": action_name,
                        "mode": "explicit",
                        "target": self.vision_gaze_snapshot(),
                        "attention_boost": boost,
                    }
                )
                continue
            if action_name == "continue_focus":
                has_visual_action = True
                self.vision_sensor.set_attention_mode("visual_focus")
                target = self._pick_gaze_target_from_image_packet(image_packet, mode="focus")
                if target:
                    self.vision_sensor.move_gaze(target["x"], target["y"])
                    boost = self.vision_sensor.apply_attention_boost(
                        source_action=action_name,
                        firmness_norm=firmness_norm,
                        target_gaze=(float(target["x"]), float(target["y"])),
                    )
                    applied.append({"action_name": action_name, **target, "attention_boost": boost})
                continue
            if action_name == "inspect_residual":
                has_visual_action = True
                self.vision_sensor.set_attention_mode("visual_focus")
                target = self._pick_gaze_target_from_image_packet(image_packet, mode="residual")
                if target:
                    self.vision_sensor.move_gaze(target["x"], target["y"])
                    boost = self.vision_sensor.apply_attention_boost(
                        source_action=action_name,
                        firmness_norm=firmness_norm,
                        target_gaze=(float(target["x"]), float(target["y"])),
                    )
                    applied.append({"action_name": action_name, **target, "attention_boost": boost})
                continue
            if action_name == "move_audio_focus":
                has_audio_action = True
                self.hearing_sensor.set_attention_mode("auditory_focus")
                center_hz = float(
                    params.get(
                        "center_hz",
                        applied_params.get(
                            "center_hz",
                            before_audio.get("center_hz", 1200.0),
                        ),
                    )
                    or before_audio.get("center_hz", 1200.0)
                )
                bandwidth_octaves = float(
                    params.get(
                        "bandwidth_octaves",
                        applied_params.get(
                            "bandwidth_octaves",
                            before_audio.get("bandwidth_octaves", 1.15),
                        ),
                    )
                    or before_audio.get("bandwidth_octaves", 1.15)
                )
                self.hearing_sensor.move_audio_focus(center_hz, bandwidth_octaves=bandwidth_octaves)
                boost = self.hearing_sensor.apply_attention_boost(
                    source_action=action_name,
                    firmness_norm=firmness_norm,
                    target_center_hz=center_hz,
                    target_bandwidth_octaves=bandwidth_octaves,
                )
                applied.append(
                    {
                        "action_name": action_name,
                        "mode": "explicit",
                        "audio_focus": dict(self.hearing_sensor.audio_focus_snapshot() or {}),
                        "attention_boost": boost,
                    }
                )
                continue
            if action_name == "continue_audio_focus":
                has_audio_action = True
                self.hearing_sensor.set_attention_mode("auditory_focus")
                target = self._pick_audio_focus_target_from_audio_packet(audio_packet, mode="focus")
                if target:
                    center_hz = float(target.get("center_hz", before_audio.get("center_hz", 1200.0)) or before_audio.get("center_hz", 1200.0))
                    bandwidth_octaves = float(target.get("bandwidth_octaves", before_audio.get("bandwidth_octaves", 1.15)) or before_audio.get("bandwidth_octaves", 1.15))
                    self.hearing_sensor.move_audio_focus(center_hz, bandwidth_octaves=bandwidth_octaves)
                    boost = self.hearing_sensor.apply_attention_boost(
                        source_action=action_name,
                        firmness_norm=firmness_norm,
                        target_center_hz=center_hz,
                        target_bandwidth_octaves=bandwidth_octaves,
                    )
                    applied.append({"action_name": action_name, **target, "attention_boost": boost})
                continue
            if action_name == "inspect_audio_residual":
                has_audio_action = True
                self.hearing_sensor.set_attention_mode("auditory_focus")
                target = self._pick_audio_focus_target_from_audio_packet(audio_packet, mode="residual")
                if target:
                    center_hz = float(target.get("center_hz", before_audio.get("center_hz", 1200.0)) or before_audio.get("center_hz", 1200.0))
                    bandwidth_octaves = float(target.get("bandwidth_octaves", before_audio.get("bandwidth_octaves", 1.15)) or before_audio.get("bandwidth_octaves", 1.15))
                    self.hearing_sensor.move_audio_focus(center_hz, bandwidth_octaves=bandwidth_octaves)
                    boost = self.hearing_sensor.apply_attention_boost(
                        source_action=action_name,
                        firmness_norm=firmness_norm,
                        target_center_hz=center_hz,
                        target_bandwidth_octaves=bandwidth_octaves,
                    )
                    applied.append({"action_name": action_name, **target, "attention_boost": boost})
                continue
        if not has_visual_action:
            self.vision_sensor.set_attention_mode("suppressed" if has_any_action else "background")
        if not has_audio_action:
            self.hearing_sensor.set_attention_mode("suppressed" if has_any_action else "background")
        after = self.vision_gaze_snapshot()
        after_audio = dict(self.hearing_sensor.audio_focus_snapshot() or {})
        modulation = self._derive_attention_modulated_controls(
            selected_actions=[dict(item) for item in selected_actions if isinstance(item, dict)],
            tick_index=int(self.state_pool._tick_index),
        )
        self._attention_modulation_state = dict(modulation)
        return {
            "gaze_center_before": before,
            "gaze_center_after": after,
            "moved": before != after,
            "audio_focus_before": before_audio,
            "audio_focus_after": after_audio,
            "audio_moved": before_audio != after_audio,
            "applied_actions": applied,
            "attention_boost": self.vision_sensor.attention_boost_snapshot(),
            "audio_attention_boost": self.hearing_sensor.attention_boost_snapshot(),
            "attention_modulation": dict(modulation),
        }

    def _pick_gaze_target_from_image_packet(self, image_packet: dict[str, Any], *, mode: str) -> dict[str, Any] | None:
        dynamic_motion = [item for item in (image_packet.get("dynamic_motion_samples", []) or []) if isinstance(item, dict)]
        focus_priority = [item for item in (image_packet.get("focus_priority_samples", []) or []) if isinstance(item, dict)]
        memory_write = [item for item in (image_packet.get("memory_write_samples", []) or []) if isinstance(item, dict)]
        raw_patches = [item for item in (image_packet.get("patches", []) or []) if isinstance(item, dict)]
        if mode == "residual":
            patches = dynamic_motion + focus_priority + memory_write + raw_patches
        else:
            patches = focus_priority + dynamic_motion + memory_write + raw_patches
        if not patches:
            return None
        current = self.vision_gaze_snapshot()
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in patches:
            coords = dict(item.get("coords", {}) or {})
            if "cx" not in coords or "cy" not in coords:
                continue
            cx = float(coords.get("cx", current["x"]) or current["x"])
            cy = float(coords.get("cy", current["y"]) or current["y"])
            energy = float(item.get("energy", 0.0) or 0.0)
            attrs = dict(item.get("attributes", {}) or {})
            brightness = float(attrs.get("brightness", 0.0) or 0.0)
            distance = abs(cx - current["x"]) + abs(cy - current["y"])
            dynamic_objectness = float(attrs.get("dynamic_objectness", 0.0) or 0.0)
            motion_speed = float(attrs.get("motion_speed", 0.0) or 0.0)
            motion_surprise = float(attrs.get("motion_surprise", 0.0) or 0.0)
            persistence = float(attrs.get("temporal_persistence", 0.0) or 0.0)
            coherence = float(attrs.get("motion_coherence", 0.0) or 0.0)
            is_dynamic = str(item.get("sa_label", "") or "").startswith("vision_dyn::")
            if mode == "residual":
                score = (
                    brightness * (0.35 + distance * 0.35)
                    + energy * 0.22
                    + dynamic_objectness * 1.15
                    + motion_speed * 0.65
                    + motion_surprise * 0.55
                    + coherence * 0.20
                    + persistence * 0.10
                )
                if is_dynamic:
                    score += 0.22
            else:
                score = energy * 0.62 + brightness * 0.18 + dynamic_objectness * 0.42 + motion_speed * 0.18 + persistence * 0.10
                if is_dynamic:
                    score += 0.10
            scored.append(
                (
                    score,
                    {
                        "mode": mode,
                        "x": _round4(cx),
                        "y": _round4(cy),
                        "score": _round4(score),
                        "sa_label": str(item.get("sa_label", "") or ""),
                    },
                )
            )
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]["sa_label"]))
        return scored[0][1]

    def _pick_audio_focus_target_from_audio_packet(self, audio_packet: dict[str, Any], *, mode: str) -> dict[str, Any] | None:
        patches: list[dict[str, Any]] = []
        for key in ("focus_priority_samples", "memory_write_samples", "windows", "global_structure_samples"):
            patches.extend([item for item in (audio_packet.get(key, []) or []) if isinstance(item, dict)])
        if not patches:
            return None
        current = dict(self.hearing_sensor.audio_focus_snapshot() or {})
        current_center = float(current.get("center_hz", 1200.0) or 1200.0)
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in patches:
            coords = dict(item.get("coords", {}) or {})
            attrs = dict(item.get("attributes", {}) or {})
            center_hz = float(coords.get("freq_center_hz", attrs.get("dominant_hz", 0.0)) or 0.0)
            if center_hz <= 0.0:
                continue
            energy = float(item.get("energy", 0.0) or 0.0)
            focus_priority = float(attrs.get("focus_priority", 0.0) or 0.0)
            novelty = float(attrs.get("novelty", 0.0) or 0.0)
            onset_strength = float(attrs.get("onset_strength", 0.0) or 0.0)
            focus_bonus = float(attrs.get("focus_bonus", 0.0) or 0.0)
            octave_distance = abs(math.log(max(1e-6, center_hz), 2.0) - math.log(max(1e-6, current_center), 2.0))
            if mode == "residual":
                score = novelty * 0.34 + onset_strength * 0.30 + energy * 0.18 + focus_priority * 0.10 + octave_distance * 0.08
            else:
                score = focus_priority * 0.34 + energy * 0.24 + focus_bonus * 0.18 + novelty * 0.14 + onset_strength * 0.10
            scored.append(
                (
                    score,
                    {
                        "mode": mode,
                        "center_hz": _round4(center_hz),
                        "bandwidth_octaves": _round4(max(0.18, 0.42 + (1.0 - min(1.0, focus_priority)) * 0.9)),
                        "score": _round4(score),
                        "sa_label": str(item.get("sa_label", "") or ""),
                    },
                )
            )
        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], item[1]["sa_label"]))
        return scored[0][1]

    def process_text_tick(self, *, text: str, tick_index: int, source_type: str = "external_text") -> dict[str, Any]:
        text_packet = self.text_sensor.ingest(text, tick_index=tick_index, source_type=source_type)
        return self.process_multimodal_tick(
            tick_index=tick_index,
            text_packet=text_packet,
            source_type=source_type,
        )

    def _focus_items_for_memory(self, focus_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        seen_labels: set[str] = set()
        for item in focus_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen_labels:
                continue
            if not label.startswith(("text::", "phrase::", "vision_mem::", "audio::", "hearingfelt::")):
                continue
            seen_labels.add(label)
            cleaned.append(_clone_item_shallow(item))
        return cleaned

    def _focus_units_for_memory(self, focus_items: list[dict[str, Any]]) -> list[str]:
        units: list[str] = []
        seen_units: set[str] = set()
        for item in focus_items:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label.startswith(("text::", "phrase::", "attr::")):
                continue
            display = str(item.get("display_text", "") or "")
            if not display or display in seen_units:
                continue
            seen_units.add(display)
            units.append(display)
        return units

    def _latent_snapshot_items_for_memory(self, query_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for item in query_items:
            if not isinstance(item, dict):
                continue
            if str(item.get("source_type", "") or "") == "prediction":
                continue
            energy = max(
                0.0,
                float(item.get("query_weight", 0.0) or 0.0),
                float(item.get("energy", 0.0) or 0.0),
            )
            if energy <= 0.0:
                continue
            cloned = _clone_item_shallow(item)
            cloned["energy"] = _round4(energy)
            selected.append(cloned)
        selected.sort(
            key=lambda row: (
                -float(row.get("energy", 0.0) or 0.0),
                -float(row.get("attention_score", 0.0) or 0.0),
                str(row.get("sa_label", "") or ""),
            )
        )
        return selected[:48]

    def process_multimodal_tick(
        self,
        *,
        tick_index: int,
        text_packet: dict[str, Any] | None = None,
        image_packet: dict[str, Any] | None = None,
        audio_packet: dict[str, Any] | None = None,
        source_type: str = "multimodal_input",
    ) -> dict[str, Any]:
        tick_started = time.perf_counter()
        stage_started = tick_started
        stage_timing: dict[str, float] = {}

        text_packet = text_packet or self.text_sensor.ingest("", tick_index=tick_index, source_type=source_type)
        base_units = [str(item or "") for item in ((text_packet.get("full_stream") or {}).get("units", []) or [])]
        pending_feedback = dict(self._pending_feedback_metrics or {})
        pending_feedback_breakdown = copy.deepcopy(self._pending_feedback_breakdown or self._blank_feedback_breakdown())
        queued_intrinsic_feedback = dict(self._queued_intrinsic_feedback or {})
        if self._feedback_breakdown_has_signal(pending_feedback_breakdown):
            pending_feedback = {
                "reward": _round4(float(pending_feedback_breakdown.get("reward", 0.0) or 0.0)),
                "punishment": _round4(float(pending_feedback_breakdown.get("punishment", 0.0) or 0.0)),
            }
        elif self._feedback_has_signal(queued_intrinsic_feedback):
            pending_feedback_breakdown = self.merge_feedback_channels(
                external_feedback=pending_feedback if self._feedback_has_signal(pending_feedback) else {},
                teacher_feedback={},
                intrinsic_feedback=queued_intrinsic_feedback,
            )
            pending_feedback = {
                "reward": _round4(float(pending_feedback_breakdown.get("reward", 0.0) or 0.0)),
                "punishment": _round4(float(pending_feedback_breakdown.get("punishment", 0.0) or 0.0)),
            }
        self._pending_feedback_metrics = {"reward": 0.0, "punishment": 0.0}
        self._pending_feedback_breakdown = self._blank_feedback_breakdown()
        self._queued_intrinsic_feedback = {}
        self.sa_registry.observe_sequence(base_units)
        sampling_budget = max(1, int(round(self._runtime_control("sampling.increment_budget"))))
        competition = self.sa_registry.compete(
            base_units,
            source_type=source_type,
            max_items=sampling_budget,
        )
        stage_timing["01_text_competition_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()

        multimodal_items = list(competition["selected_items"])
        summary_items = list(competition["selected_items"])
        multimodal_channels: list[dict[str, Any]] = []
        raw_samples: list[dict[str, Any]] = []
        memory_write_samples: list[dict[str, Any]] = []
        focus_priority_samples: list[dict[str, Any]] = []
        global_structure_samples: list[dict[str, Any]] = []
        dynamic_motion_samples: list[dict[str, Any]] = []
        windows: list[dict[str, Any]] = []
        audio_focus_priority_samples: list[dict[str, Any]] = []
        audio_memory_write_samples: list[dict[str, Any]] = []
        audio_global_structure_samples: list[dict[str, Any]] = []
        if image_packet:
            raw_samples = [item for item in (image_packet.get("raw_samples", []) or []) if isinstance(item, dict)]
            memory_write_samples = [item for item in (image_packet.get("memory_write_samples", image_packet.get("patches", [])) or []) if isinstance(item, dict)]
            focus_priority_samples = [item for item in (image_packet.get("focus_priority_samples", []) or []) if isinstance(item, dict)]
            global_structure_samples = [item for item in (image_packet.get("global_structure_samples", []) or []) if isinstance(item, dict)]
            dynamic_motion_samples = [item for item in (image_packet.get("dynamic_motion_samples", []) or []) if isinstance(item, dict)]
            multimodal_items.extend(raw_samples)
            multimodal_items.extend(memory_write_samples)
            multimodal_items.extend(global_structure_samples)
            multimodal_items.extend(dynamic_motion_samples[:4])
            summary_items.extend(memory_write_samples)
            summary_items.extend(global_structure_samples)
            summary_items.extend(dynamic_motion_samples[:4])
            multimodal_channels.append(
                {
                    "channel": "vision",
                    "count": len(raw_samples) + len(memory_write_samples) + len(global_structure_samples) + min(4, len(dynamic_motion_samples)),
                    "preview": [item.get("display_text", "") for item in (dynamic_motion_samples[:2] + global_structure_samples[:2] + memory_write_samples[:2])],
                    "raw_count": len(raw_samples),
                    "memory_write_count": len(memory_write_samples),
                    "focus_priority_count": len(focus_priority_samples),
                    "global_structure_count": len(global_structure_samples),
                    "dynamic_motion_count": len(dynamic_motion_samples),
                }
            )
        if audio_packet:
            windows = [item for item in (audio_packet.get("windows", []) or []) if isinstance(item, dict)]
            audio_focus_priority_samples = [item for item in (audio_packet.get("focus_priority_samples", []) or []) if isinstance(item, dict)]
            audio_memory_write_samples = [item for item in (audio_packet.get("memory_write_samples", []) or []) if isinstance(item, dict)]
            audio_global_structure_samples = [item for item in (audio_packet.get("global_structure_samples", []) or []) if isinstance(item, dict)]
            multimodal_items.extend(windows)
            multimodal_items.extend(audio_memory_write_samples)
            multimodal_items.extend(audio_global_structure_samples)
            summary_items.extend(audio_memory_write_samples)
            summary_items.extend(audio_global_structure_samples)
            multimodal_channels.append(
                {
                    "channel": "hearing",
                    "count": len(windows) + len(audio_memory_write_samples) + len(audio_global_structure_samples),
                    "preview": [item.get("display_text", "") for item in (audio_global_structure_samples[:2] + audio_memory_write_samples[:2] + windows[:2])],
                    "window_count": len(windows),
                    "memory_write_count": len(audio_memory_write_samples),
                    "focus_priority_count": len(audio_focus_priority_samples),
                    "global_structure_count": len(audio_global_structure_samples),
                }
            )

        competition_packet = dict(text_packet)
        competition_packet["sa_items"] = summary_items
        competition_packet["state_pool_sa_items"] = multimodal_items
        competition_packet["summary_sa_items"] = summary_items
        competition_packet["sa_flow"] = [item["sa_label"] for item in multimodal_items if isinstance(item, dict) and item.get("sa_label")]
        competition_packet["competition_summary"] = {
            "prototype_count": competition["prototype_count"],
            "phrase_hit_count": len(competition["phrase_hits"]),
            "phrase_hit_preview": [item.get("display_text", "") for item in competition["phrase_hits"][:6]],
            "multimodal_channels": multimodal_channels,
        }

        pool_result_external = self.state_pool.apply_text_packet(competition_packet, tick_index=tick_index)
        r_state = self.state_pool.read_r_state()
        effective_attention_controls = self._effective_attention_controls(tick_index)
        query_items = self.state_pool.read_query_items(
            limit=max(16, int(round(self._runtime_control("sampling.increment_budget")))),
            focus_gain=float(effective_attention_controls.get("attention.focus_gain", self._runtime_control("attention.focus_gain"))),
            anchor_bias_gain=self._runtime_control("state.anchor_bias_gain"),
            current_input_gain=float(effective_attention_controls.get("state.current_input_gain", self._runtime_control("state.current_input_gain"))),
            history_suppression_gain=float(effective_attention_controls.get("state.history_suppression_gain", self._runtime_control("state.history_suppression_gain"))),
            prediction_suppression_gain=float(effective_attention_controls.get("state.prediction_suppression_gain", self._runtime_control("state.prediction_suppression_gain"))),
            surprise_focus_gain=float(effective_attention_controls.get("state.surprise_focus_gain", self._runtime_control("state.surprise_focus_gain"))),
        )
        stage_timing["02_state_query_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()
        source_hist: dict[str, int] = {}
        channel_hist: dict[str, int] = {}
        bucket_hist: dict[str, int] = {}
        for item in query_items:
            source_type = str(item.get("source_type", "") or "")
            channel = str(item.get("channel", "") or "")
            bucket = str(item.get("query_bucket", "") or "")
            source_hist[source_type] = int(source_hist.get(source_type, 0) + 1)
            channel_hist[channel] = int(channel_hist.get(channel, 0) + 1)
            bucket_hist[bucket] = int(bucket_hist.get(bucket, 0) + 1)
        recall_query_preview = {
            "count": len(query_items),
            "source_histogram": source_hist,
            "channel_histogram": channel_hist,
            "bucket_histogram": bucket_hist,
            "preview": [
                {
                    "sa_label": str(item.get("sa_label", "") or ""),
                    "display_text": str(item.get("display_text", "") or ""),
                    "energy": _round4(float(item.get("energy", 0.0) or 0.0)),
                    "query_weight": _round4(float(item.get("query_weight", 0.0) or 0.0)),
                    "attention_score": _round4(float(item.get("attention_score", 0.0) or 0.0)),
                    "source_type": str(item.get("source_type", "") or ""),
                    "channel": str(item.get("channel", "") or ""),
                    "query_bucket": str(item.get("query_bucket", "") or ""),
                }
                for item in query_items[:12]
            ],
        }
        query_labels: list[str] = []
        query_weights: dict[str, float] = {}
        for item in query_items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            query_labels.append(label)
            query_weights[label] = _round4(
                query_weights.get(label, 0.0)
                + float(item.get("query_weight", item.get("energy", item.get("unresolved_mass", 0.0))) or 0.0)
            )

        recent_focus_units = self.short_term.recent_focus_units(limit=8)
        query_spacetime = self.memory_store._infer_spacetime(
            tick_index=tick_index,
            units=base_units,
            items=[item for item in multimodal_items if isinstance(item, dict)],
        )
        recall_rows_seed = self.memory_store.recall_bn(
            query_labels=query_labels,
            query_weights=query_weights,
            top_k=max(10, int(self.config.memory_ann_top_k // 4)),
            tick_index=tick_index,
            query_items=query_items,
            query_units=base_units,
            recent_focus_units=recent_focus_units,
            successor_bias_gain=self._runtime_control("prediction.successor_bias_gain"),
            query_spacetime=query_spacetime,
        )
        bn_seed = [row for row in recall_rows_seed if str(row.get("memory_kind", "") or "") != "latent_state_snapshot"][:6]
        latent_seed = [row for row in recall_rows_seed if str(row.get("memory_kind", "") or "") == "latent_state_snapshot"][:3]
        seed_bn_virtual_mass = sum(max(0.0, float(row.get("score", 0.0) or 0.0)) for row in bn_seed)
        seed_latent_virtual_budget = min(
            max(0.0, seed_bn_virtual_mass * 0.55),
            sum(max(0.0, float(row.get("score", 0.0) or 0.0)) for row in latent_seed),
        )
        c_i_seed, c_star_seed = self.memory_store.build_prediction_branches(
            bn_list=bn_seed,
            tick_index=tick_index,
            recent_focus_units=recent_focus_units,
            max_neighbors=4,
            successor_bias_gain=self._runtime_control("prediction.successor_bias_gain"),
            latent_candidates=latent_seed,
            latent_total_virtual_energy=seed_latent_virtual_budget,
        )
        stage_timing["03_seed_recall_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()
        time_feeling_item, time_feeling_trace = self._build_time_feeling(
            tick_index=tick_index,
            bn_list=bn_seed,
        )
        motion_feeling_item, motion_feeling_trace = self._build_motion_feeling(
            tick_index=tick_index,
            dynamic_motion_samples=dynamic_motion_samples,
        )
        # Rhythm should follow the currently summarized, reality-backed scene
        # rather than the noisiest raw sample tail, especially for vision.
        rhythm_feeling_items, rhythm_feeling_trace = self._build_rhythm_feelings(
            tick_index=tick_index,
            external_items=summary_items,
            bn_list=bn_seed,
            c_star=c_star_seed,
            dynamic_motion_samples=dynamic_motion_samples,
        )
        hearing_feeling_items, hearing_feeling_trace = self._build_hearing_feelings(
            tick_index=tick_index,
            audio_packet=audio_packet,
        )
        feedback_signal_items, feedback_signal_trace = self._build_feedback_signal_feeling(
            tick_index=tick_index,
            pending_feedback=pending_feedback,
        )
        channel_feeling_items = [item for item in [time_feeling_item, motion_feeling_item] if isinstance(item, dict)]
        channel_feeling_items.extend([item for item in rhythm_feeling_items if isinstance(item, dict)])
        channel_feeling_items.extend([item for item in hearing_feeling_items if isinstance(item, dict)])
        channel_feeling_items.extend([item for item in feedback_signal_items if isinstance(item, dict)])
        if time_feeling_item:
            time_scale = max(1.0, float(getattr(self.config, "time_feeling_default_radius_ticks", 4.0)) * 4.0)
            time_attrs = dict((time_feeling_item.get("attributes", {}) or {}))
            target_delta_t = float(time_attrs.get("delta_t_norm", 0.0) or 0.0) * time_scale
            time_sigma = max(0.25, float(time_attrs.get("delta_sigma_norm", 0.0) or 0.0) * time_scale)
            query_spacetime["target_t"] = _round4(float(tick_index) - target_delta_t)
            query_spacetime["target_delta_t"] = _round4(target_delta_t)
            query_spacetime["time_sigma"] = _round4(time_sigma)
            query_spacetime["time_confidence"] = _round4(float(time_attrs.get("confidence", 0.0) or 0.0))
            query_spacetime["time_recall_gain"] = _round4(float(getattr(self.config, "time_feeling_recall_gain", 0.22)))
        if motion_feeling_item:
            motion_attrs = dict((motion_feeling_item.get("attributes", {}) or {}))
            query_spacetime["motion_center_speed"] = _round4(float(motion_attrs.get("motion_center_speed", 0.0) or 0.0))
            query_spacetime["motion_sigma"] = _round4(max(0.05, float(motion_attrs.get("motion_sigma", 0.18) or 0.18)))
            query_spacetime["motion_confidence"] = _round4(float(motion_attrs.get("confidence", 0.0) or 0.0))
            query_spacetime["motion_recall_gain"] = _round4(float(getattr(self.config, "motion_feeling_recall_gain", 0.18)))
        rhythm_pulse_item = next((item for item in rhythm_feeling_items if str((item or {}).get("sa_label", "") or "") == "rhythmfelt::pulse"), None)
        if rhythm_pulse_item:
            rhythm_attrs = dict((rhythm_pulse_item.get("attributes", {}) or {}))
            query_spacetime["rhythm_period_ticks"] = _round4(float(rhythm_attrs.get("period_ticks", 0.0) or 0.0))
            query_spacetime["rhythm_period_sigma"] = _round4(max(0.05, float(rhythm_attrs.get("period_sigma_ticks", 0.0) or 0.0)))
            query_spacetime["rhythm_confidence"] = _round4(float(rhythm_attrs.get("confidence", 0.0) or 0.0))
            query_spacetime["rhythm_recall_gain"] = _round4(float(getattr(self.config, "rhythm_recall_gain", 0.18)))
            query_spacetime["rhythm_family_key"] = str(rhythm_attrs.get("family_key", "") or "")
        rhythm_phase_item = next((item for item in rhythm_feeling_items if str((item or {}).get("sa_label", "") or "") == "rhythmfelt::phase_expectation"), None)
        if rhythm_phase_item:
            rhythm_phase_attrs = dict((rhythm_phase_item.get("attributes", {}) or {}))
            query_spacetime["rhythm_phase_error"] = _round4(float(rhythm_phase_attrs.get("phase_error", 0.0) or 0.0))
            query_spacetime["rhythm_time_to_next"] = _round4(float(rhythm_phase_attrs.get("time_to_next", 0.0) or 0.0))
        hearing_timbre_item = next((item for item in hearing_feeling_items if str((item or {}).get("sa_label", "") or "") == "hearingfelt::timbre_clarity"), None)
        if hearing_timbre_item:
            hearing_timbre_attrs = dict((hearing_timbre_item.get("attributes", {}) or {}))
            query_spacetime["hearing_timbre_center"] = _round4(float(hearing_timbre_attrs.get("hearing_timbre_center", 0.0) or 0.0))
            query_spacetime["hearing_timbre_sigma"] = _round4(max(0.05, float(hearing_timbre_attrs.get("hearing_timbre_sigma", 0.18) or 0.18)))
            query_spacetime["hearing_confidence"] = _round4(float(hearing_timbre_attrs.get("confidence", 0.0) or 0.0))
            query_spacetime["hearing_timbre_recall_gain"] = _round4(float(hearing_timbre_attrs.get("hearing_timbre_recall_gain", getattr(self.config, "hearing_timbre_recall_gain", 0.16)) or 0.16))
        hearing_noise_item = next((item for item in hearing_feeling_items if str((item or {}).get("sa_label", "") or "") == "hearingfelt::noisiness"), None)
        if hearing_noise_item:
            hearing_noise_attrs = dict((hearing_noise_item.get("attributes", {}) or {}))
            query_spacetime["hearing_noise_center"] = _round4(float(hearing_noise_attrs.get("hearing_noise_center", 0.0) or 0.0))
            query_spacetime["hearing_noise_sigma"] = _round4(max(0.05, float(hearing_noise_attrs.get("hearing_noise_sigma", 0.18) or 0.18)))
            query_spacetime["hearing_noise_recall_gain"] = _round4(float(hearing_noise_attrs.get("hearing_noise_recall_gain", getattr(self.config, "hearing_noise_recall_gain", 0.16)) or 0.16))
            query_spacetime["hearing_confidence"] = max(float(query_spacetime.get("hearing_confidence", 0.0) or 0.0), _round4(float(hearing_noise_attrs.get("confidence", 0.0) or 0.0)))
        hearing_pitch_item = next((item for item in hearing_feeling_items if str((item or {}).get("sa_label", "") or "") == "hearingfelt::pitch_stability"), None)
        if hearing_pitch_item:
            hearing_pitch_attrs = dict((hearing_pitch_item.get("attributes", {}) or {}))
            query_spacetime["hearing_pitch_stability_center"] = _round4(float(hearing_pitch_attrs.get("hearing_pitch_stability_center", 0.0) or 0.0))
            query_spacetime["hearing_pitch_stability_sigma"] = _round4(max(0.05, float(hearing_pitch_attrs.get("hearing_pitch_stability_sigma", 0.16) or 0.16)))
            query_spacetime["hearing_pitch_recall_gain"] = _round4(float(hearing_pitch_attrs.get("hearing_pitch_recall_gain", getattr(self.config, "hearing_pitch_recall_gain", 0.18)) or 0.18))
            query_spacetime["hearing_dominant_hz"] = _round4(float(hearing_pitch_attrs.get("dominant_hz", 0.0) or 0.0))
            query_spacetime["hearing_confidence"] = max(float(query_spacetime.get("hearing_confidence", 0.0) or 0.0), _round4(float(hearing_pitch_attrs.get("confidence", 0.0) or 0.0)))
        hearing_percussive_item = next((item for item in hearing_feeling_items if str((item or {}).get("sa_label", "") or "") == "hearingfelt::percussive_burst"), None)
        if hearing_percussive_item:
            hearing_perc_attrs = dict((hearing_percussive_item.get("attributes", {}) or {}))
            query_spacetime["hearing_percussive_center"] = _round4(float(hearing_perc_attrs.get("hearing_percussive_center", 0.0) or 0.0))
            query_spacetime["hearing_percussive_sigma"] = _round4(max(0.05, float(hearing_perc_attrs.get("hearing_percussive_sigma", 0.16) or 0.16)))
            query_spacetime["hearing_percussive_recall_gain"] = _round4(float(hearing_perc_attrs.get("hearing_percussive_recall_gain", getattr(self.config, "hearing_percussive_recall_gain", 0.16)) or 0.16))
            query_spacetime["hearing_confidence"] = max(float(query_spacetime.get("hearing_confidence", 0.0) or 0.0), _round4(float(hearing_perc_attrs.get("confidence", 0.0) or 0.0)))
        if feedback_signal_items:
            feedback_attrs = dict((feedback_signal_items[0] or {}).get("attributes", {}) or {})
            query_spacetime["feedback_valence"] = _round4(float(feedback_attrs.get("feedback_valence", 0.0) or 0.0))
            query_spacetime["feedback_sigma"] = _round4(max(0.05, float(feedback_attrs.get("feedback_sigma", 0.18) or 0.18)))
            query_spacetime["feedback_confidence"] = _round4(float(feedback_attrs.get("confidence", 0.0) or 0.0))
            query_spacetime["feedback_recall_gain"] = _round4(float(getattr(self.config, "feedback_signal_recall_gain", 0.14)))
        stage_timing["04_channel_feelings_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()
        recall_rows = self.memory_store.recall_bn(
            query_labels=query_labels,
            query_weights=query_weights,
            top_k=max(10, int(self.config.memory_ann_top_k // 4)),
            tick_index=tick_index,
            query_items=query_items,
            query_units=base_units,
            recent_focus_units=recent_focus_units,
            successor_bias_gain=self._runtime_control("prediction.successor_bias_gain"),
            query_spacetime=query_spacetime,
        )
        bn_list = [row for row in recall_rows if str(row.get("memory_kind", "") or "") != "latent_state_snapshot"][:6]
        latent_rows = [row for row in recall_rows if str(row.get("memory_kind", "") or "") == "latent_state_snapshot"][:3]
        bn_virtual_mass = sum(max(0.0, float(row.get("score", 0.0) or 0.0)) for row in bn_list)
        latent_virtual_budget = min(
            max(0.0, bn_virtual_mass * 0.55),
            sum(max(0.0, float(row.get("score", 0.0) or 0.0)) for row in latent_rows),
        )
        c_i_list, c_star = self.memory_store.build_prediction_branches(
            bn_list=bn_list,
            tick_index=tick_index,
            recent_focus_units=recent_focus_units,
            max_neighbors=4,
            successor_bias_gain=self._runtime_control("prediction.successor_bias_gain"),
            latent_candidates=latent_rows,
            latent_total_virtual_energy=latent_virtual_budget,
        )
        stage_timing["05_main_recall_prediction_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()

        raw_predicted_items = [
            {
                "sa_label": str(item.get("sa_label", "") or ""),
                "display_text": str(item.get("display_text", "") or ""),
                "raw_energy": max(0.0, float(item.get("energy", 0.0) or 0.0)),
                "commitment": _round4(float(item.get("commitment", 0.0) or 0.0)),
                "prediction_role": str(item.get("prediction_role", "") or ""),
                "support": copy.deepcopy(dict(item.get("support", {}) or {})),
            }
            for item in (c_star.get("items", []) or [])[:8]
            if str(item.get("sa_label", "") or "")
        ]
        predicted_peak = max((float(item.get("raw_energy", 0.0) or 0.0) for item in raw_predicted_items), default=0.0)
        predicted_items = []
        for item in raw_predicted_items:
            raw_energy = float(item.get("raw_energy", 0.0) or 0.0)
            if raw_energy <= 0.0:
                continue
            normalized = raw_energy / max(1e-6, predicted_peak)
            predicted_items.append(
                {
                    "sa_label": str(item.get("sa_label", "") or ""),
                    "display_text": str(item.get("display_text", "") or ""),
                    "energy": _round4(max(0.08, min(1.5, normalized * 1.5))),
                    "commitment": _round4(float(item.get("commitment", 0.0) or 0.0)),
                    "prediction_role": str(item.get("prediction_role", "") or ""),
                    "support": copy.deepcopy(dict(item.get("support", {}) or {})),
                    "attributes": {
                        "prediction_commitment": _round4(float(item.get("commitment", 0.0) or 0.0)),
                        "prediction_role": str(item.get("prediction_role", "") or ""),
                        "grasp_hint": _round4(float(item.get("commitment", 0.0) or 0.0)),
                    },
                }
            )
        pool_result_predict = (
            self.state_pool.inject_runtime_items(
                predicted_items,
                tick_index=tick_index,
                source_type="prediction",
                channel="c_star",
                record_handle=False,
            )
            if predicted_items
            else {}
        )
        pool_result_channel_feelings = (
            self.state_pool.inject_runtime_items(
                channel_feeling_items,
                tick_index=tick_index,
                source_type="channel_feeling",
                channel="channel_feeling",
                record_handle=False,
            )
            if channel_feeling_items
            else {}
        )
        self.state_pool.set_pending_prediction_items(predicted_items)
        self.state_pool.refresh_prediction_trace(
            competition_packet,
            predicted_items=predicted_items,
        )
        branch_credibility_update = self.memory_store.update_branch_credibility(
            c_i_list=c_i_list,
            actual_items=competition_packet.get("state_pool_sa_items", competition_packet.get("sa_items", [])) or [],
            tick_index=tick_index,
        )

        rules_context = {
            "tick_index": tick_index,
            "state_top": self.state_pool.snapshot_top(limit=10),
            "state_pool_summary": self.state_pool.snapshot_summary(),
            "bn_list": bn_list,
            "c_star": c_star,
            "branch_credibility_update": branch_credibility_update,
            "sensor_packet": text_packet,
            "multimodal_summary": {
                "has_image": bool(image_packet),
                "has_audio": bool(audio_packet),
                "image_patch_budget_used": int(image_packet.get("budget_used", 0) or 0) if image_packet else 0,
                "image_raw_sample_count": int(len(raw_samples)) if image_packet else 0,
                "image_raw_state_budget": int(image_packet.get("raw_state_budget", 0) or 0) if image_packet else 0,
                "image_memory_write_count": int(len(memory_write_samples)) if image_packet else 0,
                "image_focus_priority_count": int(len(focus_priority_samples)) if image_packet else 0,
                "image_global_structure_count": int(len(global_structure_samples)) if image_packet else 0,
                "image_dynamic_motion_count": int(len(dynamic_motion_samples)) if image_packet else 0,
                "image_reconstruction_cell_count": int((((image_packet or {}).get("reconstruction_grid", {}) or {}).get("cell_count", 0) or 0)) if image_packet else 0,
                "audio_window_budget_used": int(audio_packet.get("budget_used", 0) or 0) if audio_packet else 0,
                "audio_window_count": int(len(windows)) if audio_packet else 0,
                "audio_memory_write_count": int(len(audio_memory_write_samples)) if audio_packet else 0,
                "audio_focus_priority_count": int(len(audio_focus_priority_samples)) if audio_packet else 0,
                "audio_global_structure_count": int(len(audio_global_structure_samples)) if audio_packet else 0,
            },
            "runtime_metrics": {
                "logic_ms": self._last_logic_ms,
                "text_budget_used": int(text_packet.get("budget_used", 0) or 0),
                "vision_budget_used": int(image_packet.get("budget_used", 0) or 0) if image_packet else 0,
                "vision_raw_sample_count": int(len(raw_samples)) if image_packet else 0,
                "vision_dynamic_motion_count": int(len(dynamic_motion_samples)) if image_packet else 0,
                "audio_budget_used": int(audio_packet.get("budget_used", 0) or 0) if audio_packet else 0,
                "audio_window_count": int(len(windows)) if audio_packet else 0,
                "audio_memory_write_count": int(len(audio_memory_write_samples)) if audio_packet else 0,
                "audio_focus_priority_count": int(len(audio_focus_priority_samples)) if audio_packet else 0,
                "audio_global_structure_count": int(len(audio_global_structure_samples)) if audio_packet else 0,
                "feedback_reward": float(pending_feedback.get("reward", 0.0) or 0.0),
                "feedback_punishment": float(pending_feedback.get("punishment", 0.0) or 0.0),
                "feedback_external_reward": self._feedback_source_metric(pending_feedback_breakdown, "external", "reward"),
                "feedback_external_punishment": self._feedback_source_metric(pending_feedback_breakdown, "external", "punishment"),
                "feedback_teacher_reward": self._feedback_source_metric(pending_feedback_breakdown, "teacher", "reward"),
                "feedback_teacher_punishment": self._feedback_source_metric(pending_feedback_breakdown, "teacher", "punishment"),
                "feedback_intrinsic_reward": self._feedback_source_metric(pending_feedback_breakdown, "intrinsic", "reward"),
                "feedback_intrinsic_punishment": self._feedback_source_metric(pending_feedback_breakdown, "intrinsic", "punishment"),
            },
            "channel_feelings": {
                "time": dict(time_feeling_trace or {}),
                "motion": dict(motion_feeling_trace or {}),
                "rhythm": dict(rhythm_feeling_trace or {}),
                "hearing": dict(hearing_feeling_trace or {}),
                "feedback": dict(feedback_signal_trace or {}),
                "injected_count": len(channel_feeling_items),
            },
        }
        rules_result_raw = self.rules_engine.evaluate(
            rules_context,
            dissonance_gain=self._runtime_control("rules.dissonance_gain"),
        )
        habituation_trace = self._compute_cognitive_feeling_habituation(
            tick_index=tick_index,
            text_packet=text_packet,
            image_packet=image_packet,
            rules_result=rules_result_raw,
        )
        self._cognitive_feeling_habituation = self._normalize_habituation_payload(habituation_trace.get("next_payload", {}))
        rules_result = self.rules_engine.evaluate(
            rules_context,
            dissonance_gain=self._runtime_control("rules.dissonance_gain"),
            emotion_channel_gains=dict(habituation_trace.get("gains", {}) or {}),
        )
        stage_timing["06_rules_feelings_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()
        rules_result["raw_emotion_channels"] = dict((rules_result_raw or {}).get("emotion_channels", {}) or {})
        rules_result["raw_metrics_snapshot"] = dict((rules_result_raw or {}).get("metrics_snapshot", {}) or {})
        rules_result["cognitive_feeling_habituation"] = {
            key: value
            for key, value in dict(habituation_trace or {}).items()
            if key != "next_payload"
        }
        planner_action_learning_context = self._build_action_learning_context(
            text_packet=text_packet,
            query_units=base_units,
            final_focus_units=[],
        )
        action_learning_view = self.action_learning.score_action_drives(
            list(rules_result.get("action_drives", []) or []),
            context_hints=planner_action_learning_context,
        )
        planner_view = self.action_planner.plan_actions(
            tick_index=tick_index,
            raw_action_drives=list(rules_result.get("action_drives", []) or []),
            rules_result=rules_result,
            bn_list=bn_list,
            c_star=c_star,
            action_learning=self.action_learning,
            context_hints=planner_action_learning_context,
            image_packet=image_packet or {},
            audio_packet=audio_packet or {},
            pending_feedback={
                "reward": float(pending_feedback.get("reward", 0.0) or 0.0),
                "punishment": float(pending_feedback.get("punishment", 0.0) or 0.0),
            },
            recent_focus_units=recent_focus_units,
        )
        rules_result["action_drives_raw"] = list(rules_result.get("action_drives", []) or [])
        rules_result["action_drives_scored_legacy"] = list(action_learning_view.get("scored_action_drives", []) or [])
        rules_result["action_drives"] = list(planner_view.get("planned_action_drives", []) or [])
        rules_result["action_learning_bias_summary"] = list(action_learning_view.get("bias_summary", []) or [])
        rules_result["planned_action_drives"] = list(planner_view.get("planned_action_drives", []) or [])
        rules_result["planned_selected_actions_preview"] = list(planner_view.get("selected_actions_preview", []) or [])
        rules_result["action_actuator_reports"] = list(planner_view.get("actuator_reports", []) or [])
        rules_result["action_planner_state"] = dict(planner_view.get("actuator_state", {}) or {})
        current_tick_attention_modulation = self._derive_attention_modulated_controls(
            selected_actions=list(planner_view.get("selected_actions_preview", []) or []),
            tick_index=tick_index,
        )
        applied_controls = self._apply_tuner_adjustments(rules_result.get("tuner_result", {}) or {})
        tuner_learning_view = self._apply_tuner_learning_offsets(matched_profiles=list((rules_result.get("tuner_result", {}) or {}).get("matched_profiles", []) or []))
        rules_result["tuner_learning"] = {
            "applied_offsets": list(tuner_learning_view.get("applied_offsets", []) or []),
            "matched_profile_ids": list(tuner_learning_view.get("matched_profile_ids", []) or []),
            "target_bias_summary": list(tuner_learning_view.get("target_bias_summary", []) or []),
            "profile_bias_summary": list(tuner_learning_view.get("profile_bias_summary", []) or []),
        }
        rules_result["current_tick_attention_modulation"] = dict(current_tick_attention_modulation)
        surprise_reorient = self._auto_visual_reorient_from_surprise(
            rules_result=rules_result,
            image_packet=image_packet or {},
        )
        if surprise_reorient:
            rules_result["auto_visual_reorient"] = surprise_reorient
        stage_timing["07_planner_modulation_ms"] = _stage_ms(stage_started)
        stage_started = time.perf_counter()
        pool_result_rules = (
            self.state_pool.inject_runtime_items(
                rules_result.get("injected_items", []) or [],
                tick_index=tick_index,
                source_type="rules",
                channel="rules",
                record_handle=False,
            )
            if (rules_result.get("injected_items") or [])
            else {}
        )

        final_focus_controls = self._merge_attention_control_sets(
            effective_attention_controls,
            dict(current_tick_attention_modulation.get("modulated_controls", {}) or {}),
        )
        final_focus = self.state_pool.read_a_focus_with_bias(
            limit=min(4, self.config.r_state_items_per_head),
            focus_gain=float(final_focus_controls.get("attention.focus_gain", self._runtime_control("attention.focus_gain"))),
            anchor_bias_gain=self._runtime_control("state.anchor_bias_gain"),
            current_input_gain=float(final_focus_controls.get("state.current_input_gain", self._runtime_control("state.current_input_gain"))),
            history_suppression_gain=float(final_focus_controls.get("state.history_suppression_gain", self._runtime_control("state.history_suppression_gain"))),
            prediction_suppression_gain=float(final_focus_controls.get("state.prediction_suppression_gain", self._runtime_control("state.prediction_suppression_gain"))),
            surprise_focus_gain=float(final_focus_controls.get("state.surprise_focus_gain", self._runtime_control("state.surprise_focus_gain"))),
            commit=True,
        )
        focus_items_raw = [dict(item) for item in (final_focus.get("focus_items", []) or []) if isinstance(item, dict)]
        focus_items = self._focus_items_for_memory(focus_items_raw)
        focus_units = self._focus_units_for_memory(focus_items)
        pending_memory_rows: list[dict[str, Any]] = []
        focus_memory_index: int | None = None
        exact_memory_index: int | None = None
        latent_memory_index: int | None = None
        if focus_items:
            focus_memory_index = len(pending_memory_rows)
            pending_memory_rows.append(
                {
                    "tick_index": tick_index,
                    "memory_kind": "focus_chain",
                    "units": focus_units,
                    "items": focus_items,
                    "source_refs": [item.get("memory_id", "") for item in bn_list[:3]],
                    "text": join_text_units(focus_units),
                    "reality_weight": 0.6,
                    "meta": {},
                }
            )

        exact_items = list(summary_items)
        if image_packet and memory_write_samples:
            exact_items = (
                list(competition["selected_items"])
                + memory_write_samples
                + global_structure_samples
                + dynamic_motion_samples[:4]
                + audio_memory_write_samples
                + audio_global_structure_samples
                + (windows if audio_packet else [])
            )
            exact_items = [item for item in exact_items if isinstance(item, dict)]
        elif audio_packet and (audio_memory_write_samples or audio_global_structure_samples or windows):
            exact_items = (
                list(competition["selected_items"])
                + audio_memory_write_samples
                + audio_global_structure_samples
                + windows
            )
            exact_items = [item for item in exact_items if isinstance(item, dict)]
        if (base_units or exact_items):
            exact_memory_index = len(pending_memory_rows)
            pending_memory_rows.append(
                {
                    "tick_index": tick_index,
                    "memory_kind": "exact_external",
                    "units": base_units,
                    "items": exact_items,
                    "source_refs": [],
                    "text": str(text_packet.get("normalized_text", "") or ""),
                    "reality_weight": 1.0,
                    "meta": {},
                }
            )
        latent_snapshot_items = self._latent_snapshot_items_for_memory(query_items)
        if latent_snapshot_items:
            latent_memory_index = len(pending_memory_rows)
            pending_memory_rows.append(
                {
                    "tick_index": tick_index,
                    "memory_kind": "latent_state_snapshot",
                    "units": base_units,
                    "items": latent_snapshot_items,
                    "source_refs": [item.get("memory_id", "") for item in bn_list[:3]],
                    "text": "",
                    "reality_weight": 0.95,
                    "meta": {
                        "query_item_count": len(latent_snapshot_items),
                        "query_preview_labels": [str(item.get("sa_label", "") or "") for item in latent_snapshot_items[:12]],
                    },
                }
            )
        written_memories = self.memory_store.write_memory_batch(pending_memory_rows) if pending_memory_rows else []
        focus_memory = written_memories[focus_memory_index] if focus_memory_index is not None and focus_memory_index < len(written_memories) else {}
        exact_memory = written_memories[exact_memory_index] if exact_memory_index is not None and exact_memory_index < len(written_memories) else {}
        latent_snapshot_memory = written_memories[latent_memory_index] if latent_memory_index is not None and latent_memory_index < len(written_memories) else {}

        if focus_units:
            self.short_term.append(
                {
                    "tick_index": tick_index,
                    "focus_units": focus_units,
                    "focus_text": final_focus.get("focus_text", ""),
                    "focus_memory_id": focus_memory.get("memory_id", ""),
                    "bn_ids": [item.get("memory_id", "") for item in bn_list[:3]],
                }
            )
        post_focus_action_learning_context = self._build_action_learning_context(
            text_packet=text_packet,
            query_units=base_units,
            final_focus_units=focus_units,
        )

        self._last_control_feedback_context = {
            "runtime_controls": self.runtime_controls_snapshot(),
            "matched_profiles": list((rules_result.get("tuner_result", {}) or {}).get("matched_profiles", []) or []),
            "applied_tuner_adjustments": list(applied_controls),
            "learned_tuner_offsets": list((tuner_learning_view.get("applied_offsets", []) or [])),
            "action_learning_context": planner_action_learning_context,
            "post_focus_action_learning_context": post_focus_action_learning_context,
        }
        self._queued_intrinsic_feedback = self._build_intrinsic_feedback(
            emotion_channels=rules_result.get("emotion_channels", {}) or {},
            balance_metrics={
                "alignment_score": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_alignment_score", 0.0) or 0.0),
                "grasp_score": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_grasp_score", 0.0) or 0.0),
                "overprediction_ratio": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_overprediction_ratio", 0.0) or 0.0),
                "underprediction_ratio": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_underprediction_ratio", 0.0) or 0.0),
                "committed_alignment_score": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_committed_alignment_score", 0.0) or 0.0),
                "committed_grasp_score": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_committed_grasp_score", 0.0) or 0.0),
                "committed_overprediction_ratio": float((rules_result.get("metrics_snapshot", {}) or {}).get("state.prediction_committed_overprediction_ratio", 0.0) or 0.0),
            },
        )
        stage_timing["08_memory_feedback_ms"] = _stage_ms(stage_started)
        stage_timing["09_total_runtime_ms"] = _stage_ms(tick_started)

        return {
            "sensor_packet": text_packet,
            "image_packet": image_packet or {},
            "audio_packet": audio_packet or {},
            "competition_packet": competition_packet,
            "competition_summary": competition_packet.get("competition_summary", {}),
            "pool_result_external": pool_result_external,
            "r_state": r_state,
            "bn_list": bn_list,
            "latent_recall_list": latent_rows,
            "c_i_list": c_i_list,
            "c_star": c_star,
            "branch_credibility_update": branch_credibility_update,
            "pool_result_predict": pool_result_predict,
            "pool_result_channel_feelings": pool_result_channel_feelings,
            "channel_feeling_items": [copy.deepcopy(item) for item in channel_feeling_items],
            "channel_feeling_trace": {
                "time": dict(time_feeling_trace or {}),
                "motion": dict(motion_feeling_trace or {}),
                "rhythm": dict(rhythm_feeling_trace or {}),
                "hearing": dict(hearing_feeling_trace or {}),
                "feedback": dict(feedback_signal_trace or {}),
            },
            "hearing_feeling_trace": dict(hearing_feeling_trace or {}),
            "rules_result": rules_result,
            "pool_result_rules": pool_result_rules,
            "a_focus": final_focus,
            "focus_memory": focus_memory,
            "exact_memory": exact_memory,
            "latent_snapshot_memory": latent_snapshot_memory,
            "short_term_snapshot": self.short_term.snapshot(),
            "state_pool_summary": self.state_pool.snapshot_summary(),
            "state_pool_sidecar": self.state_pool.snapshot_sidecar(),
            "recall_query_preview": recall_query_preview,
            "memory_count": self.memory_store.count(),
            "memory_index_summary": self.memory_store.index_summary(),
            "query_spacetime": dict(query_spacetime),
            "runtime_controls": self.runtime_controls_snapshot(),
            "effective_attention_controls": dict(effective_attention_controls),
            "final_focus_attention_controls": dict(final_focus_controls),
            "attention_modulation_state": self._attention_modulation_snapshot_for_tick(tick_index),
            "applied_tuner_adjustments": applied_controls,
            "tuner_learning_summary": {
                "applied_offsets": list(tuner_learning_view.get("applied_offsets", []) or []),
                "target_bias_summary": self.tuner_learning.target_bias_summary(limit=12),
                "profile_bias_summary": self.tuner_learning.profile_bias_summary(limit=12),
            },
            "logic_feedback": {
                "previous_tick_logic_ms": _round4(self._last_logic_ms),
                "sampling_budget": sampling_budget,
                "pending_feedback_used": {
                    "reward": _round4(float(pending_feedback.get("reward", 0.0) or 0.0)),
                    "punishment": _round4(float(pending_feedback.get("punishment", 0.0) or 0.0)),
                },
                "pending_feedback_breakdown": pending_feedback_breakdown,
                "runtime_stage_timing_ms": dict(stage_timing),
            },
            "runtime_stage_timing_ms": dict(stage_timing),
            "queued_intrinsic_feedback_preview": copy.deepcopy(self._queued_intrinsic_feedback or {}),
            "action_learning_bias_summary": self.action_learning.bias_summary(limit=12),
            "action_learning_context_bias_summary": self.action_learning.context_bias_summary(limit=12),
            "action_planner_state": self.action_planner.snapshot_actuator_state(),
            "pending_feedback_metrics": {
                "reward": _round4(float(pending_feedback.get("reward", 0.0) or 0.0)),
                "punishment": _round4(float(pending_feedback.get("punishment", 0.0) or 0.0)),
            },
            "pending_feedback_breakdown": pending_feedback_breakdown,
        }

    def apply_action_feedback(
        self,
        *,
        tick_index: int,
        selected_actions: list[dict[str, Any]],
        emotion_channels: dict[str, Any],
        runtime_action_effects: dict[str, Any] | None = None,
        external_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        action_feedback = self.action_learning.record_feedback(
            tick_index=tick_index,
            selected_actions=selected_actions,
            emotion_channels=emotion_channels,
            runtime_action_effects=runtime_action_effects,
            external_feedback=external_feedback,
            context_hints=dict((self._last_control_feedback_context or {}).get("action_learning_context", {}) or {}),
        )
        if isinstance(external_feedback, dict) and "sources" not in external_feedback and self._feedback_has_signal(self._queued_intrinsic_feedback):
            external_feedback = self.merge_feedback_channels(
                external_feedback=external_feedback,
                teacher_feedback={},
                intrinsic_feedback=self._queued_intrinsic_feedback,
            )
        self._pending_feedback_metrics = {
            "reward": _round4(float((external_feedback or {}).get("reward", 0.0) or 0.0)),
            "punishment": _round4(float((external_feedback or {}).get("punishment", 0.0) or 0.0)),
        }
        self._queued_intrinsic_feedback = {}
        if isinstance(external_feedback, dict) and "sources" in external_feedback:
            self._pending_feedback_breakdown = copy.deepcopy(external_feedback)
        else:
            self._pending_feedback_breakdown = {
                **self._blank_feedback_breakdown(),
                "reward": self._pending_feedback_metrics["reward"],
                "punishment": self._pending_feedback_metrics["punishment"],
                "notes": list((external_feedback or {}).get("notes", []) or []),
                "sources": {
                    "external": {
                        "reward": self._pending_feedback_metrics["reward"],
                        "punishment": self._pending_feedback_metrics["punishment"],
                        "notes": list((external_feedback or {}).get("notes", []) or []),
                    },
                    "teacher": {"reward": 0.0, "punishment": 0.0, "notes": []},
                    "intrinsic": {"reward": 0.0, "punishment": 0.0, "notes": []},
                },
            }
        self.action_planner.record_execution_feedback(
            tick_index=tick_index,
            selected_actions=selected_actions,
            external_feedback=external_feedback,
            runtime_action_effects=runtime_action_effects,
        )
        tuner_feedback = self.tuner_learning.record_feedback(
            tick_index=tick_index,
            control_feedback_context=self._last_control_feedback_context,
            emotion_channels=emotion_channels,
            action_feedback=action_feedback,
            logic_ms=self._last_logic_ms,
        )
        self._last_control_feedback_context = {
            **dict(self._last_control_feedback_context or {}),
            "last_action_feedback": dict(action_feedback),
            "last_tuner_feedback": dict(tuner_feedback),
        }
        return {
            **action_feedback,
            "tuner_learning_feedback": tuner_feedback,
        }

    def inject_feedback_signals(
        self,
        *,
        tick_index: int,
        feedback: dict[str, Any],
        provenance: dict[str, Any] | None = None,
        source_type: str = "runtime_feedback",
        channel: str = "feedback",
        meta_extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        feedback = dict(feedback or {})
        provenance = dict(provenance or {})
        meta_extra = dict(meta_extra or {})
        reward = float(feedback.get("reward", 0.0) or 0.0)
        punishment = float(feedback.get("punishment", 0.0) or 0.0)
        self._pending_feedback_metrics = {
            "reward": _round4(reward),
            "punishment": _round4(punishment),
        }
        if isinstance(feedback, dict) and "sources" in feedback:
            self._pending_feedback_breakdown = copy.deepcopy(feedback)
        injected_items: list[dict[str, Any]] = []
        if reward > 0:
            injected_items.append(
                {
                    "sa_label": "attr::reward_signal",
                    "display_text": "奖励信号",
                    "energy": _round4(min(1.5, reward)),
                }
            )
        if punishment > 0:
            injected_items.append(
                {
                    "sa_label": "attr::punishment_signal",
                    "display_text": "惩罚信号",
                    "energy": _round4(min(1.5, punishment)),
                }
            )
        payload = {}
        if injected_items:
            payload = self.state_pool.inject_runtime_items(
                injected_items,
                tick_index=tick_index,
                source_type=source_type,
                channel=channel,
            )
        if injected_items:
            source_refs: list[str] = []
            for key in ("focus_memory_id", "exact_memory_id"):
                clean = str(provenance.get(key, "") or "")
                if clean and clean not in source_refs:
                    source_refs.append(clean)
            for key in ("bn_ids", "selected_action_ids"):
                for clean in provenance.get(key, []) or []:
                    value = str(clean or "")
                    if value and value not in source_refs:
                        source_refs.append(value)
            self.memory_store.write_memory(
                tick_index=tick_index,
                memory_kind=channel,
                units=[item["display_text"] for item in injected_items],
                items=injected_items,
                source_refs=source_refs,
                text=" ".join(item["display_text"] for item in injected_items),
                reality_weight=0.9,
                meta={
                    "provenance": provenance,
                    "notes": list(feedback.get("notes", []) or []),
                    **meta_extra,
                },
            )
        return {
            "reward": _round4(reward),
            "punishment": _round4(punishment),
            "notes": list(feedback.get("notes", []) or []),
            "sources": copy.deepcopy(feedback.get("sources", {}) or {}),
            "intrinsic_detail": copy.deepcopy(feedback.get("intrinsic_detail", {}) or {}),
            "injected_items": injected_items,
            "pool_result": payload,
            "provenance": provenance,
            "source_type": source_type,
            "channel": channel,
        }

    def inject_teacher_feedback(
        self,
        *,
        tick_index: int,
        teacher_feedback: dict[str, Any],
        teacher_provenance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        teacher_feedback = dict(teacher_feedback or {})
        teacher_provenance = dict(teacher_provenance or {})
        payload = self.inject_feedback_signals(
            tick_index=tick_index,
            feedback=teacher_feedback,
            provenance=teacher_provenance,
            source_type="teacher_feedback",
            channel="teacher_feedback",
            meta_extra={
                "teacher_provenance": teacher_provenance,
                "teacher_review": dict(teacher_feedback.get("teacher_review", {}) or {}),
                "external_teacher_review": dict(teacher_feedback.get("external_teacher_review", {}) or {}),
            },
        )
        return {
            **payload,
            "teacher_review": dict(teacher_feedback.get("teacher_review", {}) or {}),
            "external_teacher_review": dict(teacher_feedback.get("external_teacher_review", {}) or {}),
            "teacher_provenance": teacher_provenance,
        }

    def export_payload(self) -> dict[str, Any]:
        return {
            "text_sensor": self.text_sensor.export_payload(),
            "vision_sensor": self.vision_sensor.export_payload(),
            "hearing_sensor": self.hearing_sensor.export_payload(),
            "state_pool": self.state_pool.export_payload(),
            "sa_registry": self.sa_registry.export_payload(),
            "memory_store": self.memory_store.export_payload(),
            "short_term": self.short_term.export_payload(),
            "action_learning": self.action_learning.export_payload(),
            "action_planner": self.action_planner.export_payload(),
            "tuner_learning": self.tuner_learning.export_payload(),
            "teacher_layer": self.teacher_layer.export_payload(),
            "runtime_controls": self.runtime_controls_snapshot(),
            "last_logic_ms": _round4(self._last_logic_ms),
            "last_control_feedback_context": dict(self._last_control_feedback_context or {}),
            "attention_modulation_state": dict(self._attention_modulation_state or {}),
            "pending_feedback_metrics": dict(self._pending_feedback_metrics or {}),
            "pending_feedback_breakdown": copy.deepcopy(self._pending_feedback_breakdown or {}),
            "queued_intrinsic_feedback": copy.deepcopy(self._queued_intrinsic_feedback or {}),
            "last_emotion_channels": dict(self._last_emotion_channels or {}),
            "last_cognitive_balance": dict(self._last_cognitive_balance or {}),
            "cognitive_feeling_habituation": copy.deepcopy(self._cognitive_feeling_habituation or {}),
            "channel_feeling_fatigue": copy.deepcopy(self._channel_feeling_fatigue or {}),
            "rhythm_tracker": copy.deepcopy(self._rhythm_tracker or {}),
            "rules_engine": {
                "rules": self.rules_engine.export_rules(),
                "tuner": self.rules_engine.export_tuner(),
            },
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        if isinstance(payload.get("text_sensor"), dict):
            self.text_sensor.import_payload(payload["text_sensor"])
        if isinstance(payload.get("vision_sensor"), dict):
            self.vision_sensor.import_payload(payload["vision_sensor"])
        if isinstance(payload.get("hearing_sensor"), dict):
            self.hearing_sensor.import_payload(payload["hearing_sensor"])
        if isinstance(payload.get("state_pool"), dict):
            self.state_pool.import_payload(payload["state_pool"])
        if isinstance(payload.get("sa_registry"), dict):
            self.sa_registry.import_payload(payload["sa_registry"])
        if isinstance(payload.get("memory_store"), dict):
            self.memory_store.import_payload(payload["memory_store"])
        if isinstance(payload.get("short_term"), dict):
            self.short_term.import_payload(payload["short_term"])
        if isinstance(payload.get("action_learning"), dict):
            self.action_learning.import_payload(payload["action_learning"])
        if isinstance(payload.get("action_planner"), dict):
            self.action_planner.import_payload(payload["action_planner"])
        if isinstance(payload.get("tuner_learning"), dict):
            self.tuner_learning.import_payload(payload["tuner_learning"])
        if isinstance(payload.get("teacher_layer"), dict):
            self.teacher_layer.import_payload(payload["teacher_layer"])
        runtime_controls = payload.get("runtime_controls")
        if isinstance(runtime_controls, dict):
            defaults = self._default_runtime_controls()
            merged = dict(defaults)
            for key, default_value in defaults.items():
                if key in runtime_controls:
                    try:
                        merged[key] = float(runtime_controls[key])
                    except Exception:
                        merged[key] = float(default_value)
            self._runtime_controls = merged
        else:
            self._runtime_controls = self._default_runtime_controls()
        attention_modulation_state = payload.get("attention_modulation_state", {})
        if isinstance(attention_modulation_state, dict):
            merged_attention_state = self._blank_attention_modulation_state()
            merged_attention_state.update(attention_modulation_state)
            self._attention_modulation_state = merged_attention_state
        else:
            self._attention_modulation_state = self._blank_attention_modulation_state()
        self._last_logic_ms = max(0.0, float(payload.get("last_logic_ms", 0.0) or 0.0))
        pending_feedback_metrics = payload.get("pending_feedback_metrics", {})
        if isinstance(pending_feedback_metrics, dict):
            self._pending_feedback_metrics = {
                "reward": _round4(float(pending_feedback_metrics.get("reward", 0.0) or 0.0)),
                "punishment": _round4(float(pending_feedback_metrics.get("punishment", 0.0) or 0.0)),
            }
        else:
            self._pending_feedback_metrics = {"reward": 0.0, "punishment": 0.0}
        pending_feedback_breakdown = payload.get("pending_feedback_breakdown", {})
        if isinstance(pending_feedback_breakdown, dict):
            self._pending_feedback_breakdown = copy.deepcopy(pending_feedback_breakdown)
        else:
            self._pending_feedback_breakdown = self._blank_feedback_breakdown()
        queued_intrinsic_feedback = payload.get("queued_intrinsic_feedback", {})
        if isinstance(queued_intrinsic_feedback, dict):
            self._queued_intrinsic_feedback = copy.deepcopy(queued_intrinsic_feedback)
        else:
            self._queued_intrinsic_feedback = {}
        last_emotion_channels = payload.get("last_emotion_channels", {})
        if isinstance(last_emotion_channels, dict):
            self._last_emotion_channels = self._normalize_emotion_channels(last_emotion_channels)
        else:
            self._last_emotion_channels = self._normalize_emotion_channels({})
        last_cognitive_balance = payload.get("last_cognitive_balance", {})
        if isinstance(last_cognitive_balance, dict):
            self._last_cognitive_balance = self._normalize_cognitive_balance(last_cognitive_balance)
        else:
            self._last_cognitive_balance = self._normalize_cognitive_balance({})
        habituation_payload = payload.get("cognitive_feeling_habituation", {})
        if isinstance(habituation_payload, dict):
            self._cognitive_feeling_habituation = self._normalize_habituation_payload(habituation_payload)
        else:
            self._cognitive_feeling_habituation = self._blank_cognitive_feeling_habituation()
        channel_feeling_fatigue = payload.get("channel_feeling_fatigue", {})
        if isinstance(channel_feeling_fatigue, dict):
            normalized_fatigue: dict[str, dict[str, float | int]] = {}
            for channel_key, channel_payload in channel_feeling_fatigue.items():
                clean_channel = str(channel_key or "")
                if not clean_channel or not isinstance(channel_payload, dict):
                    continue
                next_channel: dict[str, float | int] = {}
                for signal_key, signal_payload in channel_payload.items():
                    clean_signal = str(signal_key or "")
                    if not clean_signal or not isinstance(signal_payload, dict):
                        continue
                    next_channel[clean_signal] = {
                        "value": _round4(max(0.0, float(signal_payload.get("value", 0.0) or 0.0))),
                        "tick_index": int(signal_payload.get("tick_index", 0) or 0),
                    }
                if next_channel:
                    normalized_fatigue[clean_channel] = next_channel
            self._channel_feeling_fatigue = normalized_fatigue
        else:
            self._channel_feeling_fatigue = {}
        rhythm_tracker = payload.get("rhythm_tracker", {})
        if isinstance(rhythm_tracker, dict):
            self._rhythm_tracker = self._normalize_rhythm_tracker(rhythm_tracker)
        else:
            self._rhythm_tracker = {"families": {}, "last_tick": -1}
        last_control_feedback_context = payload.get("last_control_feedback_context", {})
        if isinstance(last_control_feedback_context, dict):
            self._last_control_feedback_context = dict(last_control_feedback_context)
        else:
            self._last_control_feedback_context = {
                "runtime_controls": self.runtime_controls_snapshot(),
                "matched_profiles": [],
                "applied_tuner_adjustments": [],
                "learned_tuner_offsets": [],
            }
        rules_engine_payload = payload.get("rules_engine", {})
        if isinstance(rules_engine_payload, dict):
            rules_payload = rules_engine_payload.get("rules")
            tuner_payload = rules_engine_payload.get("tuner")
            if isinstance(rules_payload, dict):
                self.rules_engine.save_rules(rules_payload)
            if isinstance(tuner_payload, dict):
                self.rules_engine.save_tuner(tuner_payload)

    def _build_action_learning_context(
        self,
        *,
        text_packet: dict[str, Any],
        query_units: list[str],
        final_focus_units: list[str],
    ) -> dict[str, Any]:
        normalized_text = str(text_packet.get("normalized_text", "") or "")
        context_keys: list[str] = []
        joined_units = "".join(str(unit or "") for unit in query_units if str(unit or ""))
        if joined_units:
            context_keys.append(f"text::{joined_units[:96]}")
        joined_focus = "".join(str(unit or "") for unit in final_focus_units if str(unit or ""))
        if joined_focus and f"focus_text::{joined_focus[:96]}" not in context_keys:
            context_keys.append(f"focus_text::{joined_focus[:96]}")
        for row in self.short_term.snapshot()[-2:]:
            focus_text = str(row.get("focus_text", "") or "")
            if focus_text:
                key = f"short_term::{focus_text[:96]}"
                if key not in context_keys:
                    context_keys.append(key)
        return {
            "normalized_text": normalized_text,
            "query_units": list(query_units),
            "focus_units": list(final_focus_units),
            "context_keys": context_keys[:8],
        }

