# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from sensors.text_sensor_v2 import normalize_text

RULES_SCHEMA_ID = "innate_rules_v2"
TUNER_SCHEMA_ID = "auto_tuner_v2"
SUPPORTED_SCHEMA_VERSION = "1.0"
CONDITION_METRICS = {
    "state.residual_count",
    "state.residual_mass",
    "state.prediction_match_count",
    "state.prediction_unexpected_count",
    "state.prediction_missed_count",
    "state.prediction_mismatch_mass",
    "state.prediction_match_mass",
    "state.prediction_overprediction_mass",
    "state.prediction_underprediction_mass",
    "state.prediction_missed_expected_mass",
    "state.prediction_unexpected_novelty_mass",
    "state.prediction_predicted_mass",
    "state.prediction_actual_mass",
    "state.prediction_alignment_score",
    "state.prediction_overprediction_ratio",
    "state.prediction_underprediction_ratio",
    "state.prediction_grasp_score",
    "state.prediction_committed_alignment_score",
    "state.prediction_committed_overprediction_ratio",
    "state.prediction_committed_grasp_score",
    "state.prediction_mismatch_balance",
    "state.top_energy",
    "state.total_top_energy",
    "state.entry_count",
    "state.anchor_count",
    "state.handle_count",
    "bn.count",
    "bn.top_score",
    "c_star.count",
    "c_star.top_energy",
    "metrics.logic_ms",
    "metrics.text_budget_used",
    "metrics.vision_budget_used",
    "metrics.audio_budget_used",
    "metrics.audio_window_count",
    "metrics.audio_memory_write_count",
    "metrics.audio_focus_priority_count",
    "metrics.audio_global_structure_count",
    "feedback.reward",
    "feedback.punishment",
    "feedback.external_reward",
    "feedback.external_punishment",
    "feedback.teacher_reward",
    "feedback.teacher_punishment",
    "feedback.intrinsic_reward",
    "feedback.intrinsic_punishment",
    "text.contains_cold",
    "text.contains_cool",
    "text.contains_winter",
    "text.contains_weather",
    "text.contains_open_app",
    "text.contains_notepad",
    "text.contains_calc",
    "text.contains_hello",
    "text.contains_exit",
    "emotion.dissonance",
    "emotion.surprise",
    "emotion.correctness",
    "emotion.expectation",
    "emotion.pressure",
    "emotion.expectation_minus_pressure",
    "emotion.correctness_minus_dissonance",
    "emotion.surprise_plus_dissonance",
    "modal.has_image",
    "modal.has_audio",
    "modal.audio_window_count",
    "modal.audio_memory_write_count",
    "modal.audio_focus_priority_count",
    "modal.audio_global_structure_count",
}
CONDITION_OPS = {">", ">=", "<", "<=", "==", "=", "!=",}
EMOTION_CHANNELS = {"dissonance", "surprise", "correctness", "expectation", "pressure", "grasp"}
EFFECT_TYPES = {"set_emotion_floor", "inject_sa", "add_action_drive", "append_rule_log"}
FORMULA_KINDS = {"constant", "metric", "mul", "affine", "max_metric", "threshold_excess"}
TUNER_TARGETS = {
    "attention.focus_gain",
    "sampling.increment_budget",
    "prediction.successor_bias_gain",
    "state.anchor_bias_gain",
    "state.current_input_gain",
    "state.history_suppression_gain",
    "state.prediction_suppression_gain",
    "state.surprise_focus_gain",
    "rules.dissonance_gain",
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _safe_id(text: str, fallback: str = "item") -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("_", "-", ".", ":") else "_" for ch in str(text or "").strip())
    clean = "_".join(part for part in clean.split("_") if part)
    return clean or fallback


def _float_or(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _warning(*, code: str, path: str, message: str, level: str = "warning") -> dict[str, Any]:
    return {
        "level": str(level or "warning"),
        "code": str(code or "warning"),
        "path": str(path or ""),
        "message": str(message or ""),
    }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class RuleStorePaths:
    rules_path: Path
    tuner_path: Path


def default_rule_store_paths(repo_root: Path) -> RuleStorePaths:
    root = Path(repo_root).resolve()
    return RuleStorePaths(
        rules_path=root / "config" / "innate_rules_v2.json",
        tuner_path=root / "config" / "auto_tuner_v2.json",
    )


def default_rules_payload() -> dict[str, Any]:
    packaged_path = Path(__file__).resolve().parents[1] / "config" / "innate_rules_v2.json"
    payload = _read_json(packaged_path, default=None)
    if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        return payload
    return {
        "schema_id": RULES_SCHEMA_ID,
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "rules": [],
    }


def default_tuner_payload() -> dict[str, Any]:
    return {
        "schema_id": TUNER_SCHEMA_ID,
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "enabled": True,
        "profiles": [
            {
                "profile_id": "baseline_default",
                "enabled": True,
                "display_name": "默认基线",
                "description": "稳定环境下的默认调参基线。",
                "when": [
                    {"metric": "metrics.logic_ms", "op": "<", "value": 150.0},
                ],
                "adjustments": [
                    {"target": "attention.focus_gain", "value": 1.35},
                    {"target": "sampling.increment_budget", "value": 48.0},
                    {"target": "prediction.successor_bias_gain", "value": 1.18},
                    {"target": "state.current_input_gain", "value": 1.0},
                    {"target": "state.history_suppression_gain", "value": 1.0},
                    {"target": "state.prediction_suppression_gain", "value": 1.0},
                    {"target": "state.surprise_focus_gain", "value": 1.0},
                ],
            },
            {
                "profile_id": "high_load_guard",
                "enabled": True,
                "display_name": "高负载保护",
                "description": "当单 tick 耗时变大时，收紧采样增量和注意力预算，防止性能失控。",
                "when": [
                    {"metric": "metrics.logic_ms", "op": ">=", "value": 150.0},
                ],
                "adjustments": [
                    {"target": "attention.focus_gain", "value": 1.05},
                    {"target": "sampling.increment_budget", "value": 32.0},
                    {"target": "prediction.successor_bias_gain", "value": 1.05},
                    {"target": "state.current_input_gain", "value": 0.9},
                    {"target": "state.history_suppression_gain", "value": 1.1},
                    {"target": "state.prediction_suppression_gain", "value": 1.15},
                    {"target": "state.surprise_focus_gain", "value": 1.0},
                ],
            },
        ],
    }


class RulesEngineV2:
    def __init__(self, *, repo_root: Path | str | None = None) -> None:
        root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
        self.repo_root = root
        self.paths = default_rule_store_paths(root)
        self._rules_payload = self._load_or_seed_rules()
        self._tuner_payload = self._load_or_seed_tuner()

    def export_rules(self) -> dict[str, Any]:
        return copy.deepcopy(self._rules_payload)

    def export_tuner(self) -> dict[str, Any]:
        return copy.deepcopy(self._tuner_payload)

    def validate_rules(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._validate_and_normalize_rules_payload(payload)
        return {
            "payload": copy.deepcopy(result["payload"]),
            "warnings": copy.deepcopy(result["warnings"]),
            "stats": copy.deepcopy(result["stats"]),
        }

    def validate_tuner(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._validate_and_normalize_tuner_payload(payload)
        return {
            "payload": copy.deepcopy(result["payload"]),
            "warnings": copy.deepcopy(result["warnings"]),
            "stats": copy.deepcopy(result["stats"]),
        }

    def save_rules(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._validate_and_normalize_rules_payload(payload)
        self._rules_payload = result["payload"]
        _write_json(self.paths.rules_path, self._rules_payload)
        return {
            "payload": self.export_rules(),
            "warnings": copy.deepcopy(result["warnings"]),
            "stats": copy.deepcopy(result["stats"]),
        }

    def save_tuner(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._validate_and_normalize_tuner_payload(payload)
        self._tuner_payload = result["payload"]
        _write_json(self.paths.tuner_path, self._tuner_payload)
        return {
            "payload": self.export_tuner(),
            "warnings": copy.deepcopy(result["warnings"]),
            "stats": copy.deepcopy(result["stats"]),
        }

    def evaluate(
        self,
        context: dict[str, Any],
        *,
        dissonance_gain: float = 1.0,
        emotion_channel_gains: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return self._evaluate_with_payloads(
            context,
            rules_payload=self._rules_payload,
            tuner_payload=self._tuner_payload,
            dissonance_gain=dissonance_gain,
            emotion_channel_gains=emotion_channel_gains,
        )

    def _evaluate_with_payloads(
        self,
        context: dict[str, Any],
        *,
        rules_payload: dict[str, Any],
        tuner_payload: dict[str, Any],
        dissonance_gain: float = 1.0,
        emotion_channel_gains: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        metrics = self._build_metrics(context)
        channel_gains = {
            str(key): max(0.0, float(value))
            for key, value in dict(emotion_channel_gains or {}).items()
        }
        emotion_channels = {
            "dissonance": 0.0,
            "surprise": 0.0,
            "correctness": 0.0,
            "expectation": 0.0,
            "pressure": 0.0,
            "grasp": 0.0,
        }
        injected_items: list[dict[str, Any]] = []
        action_drives: list[dict[str, Any]] = []
        rules_fired: list[dict[str, Any]] = []
        rule_logs: list[dict[str, Any]] = []

        rules = sorted(
            [self._normalize_rule(rule, fallback_index=index) for index, rule in enumerate(rules_payload.get("rules", []) or [])],
            key=lambda item: (-int(item.get("priority", 0) or 0), str(item.get("rule_id", "") or "")),
        )

        for rule in rules:
            if not bool(rule.get("enabled", True)):
                continue
            condition_results = self._evaluate_conditions(rule.get("conditions", []) or [], metrics)
            if not all(item["passed"] for item in condition_results):
                continue
            fired_entry = {
                "rule_id": str(rule.get("rule_id", "") or ""),
                "display_name": str(rule.get("display_name", "") or ""),
                "family": str(rule.get("family", "") or ""),
                "priority": int(rule.get("priority", 0) or 0),
                "condition_results": condition_results,
                "effects_applied": [],
            }
            for effect in rule.get("effects", []) or []:
                if not isinstance(effect, dict):
                    continue
                applied = self._apply_effect(
                    effect=effect,
                    metrics=metrics,
                    emotion_channels=emotion_channels,
                    injected_items=injected_items,
                    action_drives=action_drives,
                    rule_logs=rule_logs,
                    rule_id=str(rule.get("rule_id", "") or ""),
                    dissonance_gain=dissonance_gain,
                    emotion_channel_gains=channel_gains,
                )
                if applied:
                    fired_entry["effects_applied"].append(applied)
            fired_entry["score"] = _round4(max((float(item.get("score", 0.0) or 0.0) for item in fired_entry["effects_applied"]), default=0.0))
            rules_fired.append(fired_entry)
            metrics = self._refresh_derived_emotion_metrics(metrics, emotion_channels)

        tuner_result = self._evaluate_tuner(metrics, tuner_payload=tuner_payload)
        metrics = {**metrics, "tuner": tuner_result}
        return {
            "rules_fired": rules_fired,
            "injected_items": injected_items,
            "emotion_channels": {key: _round4(value) for key, value in emotion_channels.items()},
            "action_drives": self._merge_action_drives(action_drives),
            "rule_logs": rule_logs,
            "metrics_snapshot": {
                **metrics,
                **{f"emotion_gain.{key}": _round4(value) for key, value in channel_gains.items()},
            },
            "tuner_result": tuner_result,
        }

    def simulate(
        self,
        context: dict[str, Any],
        *,
        rules_payload: dict[str, Any] | None = None,
        tuner_payload: dict[str, Any] | None = None,
        dissonance_gain: float = 1.0,
    ) -> dict[str, Any]:
        sim_rules_payload = self._rules_payload
        sim_tuner_payload = self._tuner_payload
        if rules_payload is not None:
            sim_rules_payload = self._validate_and_normalize_rules_payload(rules_payload)["payload"]
        if tuner_payload is not None:
            sim_tuner_payload = self._validate_and_normalize_tuner_payload(tuner_payload)["payload"]
        return self._evaluate_with_payloads(
            context,
            rules_payload=sim_rules_payload,
            tuner_payload=sim_tuner_payload,
            dissonance_gain=dissonance_gain,
        )

    def _load_or_seed_rules(self) -> dict[str, Any]:
        payload = _read_json(self.paths.rules_path, default=None)
        if isinstance(payload, dict) and isinstance(payload.get("rules"), list):
            return payload
        payload = default_rules_payload()
        _write_json(self.paths.rules_path, payload)
        return payload

    def _load_or_seed_tuner(self) -> dict[str, Any]:
        payload = _read_json(self.paths.tuner_path, default=None)
        if isinstance(payload, dict) and isinstance(payload.get("profiles"), list):
            return payload
        payload = default_tuner_payload()
        _write_json(self.paths.tuner_path, payload)
        return payload

    def _build_metrics(self, context: dict[str, Any]) -> dict[str, float]:
        state_top = list(context.get("state_top", []) or [])
        state_pool_summary = dict(context.get("state_pool_summary", {}) or {})
        bn_list = list(context.get("bn_list", []) or [])
        c_star = dict(context.get("c_star", {}) or {})
        tick_index = int(context.get("tick_index", 0) or 0)
        runtime_metrics = dict(context.get("runtime_metrics", {}) or {})

        residual_summary = dict(state_pool_summary.get("residual_summary", {}) or {})
        anchor_summary = dict(state_pool_summary.get("anchor_summary", {}) or {})
        handle_summary = dict(state_pool_summary.get("handle_summary", {}) or {})
        prediction_trace = dict(state_pool_summary.get("prediction_trace", {}) or {})
        multimodal_summary = dict(context.get("multimodal_summary", {}) or {})
        sensor_packet = dict(context.get("sensor_packet", {}) or {})
        full_stream = dict(sensor_packet.get("full_stream", {}) or {})
        input_units = [str(item or "") for item in (full_stream.get("units", []) or []) if str(item or "")]
        unit_set = set(input_units)
        normalized_text = str(sensor_packet.get("normalized_text", "") or "")
        word_units = [part for part in normalize_text(normalized_text).split(" ") if part]
        dynamic_text_metrics: dict[str, float] = {}
        for unit in input_units:
            dynamic_text_metrics[f"text.unit::{unit}"] = 1.0
        for size in range(2, min(4, len(input_units)) + 1):
            for start in range(0, len(input_units) - size + 1):
                joined = "".join(input_units[start : start + size])
                dynamic_text_metrics[f"text.ngram::{joined}"] = 1.0
        for unit in word_units:
            dynamic_text_metrics[f"text.word::{unit}"] = 1.0
            dynamic_text_metrics[f"text.unit::{unit}"] = 1.0
        for size in range(2, min(4, len(word_units)) + 1):
            for start in range(0, len(word_units) - size + 1):
                joined = "".join(word_units[start : start + size])
                dynamic_text_metrics[f"text.ngram::{joined}"] = 1.0
                dynamic_text_metrics[f"text.word_ngram::{joined}"] = 1.0

        top_energy = max((float(item.get("energy", 0.0) or 0.0) for item in state_top), default=0.0)
        total_top_energy = sum(float(item.get("energy", 0.0) or 0.0) for item in state_top)
        bn_top_score = max((float(item.get("score", 0.0) or 0.0) for item in bn_list), default=0.0)
        c_star_items = list(c_star.get("items", []) or [])
        c_star_top_energy = max((float(item.get("energy", 0.0) or 0.0) for item in c_star_items), default=0.0)

        metrics = {
            "tick.index": float(tick_index),
            "state.top_energy": float(top_energy),
            "state.total_top_energy": float(total_top_energy),
            "state.entry_count": float(state_pool_summary.get("state_pool_size", 0) or 0),
            "state.anchor_count": float(anchor_summary.get("count", 0) or 0),
            "state.residual_count": float(residual_summary.get("count", 0) or 0),
            "state.residual_mass": float(residual_summary.get("total_unresolved_mass", 0.0) or 0.0),
            "state.prediction_match_count": float(prediction_trace.get("match_count", 0) or 0),
            "state.prediction_unexpected_count": float(prediction_trace.get("unexpected_count", 0) or 0),
            "state.prediction_missed_count": float(prediction_trace.get("missed_count", 0) or 0),
            "state.prediction_mismatch_mass": float(prediction_trace.get("mismatch_mass", 0.0) or 0.0),
            "state.prediction_match_mass": float(prediction_trace.get("match_mass", 0.0) or 0.0),
            "state.prediction_overprediction_mass": float(prediction_trace.get("overprediction_mass", 0.0) or 0.0),
            "state.prediction_underprediction_mass": float(prediction_trace.get("underprediction_mass", 0.0) or 0.0),
            "state.prediction_missed_expected_mass": float(prediction_trace.get("missed_expected_mass", 0.0) or 0.0),
            "state.prediction_unexpected_novelty_mass": float(prediction_trace.get("unexpected_novelty_mass", 0.0) or 0.0),
            "state.prediction_predicted_mass": float(prediction_trace.get("predicted_mass", 0.0) or 0.0),
            "state.prediction_actual_mass": float(prediction_trace.get("actual_mass", 0.0) or 0.0),
            "state.prediction_alignment_score": 0.0,
            "state.prediction_overprediction_ratio": 0.0,
            "state.prediction_underprediction_ratio": 0.0,
            "state.prediction_grasp_score": 0.0,
            "state.prediction_committed_alignment_score": 0.0,
            "state.prediction_committed_overprediction_ratio": 0.0,
            "state.prediction_committed_grasp_score": 0.0,
            "state.prediction_mismatch_balance": 0.0,
            "state.handle_count": float(handle_summary.get("count", 0) or 0),
            "bn.count": float(len(bn_list)),
            "bn.top_score": float(bn_top_score),
            "c_star.count": float(len(c_star_items)),
            "c_star.top_energy": float(c_star_top_energy),
            "metrics.logic_ms": float(runtime_metrics.get("logic_ms", 0.0) or 0.0),
            "metrics.text_budget_used": float(runtime_metrics.get("text_budget_used", 0.0) or 0.0),
            "metrics.vision_budget_used": float(runtime_metrics.get("vision_budget_used", 0.0) or 0.0),
            "metrics.vision_raw_sample_count": float(runtime_metrics.get("vision_raw_sample_count", 0.0) or 0.0),
            "metrics.audio_budget_used": float(runtime_metrics.get("audio_budget_used", 0.0) or 0.0),
            "metrics.audio_window_count": float(runtime_metrics.get("audio_window_count", 0.0) or 0.0),
            "metrics.audio_memory_write_count": float(runtime_metrics.get("audio_memory_write_count", 0.0) or 0.0),
            "metrics.audio_focus_priority_count": float(runtime_metrics.get("audio_focus_priority_count", 0.0) or 0.0),
            "metrics.audio_global_structure_count": float(runtime_metrics.get("audio_global_structure_count", 0.0) or 0.0),
            "feedback.reward": float(runtime_metrics.get("feedback_reward", 0.0) or 0.0),
            "feedback.punishment": float(runtime_metrics.get("feedback_punishment", 0.0) or 0.0),
            "feedback.external_reward": float(runtime_metrics.get("feedback_external_reward", 0.0) or 0.0),
            "feedback.external_punishment": float(runtime_metrics.get("feedback_external_punishment", 0.0) or 0.0),
            "feedback.teacher_reward": float(runtime_metrics.get("feedback_teacher_reward", 0.0) or 0.0),
            "feedback.teacher_punishment": float(runtime_metrics.get("feedback_teacher_punishment", 0.0) or 0.0),
            "feedback.intrinsic_reward": float(runtime_metrics.get("feedback_intrinsic_reward", 0.0) or 0.0),
            "feedback.intrinsic_punishment": float(runtime_metrics.get("feedback_intrinsic_punishment", 0.0) or 0.0),
            "text.contains_cold": 1.0 if ("冷" in unit_set or "cold" in unit_set) else 0.0,
            "text.contains_cool": 1.0 if ("凉" in unit_set or "cool" in unit_set) else 0.0,
            "text.contains_winter": 1.0 if ("冬" in unit_set or "天" in unit_set or "winter" in unit_set) else 0.0,
            "text.contains_weather": 1.0 if ("天气" in "".join(input_units) or "weather" in unit_set) else 0.0,
            "text.contains_open_app": 1.0 if ("打开" in "".join(input_units) or "open" in unit_set) else 0.0,
            "text.contains_notepad": 1.0 if ("记事本" in "".join(input_units) or "notepad" in unit_set) else 0.0,
            "text.contains_calc": 1.0 if ("计算器" in "".join(input_units) or "calc" in unit_set or "calculator" in unit_set) else 0.0,
            "text.contains_hello": 1.0 if ("你好" in "".join(input_units) or "hello" in unit_set) else 0.0,
            "text.contains_exit": 1.0 if ("退出" in "".join(input_units) or "exit" in unit_set) else 0.0,
            "modal.has_image": 1.0 if bool(multimodal_summary.get("has_image", False)) else 0.0,
            "modal.has_audio": 1.0 if bool(multimodal_summary.get("has_audio", False)) else 0.0,
            "modal.image_raw_sample_count": float(multimodal_summary.get("image_raw_sample_count", 0.0) or 0.0),
            "modal.image_memory_write_count": float(multimodal_summary.get("image_memory_write_count", 0.0) or 0.0),
            "modal.image_focus_priority_count": float(multimodal_summary.get("image_focus_priority_count", 0.0) or 0.0),
            "modal.audio_window_count": float(multimodal_summary.get("audio_window_count", 0.0) or 0.0),
            "modal.audio_memory_write_count": float(multimodal_summary.get("audio_memory_write_count", 0.0) or 0.0),
            "modal.audio_focus_priority_count": float(multimodal_summary.get("audio_focus_priority_count", 0.0) or 0.0),
            "modal.audio_global_structure_count": float(multimodal_summary.get("audio_global_structure_count", 0.0) or 0.0),
            "emotion.dissonance": 0.0,
            "emotion.surprise": 0.0,
            "emotion.correctness": 0.0,
            "emotion.expectation": 0.0,
            "emotion.pressure": 0.0,
            "emotion.expectation_minus_pressure": 0.0,
            "emotion.correctness_minus_dissonance": 0.0,
            "emotion.surprise_plus_dissonance": 0.0,
        }
        predicted_mass = float(metrics.get("state.prediction_predicted_mass", 0.0) or 0.0)
        actual_mass = float(metrics.get("state.prediction_actual_mass", 0.0) or 0.0)
        match_mass = float(metrics.get("state.prediction_match_mass", 0.0) or 0.0)
        over_mass = float(metrics.get("state.prediction_overprediction_mass", 0.0) or 0.0)
        under_mass = float(metrics.get("state.prediction_underprediction_mass", 0.0) or 0.0)
        committed_predicted_mass = float(prediction_trace.get("predicted_commitment_mass", 0.0) or 0.0)
        committed_match_mass = float(prediction_trace.get("committed_match_mass", 0.0) or 0.0)
        committed_over_mass = float(prediction_trace.get("committed_overprediction_mass", 0.0) or 0.0)
        align_den = max(1e-6, predicted_mass + actual_mass)
        predicted_den = max(1e-6, predicted_mass)
        actual_den = max(1e-6, actual_mass)
        alignment_score = _clamp((2.0 * match_mass) / align_den, 0.0, 1.0)
        over_ratio = _clamp(over_mass / predicted_den if predicted_mass > 0.0 else 0.0, 0.0, 4.0)
        under_ratio = _clamp(under_mass / actual_den if actual_mass > 0.0 else 0.0, 0.0, 4.0)
        grasp_score = _clamp(alignment_score - 0.72 * min(1.0, over_ratio) - 0.58 * min(1.0, under_ratio), 0.0, 1.0)
        committed_align_den = max(1e-6, committed_predicted_mass + actual_mass)
        committed_alignment_score = _clamp((2.0 * committed_match_mass) / committed_align_den, 0.0, 1.0)
        committed_over_ratio = _clamp(committed_over_mass / max(1e-6, committed_predicted_mass) if committed_predicted_mass > 0.0 else 0.0, 0.0, 4.0)
        committed_grasp_score = _clamp(committed_alignment_score - 0.82 * min(1.0, committed_over_ratio) - 0.42 * min(1.0, under_ratio), 0.0, 1.0)
        balance_den = max(1e-6, predicted_mass + actual_mass)
        mismatch_balance = _clamp((under_mass - over_mass) / balance_den, -1.0, 1.0)
        metrics["state.prediction_alignment_score"] = float(alignment_score)
        metrics["state.prediction_overprediction_ratio"] = float(over_ratio)
        metrics["state.prediction_underprediction_ratio"] = float(under_ratio)
        metrics["state.prediction_grasp_score"] = float(grasp_score)
        metrics["state.prediction_committed_alignment_score"] = float(committed_alignment_score)
        metrics["state.prediction_committed_overprediction_ratio"] = float(committed_over_ratio)
        metrics["state.prediction_committed_grasp_score"] = float(committed_grasp_score)
        metrics["state.prediction_mismatch_balance"] = float(mismatch_balance)
        metrics.update(dynamic_text_metrics)
        return metrics

    def _validate_and_normalize_rules_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("rules payload must be a dict")
        warnings: list[dict[str, Any]] = []
        schema_id = str(payload.get("schema_id", RULES_SCHEMA_ID) or RULES_SCHEMA_ID)
        if schema_id != RULES_SCHEMA_ID:
            warnings.append(_warning(code="schema_id_mismatch", path="schema_id", message=f"收到 {schema_id}，已按 {RULES_SCHEMA_ID} 处理。"))
        raw_rules = payload.get("rules", [])
        if raw_rules is None:
            raw_rules = []
        if not isinstance(raw_rules, list):
            raise ValueError("rules payload.rules must be a list")

        normalized: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}
        for index, rule in enumerate(raw_rules):
            if not isinstance(rule, dict):
                warnings.append(_warning(code="rule_item_skipped", path=f"rules[{index}]", message="该规则不是对象，已跳过。"))
                continue
            normalized.append(self._normalize_rule(rule, fallback_index=index, warnings=warnings, path=f"rules[{index}]", seen_ids=seen_ids))

        result = {
            "schema_id": RULES_SCHEMA_ID,
            "schema_version": str(payload.get("schema_version", SUPPORTED_SCHEMA_VERSION) or SUPPORTED_SCHEMA_VERSION),
            "rules": normalized,
        }
        stats = {
            "rule_count": len(normalized),
            "enabled_rule_count": sum(1 for item in normalized if bool(item.get("enabled", True))),
            "condition_count": sum(len(item.get("conditions", []) or []) for item in normalized),
            "effect_count": sum(len(item.get("effects", []) or []) for item in normalized),
            "always_on_rule_count": sum(1 for item in normalized if not list(item.get("conditions", []) or [])),
            "noop_rule_count": sum(1 for item in normalized if not list(item.get("effects", []) or [])),
        }
        if not normalized:
            warnings.append(_warning(code="empty_ruleset", path="rules", message="规则列表为空，系统将不会触发任何先天规则。"))
        return {"payload": result, "warnings": warnings, "stats": stats}

    def _validate_and_normalize_tuner_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("tuner payload must be a dict")
        warnings: list[dict[str, Any]] = []
        schema_id = str(payload.get("schema_id", TUNER_SCHEMA_ID) or TUNER_SCHEMA_ID)
        if schema_id != TUNER_SCHEMA_ID:
            warnings.append(_warning(code="schema_id_mismatch", path="schema_id", message=f"收到 {schema_id}，已按 {TUNER_SCHEMA_ID} 处理。"))
        raw_profiles = payload.get("profiles", [])
        if raw_profiles is None:
            raw_profiles = []
        if not isinstance(raw_profiles, list):
            raise ValueError("tuner payload.profiles must be a list")

        normalized_profiles: list[dict[str, Any]] = []
        seen_ids: dict[str, int] = {}
        for index, profile in enumerate(raw_profiles):
            if not isinstance(profile, dict):
                warnings.append(_warning(code="profile_item_skipped", path=f"profiles[{index}]", message="该调参档不是对象，已跳过。"))
                continue
            normalized_profiles.append(self._normalize_profile(profile, fallback_index=index, warnings=warnings, path=f"profiles[{index}]", seen_ids=seen_ids))

        result = {
            "schema_id": TUNER_SCHEMA_ID,
            "schema_version": str(payload.get("schema_version", SUPPORTED_SCHEMA_VERSION) or SUPPORTED_SCHEMA_VERSION),
            "enabled": bool(payload.get("enabled", True)),
            "profiles": normalized_profiles,
        }
        stats = {
            "profile_count": len(normalized_profiles),
            "enabled_profile_count": sum(1 for item in normalized_profiles if bool(item.get("enabled", True))),
            "when_count": sum(len(item.get("when", []) or []) for item in normalized_profiles),
            "adjustment_count": sum(len(item.get("adjustments", []) or []) for item in normalized_profiles),
            "always_on_profile_count": sum(1 for item in normalized_profiles if not list(item.get("when", []) or [])),
            "empty_adjustment_profile_count": sum(1 for item in normalized_profiles if not list(item.get("adjustments", []) or [])),
        }
        if not normalized_profiles:
            warnings.append(_warning(code="empty_tuner", path="profiles", message="调参档列表为空，自动调参将不会生效。"))
        return {"payload": result, "warnings": warnings, "stats": stats}

    def _normalize_rule(
        self,
        rule: dict[str, Any],
        *,
        fallback_index: int,
        warnings: list[dict[str, Any]] | None = None,
        path: str = "",
        seen_ids: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        warning_list = warnings if warnings is not None else []
        base_rule_id = str(rule.get("rule_id", "") or "")
        if not base_rule_id:
            warning_list.append(_warning(code="rule_id_missing", path=f"{path}.rule_id", message="rule_id 为空，已自动补全。"))
        rule_id = self._make_unique_id(
            _safe_id(base_rule_id, fallback=f"rule::{fallback_index:03d}"),
            seen_ids=seen_ids,
            path=f"{path}.rule_id",
            warnings=warning_list,
            duplicate_code="duplicate_rule_id",
            duplicate_message="rule_id 重复，已自动改名为 {value}。",
        )
        display_name = str(rule.get("display_name", "") or "").strip()
        if not display_name:
            display_name = f"规则 {fallback_index + 1}"
            warning_list.append(_warning(code="rule_display_name_missing", path=f"{path}.display_name", message=f"display_name 为空，已补成 {display_name}。"))

        conditions = self._normalize_conditions(rule.get("conditions", []) or [], warnings=warning_list, path=f"{path}.conditions")
        effects = self._normalize_effects(rule.get("effects", []) or [], warnings=warning_list, path=f"{path}.effects")
        if not conditions:
            warning_list.append(_warning(code="rule_always_on", path=f"{path}.conditions", message=f"{rule_id} 没有条件，将视为总是可触发。"))
        if not effects:
            warning_list.append(_warning(code="rule_no_effect", path=f"{path}.effects", message=f"{rule_id} 没有效果，触发后不会产生变化。"))
        return {
            "rule_id": rule_id,
            "enabled": bool(rule.get("enabled", True)),
            "priority": int(_float_or(rule.get("priority", 0), 0.0)),
            "display_name": display_name,
            "family": str(rule.get("family", "generic") or "generic"),
            "description": str(rule.get("description", "") or ""),
            "conditions": conditions,
            "effects": effects,
        }

    def _normalize_profile(
        self,
        profile: dict[str, Any],
        *,
        fallback_index: int,
        warnings: list[dict[str, Any]] | None = None,
        path: str = "",
        seen_ids: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        warning_list = warnings if warnings is not None else []
        base_profile_id = str(profile.get("profile_id", "") or "")
        if not base_profile_id:
            warning_list.append(_warning(code="profile_id_missing", path=f"{path}.profile_id", message="profile_id 为空，已自动补全。"))
        profile_id = self._make_unique_id(
            _safe_id(base_profile_id, fallback=f"profile::{fallback_index:03d}"),
            seen_ids=seen_ids,
            path=f"{path}.profile_id",
            warnings=warning_list,
            duplicate_code="duplicate_profile_id",
            duplicate_message="profile_id 重复，已自动改名为 {value}。",
        )
        display_name = str(profile.get("display_name", "") or "").strip()
        if not display_name:
            display_name = f"调参档 {fallback_index + 1}"
            warning_list.append(_warning(code="profile_display_name_missing", path=f"{path}.display_name", message=f"display_name 为空，已补成 {display_name}。"))

        when = self._normalize_conditions(profile.get("when", []) or [], warnings=warning_list, path=f"{path}.when")
        adjustments = self._normalize_adjustments(profile.get("adjustments", []) or [], warnings=warning_list, path=f"{path}.adjustments")
        if not when:
            warning_list.append(_warning(code="profile_always_on", path=f"{path}.when", message=f"{profile_id} 没有条件，将视为总是命中。"))
        if not adjustments:
            warning_list.append(_warning(code="profile_no_adjustment", path=f"{path}.adjustments", message=f"{profile_id} 没有调参项，命中后不会改变参数。"))
        return {
            "profile_id": profile_id,
            "enabled": bool(profile.get("enabled", True)),
            "display_name": display_name,
            "description": str(profile.get("description", "") or ""),
            "when": when,
            "adjustments": adjustments,
        }

    def _make_unique_id(
        self,
        base_id: str,
        *,
        seen_ids: dict[str, int] | None,
        path: str,
        warnings: list[dict[str, Any]],
        duplicate_code: str,
        duplicate_message: str,
    ) -> str:
        if seen_ids is None:
            return base_id
        count = int(seen_ids.get(base_id, 0) or 0)
        if count <= 0:
            seen_ids[base_id] = 1
            return base_id
        seen_ids[base_id] = count + 1
        unique_id = f"{base_id}__dup{count + 1}"
        while unique_id in seen_ids:
            seen_ids[base_id] += 1
            unique_id = f"{base_id}__dup{seen_ids[base_id]}"
        seen_ids[unique_id] = 1
        warnings.append(_warning(code=duplicate_code, path=path, message=duplicate_message.format(value=unique_id)))
        return unique_id

    def _normalize_conditions(self, conditions: Any, *, warnings: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
        if not isinstance(conditions, list):
            warnings.append(_warning(code="conditions_not_list", path=path, message="条件列表不是数组，已按空列表处理。"))
            return []
        result: list[dict[str, Any]] = []
        for index, condition in enumerate(conditions):
            item_path = f"{path}[{index}]"
            if not isinstance(condition, dict):
                warnings.append(_warning(code="condition_item_skipped", path=item_path, message="该条件不是对象，已跳过。"))
                continue
            metric = str(condition.get("metric", "") or "").strip()
            if not metric:
                metric = "state.top_energy"
                warnings.append(_warning(code="condition_metric_missing", path=f"{item_path}.metric", message="metric 为空，已改为 state.top_energy。"))
            elif metric not in CONDITION_METRICS:
                warnings.append(_warning(code="condition_metric_unknown", path=f"{item_path}.metric", message=f"{metric} 当前不在内建指标列表中，运行时会按实际暴露指标解释。"))
            op = str(condition.get("op", ">") or ">").strip()
            if op not in CONDITION_OPS:
                warnings.append(_warning(code="condition_op_invalid", path=f"{item_path}.op", message=f"比较符 {op} 无效，已改为 >。"))
                op = ">"
            raw_value = condition.get("value", 0.0)
            try:
                value = float(raw_value)
            except Exception:
                warnings.append(_warning(code="condition_value_invalid", path=f"{item_path}.value", message=f"value={raw_value!r} 不是数字，已改为 0。"))
                value = 0.0
            result.append({"metric": metric, "op": op, "value": value})
        return result

    def _normalize_effects(self, effects: Any, *, warnings: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
        if not isinstance(effects, list):
            warnings.append(_warning(code="effects_not_list", path=path, message="效果列表不是数组，已按空列表处理。"))
            return []
        result: list[dict[str, Any]] = []
        for index, effect in enumerate(effects):
            item_path = f"{path}[{index}]"
            if not isinstance(effect, dict):
                warnings.append(_warning(code="effect_item_skipped", path=item_path, message="该效果不是对象，已跳过。"))
                continue
            effect_type = str(effect.get("type", "") or "").strip()
            if effect_type not in EFFECT_TYPES:
                warnings.append(_warning(code="effect_type_invalid", path=f"{item_path}.type", message=f"effect type={effect_type or '[empty]'} 无效，已跳过该效果。"))
                continue
            row = {"type": effect_type}
            if effect_type == "set_emotion_floor":
                channel = str(effect.get("channel", "") or "").strip()
                if channel not in EMOTION_CHANNELS:
                    warnings.append(_warning(code="effect_channel_invalid", path=f"{item_path}.channel", message=f"channel={channel or '[empty]'} 无效，已跳过该效果。"))
                    continue
                row["channel"] = channel
                row["formula"] = self._normalize_formula(effect.get("formula", {}), warnings=warnings, path=f"{item_path}.formula")
            elif effect_type == "inject_sa":
                row["sa_label"] = str(effect.get("sa_label", "") or "").strip()
                row["display_text"] = str(effect.get("display_text", "") or "").strip()
                row["when_channel"] = str(effect.get("when_channel", "") or "").strip()
                row["threshold"] = _float_or(effect.get("threshold", 0.0), 0.0)
                row["formula"] = self._normalize_formula(effect.get("formula", {}), warnings=warnings, path=f"{item_path}.formula")
                if not row["sa_label"]:
                    warnings.append(_warning(code="inject_sa_missing_label", path=f"{item_path}.sa_label", message="inject_sa 缺少 sa_label，运行时会被忽略。"))
                if row["when_channel"] and row["when_channel"] not in EMOTION_CHANNELS:
                    warnings.append(_warning(code="inject_sa_when_channel_invalid", path=f"{item_path}.when_channel", message=f"when_channel={row['when_channel']} 不在已知情绪通道中，运行时可能永远不触发。"))
            elif effect_type == "add_action_drive":
                row["action_id"] = str(effect.get("action_id", "") or "").strip()
                row["reason"] = str(effect.get("reason", "") or "").strip()
                row["params"] = dict(effect.get("params", {}) or {}) if isinstance(effect.get("params"), dict) else {}
                row["formula"] = self._normalize_formula(effect.get("formula", {}), warnings=warnings, path=f"{item_path}.formula")
                if not row["action_id"]:
                    warnings.append(_warning(code="action_drive_missing_action_id", path=f"{item_path}.action_id", message="add_action_drive 缺少 action_id，运行时会被忽略。"))
            elif effect_type == "append_rule_log":
                row["message"] = str(effect.get("message", "") or "").strip()
                if not row["message"]:
                    warnings.append(_warning(code="rule_log_message_missing", path=f"{item_path}.message", message="append_rule_log 缺少 message，运行时不会产生日志。"))
            result.append(row)
        return result

    def _normalize_formula(self, formula: Any, *, warnings: list[dict[str, Any]], path: str) -> dict[str, Any]:
        if not isinstance(formula, dict):
            warnings.append(_warning(code="formula_not_object", path=path, message="公式不是对象，已改为 constant=0。"))
            return {"kind": "constant", "value": 0.0}
        kind = str(formula.get("kind", "constant") or "constant").strip()
        if kind not in FORMULA_KINDS:
            warnings.append(_warning(code="formula_kind_invalid", path=f"{path}.kind", message=f"formula kind={kind or '[empty]'} 无效，已改为 constant=0。"))
            return {"kind": "constant", "value": 0.0}
        result: dict[str, Any] = {"kind": kind}
        if kind == "constant":
            result["value"] = _float_or(formula.get("value", 0.0), 0.0)
        elif kind == "metric":
            metric_name = str(formula.get("metric", "") or "").strip()
            if not metric_name:
                warnings.append(_warning(code="formula_metric_missing", path=f"{path}.metric", message="metric 为空，结果将按 0 处理。"))
            elif metric_name not in CONDITION_METRICS and not metric_name.startswith("emotion."):
                warnings.append(_warning(code="formula_metric_unknown", path=f"{path}.metric", message=f"{metric_name} 当前不在内建指标列表中。"))
            result["metric"] = metric_name
        elif kind == "mul":
            metric_name = str(formula.get("metric", "") or "").strip()
            if not metric_name:
                warnings.append(_warning(code="formula_metric_missing", path=f"{path}.metric", message="metric 为空，结果将按 0 处理。"))
            elif metric_name not in CONDITION_METRICS and not metric_name.startswith("emotion."):
                warnings.append(_warning(code="formula_metric_unknown", path=f"{path}.metric", message=f"{metric_name} 当前不在内建指标列表中。"))
            result["metric"] = metric_name
            result["factor"] = _float_or(formula.get("factor", 1.0), 1.0)
        elif kind == "affine":
            metric_name = str(formula.get("metric", "") or "").strip()
            if not metric_name:
                warnings.append(_warning(code="formula_metric_missing", path=f"{path}.metric", message="metric 为空，结果将按 base 处理。"))
            elif metric_name not in CONDITION_METRICS and not metric_name.startswith("emotion."):
                warnings.append(_warning(code="formula_metric_unknown", path=f"{path}.metric", message=f"{metric_name} 当前不在内建指标列表中。"))
            result["metric"] = metric_name
            result["base"] = _float_or(formula.get("base", 0.0), 0.0)
            result["factor"] = _float_or(formula.get("factor", 1.0), 1.0)
        elif kind == "max_metric":
            metrics = [str(item or "").strip() for item in (formula.get("metrics", []) or []) if str(item or "").strip()]
            if not metrics:
                warnings.append(_warning(code="formula_metrics_missing", path=f"{path}.metrics", message="metrics 为空，结果将按 0 处理。"))
            for metric_name in metrics:
                if metric_name not in CONDITION_METRICS and not metric_name.startswith("emotion."):
                    warnings.append(_warning(code="formula_metric_unknown", path=f"{path}.metrics", message=f"{metric_name} 当前不在内建指标列表中。"))
            result["metrics"] = metrics
        minimum = formula.get("min")
        maximum = formula.get("max")
        if minimum is not None:
            result["min"] = _float_or(minimum, 0.0)
        if maximum is not None:
            result["max"] = _float_or(maximum, 0.0)
        if "min" in result and "max" in result and float(result["min"]) > float(result["max"]):
            result["min"], result["max"] = result["max"], result["min"]
            warnings.append(_warning(code="formula_min_max_swapped", path=path, message="公式中的 min 大于 max，已自动交换。"))
        return result

    def _normalize_adjustments(self, adjustments: Any, *, warnings: list[dict[str, Any]], path: str) -> list[dict[str, Any]]:
        if not isinstance(adjustments, list):
            warnings.append(_warning(code="adjustments_not_list", path=path, message="调参项不是数组，已按空列表处理。"))
            return []
        result: list[dict[str, Any]] = []
        for index, adjustment in enumerate(adjustments):
            item_path = f"{path}[{index}]"
            if not isinstance(adjustment, dict):
                warnings.append(_warning(code="adjustment_item_skipped", path=item_path, message="该调参项不是对象，已跳过。"))
                continue
            target = str(adjustment.get("target", "") or "").strip()
            if not target:
                target = "attention.focus_gain"
                warnings.append(_warning(code="adjustment_target_missing", path=f"{item_path}.target", message="target 为空，已改为 attention.focus_gain。"))
            elif target not in TUNER_TARGETS:
                warnings.append(_warning(code="adjustment_target_unknown", path=f"{item_path}.target", message=f"{target} 不在当前已知调参目标白名单中，但会被保留。"))
            raw_value = adjustment.get("value", 0.0)
            try:
                value = float(raw_value)
            except Exception:
                warnings.append(_warning(code="adjustment_value_invalid", path=f"{item_path}.value", message=f"value={raw_value!r} 不是数字，已改为 0。"))
                value = 0.0
            result.append({"target": target, "value": value})
        return result

    def _evaluate_conditions(self, conditions: list[dict[str, Any]], metrics: dict[str, float]) -> list[dict[str, Any]]:
        results = []
        for condition in conditions:
            metric_name = str(condition.get("metric", "") or "")
            op = str(condition.get("op", ">") or ">")
            threshold = float(condition.get("value", 0.0) or 0.0)
            current = float(metrics.get(metric_name, 0.0) or 0.0)
            passed = self._compare(current=current, op=op, threshold=threshold)
            results.append(
                {
                    "metric": metric_name,
                    "op": op,
                    "threshold": _round4(threshold),
                    "current": _round4(current),
                    "passed": bool(passed),
                }
            )
        return results

    def _compare(self, *, current: float, op: str, threshold: float) -> bool:
        if op == ">":
            return current > threshold
        if op == ">=":
            return current >= threshold
        if op == "<":
            return current < threshold
        if op == "<=":
            return current <= threshold
        if op in ("==", "="):
            return abs(current - threshold) <= 1e-9
        if op == "!=":
            return abs(current - threshold) > 1e-9
        return False

    def _apply_effect(
        self,
        *,
        effect: dict[str, Any],
        metrics: dict[str, float],
        emotion_channels: dict[str, float],
        injected_items: list[dict[str, Any]],
        action_drives: list[dict[str, Any]],
        rule_logs: list[dict[str, Any]],
        rule_id: str,
        dissonance_gain: float,
        emotion_channel_gains: dict[str, float],
    ) -> dict[str, Any] | None:
        effect_type = str(effect.get("type", "") or "")
        if effect_type == "set_emotion_floor":
            channel = str(effect.get("channel", "") or "")
            if channel not in emotion_channels:
                return None
            value = self._evaluate_formula(effect.get("formula", {}), metrics)
            if channel == "dissonance":
                value *= max(0.0, float(dissonance_gain))
            value *= max(0.0, float(emotion_channel_gains.get(channel, 1.0) or 1.0))
            emotion_channels[channel] = max(float(emotion_channels.get(channel, 0.0) or 0.0), _clamp(value, 0.0, 1.0))
            return {"type": effect_type, "channel": channel, "score": _round4(emotion_channels[channel])}
        if effect_type == "inject_sa":
            channel = str(effect.get("when_channel", "") or "")
            threshold = float(effect.get("threshold", 0.0) or 0.0)
            channel_value = float(emotion_channels.get(channel, 0.0) or 0.0)
            if channel and channel_value < threshold:
                return None
            energy = _clamp(self._evaluate_formula(effect.get("formula", {}), {**metrics, **{f"emotion.{key}": value for key, value in emotion_channels.items()}}), 0.0, 1.0)
            sa_label = str(effect.get("sa_label", "") or "")
            if not sa_label or energy <= 0.0:
                return None
            injected_items.append(
                {
                    "sa_label": sa_label,
                    "display_text": str(effect.get("display_text", "") or sa_label),
                    "energy": _round4(energy),
                }
            )
            return {"type": effect_type, "sa_label": sa_label, "score": _round4(energy)}
        if effect_type == "add_action_drive":
            action_id = str(effect.get("action_id", "") or "")
            drive = _clamp(self._evaluate_formula(effect.get("formula", {}), {**metrics, **{f"emotion.{key}": value for key, value in emotion_channels.items()}}), 0.0, 1.0)
            if not action_id or drive <= 0.0:
                return None
            action_drives.append(
                {
                    "action_id": action_id,
                    "drive": _round4(drive),
                    "reason": str(effect.get("reason", "") or rule_id),
                    "params": dict(effect.get("params", {}) or {}) if isinstance(effect.get("params"), dict) else {},
                }
            )
            return {"type": effect_type, "action_id": action_id, "score": _round4(drive)}
        if effect_type == "append_rule_log":
            message = str(effect.get("message", "") or "")
            if not message:
                return None
            rule_logs.append({"rule_id": rule_id, "message": message})
            return {"type": effect_type, "score": 0.0, "message": message}
        return None

    def _evaluate_formula(self, formula: dict[str, Any], metrics: dict[str, float]) -> float:
        if not isinstance(formula, dict):
            return 0.0
        kind = str(formula.get("kind", "constant") or "constant")
        if kind == "constant":
            value = float(formula.get("value", 0.0) or 0.0)
        elif kind == "metric":
            value = float(metrics.get(str(formula.get("metric", "") or ""), 0.0) or 0.0)
        elif kind == "mul":
            metric_name = str(formula.get("metric", "") or "")
            factor = float(formula.get("factor", 1.0) or 1.0)
            value = float(metrics.get(metric_name, 0.0) or 0.0) * factor
        elif kind == "affine":
            base = float(formula.get("base", 0.0) or 0.0)
            metric_name = str(formula.get("metric", "") or "")
            factor = float(formula.get("factor", 1.0) or 1.0)
            value = base + float(metrics.get(metric_name, 0.0) or 0.0) * factor
        elif kind == "threshold_excess":
            metric_name = str(formula.get("metric", "") or "")
            threshold = float(formula.get("threshold", 0.0) or 0.0)
            factor = float(formula.get("factor", 1.0) or 1.0)
            base = float(formula.get("base", 0.0) or 0.0)
            metric_value = float(metrics.get(metric_name, 0.0) or 0.0)
            value = base + max(0.0, metric_value - threshold) * factor
        elif kind == "max_metric":
            metric_names = [str(item or "") for item in (formula.get("metrics", []) or []) if str(item or "")]
            value = max((float(metrics.get(name, 0.0) or 0.0) for name in metric_names), default=0.0)
        else:
            value = 0.0
        minimum = formula.get("min")
        maximum = formula.get("max")
        if minimum is not None or maximum is not None:
            low = float(minimum if minimum is not None else value)
            high = float(maximum if maximum is not None else value)
            value = _clamp(value, low, high)
        return float(value)

    def _refresh_derived_emotion_metrics(self, metrics: dict[str, float], emotion_channels: dict[str, float]) -> dict[str, float]:
        next_metrics = dict(metrics)
        for key, value in emotion_channels.items():
            next_metrics[f"emotion.{key}"] = float(value)
        next_metrics["emotion.expectation_minus_pressure"] = float(emotion_channels.get("expectation", 0.0) or 0.0) - float(emotion_channels.get("pressure", 0.0) or 0.0)
        next_metrics["emotion.correctness_minus_dissonance"] = float(emotion_channels.get("correctness", 0.0) or 0.0) - float(emotion_channels.get("dissonance", 0.0) or 0.0)
        next_metrics["emotion.surprise_plus_dissonance"] = float(emotion_channels.get("surprise", 0.0) or 0.0) + float(emotion_channels.get("dissonance", 0.0) or 0.0)
        return next_metrics

    def _merge_action_drives(self, action_drives: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for row in action_drives:
            action_id = str(row.get("action_id", "") or "")
            if not action_id:
                continue
            params = dict(row.get("params", {}) or {}) if isinstance(row.get("params"), dict) else {}
            merge_key = f"{action_id}||{json.dumps(params, ensure_ascii=False, sort_keys=True, separators=(',', ':'))}"
            entry = merged.get(merge_key)
            drive = float(row.get("drive", 0.0) or 0.0)
            if entry is None:
                merged[merge_key] = {
                    "action_id": action_id,
                    "drive": drive,
                    "reason": str(row.get("reason", "") or ""),
                    "params": params,
                }
            else:
                if drive > float(entry.get("drive", 0.0) or 0.0):
                    entry["drive"] = drive
                    entry["reason"] = str(row.get("reason", "") or entry.get("reason", "") or "")
                    entry["params"] = params
        rows = list(merged.values())
        rows.sort(key=lambda item: (-float(item.get("drive", 0.0) or 0.0), item.get("action_id", "")))
        return [
            {
                "action_id": row["action_id"],
                "drive": _round4(row["drive"]),
                "reason": row["reason"],
                "params": dict(row.get("params", {}) or {}) if isinstance(row.get("params"), dict) else {},
            }
            for row in rows
        ]

    def _evaluate_tuner(self, metrics: dict[str, float], *, tuner_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = tuner_payload if isinstance(tuner_payload, dict) else self._tuner_payload
        if not bool(payload.get("enabled", True)):
            return {"enabled": False, "matched_profiles": [], "adjustments": []}
        matched_profiles: list[dict[str, Any]] = []
        adjustments: list[dict[str, Any]] = []
        for index, raw_profile in enumerate(payload.get("profiles", []) or []):
            if not isinstance(raw_profile, dict):
                continue
            profile = self._normalize_profile(raw_profile, fallback_index=index)
            if not bool(profile.get("enabled", True)):
                continue
            condition_results = self._evaluate_conditions(profile.get("when", []) or [], metrics)
            if not all(item["passed"] for item in condition_results):
                continue
            matched_profiles.append(
                {
                    "profile_id": profile["profile_id"],
                    "display_name": profile["display_name"],
                    "condition_results": condition_results,
                }
            )
            for adjustment in profile.get("adjustments", []) or []:
                if not isinstance(adjustment, dict):
                    continue
                target = str(adjustment.get("target", "") or "")
                if not target:
                    continue
                adjustments.append(
                    {
                        "target": target,
                        "value": float(adjustment.get("value", 0.0) or 0.0),
                        "profile_id": profile["profile_id"],
                    }
                )
        return {"enabled": True, "matched_profiles": matched_profiles, "adjustments": adjustments}

