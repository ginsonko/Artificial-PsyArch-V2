from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class AdaptiveTuner:
    """
    Conservative long-horizon tuner.

    It does not mutate RuntimeConfig directly. It observes tick summaries and
    emits bounded modulation suggestions that other modules may consume.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        ema_alpha: float = 0.04,
        min_support_ticks: int = 12,
        target_prediction_alignment: float = 0.58,
        max_normal_pressure: float = 3.5,
        target_action_success: float = 0.52,
        adjustment_rate: float = 0.025,
        rollback_threshold: float = 0.18,
    ) -> None:
        self.enabled = bool(enabled)
        self.ema_alpha = _clamp(float(ema_alpha), 0.005, 0.25)
        self.min_support_ticks = max(1, int(min_support_ticks))
        self.target_prediction_alignment = _clamp(float(target_prediction_alignment), 0.0, 1.0)
        self.max_normal_pressure = max(0.1, float(max_normal_pressure))
        self.target_action_success = _clamp(float(target_action_success), 0.0, 1.0)
        self.adjustment_rate = _clamp(float(adjustment_rate), 0.001, 0.2)
        self.rollback_threshold = _clamp(float(rollback_threshold), 0.01, 1.0)
        self._tick_count = 0
        self._metrics: dict[str, float] = {}
        self._modulation = self._neutral_modulation()
        self._history: list[dict] = []
        self._active_experiment: dict | None = None
        self._last_experiment_decision_tick = -1

    def observe_tick(self, trace: dict) -> dict:
        self._tick_count += 1
        sample = self._extract_sample(trace)
        for key, value in sample.items():
            old = float(self._metrics.get(key, value) or 0.0)
            self._metrics[key] = old * (1.0 - self.ema_alpha) + float(value) * self.ema_alpha
        self._evaluate_active_experiment()
        recommendation = self._recommend()
        just_decided_experiment = int(self._last_experiment_decision_tick) == int(self._tick_count)
        if self.enabled and self._tick_count >= self.min_support_ticks and not just_decided_experiment:
            self._apply_recommendation(recommendation)
        snapshot = self.snapshot()
        snapshot["last_sample"] = {key: _round4(value) for key, value in sample.items()}
        snapshot["recommendation"] = recommendation
        return snapshot

    def modulation(self) -> dict:
        return {
            "schema_id": "adaptive_tuner_modulation/v1",
            "enabled": bool(self.enabled),
            "support_ready": self._tick_count >= self.min_support_ticks,
            "values": {
                section: {key: _round4(value) for key, value in dict(values).items()}
                for section, values in self._modulation.items()
            },
        }

    def snapshot(self) -> dict:
        return {
            "schema_id": "adaptive_tuner_trace/v1",
            "enabled": bool(self.enabled),
            "tick_count": int(self._tick_count),
            "support_ready": self._tick_count >= self.min_support_ticks,
            "policy": {
                "ema_alpha": _round4(self.ema_alpha),
                "min_support_ticks": int(self.min_support_ticks),
                "target_prediction_alignment": _round4(self.target_prediction_alignment),
                "max_normal_pressure": _round4(self.max_normal_pressure),
                "target_action_success": _round4(self.target_action_success),
                "adjustment_rate": _round4(self.adjustment_rate),
                "rollback_threshold": _round4(self.rollback_threshold),
            },
            "metrics": {key: _round4(value) for key, value in sorted(self._metrics.items())},
            "modulation": self.modulation(),
            "experiment": self._experiment_trace(),
            "history": list(self._history[-12:]),
        }

    def _extract_sample(self, trace: dict) -> dict[str, float]:
        state_pool = dict((trace or {}).get("state_pool", {}) or {})
        snapshot = dict(state_pool.get("snapshot", {}) or {})
        items = list(snapshot.get("items", []) or [])
        total_pressure = 0.0
        total_abs_pressure = 0.0
        pressure_count = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            real = float(item.get("real_energy", 0.0) or 0.0)
            virtual = float(item.get("virtual_energy", 0.0) or 0.0)
            pressure = float(item.get("cognitive_pressure", real - virtual) or 0.0)
            total_pressure += pressure
            total_abs_pressure += abs(pressure)
            pressure_count += 1
        prediction_trace = dict(state_pool.get("prediction_trace", {}) or {})
        action = dict((trace or {}).get("action", {}) or {})
        selected = list(action.get("selected_actions", []) or [])
        success_scores = []
        punishment_scores = []
        for row in selected:
            predicted = dict((row or {}).get("predicted_outcome", {}) or {})
            reward = float(predicted.get("reward", 0.0) or 0.0)
            correctness = float(predicted.get("correctness", 0.0) or 0.0)
            punishment = float(predicted.get("punishment", 0.0) or 0.0)
            pressure = float(predicted.get("pressure", 0.0) or 0.0)
            success_scores.append(_clamp(reward + correctness * 0.45 - punishment * 0.65 - pressure * 0.25, 0.0, 1.0))
            punishment_scores.append(_clamp(punishment + pressure * 0.5, 0.0, 1.0))
        observed_feedback = dict(((trace or {}).get("action_feedback", {}) or {}).get("observed_feedback", {}) or {})
        feedback_reward = float(observed_feedback.get("reward", 0.0) or 0.0)
        feedback_punishment = float(observed_feedback.get("punishment", 0.0) or 0.0)
        feedback_correctness = float(observed_feedback.get("correctness", 0.0) or 0.0)
        feedback_success = _clamp(feedback_reward + feedback_correctness * 0.45 - feedback_punishment * 0.7, 0.0, 1.0)
        expectation_pressure = dict((trace or {}).get("expectation_pressure", {}) or {})
        anchor_trace = dict(expectation_pressure.get("anchor_verification", {}) or {})
        active_anchors = list(anchor_trace.get("anchors", []) or [])
        pressure_anchor_levels = [
            _clamp(float((anchor or {}).get("level", 0.0) or 0.0), 0.0, 1.0)
            for anchor in active_anchors
            if isinstance(anchor, dict) and str(anchor.get("anchor_type", "") or "") == "pressure"
        ]
        expectation_anchor_levels = [
            _clamp(float((anchor or {}).get("level", 0.0) or 0.0), 0.0, 1.0)
            for anchor in active_anchors
            if isinstance(anchor, dict) and str(anchor.get("anchor_type", "") or "") == "expectation"
        ]
        selected_action_ids = {str((row or {}).get("action_id", "") or "") for row in selected if isinstance(row, dict)}
        safety_gate = dict(action.get("safety_gate", {}) or {})
        anchor_verified = len(list(anchor_trace.get("verified", []) or []))
        anchor_missed = len(list(anchor_trace.get("missed", []) or []))
        runtime_load = dict((trace or {}).get("runtime_load_feeling", {}) or {})
        runtime_load_channels = dict(runtime_load.get("channels", {}) or {})
        performance = dict((trace or {}).get("performance", {}) or {})
        target_tick_ms = max(1.0, float(performance.get("target_tick_ms", runtime_load_channels.get("target_tick_ms", 100.0)) or 100.0))
        total_ms = max(0.0, float(performance.get("total_ms", runtime_load_channels.get("elapsed_ms", 0.0)) or 0.0))
        load_ratio = float(runtime_load_channels.get("load_ratio", total_ms / target_tick_ms) or 0.0)
        return {
            "mean_cognitive_pressure": total_pressure / max(1, pressure_count),
            "mean_abs_cognitive_pressure": total_abs_pressure / max(1, pressure_count),
            "prediction_alignment": _clamp(float(prediction_trace.get("alignment_score", 0.0) or 0.0), 0.0, 1.0),
            "prediction_mismatch": _clamp(float(prediction_trace.get("mismatch_ratio", 0.0) or 0.0), 0.0, 1.0),
            "action_success": (sum(success_scores) / max(1, len(success_scores))) if success_scores else feedback_success,
            "action_punishment": (sum(punishment_scores) / max(1, len(punishment_scores))) if punishment_scores else _clamp(feedback_punishment, 0.0, 1.0),
            "expectation_anchor_active": _clamp(len(expectation_anchor_levels) / 8.0, 0.0, 1.0),
            "pressure_anchor_active": _clamp(len(pressure_anchor_levels) / 8.0, 0.0, 1.0),
            "expectation_anchor_level": (sum(expectation_anchor_levels) / max(1, len(expectation_anchor_levels))) if expectation_anchor_levels else 0.0,
            "pressure_anchor_level": (sum(pressure_anchor_levels) / max(1, len(pressure_anchor_levels))) if pressure_anchor_levels else 0.0,
            "anchor_validation_rate": _clamp(anchor_verified / max(1, anchor_verified + anchor_missed), 0.0, 1.0) if (anchor_verified or anchor_missed) else 0.0,
            "anchor_miss_rate": _clamp(anchor_missed / max(1, anchor_verified + anchor_missed), 0.0, 1.0) if (anchor_verified or anchor_missed) else 0.0,
            "expectation_recall_selected": 1.0 if "action::recall_by_expectation" in selected_action_ids else 0.0,
            "safety_anchor_pressure": _clamp(float((safety_gate.get("anchor_risk", {}) or {}).get("pressure", 0.0) or 0.0), 0.0, 1.0),
            "runtime_load_ratio": _clamp(load_ratio, 0.0, 4.0),
            "runtime_complexity": _clamp(float(runtime_load_channels.get("complexity", 0.0) or 0.0), 0.0, 1.0),
            "runtime_simplicity": _clamp(float(runtime_load_channels.get("simplicity", 0.0) or 0.0), 0.0, 1.0),
            "runtime_pending_index": _clamp(float(runtime_load_channels.get("pending_index_total", 0.0) or 0.0) / 64.0, 0.0, 1.0),
        }

    def _recommend(self) -> dict:
        pressure = float(self._metrics.get("mean_abs_cognitive_pressure", 0.0) or 0.0)
        alignment = float(self._metrics.get("prediction_alignment", 0.0) or 0.0)
        mismatch = float(self._metrics.get("prediction_mismatch", 0.0) or 0.0)
        action_success = float(self._metrics.get("action_success", 0.0) or 0.0)
        action_punishment = float(self._metrics.get("action_punishment", 0.0) or 0.0)
        runtime_complexity = float(self._metrics.get("runtime_complexity", 0.0) or 0.0)
        runtime_simplicity = float(self._metrics.get("runtime_simplicity", 0.0) or 0.0)
        runtime_overload = _clamp(float(self._metrics.get("runtime_load_ratio", 0.0) or 0.0) - 1.0, 0.0, 1.0)
        pending_pressure = float(self._metrics.get("runtime_pending_index", 0.0) or 0.0)
        pressure_anchor_level = float(self._metrics.get("pressure_anchor_level", 0.0) or 0.0)
        anchor_miss_rate = float(self._metrics.get("anchor_miss_rate", 0.0) or 0.0)
        safety_anchor_pressure = float(self._metrics.get("safety_anchor_pressure", 0.0) or 0.0)
        pressure_excess = _clamp((pressure - self.max_normal_pressure) / max(1.0, self.max_normal_pressure), 0.0, 1.0)
        alignment_deficit = _clamp(self.target_prediction_alignment - alignment, 0.0, 1.0)
        action_deficit = _clamp(self.target_action_success - action_success, 0.0, 1.0)
        action_risk = _clamp(action_punishment + action_deficit * 0.5 + pressure_anchor_level * 0.22 + safety_anchor_pressure * 0.20, 0.0, 1.0)
        runtime_pressure = _clamp(runtime_complexity + runtime_overload * 0.65 + pending_pressure * 0.35 - runtime_simplicity * 0.35, 0.0, 1.0)
        return {
            "schema_id": "adaptive_tuner_recommendation/v1",
            "support_ready": self._tick_count >= self.min_support_ticks,
            "pressure_excess": _round4(pressure_excess),
            "alignment_deficit": _round4(alignment_deficit),
            "mismatch": _round4(mismatch),
            "action_risk": _round4(action_risk),
            "runtime_pressure": _round4(runtime_pressure),
            "anchor_miss_rate": _round4(anchor_miss_rate),
            "pressure_anchor_level": _round4(pressure_anchor_level),
            "suggested": {
                "attention_threshold_delta": _round4(pressure_excess * 0.04 + runtime_pressure * 0.018 - alignment_deficit * 0.025),
                "prediction_gain_delta": _round4(alignment_deficit * 0.05 + mismatch * 0.025 - runtime_pressure * 0.012),
                "action_threshold_delta": _round4(action_risk * 0.045 + pressure_anchor_level * 0.018 - action_success * 0.015),
                "learning_rate_multiplier_delta": _round4(alignment_deficit * 0.035 + anchor_miss_rate * 0.018 - pressure_excess * 0.02 - runtime_pressure * 0.018),
            },
        }

    def _apply_recommendation(self, recommendation: dict) -> None:
        suggested = dict((recommendation or {}).get("suggested", {}) or {})
        old = self._copy_modulation()
        self._modulation["attention"]["threshold_adjustment"] = self._bounded_step(
            self._modulation["attention"]["threshold_adjustment"],
            float(suggested.get("attention_threshold_delta", 0.0) or 0.0),
            -0.08,
            0.08,
        )
        self._modulation["memory"]["prediction_gain_multiplier"] = self._bounded_step(
            self._modulation["memory"]["prediction_gain_multiplier"],
            float(suggested.get("prediction_gain_delta", 0.0) or 0.0),
            0.85,
            1.18,
            center=1.0,
        )
        self._modulation["action"]["threshold_adjustment"] = self._bounded_step(
            self._modulation["action"]["threshold_adjustment"],
            float(suggested.get("action_threshold_delta", 0.0) or 0.0),
            -0.06,
            0.10,
        )
        self._modulation["learning"]["rate_multiplier"] = self._bounded_step(
            self._modulation["learning"]["rate_multiplier"],
            float(suggested.get("learning_rate_multiplier_delta", 0.0) or 0.0),
            0.85,
            1.15,
            center=1.0,
        )
        if self._is_degraded(old, self._modulation):
            self._modulation = old
            event = {"tick_count": int(self._tick_count), "event": "rollback", "reason": "modulation_jump_too_large"}
        else:
            event = {"tick_count": int(self._tick_count), "event": "apply", "suggested": dict(suggested)}
            self._start_or_update_experiment(old=old, suggested=suggested, recommendation=recommendation)
        self._history.append(event)
        self._history = self._history[-64:]

    def _bounded_step(self, current: float, delta: float, low: float, high: float, *, center: float = 0.0) -> float:
        target = float(center) + float(delta)
        next_value = float(current) + (target - float(current)) * self.adjustment_rate
        return _clamp(next_value, low, high)

    def _is_degraded(self, old: dict, new: dict) -> bool:
        distance = 0.0
        for section, values in new.items():
            for key, value in dict(values).items():
                distance = max(distance, abs(float(value) - float((old.get(section, {}) or {}).get(key, value))))
        return distance > self.rollback_threshold

    def _neutral_modulation(self) -> dict:
        return {
            "attention": {"threshold_adjustment": 0.0},
            "memory": {"prediction_gain_multiplier": 1.0},
            "action": {"threshold_adjustment": 0.0},
            "learning": {"rate_multiplier": 1.0},
        }

    def _copy_modulation(self) -> dict:
        return {section: dict(values) for section, values in self._modulation.items()}

    def _start_or_update_experiment(self, *, old: dict, suggested: dict, recommendation: dict) -> None:
        """
        Track a small modulation as an experiment, not as a permanent truth.

        This keeps the AP philosophy conservative: the tuner may try a slightly
        different cognitive posture, but it must watch several ticks before
        confirming it and must be able to roll back if the field becomes worse.
        """

        if self._active_experiment and self._active_experiment.get("state") == "probing":
            return
        self._active_experiment = {
            "schema_id": "adaptive_tuner_experiment/v1",
            "state": "probing",
            "started_tick_count": int(self._tick_count),
            "baseline_metrics": self._metrics_for_experiment(),
            "previous_modulation": self._round_modulation(old),
            "candidate_modulation": self._round_modulation(self._modulation),
            "suggested": dict(suggested),
            "recommendation": {
                key: recommendation.get(key)
                for key in ("pressure_excess", "alignment_deficit", "mismatch", "action_risk", "runtime_pressure")
                if key in recommendation
            },
            "probe_ticks": 0,
            "last_degradation": {},
            "decision": "pending",
        }

    def _evaluate_active_experiment(self) -> None:
        experiment = self._active_experiment
        if not experiment or experiment.get("state") != "probing":
            return
        experiment["probe_ticks"] = max(0, int(self._tick_count) - int(experiment.get("started_tick_count", self._tick_count) or self._tick_count))
        if int(experiment.get("probe_ticks", 0) or 0) < max(3, self.min_support_ticks // 2):
            return
        current = self._metrics_for_experiment()
        degradation = self._experiment_degradation(dict(experiment.get("baseline_metrics", {}) or {}), current)
        experiment["current_metrics"] = current
        experiment["last_degradation"] = degradation
        if bool(degradation.get("should_rollback", False)):
            self._modulation = self._restore_modulation(dict(experiment.get("previous_modulation", {}) or {}))
            experiment["state"] = "rolled_back"
            experiment["decision"] = "rollback"
            experiment["rollback_reason"] = list(degradation.get("reasons", []) or [])
            self._last_experiment_decision_tick = int(self._tick_count)
            self._history.append(
                {
                    "tick_count": int(self._tick_count),
                    "event": "experiment_rollback",
                    "reason": experiment["rollback_reason"],
                }
            )
            self._history = self._history[-64:]
            return
        if self._experiment_improved(dict(experiment.get("baseline_metrics", {}) or {}), current):
            experiment["state"] = "confirmed"
            experiment["decision"] = "confirmed"
            self._last_experiment_decision_tick = int(self._tick_count)
            self._history.append(
                {
                    "tick_count": int(self._tick_count),
                    "event": "experiment_confirmed",
                    "current_metrics": current,
                }
            )
            self._history = self._history[-64:]

    def _metrics_for_experiment(self) -> dict:
        keys = (
            "mean_abs_cognitive_pressure",
            "prediction_alignment",
            "prediction_mismatch",
            "action_success",
            "action_punishment",
            "runtime_load_ratio",
            "runtime_complexity",
            "runtime_pending_index",
        )
        return {key: _round4(float(self._metrics.get(key, 0.0) or 0.0)) for key in keys}

    def _experiment_degradation(self, baseline: dict, current: dict) -> dict:
        pressure_delta = float(current.get("mean_abs_cognitive_pressure", 0.0) or 0.0) - float(baseline.get("mean_abs_cognitive_pressure", 0.0) or 0.0)
        alignment_delta = float(current.get("prediction_alignment", 0.0) or 0.0) - float(baseline.get("prediction_alignment", 0.0) or 0.0)
        punishment_delta = float(current.get("action_punishment", 0.0) or 0.0) - float(baseline.get("action_punishment", 0.0) or 0.0)
        runtime_delta = float(current.get("runtime_load_ratio", 0.0) or 0.0) - float(baseline.get("runtime_load_ratio", 0.0) or 0.0)
        reasons = []
        if pressure_delta > max(0.18, self.max_normal_pressure * 0.10):
            reasons.append("abs_pressure_increased")
        if alignment_delta < -0.10:
            reasons.append("prediction_alignment_dropped")
        if punishment_delta > 0.12:
            reasons.append("action_punishment_increased")
        if runtime_delta > 0.35:
            reasons.append("runtime_load_increased")
        return {
            "pressure_delta": _round4(pressure_delta),
            "alignment_delta": _round4(alignment_delta),
            "punishment_delta": _round4(punishment_delta),
            "runtime_delta": _round4(runtime_delta),
            "reasons": reasons,
            "should_rollback": bool(reasons),
        }

    def _experiment_improved(self, baseline: dict, current: dict) -> bool:
        pressure_delta = float(current.get("mean_abs_cognitive_pressure", 0.0) or 0.0) - float(baseline.get("mean_abs_cognitive_pressure", 0.0) or 0.0)
        alignment_delta = float(current.get("prediction_alignment", 0.0) or 0.0) - float(baseline.get("prediction_alignment", 0.0) or 0.0)
        punishment_delta = float(current.get("action_punishment", 0.0) or 0.0) - float(baseline.get("action_punishment", 0.0) or 0.0)
        return alignment_delta >= 0.04 or pressure_delta <= -0.12 or punishment_delta <= -0.08

    def _experiment_trace(self) -> dict:
        if not self._active_experiment:
            return {
                "schema_id": "adaptive_tuner_experiment/v1",
                "state": "idle",
                "humanlike_policy": "multi_tick_ema_before_confirm_or_rollback",
            }
        trace = dict(self._active_experiment)
        trace["humanlike_policy"] = "multi_tick_ema_before_confirm_or_rollback"
        return trace

    def _round_modulation(self, modulation: dict) -> dict:
        return {
            section: {key: _round4(value) for key, value in dict(values).items()}
            for section, values in dict(modulation or {}).items()
        }

    def _restore_modulation(self, modulation: dict) -> dict:
        neutral = self._neutral_modulation()
        restored = self._copy_modulation_from(neutral)
        for section, values in dict(modulation or {}).items():
            if section not in restored:
                continue
            for key, value in dict(values or {}).items():
                if key in restored[section]:
                    restored[section][key] = float(value or restored[section][key])
        return restored

    def _copy_modulation_from(self, modulation: dict) -> dict:
        return {section: dict(values) for section, values in dict(modulation or {}).items()}
