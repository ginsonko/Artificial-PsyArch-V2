from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ExpectationPressureChannel:
    """
    Slow expectation/pressure field.

    This channel is separate from cognitive feelings:
    - cognitive feelings say "what the system feels about the current cognitive act";
    - this channel keeps a decaying field of expectation, unresolved pressure, and
      satisfaction that can enter the state pool and modulate action/emotion.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        min_activation: float,
        expectation_decay: float,
        pressure_decay: float,
        satisfaction_decay: float,
        expectation_gain: float,
        pressure_gain: float,
        satisfaction_gain: float,
        residual_gain: float,
        feedback_gain: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_activation = max(0.0, float(min_activation))
        self.expectation_decay = _clamp(expectation_decay, 0.0, 1.0)
        self.pressure_decay = _clamp(pressure_decay, 0.0, 1.0)
        self.satisfaction_decay = _clamp(satisfaction_decay, 0.0, 1.0)
        self.expectation_gain = max(0.0, float(expectation_gain))
        self.pressure_gain = max(0.0, float(pressure_gain))
        self.satisfaction_gain = max(0.0, float(satisfaction_gain))
        self.residual_gain = max(0.0, float(residual_gain))
        self.feedback_gain = max(0.0, float(feedback_gain))
        self._expectation_level = 0.0
        self._pressure_level = 0.0
        self._satisfaction_level = 0.0
        self._last_tick = -1

    def derive(
        self,
        *,
        tick_index: int,
        cognitive_feelings: dict,
        prediction_trace: dict | None = None,
        residual_summary: dict | None = None,
        action_feedback_trace: dict | None = None,
        rhythm_trace: dict | None = None,
        time_trace: dict | None = None,
    ) -> dict:
        self._advance_tick(int(tick_index))
        if not self.enabled:
            return {"channels": {}, "items": [], "field_state": self._field_state()}

        feelings = dict((cognitive_feelings or {}).get("channels", {}) or {})
        prediction = dict(prediction_trace or {})
        residual = dict(residual_summary or {})
        feedback = dict((action_feedback_trace or {}).get("observed_feedback", {}) or {})
        rhythm = dict((rhythm_trace or {}).get("channels", {}) or {})
        time_channels = dict((time_trace or {}).get("channels", {}) or {})

        cfs_expectation = float(feelings.get("expectation", 0.0) or 0.0)
        cfs_pressure = float(feelings.get("pressure", 0.0) or 0.0)
        cfs_dissonance = float(feelings.get("dissonance", 0.0) or 0.0)
        cfs_correctness = float(feelings.get("correctness", 0.0) or 0.0)
        cfs_grasp = float(feelings.get("grasp", 0.0) or 0.0)
        mismatch_ratio = float(prediction.get("mismatch_ratio", 0.0) or 0.0)
        alignment_score = float(prediction.get("alignment_score", 0.0) or 0.0)
        predicted_count = int(prediction.get("predicted_labels_count", len(prediction.get("predicted_labels", []) or [])) or 0)
        residual_mass = float(residual.get("total_unresolved_mass", 0.0) or 0.0)
        residual_pressure = _clamp(residual_mass / max(1.0, residual_mass + 12.0), 0.0, 1.0)
        reward = float(feedback.get("reward", 0.0) or 0.0)
        punishment = float(feedback.get("punishment", 0.0) or 0.0)
        rhythm_expectation = float(rhythm.get("phase_expectation", 0.0) or 0.0)
        time_confidence = float(time_channels.get("confidence", 0.0) or 0.0)

        expectation_input = _clamp(
            cfs_expectation * 0.42
            + min(1.0, predicted_count / 8.0) * 0.2
            + alignment_score * 0.24
            + rhythm_expectation * 0.14
            + time_confidence * 0.08
            + reward * self.feedback_gain * 0.18
            - mismatch_ratio * 0.12,
            0.0,
            1.0,
        )
        pressure_input = _clamp(
            cfs_pressure * 0.35
            + cfs_dissonance * 0.25
            + mismatch_ratio * 0.34
            + residual_pressure * self.residual_gain * 0.28
            + punishment * self.feedback_gain * 0.22
            - alignment_score * 0.1
            - cfs_grasp * 0.08,
            0.0,
            1.0,
        )
        satisfaction_input = _clamp(
            alignment_score * 0.38
            + cfs_correctness * 0.24
            + cfs_grasp * 0.16
            + reward * self.feedback_gain * 0.18
            - mismatch_ratio * 0.2
            - punishment * self.feedback_gain * 0.18,
            0.0,
            1.0,
        )

        self._expectation_level = _clamp(self._expectation_level + expectation_input * self.expectation_gain * (1.0 - self._expectation_level * 0.35), 0.0, 1.0)
        self._pressure_level = _clamp(self._pressure_level + pressure_input * self.pressure_gain * (1.0 - self._pressure_level * 0.3), 0.0, 1.0)
        self._satisfaction_level = _clamp(self._satisfaction_level + satisfaction_input * self.satisfaction_gain * (1.0 - self._satisfaction_level * 0.4), 0.0, 1.0)

        expectation_gap = _clamp(self._expectation_level - self._satisfaction_level + self._pressure_level * 0.25, 0.0, 1.0)
        channels = {
            "expectation_level": _round4(self._expectation_level),
            "pressure_level": _round4(self._pressure_level),
            "satisfaction_level": _round4(self._satisfaction_level),
            "expectation_gap": _round4(expectation_gap),
            "prediction_alignment": _round4(alignment_score),
            "prediction_mismatch": _round4(mismatch_ratio),
            "residual_pressure": _round4(residual_pressure),
            "feedback_reward": _round4(reward),
            "feedback_punishment": _round4(punishment),
        }

        derived_from = {
            "cfs_expectation": _round4(cfs_expectation),
            "cfs_pressure": _round4(cfs_pressure),
            "cfs_dissonance": _round4(cfs_dissonance),
            "prediction_alignment": _round4(alignment_score),
            "prediction_mismatch": _round4(mismatch_ratio),
            "predicted_count": int(predicted_count),
            "residual_unresolved_mass": _round4(residual_mass),
            "residual_pressure": _round4(residual_pressure),
            "feedback_reward": _round4(reward),
            "feedback_punishment": _round4(punishment),
        }
        items = []
        item_specs = [
            ("expectation", "expectation_pressure::expectation", "期待场", self._expectation_level),
            ("pressure", "expectation_pressure::pressure", "压力场", self._pressure_level),
            ("satisfaction", "expectation_pressure::satisfaction", "满足校验感", self._satisfaction_level),
            ("expectation_gap", "expectation_pressure::gap", "期待落差", expectation_gap),
        ]
        for key, label, display, value in item_specs:
            if value < self.min_activation:
                continue
            items.append(
                {
                    "sa_label": label,
                    "display_text": display,
                    "source_type": "expectation_pressure",
                    "family": "expectation_pressure",
                    "real_energy": _round4(value),
                    "anchor_meta": {
                        "channel_key": key,
                        "channel_value": _round4(value),
                        "derived_from": dict(derived_from),
                    },
                }
            )

        return {
            "channels": channels,
            "items": items,
            "field_state": self._field_state(),
            "derived_from": derived_from,
        }

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = int(tick_index)
            return
        delta = max(1, int(tick_index) - int(self._last_tick))
        self._expectation_level = _clamp(self._expectation_level * (self.expectation_decay**delta), 0.0, 1.0)
        self._pressure_level = _clamp(self._pressure_level * (self.pressure_decay**delta), 0.0, 1.0)
        self._satisfaction_level = _clamp(self._satisfaction_level * (self.satisfaction_decay**delta), 0.0, 1.0)
        self._last_tick = int(tick_index)

    def _field_state(self) -> dict:
        return {
            "expectation_level": _round4(self._expectation_level),
            "pressure_level": _round4(self._pressure_level),
            "satisfaction_level": _round4(self._satisfaction_level),
            "last_tick": int(self._last_tick),
        }
