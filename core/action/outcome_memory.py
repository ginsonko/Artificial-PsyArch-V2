from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ActionOutcomeMemory:
    """
    Bounded long-term action outcome memory.

    This layer is intentionally separate from cognitive association learning:
    reward / punishment evidence changes action tendency, not concept distance.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        learning_rate: float = 0.18,
        decay_per_tick: float = 0.992,
        support_scale: float = 6.0,
        max_weighted_support: float = 48.0,
        max_drive_bias: float = 0.75,
    ) -> None:
        self.enabled = bool(enabled)
        self.learning_rate = _clamp(float(learning_rate), 0.001, 1.0)
        self.decay_per_tick = _clamp(float(decay_per_tick), 0.90, 1.0)
        self.support_scale = max(1.0, float(support_scale))
        self.max_weighted_support = max(1.0, float(max_weighted_support))
        self.max_drive_bias = max(0.0, float(max_drive_bias))
        self._stats: dict[str, dict] = {}
        self._last_tick = -1

    def advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = int(tick_index)
            return
        delta = max(0, int(tick_index) - int(self._last_tick))
        if delta <= 0:
            return
        decay = self.decay_per_tick**delta
        for action_id in list(self._stats.keys()):
            row = self._stats[action_id]
            row["weighted_support"] = float(row.get("weighted_support", 0.0) or 0.0) * decay
            row["drive_bias"] = float(row.get("drive_bias", 0.0) or 0.0) * decay
            row["approach_bias"] = float(row.get("approach_bias", 0.0) or 0.0) * decay
            row["avoidance_bias"] = float(row.get("avoidance_bias", 0.0) or 0.0) * decay
            if float(row.get("weighted_support", 0.0) or 0.0) < 0.001 and int(row.get("event_count", 0) or 0) <= 0:
                self._stats.pop(action_id, None)
        self._last_tick = int(tick_index)

    def record(self, *, action_id: str, observed_feedback: dict, predicted_outcome: dict | None = None) -> dict:
        action = str(action_id or "")
        if not self.enabled or not action:
            return self._empty_estimate(action)
        row = self._stats.setdefault(action, self._empty_stats(action))
        reward = max(0.0, float((observed_feedback or {}).get("reward", 0.0) or 0.0))
        punishment = max(0.0, float((observed_feedback or {}).get("punishment", 0.0) or 0.0))
        correctness = max(0.0, float((observed_feedback or {}).get("correctness", 0.0) or 0.0))
        confidence = _clamp(float((observed_feedback or {}).get("confidence", 0.0) or 0.0), 0.0, 1.0)
        pressure = max(0.0, punishment * 0.78 - reward * 0.18 - correctness * 0.08)
        utility = reward + correctness * 0.42 - punishment * 1.08 - pressure * 0.28
        predicted = dict(predicted_outcome or {})
        predicted_utility = (
            float(predicted.get("reward", 0.0) or 0.0)
            + float(predicted.get("correctness", 0.0) or 0.0) * 0.42
            - float(predicted.get("punishment", 0.0) or 0.0) * 1.08
            - float(predicted.get("pressure", 0.0) or 0.0) * 0.28
        )
        prediction_error = abs(utility - predicted_utility) if predicted else 0.0
        intensity = _clamp(reward + punishment + correctness * 0.7 + prediction_error * 0.3, 0.05, 1.0)
        sample_weight = _clamp((0.35 + confidence * 0.65) * intensity, 0.02, 1.0)
        alpha = _clamp(self.learning_rate * (0.45 + confidence * 0.55) * (0.7 + intensity * 0.3), 0.02, 0.45)
        for key, sample in (
            ("reward", reward),
            ("punishment", punishment),
            ("correctness", correctness),
            ("pressure", pressure),
            ("confidence", confidence),
            ("utility", utility),
            ("prediction_error", prediction_error),
        ):
            row[key] = self._ema(float(row.get(key, 0.0) or 0.0), sample, alpha)
        row["event_count"] = int(row.get("event_count", 0) or 0) + 1
        row["weighted_support"] = _clamp(
            float(row.get("weighted_support", 0.0) or 0.0) + sample_weight,
            0.0,
            self.max_weighted_support,
        )
        is_failure = punishment > max(0.04, reward + correctness * 0.35)
        is_success = reward + correctness * 0.35 > max(0.06, punishment * 1.05)
        if is_failure:
            row["failure_count"] = int(row.get("failure_count", 0) or 0) + 1
            row["failure_streak"] = int(row.get("failure_streak", 0) or 0) + 1
            row["success_streak"] = 0
        elif is_success:
            row["success_count"] = int(row.get("success_count", 0) or 0) + 1
            row["success_streak"] = int(row.get("success_streak", 0) or 0) + 1
            row["failure_streak"] = 0
        else:
            row["failure_streak"] = 0
            row["success_streak"] = 0
        support_gate = self._support_gate(float(row.get("weighted_support", 0.0) or 0.0))
        approach_sample = max(0.0, utility) * support_gate
        avoidance_sample = max(0.0, -utility) * support_gate
        row["approach_bias"] = self._ema(float(row.get("approach_bias", 0.0) or 0.0), approach_sample, alpha)
        row["avoidance_bias"] = self._ema(float(row.get("avoidance_bias", 0.0) or 0.0), avoidance_sample, alpha)
        drive_sample = _clamp((approach_sample - avoidance_sample) * 0.8, -self.max_drive_bias, self.max_drive_bias)
        row["drive_bias"] = _clamp(
            self._ema(float(row.get("drive_bias", 0.0) or 0.0), drive_sample, alpha),
            -self.max_drive_bias,
            self.max_drive_bias,
        )
        row["last_feedback"] = {
            "reward": _round4(reward),
            "punishment": _round4(punishment),
            "correctness": _round4(correctness),
            "pressure": _round4(pressure),
            "confidence": _round4(confidence),
            "utility": _round4(utility),
            "sample_weight": _round4(sample_weight),
            "prediction_error": _round4(prediction_error),
            "classification": "failure" if is_failure else ("success" if is_success else "neutral"),
        }
        return self.estimate(action)

    def estimate(self, action_id: str) -> dict:
        action = str(action_id or "")
        row = self._stats.get(action)
        if not self.enabled or row is None:
            return self._empty_estimate(action)
        weighted_support = float(row.get("weighted_support", 0.0) or 0.0)
        support = self._support_gate(weighted_support)
        if support <= 0.0:
            return self._empty_estimate(action)
        return {
            "schema_id": "action_outcome_estimate/v1",
            "method": "long_term_reward_punishment_drive_memory",
            "action_id": action,
            "support": _round4(support),
            "event_count": int(row.get("event_count", 0) or 0),
            "weighted_support": _round4(weighted_support),
            "reward": _round4(float(row.get("reward", 0.0) or 0.0)),
            "punishment": _round4(float(row.get("punishment", 0.0) or 0.0)),
            "correctness": _round4(float(row.get("correctness", 0.0) or 0.0)),
            "pressure": _round4(float(row.get("pressure", 0.0) or 0.0)),
            "confidence": _round4(float(row.get("confidence", 0.0) or 0.0)),
            "utility": _round4(float(row.get("utility", 0.0) or 0.0)),
            "prediction_error": _round4(float(row.get("prediction_error", 0.0) or 0.0)),
            "approach_bias": _round4(float(row.get("approach_bias", 0.0) or 0.0)),
            "avoidance_bias": _round4(float(row.get("avoidance_bias", 0.0) or 0.0)),
            "drive_bias": _round4(float(row.get("drive_bias", 0.0) or 0.0)),
            "success_count": int(row.get("success_count", 0) or 0),
            "failure_count": int(row.get("failure_count", 0) or 0),
            "success_streak": int(row.get("success_streak", 0) or 0),
            "failure_streak": int(row.get("failure_streak", 0) or 0),
            "last_feedback": dict(row.get("last_feedback", {}) or {}),
        }

    def snapshot(self) -> dict:
        estimates = [self.estimate(action_id) for action_id in sorted(self._stats)]
        active = [row for row in estimates if float(row.get("support", 0.0) or 0.0) > 0.0]
        active.sort(key=lambda item: (-abs(float(item.get("drive_bias", 0.0) or 0.0)), str(item.get("action_id", "") or "")))
        return {
            "schema_id": "action_outcome_memory/v1",
            "enabled": bool(self.enabled),
            "policy": {
                "learning_rate": _round4(self.learning_rate),
                "decay_per_tick": _round4(self.decay_per_tick),
                "support_scale": _round4(self.support_scale),
                "max_weighted_support": _round4(self.max_weighted_support),
                "max_drive_bias": _round4(self.max_drive_bias),
            },
            "action_count": len(active),
            "estimates": active,
        }

    def _support_gate(self, weighted_support: float) -> float:
        return _clamp(float(weighted_support) / (float(weighted_support) + self.support_scale), 0.0, 1.0)

    def _ema(self, old: float, sample: float, alpha: float) -> float:
        return float(old) * (1.0 - float(alpha)) + float(sample) * float(alpha)

    def _empty_stats(self, action_id: str) -> dict:
        return {
            "action_id": str(action_id or ""),
            "event_count": 0,
            "weighted_support": 0.0,
            "reward": 0.0,
            "punishment": 0.0,
            "correctness": 0.0,
            "pressure": 0.0,
            "confidence": 0.0,
            "utility": 0.0,
            "prediction_error": 0.0,
            "approach_bias": 0.0,
            "avoidance_bias": 0.0,
            "drive_bias": 0.0,
            "success_count": 0,
            "failure_count": 0,
            "success_streak": 0,
            "failure_streak": 0,
            "last_feedback": {},
        }

    def _empty_estimate(self, action_id: str) -> dict:
        return {
            "schema_id": "action_outcome_estimate/v1",
            "method": "no_long_term_outcome_evidence",
            "action_id": str(action_id or ""),
            "support": 0.0,
            "event_count": 0,
            "weighted_support": 0.0,
            "reward": 0.0,
            "punishment": 0.0,
            "correctness": 0.0,
            "pressure": 0.0,
            "confidence": 0.0,
            "utility": 0.0,
            "prediction_error": 0.0,
            "approach_bias": 0.0,
            "avoidance_bias": 0.0,
            "drive_bias": 0.0,
            "success_count": 0,
            "failure_count": 0,
            "success_streak": 0,
            "failure_streak": 0,
            "last_feedback": {},
        }
