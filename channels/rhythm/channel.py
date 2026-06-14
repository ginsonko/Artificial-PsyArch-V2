from __future__ import annotations

from collections import defaultdict, deque


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class RhythmChannel:
    def __init__(
        self,
        *,
        enabled: bool,
        window: int,
        min_hits: int,
        min_period: int,
        max_period: int,
        period_sigma_scale: float,
        phase_sigma_scale: float,
        pulse_threshold: float,
        phase_threshold: float,
        fatigue_decay: float,
        fatigue_step: float,
        fatigue_gain: float,
        fatigue_max: float,
        salience_threshold: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.window = max(4, int(window))
        self.min_hits = max(2, int(min_hits))
        self.min_period = max(1, int(min_period))
        self.max_period = max(self.min_period, int(max_period))
        self.period_sigma_scale = max(0.05, float(period_sigma_scale))
        self.phase_sigma_scale = max(0.05, float(phase_sigma_scale))
        self.pulse_threshold = max(0.0, float(pulse_threshold))
        self.phase_threshold = max(0.0, float(phase_threshold))
        self.fatigue_decay = _clamp(fatigue_decay, 0.0, 1.0)
        self.fatigue_step = max(0.0, float(fatigue_step))
        self.fatigue_gain = max(0.0, float(fatigue_gain))
        self.fatigue_max = max(0.0, float(fatigue_max))
        self.salience_threshold = max(0.0, float(salience_threshold))
        self._history: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=self.window))
        self._fatigue: dict[str, float] = defaultdict(float)
        self._last_tick = -1

    def observe(self, *, tick_index: int, focus_items: list[dict]) -> None:
        self._advance_tick(int(tick_index))
        for item in focus_items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            salience = max(
                float(item.get("focus_score", 0.0) or 0.0),
                float(item.get("attention_score", 0.0) or 0.0),
                float(item.get("real_energy", 0.0) or 0.0),
            )
            if salience < self.salience_threshold:
                continue
            family_key = self._family_key(label)
            self._history[family_key].append(
                {
                    "tick_index": int(tick_index),
                    "salience": salience,
                    "label": label,
                }
            )

    def derive(self, *, tick_index: int) -> dict:
        if not self.enabled:
            return {"channels": {}, "items": [], "family": None}
        candidates = []
        for family_key, hits in self._history.items():
            if len(hits) < self.min_hits:
                continue
            analysis = self._analyze_family(family_key, list(hits), tick_index=int(tick_index))
            if analysis is not None:
                candidates.append(analysis)
        if not candidates:
            return {"channels": {}, "items": [], "family": None}
        candidates.sort(key=lambda item: (-float(item["groove"]), -float(item["phase_expectation"]), item["family_key"]))
        best = candidates[0]
        fatigue = float(self._fatigue.get(best["family_key"], 0.0) or 0.0)
        items = []
        channels = {
            "family_key": best["family_key"],
            "period_ticks": _round4(best["period_ticks"]),
            "regularity": _round4(best["regularity"]),
            "recurrence": _round4(best["recurrence"]),
            "recovery_match": _round4(best["recovery_match"]),
            "groove": _round4(best["groove"]),
            "phase_expectation": _round4(best["phase_expectation"]),
            "fatigue": _round4(fatigue),
        }
        if best["groove"] >= self.pulse_threshold:
            items.append(
                {
                    "sa_label": "rhythmfelt::pulse",
                    "display_text": "节拍感",
                    "source_type": "rhythm_feeling",
                    "family": "rhythm_feeling",
                    "real_energy": _round4(best["groove"]),
                    "anchor_meta": {
                        "period_ticks": _round4(best["period_ticks"]),
                        "regularity": _round4(best["regularity"]),
                        "recurrence": _round4(best["recurrence"]),
                        "recovery_match": _round4(best["recovery_match"]),
                        "confidence": _round4(best["confidence"]),
                        "family_key": best["family_key"],
                    },
                }
            )
        if best["phase_expectation"] >= self.phase_threshold:
            items.append(
                {
                    "sa_label": "rhythmfelt::phase_expectation",
                    "display_text": "节奏期待感",
                    "source_type": "rhythm_feeling",
                    "family": "rhythm_feeling",
                    "real_energy": _round4(best["phase_expectation"]),
                    "anchor_meta": {
                        "period_ticks": _round4(best["period_ticks"]),
                        "time_to_next": _round4(best["time_to_next"]),
                        "phase_error": _round4(best["phase_error"]),
                        "regularity": _round4(best["regularity"]),
                        "confidence": _round4(best["confidence"]),
                        "family_key": best["family_key"],
                    },
                }
            )
        if items:
            self._fatigue[best["family_key"]] = _clamp(fatigue + self.fatigue_step, 0.0, self.fatigue_max)
        return {
            "channels": channels,
            "items": items,
            "family": best,
        }

    def _analyze_family(self, family_key: str, hits: list[dict], *, tick_index: int) -> dict | None:
        deltas = []
        for idx in range(1, len(hits)):
            delta = int(hits[idx]["tick_index"]) - int(hits[idx - 1]["tick_index"])
            if self.min_period <= delta <= self.max_period:
                deltas.append(delta)
        if len(deltas) < max(1, self.min_hits - 1):
            return None
        peaks = []
        for seed in deltas:
            sigma = max(1.0, seed * self.period_sigma_scale)
            mass = 0.0
            for other in deltas:
                mass += max(0.0, 1.0 - abs(other - seed) / max(1.0, sigma * 2.2))
            peaks.append({"seed": seed, "mass": mass, "sigma": sigma})
        peaks.sort(key=lambda item: (-float(item["mass"]), float(item["seed"])))
        tau = float(peaks[0]["seed"])
        second_mass = float(peaks[1]["mass"]) if len(peaks) > 1 else 0.0
        regularity = _clamp(float(peaks[0]["mass"]) / max(1.0, len(deltas)), 0.0, 1.0)
        recurrence = _clamp(len(hits) / max(3.0, float(self.window)), 0.0, 1.0)
        salience_support = _clamp(sum(float(hit["salience"]) for hit in hits[-self.min_hits :]) / max(1.0, float(self.min_hits)), 0.0, 1.0)
        recovery_target = max(float(self.min_period), min(float(self.max_period), 4.0))
        recovery_sigma = max(1.0, recovery_target * 0.45)
        recovery_match = _clamp(
            pow(2.718281828, -(((tau - recovery_target) ** 2) / max(1e-6, 2.0 * recovery_sigma * recovery_sigma))),
            0.0,
            1.0,
        )
        rhythmicity = regularity * recurrence * salience_support
        fatigue = float(self._fatigue.get(family_key, 0.0) or 0.0)
        confidence = _clamp(0.45 * regularity + 0.25 * recurrence + 0.30 * salience_support, 0.0, 1.0)
        anticipated_next = int(hits[-1]["tick_index"]) + tau
        phase_error = abs(float(tick_index) - anticipated_next)
        phase_sigma = max(1.0, tau * self.phase_sigma_scale)
        phase_expectation = regularity * pow(2.718281828, -((phase_error**2) / max(1e-6, 2.0 * phase_sigma * phase_sigma)))
        dominance = float(peaks[0]["mass"]) / max(1e-6, float(peaks[0]["mass"]) + second_mass)
        groove = rhythmicity * recovery_match * confidence * dominance * max(0.0, 1.0 - fatigue * self.fatigue_gain)
        return {
            "family_key": family_key,
            "period_ticks": tau,
            "regularity": regularity,
            "recurrence": recurrence,
            "salience_support": salience_support,
            "recovery_match": recovery_match,
            "confidence": confidence,
            "groove": _clamp(groove, 0.0, 1.0),
            "phase_expectation": _clamp(phase_expectation, 0.0, 1.0),
            "phase_error": phase_error,
            "time_to_next": anticipated_next - float(tick_index),
        }

    def _family_key(self, label: str) -> str:
        clean = str(label or "")
        if "::" in clean:
            return clean.split("::", 1)[1]
        return clean

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = tick_index
            return
        delta = max(1, tick_index - self._last_tick)
        for key in list(self._fatigue.keys()):
            self._fatigue[key] = _clamp(float(self._fatigue[key]) * (self.fatigue_decay**delta), 0.0, self.fatigue_max)
            if self._fatigue[key] < 0.0001:
                self._fatigue.pop(key, None)
        self._last_tick = tick_index
