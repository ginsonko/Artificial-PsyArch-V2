from __future__ import annotations

from collections import defaultdict


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class TimeFeelingChannel:
    def __init__(
        self,
        *,
        enabled: bool,
        threshold: float,
        gain: float,
        min_confidence: float,
        default_radius_ticks: float,
        recall_gain: float,
        fatigue_decay: float,
        fatigue_step: float,
        fatigue_gain: float,
        fatigue_max: float,
        max_sources: int,
    ) -> None:
        self.enabled = bool(enabled)
        self.threshold = max(0.0, float(threshold))
        self.gain = max(0.0, float(gain))
        self.min_confidence = max(0.0, float(min_confidence))
        self.default_radius_ticks = max(1.0, float(default_radius_ticks))
        self.recall_gain = max(0.0, float(recall_gain))
        self.fatigue_decay = _clamp(fatigue_decay, 0.0, 1.0)
        self.fatigue_step = max(0.0, float(fatigue_step))
        self.fatigue_gain = max(0.0, float(fatigue_gain))
        self.fatigue_max = max(0.0, float(fatigue_max))
        self.max_sources = max(1, int(max_sources))
        self._peak_fatigue: dict[int, float] = defaultdict(float)
        self._last_tick = -1

    def derive(self, *, tick_index: int, bn_rows: list[dict]) -> dict:
        self._advance_tick(int(tick_index))
        if not self.enabled:
            return {"channels": {}, "items": [], "dominant_peak": None}
        candidates = self._candidate_rows(bn_rows, tick_index=int(tick_index))
        if not candidates:
            return {"channels": {}, "items": [], "dominant_peak": None}

        peaks = []
        for seed in candidates:
            radius = max(1.0, self.default_radius_ticks * (0.65 + 0.35 * max(0.0, seed["weight"])))
            cluster = []
            cluster_mass = 0.0
            for other in candidates:
                distance = abs(float(other["delta_t"]) - float(seed["delta_t"]))
                proximity = max(0.0, 1.0 - distance / radius)
                if proximity <= 0.0:
                    continue
                contribution = float(other["weight"]) * proximity
                if contribution <= 0.0:
                    continue
                cluster.append({"memory_id": other["memory_id"], "delta_t": other["delta_t"], "weight": _round4(contribution)})
                cluster_mass += contribution
            if not cluster:
                continue
            weighted_delta = sum(float(row["delta_t"]) * float(row["weight"]) for row in cluster) / max(1e-6, cluster_mass)
            variance = sum(float(row["weight"]) * (float(row["delta_t"]) - weighted_delta) ** 2 for row in cluster) / max(1e-6, cluster_mass)
            sigma = variance ** 0.5
            peaks.append(
                {
                    "seed_delta_t": int(seed["delta_t"]),
                    "cluster_mass": cluster_mass,
                    "center_delta_t": weighted_delta,
                    "sigma": sigma,
                    "support": cluster,
                }
            )
        if not peaks:
            return {"channels": {}, "items": [], "dominant_peak": None}

        peaks.sort(key=lambda item: (-float(item["cluster_mass"]), float(item["center_delta_t"])))
        dominant = peaks[0]
        second_mass = float(peaks[1]["cluster_mass"]) if len(peaks) > 1 else 0.0
        source_count = len(dominant["support"])
        dominance = float(dominant["cluster_mass"]) / max(1e-6, float(dominant["cluster_mass"]) + second_mass)
        confidence = _clamp(
            0.38 * dominance
            + 0.22 * min(1.0, float(dominant["cluster_mass"]) / max(1.0, len(candidates)))
            + 0.20 * min(1.0, source_count / 4.0)
            + 0.20 * max(0.0, 1.0 - float(dominant["sigma"]) / max(1.0, self.default_radius_ticks * 1.5)),
            0.0,
            1.0,
        )
        peak_key = int(round(float(dominant["center_delta_t"])))
        fatigue = float(self._peak_fatigue.get(peak_key, 0.0) or 0.0)
        signal_strength = float(dominant["cluster_mass"]) * dominance
        energy = self.gain * max(0.0, signal_strength - self.threshold) * confidence * max(0.0, 1.0 - fatigue * self.fatigue_gain)
        energy = _clamp(energy, 0.0, 1.0)

        channels = {
            "dominant_delta_t": _round4(dominant["center_delta_t"]),
            "confidence": _round4(confidence),
            "cluster_mass": _round4(dominant["cluster_mass"]),
            "dominance": _round4(dominance),
            "source_count": source_count,
            "fatigue": _round4(fatigue),
            "recall_gain": _round4(self.recall_gain),
        }

        items = []
        if confidence >= self.min_confidence and energy > 0.0 and source_count >= 2:
            self._peak_fatigue[peak_key] = _clamp(fatigue + self.fatigue_step, 0.0, self.fatigue_max)
            items.append(
                {
                    "sa_label": "timefelt::elapsed",
                    "display_text": "时间间隔感",
                    "source_type": "time_feeling",
                    "family": "time_feeling",
                    "real_energy": _round4(energy),
                    "anchor_meta": {
                        "delta_t_norm": _round4(float(dominant["center_delta_t"]) / max(1.0, int(tick_index) + 1)),
                        "delta_sigma_norm": _round4(float(dominant["sigma"]) / max(1.0, int(tick_index) + 1)),
                        "confidence": _round4(confidence),
                        "cluster_mass": _round4(dominant["cluster_mass"]),
                        "dominance": _round4(dominance),
                        "source_count": source_count,
                        "support": dominant["support"][: self.max_sources],
                    },
                }
            )
        return {
            "channels": channels,
            "items": items,
            "dominant_peak": {
                "center_delta_t": _round4(dominant["center_delta_t"]),
                "sigma": _round4(dominant["sigma"]),
                "cluster_mass": _round4(dominant["cluster_mass"]),
                "dominance": _round4(dominance),
                "confidence": _round4(confidence),
                "support": dominant["support"][: self.max_sources],
            },
        }

    def _candidate_rows(self, bn_rows: list[dict], *, tick_index: int) -> list[dict]:
        rows = []
        for row in bn_rows[: self.max_sources * 2]:
            snapshot = dict(row.get("snapshot", {}) or {})
            snapshot_ref = dict(row.get("snapshot_ref", {}) or {})
            memory_id = str(snapshot.get("memory_id", snapshot_ref.get("memory_id", row.get("memory_id", ""))) or "")
            memory_tick = snapshot.get("tick_index", snapshot_ref.get("tick_index", row.get("tick_index")))
            if memory_tick is None:
                continue
            delta_t = max(0, int(tick_index) - int(memory_tick))
            score = float(row.get("score", row.get("raw_score", 0.0)) or 0.0)
            state_match = float(row.get("state_match", 0.0) or 0.0)
            weight = max(0.0, score * (0.65 + 0.35 * state_match))
            if weight <= 0.0:
                continue
            rows.append(
                {
                    "memory_id": memory_id,
                    "delta_t": delta_t,
                    "weight": weight,
                }
            )
        return rows

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = tick_index
            return
        delta = max(1, tick_index - self._last_tick)
        for key in list(self._peak_fatigue.keys()):
            self._peak_fatigue[key] = _clamp(float(self._peak_fatigue[key]) * (self.fatigue_decay**delta), 0.0, self.fatigue_max)
            if self._peak_fatigue[key] < 0.0001:
                self._peak_fatigue.pop(key, None)
        self._last_tick = tick_index
