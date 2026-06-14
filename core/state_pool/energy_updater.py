from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


class PredictionEnergyUpdater:
    """
    Compare predicted virtual evidence with actual real evidence for the tick.

    The updater is intentionally bounded: it only compares the predicted labels
    already injected into prediction_slot with the current tick's external items.
    """

    def __init__(
        self,
        *,
        match_virtual_boost: float = 0.18,
        miss_virtual_decay: float = 0.52,
        unexpected_attention_gain: float = 0.22,
        miss_attention_gain: float = 0.12,
    ) -> None:
        self.match_virtual_boost = max(0.0, float(match_virtual_boost))
        self.miss_virtual_decay = max(0.0, min(1.0, float(miss_virtual_decay)))
        self.unexpected_attention_gain = max(0.0, float(unexpected_attention_gain))
        self.miss_attention_gain = max(0.0, float(miss_attention_gain))

    def build_trace(self, *, predicted_items: list[dict], actual_items: list[dict], tick_index: int) -> dict:
        predicted_energy = _energy_by_label(predicted_items, energy_key="virtual_energy", fallback_key="real_energy")
        actual_energy = _energy_by_label(actual_items, energy_key="real_energy", fallback_key="virtual_energy")
        predicted_labels = sorted(predicted_energy.keys())
        actual_labels = sorted(actual_energy.keys())
        predicted_set = set(predicted_labels)
        actual_set = set(actual_labels)
        matched = sorted(predicted_set & actual_set)
        missed = sorted(predicted_set - actual_set)
        unexpected = sorted(actual_set - predicted_set)
        match_mass = sum(min(float(predicted_energy.get(label, 0.0) or 0.0), float(actual_energy.get(label, 0.0) or 0.0)) for label in matched)
        missed_mass = sum(float(predicted_energy.get(label, 0.0) or 0.0) for label in missed)
        unexpected_mass = sum(float(actual_energy.get(label, 0.0) or 0.0) for label in unexpected)
        predicted_mass = sum(float(value or 0.0) for value in predicted_energy.values())
        actual_mass = sum(float(value or 0.0) for value in actual_energy.values())
        mismatch_mass = missed_mass + unexpected_mass
        denom = max(1e-6, predicted_mass + actual_mass)
        alignment_score = match_mass / denom
        mismatch_ratio = mismatch_mass / max(1e-6, predicted_mass + actual_mass + match_mass)
        return {
            "schema_id": "prediction_energy_trace/v1",
            "tick_index": int(tick_index),
            "predicted_labels": predicted_labels[:64],
            "actual_labels": actual_labels[:64],
            "matched_labels": matched[:64],
            "missed_predicted_labels": missed[:64],
            "unexpected_labels": unexpected[:64],
            "predicted_energy_by_label": {key: _round4(value) for key, value in predicted_energy.items()},
            "actual_energy_by_label": {key: _round4(value) for key, value in actual_energy.items()},
            "predicted_mass": _round4(predicted_mass),
            "actual_mass": _round4(actual_mass),
            "match_mass": _round4(match_mass),
            "missed_expected_mass": _round4(missed_mass),
            "unexpected_novelty_mass": _round4(unexpected_mass),
            "mismatch_mass": _round4(mismatch_mass),
            "alignment_score": _round4(alignment_score),
            "mismatch_ratio": _round4(mismatch_ratio),
            "match_count": len(matched),
            "missed_count": len(missed),
            "unexpected_count": len(unexpected),
        }

    def update_entry_from_trace(self, entry, *, label: str, trace: dict) -> dict:
        predicted = float((trace.get("predicted_energy_by_label", {}) or {}).get(label, 0.0) or 0.0)
        actual = float((trace.get("actual_energy_by_label", {}) or {}).get(label, 0.0) or 0.0)
        before = {
            "real_energy": _round4(float(entry.real_energy)),
            "virtual_energy": _round4(float(entry.virtual_energy)),
            "attention_gain": _round4(float(entry.attention_gain)),
        }
        role = "unrelated"
        if label in set(trace.get("matched_labels", []) or []):
            role = "matched_prediction"
            entry.virtual_energy = _round4(max(float(entry.virtual_energy), predicted) + min(actual, predicted) * self.match_virtual_boost)
        elif label in set(trace.get("missed_predicted_labels", []) or []):
            role = "missed_prediction"
            entry.virtual_energy = _round4(float(entry.virtual_energy) * self.miss_virtual_decay)
            entry.attention_gain = _round4(float(entry.attention_gain) + self.miss_attention_gain)
        elif label in set(trace.get("unexpected_labels", []) or []):
            role = "unexpected_actual"
            entry.attention_gain = _round4(float(entry.attention_gain) + self.unexpected_attention_gain)
        entry.anchor_meta["last_prediction_validation"] = {
            "role": role,
            "predicted_virtual": _round4(predicted),
            "actual_real": _round4(actual),
            "tick_index": int(trace.get("tick_index", -1) or -1),
        }
        entry.refresh_pressure()
        after = {
            "real_energy": _round4(float(entry.real_energy)),
            "virtual_energy": _round4(float(entry.virtual_energy)),
            "attention_gain": _round4(float(entry.attention_gain)),
        }
        return {
            "sa_label": str(label),
            "role": role,
            "before": before,
            "after": after,
        }


def _energy_by_label(items: list[dict], *, energy_key: str, fallback_key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("sa_label", "") or "")
        if not label:
            continue
        value = float(item.get(energy_key, item.get(fallback_key, 0.0)) or 0.0)
        if value <= 0.0:
            continue
        result[label] = result.get(label, 0.0) + value
    return result
