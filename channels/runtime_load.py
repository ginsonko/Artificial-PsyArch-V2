from __future__ import annotations


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class RuntimeLoadFeelingChannel:
    """
    Trace-first feeling channel for runtime complexity/simplicity.

    This channel lets AP perceive its own operating load without immediately
    mutating semantic parameters such as decay, pruning, or recall budgets.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        min_activation: float,
        complexity_gain: float,
        simplicity_gain: float,
        target_load_ratio: float,
        ideal_load_ratio: float,
        state_item_soft_limit: int,
        r_state_item_soft_limit: int,
        attention_candidate_soft_limit: int,
        pending_index_soft_limit: int,
        family_overflow_soft_limit: int,
        residual_mass_soft_limit: float,
        mismatch_weight: float,
        fatigue_decay: float,
        fatigue_step: float,
        fatigue_gain: float,
        max_energy: float,
    ) -> None:
        self.enabled = bool(enabled)
        self.min_activation = max(0.0, float(min_activation))
        self.complexity_gain = max(0.0, float(complexity_gain))
        self.simplicity_gain = max(0.0, float(simplicity_gain))
        self.target_load_ratio = max(0.05, float(target_load_ratio))
        self.ideal_load_ratio = _clamp(float(ideal_load_ratio), 0.05, self.target_load_ratio)
        self.state_item_soft_limit = max(1, int(state_item_soft_limit))
        self.r_state_item_soft_limit = max(1, int(r_state_item_soft_limit))
        self.attention_candidate_soft_limit = max(1, int(attention_candidate_soft_limit))
        self.pending_index_soft_limit = max(1, int(pending_index_soft_limit))
        self.family_overflow_soft_limit = max(1, int(family_overflow_soft_limit))
        self.residual_mass_soft_limit = max(0.01, float(residual_mass_soft_limit))
        self.mismatch_weight = max(0.0, float(mismatch_weight))
        self.fatigue_decay = _clamp(fatigue_decay, 0.0, 1.0)
        self.fatigue_step = max(0.0, float(fatigue_step))
        self.fatigue_gain = max(0.0, float(fatigue_gain))
        self.max_energy = _clamp(float(max_energy), 0.01, 1.0)
        self._complexity_fatigue = 0.0
        self._simplicity_fatigue = 0.0
        self._last_tick = -1

    def derive(
        self,
        *,
        tick_index: int,
        target_tick_ms: float,
        elapsed_ms: float,
        r_state: dict,
        attention_candidates: list[dict],
        state_snapshot: dict,
        attention_trace: dict,
        prediction_trace: dict,
        residual_summary: dict,
        pending_index_summary: dict | None = None,
    ) -> dict:
        self._advance_tick(int(tick_index))
        target_ms = max(1.0, float(target_tick_ms or 100.0))
        elapsed = max(0.0, float(elapsed_ms or 0.0))
        load_ratio = elapsed / target_ms
        r_state_items = sum(len((head or {}).get("items", []) or []) for head in (r_state or {}).get("heads", []) or [])
        pool_size = int((r_state or {}).get("total_pool_size", 0) or 0)
        snapshot_items = len(((state_snapshot or {}).get("items", []) or []))
        candidate_count = len(attention_candidates or [])
        family_trace = dict((attention_trace or {}).get("family_budget", {}) or {})
        overflow_count = int(family_trace.get("overflow_count", 0) or 0)
        prediction = dict(prediction_trace or {})
        mismatch = _clamp(float(prediction.get("mismatch_ratio", 0.0) or 0.0), 0.0, 1.0)
        residual = dict(residual_summary or {})
        residual_mass = float(residual.get("total_unresolved_mass", 0.0) or 0.0)
        pending = dict(pending_index_summary or {})
        pending_total = int(pending.get("total", pending.get("pending", 0)) or 0)
        pending_heavy = int(pending.get("heavy", pending.get("pending_heavy", 0)) or 0)

        components = {
            "runtime": _clamp((load_ratio - self.ideal_load_ratio) / max(0.05, self.target_load_ratio - self.ideal_load_ratio), 0.0, 1.0),
            "state": _clamp(pool_size / float(self.state_item_soft_limit), 0.0, 1.0),
            "r_state": _clamp(r_state_items / float(self.r_state_item_soft_limit), 0.0, 1.0),
            "attention_candidates": _clamp(candidate_count / float(self.attention_candidate_soft_limit), 0.0, 1.0),
            "pending_index": _clamp((pending_total + pending_heavy * 1.5) / float(self.pending_index_soft_limit), 0.0, 1.0),
            "focus_overflow": _clamp(overflow_count / float(self.family_overflow_soft_limit), 0.0, 1.0),
            "residual": _clamp(residual_mass / float(self.residual_mass_soft_limit), 0.0, 1.0),
            "mismatch": mismatch,
        }
        complexity_raw = _clamp(
            components["runtime"] * 0.34
            + components["state"] * 0.12
            + components["r_state"] * 0.12
            + components["attention_candidates"] * 0.10
            + components["pending_index"] * 0.12
            + components["focus_overflow"] * 0.08
            + components["residual"] * 0.06
            + components["mismatch"] * self.mismatch_weight * 0.16,
            0.0,
            1.0,
        )
        simplicity_raw = _clamp(
            (1.0 - _clamp(load_ratio / max(0.05, self.ideal_load_ratio), 0.0, 1.0)) * 0.56
            + (1.0 - components["state"]) * 0.10
            + (1.0 - components["r_state"]) * 0.10
            + (1.0 - components["pending_index"]) * 0.10
            + (1.0 - components["mismatch"]) * 0.14,
            0.0,
            1.0,
        )
        if complexity_raw >= simplicity_raw:
            simplicity_raw *= max(0.0, 1.0 - complexity_raw * 0.75)
        else:
            complexity_raw *= max(0.0, 1.0 - simplicity_raw * 0.45)

        complexity = _clamp(complexity_raw * self.complexity_gain * (1.0 - self._complexity_fatigue * self.fatigue_gain), 0.0, self.max_energy)
        simplicity = _clamp(simplicity_raw * self.simplicity_gain * (1.0 - self._simplicity_fatigue * self.fatigue_gain), 0.0, self.max_energy)
        if not self.enabled:
            complexity = 0.0
            simplicity = 0.0

        channels = {
            "complexity": _round4(complexity),
            "simplicity": _round4(simplicity),
            "load_ratio": _round4(load_ratio),
            "target_tick_ms": _round4(target_ms),
            "elapsed_ms": _round4(elapsed),
            "pool_size": int(pool_size),
            "snapshot_item_count": int(snapshot_items),
            "r_state_item_count": int(r_state_items),
            "attention_candidate_count": int(candidate_count),
            "family_overflow_count": int(overflow_count),
            "pending_index_total": int(pending_total),
            "pending_index_heavy": int(pending_heavy),
            "prediction_mismatch": _round4(mismatch),
            "residual_unresolved_mass": _round4(residual_mass),
        }
        items = []
        if self.enabled and complexity >= self.min_activation:
            self._complexity_fatigue = _clamp(self._complexity_fatigue + self.fatigue_step, 0.0, 1.0)
            items.append(self._item("complexity", "复杂感", complexity, channels, components))
        if self.enabled and simplicity >= self.min_activation:
            self._simplicity_fatigue = _clamp(self._simplicity_fatigue + self.fatigue_step, 0.0, 1.0)
            items.append(self._item("simplicity", "简单感", simplicity, channels, components))

        return {
            "schema_id": "runtime_load_feeling/v1",
            "enabled": bool(self.enabled),
            "tick_index": int(tick_index),
            "policy": "trace_first_no_runtime_budget_mutation",
            "channels": channels,
            "components": {key: _round4(value) for key, value in sorted(components.items())},
            "items": items,
            "suggested_modulation": self._suggested_modulation(complexity=complexity, simplicity=simplicity),
        }

    def _item(self, key: str, display: str, energy: float, channels: dict, components: dict) -> dict:
        return {
            "sa_label": f"feeling::{key}",
            "display_text": display,
            "source_type": "runtime_load_feeling",
            "family": "cognitive_feeling",
            "real_energy": _round4(energy),
            "anchor_meta": {
                "feeling_key": key,
                "feeling_value": _round4(energy),
                "runtime_load": dict(channels),
                "components": {name: _round4(value) for name, value in sorted(components.items())},
                "trace_only": True,
            },
        }

    def _suggested_modulation(self, *, complexity: float, simplicity: float) -> dict:
        pressure = _clamp(float(complexity) - float(simplicity) * 0.35, 0.0, 1.0)
        spare = _clamp(float(simplicity) - float(complexity) * 0.25, 0.0, 1.0)
        return {
            "schema_id": "runtime_load_modulation_suggestion/v1",
            "applied": False,
            "attention_threshold_delta": _round4(pressure * 0.035 - spare * 0.012),
            "readout_budget_multiplier": _round4(_clamp(1.0 - pressure * 0.12 + spare * 0.08, 0.82, 1.08)),
            "index_jobs_multiplier": _round4(_clamp(1.0 - pressure * 0.35 + spare * 0.25, 0.35, 1.25)),
            "trace_detail_multiplier": _round4(_clamp(1.0 - pressure * 0.18 + spare * 0.12, 0.65, 1.12)),
            "decay_pressure_hint": _round4(pressure * 0.025 - spare * 0.008),
            "reason": "suggestion_only_until_short_term_controller_is_enabled",
        }

    def _advance_tick(self, tick_index: int) -> None:
        if self._last_tick < 0:
            self._last_tick = int(tick_index)
            return
        delta = max(1, int(tick_index) - int(self._last_tick))
        self._complexity_fatigue = _clamp(self._complexity_fatigue * (self.fatigue_decay**delta), 0.0, 1.0)
        self._simplicity_fatigue = _clamp(self._simplicity_fatigue * (self.fatigue_decay**delta), 0.0, 1.0)
        self._last_tick = int(tick_index)
