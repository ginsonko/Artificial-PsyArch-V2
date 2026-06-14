from __future__ import annotations

from math import ceil


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class RuntimeBudgetController:
    """
    Short-horizon controller for non-semantic runtime budgets.

    The controller consumes the previous tick's runtime-load suggestion and
    applies it to bounded readout, attention candidate, trace preview, and index
    maintenance budgets. It deliberately does not mutate semantic state-pool
    parameters such as real/virtual decay or prune thresholds.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        smoothing_alpha: float = 1.0,
        readout_min_multiplier: float = 0.72,
        readout_max_multiplier: float = 1.08,
        attention_candidate_min_multiplier: float = 0.68,
        attention_candidate_max_multiplier: float = 1.12,
        index_jobs_min_multiplier: float = 0.0,
        index_jobs_max_multiplier: float = 1.45,
        index_time_min_multiplier: float = 0.35,
        index_time_max_multiplier: float = 1.25,
        trace_detail_min_multiplier: float = 0.6,
        trace_detail_max_multiplier: float = 1.12,
        min_r_state_items_per_head: int = 32,
        preserve_1024_query_floor: bool = True,
        max_extra_index_jobs: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.smoothing_alpha = _clamp(float(smoothing_alpha), 0.05, 1.0)
        self.readout_min_multiplier = _clamp(readout_min_multiplier, 0.1, 1.0)
        self.readout_max_multiplier = max(1.0, float(readout_max_multiplier))
        self.attention_candidate_min_multiplier = _clamp(attention_candidate_min_multiplier, 0.1, 1.0)
        self.attention_candidate_max_multiplier = max(1.0, float(attention_candidate_max_multiplier))
        self.index_jobs_min_multiplier = _clamp(index_jobs_min_multiplier, 0.0, 1.0)
        self.index_jobs_max_multiplier = max(1.0, float(index_jobs_max_multiplier))
        self.index_time_min_multiplier = _clamp(index_time_min_multiplier, 0.05, 1.0)
        self.index_time_max_multiplier = max(1.0, float(index_time_max_multiplier))
        self.trace_detail_min_multiplier = _clamp(trace_detail_min_multiplier, 0.1, 1.0)
        self.trace_detail_max_multiplier = max(1.0, float(trace_detail_max_multiplier))
        self.min_r_state_items_per_head = max(1, int(min_r_state_items_per_head))
        self.preserve_1024_query_floor = bool(preserve_1024_query_floor)
        self.max_extra_index_jobs = max(0, int(max_extra_index_jobs))
        self._active = self._neutral()
        self._last_source_tick = -1
        self._active_tick = -1
        self._last_next_trace: dict = {}

    def begin_tick(self, tick_index: int) -> dict:
        self._active_tick = int(tick_index)
        return {
            "schema_id": "runtime_budget_controller/v1",
            "enabled": bool(self.enabled),
            "policy": "previous_tick_runtime_load_controls_nonsemantic_budgets",
            "applied": bool(self.enabled and self._last_source_tick >= 0),
            "tick_index": int(tick_index),
            "source_tick_index": int(self._last_source_tick),
            "active": self._rounded_values(self._active),
            "readout_budget": {},
            "attention_budget": {},
            "index_budget": {},
            "trace_budget": {},
            "next_budget": dict(self._last_next_trace),
            "semantic_mutation": {
                "real_decay": False,
                "virtual_decay": False,
                "prune_threshold": False,
            },
        }

    def readout_budget(self, *, base_items_per_head: int, base_head_limit: int) -> dict:
        base_items = max(1, int(base_items_per_head))
        multiplier = float(self._active["readout_budget_multiplier"] if self.enabled else 1.0)
        items = int(round(base_items * multiplier))
        if multiplier < 1.0:
            items = max(self.min_r_state_items_per_head, items)
        recent_head_count = max(1, min(4, max(1, int(base_head_limit))))
        if self.preserve_1024_query_floor and base_items * recent_head_count >= 1024:
            items = max(items, int(ceil(1024 / recent_head_count)))
        items = max(1, min(max(base_items, int(round(base_items * self.readout_max_multiplier))), items))
        return {
            "base_items_per_head": int(base_items),
            "items_per_head": int(items),
            "base_head_limit": max(1, int(base_head_limit)),
            "head_limit": max(1, int(base_head_limit)),
            "multiplier": _round4(multiplier),
            "changed": int(items) != int(base_items),
        }

    def attention_candidate_limit(self, *, base_limit: int) -> dict:
        base = max(1, int(base_limit))
        multiplier = float(self._active["attention_candidate_multiplier"] if self.enabled else 1.0)
        limit = max(1, int(round(base * multiplier)))
        return {
            "base_limit": int(base),
            "limit": int(limit),
            "multiplier": _round4(multiplier),
            "changed": int(limit) != int(base),
        }

    def index_budget(self, *, base_jobs: int, base_min_remaining_ms: float, base_max_ms: float) -> dict:
        jobs_base = max(0, int(base_jobs))
        jobs_multiplier = float(self._active["index_jobs_multiplier"] if self.enabled else 1.0)
        if jobs_multiplier < 0.999:
            jobs = int(jobs_base * jobs_multiplier)
        elif jobs_multiplier > 1.05:
            jobs = min(jobs_base + self.max_extra_index_jobs, int(ceil(jobs_base * jobs_multiplier)))
        else:
            jobs = jobs_base
        jobs = max(0, jobs)
        time_multiplier = float(self._active["index_time_multiplier"] if self.enabled else 1.0)
        min_remaining = float(base_min_remaining_ms) / max(0.05, time_multiplier)
        max_ms = max(0.0, float(base_max_ms) * max(0.0, time_multiplier))
        return {
            "base_jobs_per_tick": int(jobs_base),
            "jobs_per_tick": int(jobs),
            "jobs_multiplier": _round4(jobs_multiplier),
            "base_min_remaining_ms": _round4(base_min_remaining_ms),
            "min_remaining_ms": _round4(min_remaining),
            "base_max_ms": _round4(base_max_ms),
            "max_ms": _round4(max_ms),
            "time_multiplier": _round4(time_multiplier),
            "changed": int(jobs) != int(jobs_base) or abs(float(max_ms) - float(base_max_ms)) > 1e-9,
        }

    def trace_limit(self, *, base_limit: int, minimum: int = 1) -> int:
        base = max(1, int(base_limit))
        multiplier = float(self._active["trace_detail_multiplier"] if self.enabled else 1.0)
        return max(int(minimum), int(round(base * multiplier)))

    def trace_budget(self, *, base_item_preview_limit: int, base_r_state_preview_limit: int, base_matched_token_limit: int) -> dict:
        multiplier = float(self._active["trace_detail_multiplier"] if self.enabled else 1.0)
        return {
            "base_item_preview_limit": int(base_item_preview_limit),
            "item_preview_limit": int(self.trace_limit(base_limit=base_item_preview_limit)),
            "base_r_state_item_preview_limit": int(base_r_state_preview_limit),
            "r_state_item_preview_limit": int(self.trace_limit(base_limit=base_r_state_preview_limit)),
            "base_matched_token_preview_limit": int(base_matched_token_limit),
            "matched_token_preview_limit": int(self.trace_limit(base_limit=base_matched_token_limit)),
            "multiplier": _round4(multiplier),
        }

    def observe_runtime_load(self, runtime_load_trace: dict) -> dict:
        suggestion = dict((runtime_load_trace or {}).get("suggested_modulation", {}) or {})
        channels = dict((runtime_load_trace or {}).get("channels", {}) or {})
        desired = self._desired_from_suggestion(suggestion)
        if not self.enabled:
            desired = self._neutral()
        next_values = {}
        for key, value in desired.items():
            current = float(self._active.get(key, 1.0) or 1.0)
            next_values[key] = current * (1.0 - self.smoothing_alpha) + float(value) * self.smoothing_alpha
        self._active = self._bounded(next_values)
        self._last_source_tick = int((runtime_load_trace or {}).get("tick_index", self._active_tick))
        self._last_next_trace = {
            "schema_id": "runtime_budget_next_tick/v1",
            "queued": bool(self.enabled),
            "source_tick_index": int(self._last_source_tick),
            "source_channels": {
                "complexity": _round4(float(channels.get("complexity", 0.0) or 0.0)),
                "simplicity": _round4(float(channels.get("simplicity", 0.0) or 0.0)),
                "load_ratio": _round4(float(channels.get("load_ratio", 0.0) or 0.0)),
            },
            "desired": self._rounded_values(desired),
            "queued_values": self._rounded_values(self._active),
        }
        return dict(self._last_next_trace)

    def _desired_from_suggestion(self, suggestion: dict) -> dict:
        readout = _clamp(
            float(suggestion.get("readout_budget_multiplier", 1.0) or 1.0),
            self.readout_min_multiplier,
            self.readout_max_multiplier,
        )
        index_jobs = _clamp(
            float(suggestion.get("index_jobs_multiplier", 1.0) or 1.0),
            self.index_jobs_min_multiplier,
            self.index_jobs_max_multiplier,
        )
        trace_detail = _clamp(
            float(suggestion.get("trace_detail_multiplier", 1.0) or 1.0),
            self.trace_detail_min_multiplier,
            self.trace_detail_max_multiplier,
        )
        attention = _clamp(readout, self.attention_candidate_min_multiplier, self.attention_candidate_max_multiplier)
        index_time = _clamp(index_jobs, self.index_time_min_multiplier, self.index_time_max_multiplier)
        return {
            "readout_budget_multiplier": readout,
            "attention_candidate_multiplier": attention,
            "index_jobs_multiplier": index_jobs,
            "index_time_multiplier": index_time,
            "trace_detail_multiplier": trace_detail,
        }

    def _bounded(self, values: dict[str, float]) -> dict[str, float]:
        return {
            "readout_budget_multiplier": _clamp(values.get("readout_budget_multiplier", 1.0), self.readout_min_multiplier, self.readout_max_multiplier),
            "attention_candidate_multiplier": _clamp(values.get("attention_candidate_multiplier", 1.0), self.attention_candidate_min_multiplier, self.attention_candidate_max_multiplier),
            "index_jobs_multiplier": _clamp(values.get("index_jobs_multiplier", 1.0), self.index_jobs_min_multiplier, self.index_jobs_max_multiplier),
            "index_time_multiplier": _clamp(values.get("index_time_multiplier", 1.0), self.index_time_min_multiplier, self.index_time_max_multiplier),
            "trace_detail_multiplier": _clamp(values.get("trace_detail_multiplier", 1.0), self.trace_detail_min_multiplier, self.trace_detail_max_multiplier),
        }

    def _neutral(self) -> dict[str, float]:
        return {
            "readout_budget_multiplier": 1.0,
            "attention_candidate_multiplier": 1.0,
            "index_jobs_multiplier": 1.0,
            "index_time_multiplier": 1.0,
            "trace_detail_multiplier": 1.0,
        }

    def _rounded_values(self, values: dict[str, float]) -> dict[str, float]:
        return {key: _round4(value) for key, value in sorted(dict(values or {}).items())}
