# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


def empty_rollup(*, run_id: str = "") -> dict[str, Any]:
    return {
        "schema_id": "run_rollup/v1",
        "schema_version": "1.0",
        "run_id": run_id,
        "tick_count": 0,
        "last_tick_index": -1,
        "logic_ms": {"mean": 0.0, "max": 0.0, "sum": 0.0},
        "memory_count_last": 0,
        "state_pool_size_last": 0,
        "bn_count_last": 0,
        "c_star_count_last": 0,
        "emotion_last": {
            "dissonance": 0.0,
            "correctness": 0.0,
            "expectation": 0.0,
            "pressure": 0.0,
        },
        "runtime_stage_timing_last": {},
        "series_tail": {
            "tick_index": [],
            "logic_ms": [],
            "memory_count": [],
            "state_pool_size": [],
            "bn_count": [],
            "c_star_count": [],
            "vision_budget_used": [],
            "vision_total_patch_count": [],
            "vision_reconstruction_cell_count": [],
            "audio_budget_used": [],
            "text_budget_used": [],
            "residual_count": [],
            "top_energy": [],
            "emotion_dissonance": [],
            "emotion_correctness": [],
            "emotion_expectation": [],
            "emotion_pressure": [],
            "rules_fired_count": [],
            "action_drive_count": [],
            "sandbox_action_count": [],
            "tuner_matched_count": [],
        },
        "input_preview_tail": [],
        "focus_preview_tail": [],
        "candidate_source_histogram": {},
        "rules_fired_histogram": {},
        "last_summary": {},
        "last_metrics": {},
    }


def update_rollup(
    rollup: dict[str, Any] | None,
    *,
    summary: dict[str, Any],
    metrics: dict[str, Any],
    series_tail_limit: int = 96,
) -> dict[str, Any]:
    src = copy.deepcopy(rollup) if isinstance(rollup, dict) else empty_rollup(run_id=str(summary.get("run_id", "") or ""))
    src["run_id"] = str(summary.get("run_id", src.get("run_id", "")) or src.get("run_id", ""))

    raw_tick_index = summary.get("tick_index", -1)
    tick_index = int(-1 if raw_tick_index is None else raw_tick_index)
    prev_count = int(src.get("tick_count", 0) or 0)
    next_count = max(prev_count, tick_index + 1)
    src["tick_count"] = next_count
    raw_last_tick_index = src.get("last_tick_index", -1)
    src["last_tick_index"] = max(int(-1 if raw_last_tick_index is None else raw_last_tick_index), tick_index)

    logic_ms = float(metrics.get("logic_ms", 0.0) or 0.0)
    logic_state = dict(src.get("logic_ms", {}) or {})
    logic_sum = float(logic_state.get("sum", 0.0) or 0.0) + logic_ms
    logic_max = max(float(logic_state.get("max", 0.0) or 0.0), logic_ms)
    logic_mean = logic_sum / max(1, next_count)
    src["logic_ms"] = {"mean": _round4(logic_mean), "max": _round4(logic_max), "sum": _round4(logic_sum)}

    src["memory_count_last"] = int(summary.get("memory_index_summary", {}).get("vector", {}).get("vector_count", src.get("memory_count_last", 0)) or 0)
    src["state_pool_size_last"] = int(metrics.get("state_pool_size", 0) or 0)
    src["bn_count_last"] = int(metrics.get("bn_count", 0) or 0)
    src["c_star_count_last"] = int(metrics.get("c_star_count", 0) or 0)
    src["emotion_last"] = dict((summary.get("rules_preview", {}) or {}).get("emotion_channels", {}) or src.get("emotion_last", {}))
    src["runtime_stage_timing_last"] = dict(metrics.get("runtime_stage_timing_ms", {}) or {})

    series_tail = dict(src.get("series_tail", {}) or {})
    _push_series(series_tail, "tick_index", tick_index, series_tail_limit)
    _push_series(series_tail, "logic_ms", _round4(logic_ms), series_tail_limit)
    _push_series(series_tail, "memory_count", src["memory_count_last"], series_tail_limit)
    _push_series(series_tail, "state_pool_size", src["state_pool_size_last"], series_tail_limit)
    _push_series(series_tail, "bn_count", src["bn_count_last"], series_tail_limit)
    _push_series(series_tail, "c_star_count", src["c_star_count_last"], series_tail_limit)
    _push_series(series_tail, "text_budget_used", int(metrics.get("text_budget_used", 0) or 0), series_tail_limit)
    _push_series(series_tail, "vision_budget_used", int(metrics.get("vision_budget_used", 0) or 0), series_tail_limit)
    _push_series(series_tail, "vision_total_patch_count", int(metrics.get("vision_total_patch_count", 0) or 0), series_tail_limit)
    _push_series(series_tail, "vision_reconstruction_cell_count", int(metrics.get("vision_reconstruction_cell_count", 0) or 0), series_tail_limit)
    _push_series(series_tail, "audio_budget_used", int(metrics.get("audio_budget_used", 0) or 0), series_tail_limit)
    _push_series(series_tail, "residual_count", int((summary.get("state_pool_summary", {}).get("residual_summary", {}) or {}).get("count", 0) or 0), series_tail_limit)
    top_energy = max((float(item.get("energy", 0.0) or 0.0) for item in (summary.get("state_top", []) or [])), default=0.0)
    _push_series(series_tail, "top_energy", _round4(top_energy), series_tail_limit)
    emotion_last = dict(src.get("emotion_last", {}) or {})
    _push_series(series_tail, "emotion_dissonance", _round4(float(emotion_last.get("dissonance", 0.0) or 0.0)), series_tail_limit)
    _push_series(series_tail, "emotion_correctness", _round4(float(emotion_last.get("correctness", 0.0) or 0.0)), series_tail_limit)
    _push_series(series_tail, "emotion_expectation", _round4(float(emotion_last.get("expectation", 0.0) or 0.0)), series_tail_limit)
    _push_series(series_tail, "emotion_pressure", _round4(float(emotion_last.get("pressure", 0.0) or 0.0)), series_tail_limit)
    rule_preview = dict(summary.get("rules_preview", {}) or {})
    _push_series(series_tail, "rules_fired_count", int(rule_preview.get("rule_fired_count", len(rule_preview.get("rules_fired", []) or [])) or 0), series_tail_limit)
    _push_series(series_tail, "action_drive_count", int(rule_preview.get("action_drive_count", 0) or 0), series_tail_limit)
    _push_series(series_tail, "sandbox_action_count", int(rule_preview.get("sandbox_action_count", len(rule_preview.get("sandbox_actions", []) or [])) or 0), series_tail_limit)
    _push_series(series_tail, "tuner_matched_count", int(rule_preview.get("tuner_matched_count", len((summary.get("tuner_preview", {}) or {}).get("matched_profiles", []) or [])) or 0), series_tail_limit)
    src["series_tail"] = series_tail

    _push_tail(src, "input_preview_tail", str(summary.get("input_preview", "") or ""), limit=12)
    focus_preview = " ".join(str(item or "") for item in (summary.get("a_focus_preview", []) or []) if str(item or ""))
    if focus_preview:
        _push_tail(src, "focus_preview_tail", focus_preview, limit=12)

    for bn_item in summary.get("bn_preview", []) or []:
        for source in bn_item.get("candidate_sources", []) or []:
            _inc_hist(src, "candidate_source_histogram", str(source or "unknown"))
    for rule_id in (summary.get("rules_preview", {}) or {}).get("rules_fired", []) or []:
        _inc_hist(src, "rules_fired_histogram", str(rule_id or "unknown"))

    src["last_summary"] = copy.deepcopy(summary)
    src["last_metrics"] = copy.deepcopy(metrics)
    return src


def _push_series(target: dict[str, Any], key: str, value: Any, limit: int) -> None:
    rows = list(target.get(key, []) or [])
    rows.append(value)
    target[key] = rows[-max(1, int(limit)) :]


def _push_tail(payload: dict[str, Any], key: str, value: str, *, limit: int) -> None:
    rows = list(payload.get(key, []) or [])
    if value:
        rows.append(value)
    payload[key] = rows[-max(1, int(limit)) :]


def _inc_hist(payload: dict[str, Any], key: str, item: str) -> None:
    hist = dict(payload.get(key, {}) or {})
    hist[item] = int(hist.get(item, 0) or 0) + 1
    payload[key] = hist
