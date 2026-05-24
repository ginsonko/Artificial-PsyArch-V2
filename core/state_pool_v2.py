# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from collections import deque
from typing import Any

from sensors.text_sensor_v2 import join_text_units, split_text_units


def _round_energy(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _shrink_text(value: str, limit: int = 48) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(4, limit - 3)] + "..."


def _int_or_default(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _clone_item_light(item: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(item)
    if "coords" in cloned:
        cloned["coords"] = dict(item.get("coords", {}) or {})
    if "attributes" in cloned:
        cloned["attributes"] = dict(item.get("attributes", {}) or {})
    if "support" in cloned:
        cloned["support"] = dict(item.get("support", {}) or {})
    return cloned


class StatePoolV2:
    def __init__(
        self,
        *,
        decay: float,
        prune_threshold: float,
        recent_queue_limit: int,
        verbatim_window_chars: int,
        head_limit: int,
        items_per_head: int,
        anchor_cache_limit: int,
        residual_limit: int,
        handle_limit: int,
        residual_unit_limit: int,
        attention_object_fatigue_decay: float = 0.82,
        attention_object_fatigue_step: float = 0.16,
        attention_object_fatigue_gain: float = 0.72,
        attention_object_fatigue_max: float = 1.0,
        attention_object_min_multiplier: float = 0.42,
    ) -> None:
        self.decay = max(0.0, min(1.0, float(decay)))
        self.prune_threshold = max(0.0, float(prune_threshold))
        self.recent_queue_limit = max(1, int(recent_queue_limit))
        self.verbatim_window_chars = max(1, int(verbatim_window_chars))
        self.head_limit = max(1, int(head_limit))
        self.items_per_head = max(1, int(items_per_head))
        self.anchor_cache_limit = max(1, int(anchor_cache_limit))
        self.residual_limit = max(1, int(residual_limit))
        self.handle_limit = max(1, int(handle_limit))
        self.residual_unit_limit = max(1, int(residual_unit_limit))
        self.attention_object_fatigue_decay = _clamp(float(attention_object_fatigue_decay), 0.0, 1.0)
        self.attention_object_fatigue_step = max(0.0, float(attention_object_fatigue_step))
        self.attention_object_fatigue_gain = max(0.0, float(attention_object_fatigue_gain))
        self.attention_object_fatigue_max = max(0.0, float(attention_object_fatigue_max))
        self.attention_object_min_multiplier = _clamp(float(attention_object_min_multiplier), 0.0, 1.0)

        self._entries: dict[str, dict[str, Any]] = {}
        self._recent_external: deque[dict[str, Any]] = deque(maxlen=self.recent_queue_limit)
        self._verbatim_window: deque[str] = deque()
        self._verbatim_chars = 0
        self._tick_index = -1
        self._attention_fatigue: dict[str, dict[str, Any]] = {}
        self._attention_fatigue_cache_tick = -1
        self._attention_fatigue_value_cache: dict[str, float] = {}
        self._last_committed_focus_tick = -1

        self._hot_anchor_cache: list[dict[str, Any]] = []
        self._residual_bucket: dict[str, dict[str, Any]] = {}
        self._handle_ring: deque[dict[str, Any]] = deque(maxlen=self.handle_limit)
        self._last_pool_result: dict[str, Any] = {}
        self._prediction_trace: dict[str, Any] = {
            "tick_index": -1,
            "predicted_labels": [],
            "predicted_texts": [],
            "actual_labels": [],
            "actual_texts": [],
            "matched_labels": [],
            "unexpected_labels": [],
            "missed_predicted_labels": [],
            "match_count": 0,
            "unexpected_count": 0,
            "missed_count": 0,
            "predicted_mass": 0.0,
            "actual_mass": 0.0,
            "overprediction_mass": 0.0,
            "underprediction_mass": 0.0,
            "missed_expected_mass": 0.0,
            "unexpected_novelty_mass": 0.0,
            "mismatch_mass": 0.0,
            "match_mass": 0.0,
            "committed_match_mass": 0.0,
            "committed_overprediction_mass": 0.0,
            "committed_underprediction_mass": 0.0,
            "committed_mismatch_mass": 0.0,
            "predicted_commitment_mass": 0.0,
            "committed_labels": [],
            "halo_labels": [],
        }
        self._pending_prediction_rows: list[dict[str, Any]] = []
        self._view_cache_tick = -1
        self._live_entry_rows_cache: list[dict[str, Any]] | None = None
        self._top_items_cache: dict[int, list[dict[str, Any]]] = {}
        self._anchor_ranked_rows_cache: list[dict[str, Any]] | None = None
        self._anchor_items_cache: dict[int, list[dict[str, Any]]] = {}
        self._attention_ranked_rows_cache: dict[tuple[float, float, float, float, float, float], list[dict[str, Any]]] = {}
        self._focus_ranked_cache: dict[tuple[float, float], list[str]] = {}
        self._attention_context_cache: dict[str, Any] | None = None
        self._attention_items_cache: dict[tuple[float, float, float, float, float, float, int], list[dict[str, Any]]] = {}
        self._hot_anchor_cache_dirty = True
        self._snapshot_summary_cache: dict[str, Any] | None = None
        self._snapshot_sidecar_cache: dict[str, Any] | None = None

    def _invalidate_view_caches(self) -> None:
        self._view_cache_tick = -1
        self._live_entry_rows_cache = None
        self._top_items_cache = {}
        self._anchor_ranked_rows_cache = None
        self._anchor_items_cache = {}
        self._attention_ranked_rows_cache = {}
        self._focus_ranked_cache = {}
        self._attention_context_cache = None
        self._attention_items_cache = {}
        self._hot_anchor_cache = []
        self._hot_anchor_cache_dirty = True
        self._snapshot_summary_cache = None
        self._snapshot_sidecar_cache = None

    def _invalidate_attention_fatigue_cache(self) -> None:
        self._attention_fatigue_cache_tick = -1
        self._attention_fatigue_value_cache = {}

    def _begin_tick(self, tick_index: int) -> bool:
        next_tick = int(tick_index)
        changed = next_tick != self._tick_index
        self._tick_index = next_tick
        if changed:
            self._invalidate_view_caches()
            self._invalidate_attention_fatigue_cache()
        return changed

    def _decay_energy_value(self, value: float, *, from_tick: int, to_tick: int) -> float:
        steps = max(0, int(to_tick) - int(from_tick))
        if steps <= 0:
            return _round_energy(value)
        return _round_energy(float(value) * (self.decay ** steps))

    def _decay_attention_fatigue_value(self, value: float, *, from_tick: int, to_tick: int) -> float:
        steps = max(0, int(to_tick) - int(from_tick))
        if steps <= 0:
            return _round_energy(value)
        return _round_energy(float(value) * (self.attention_object_fatigue_decay ** steps))

    def _attention_fatigue_value(self, label: str) -> float:
        clean_label = str(label or "")
        if not clean_label:
            return 0.0
        if self._attention_fatigue_cache_tick != self._tick_index:
            self._attention_fatigue_cache_tick = self._tick_index
            self._attention_fatigue_value_cache = {}
        cached = self._attention_fatigue_value_cache.get(clean_label)
        if cached is not None:
            return cached
        entry = self._attention_fatigue.get(clean_label)
        if not isinstance(entry, dict):
            self._attention_fatigue_value_cache[clean_label] = 0.0
            return 0.0
        stored_tick = _int_or_default(entry.get("tick_index", self._tick_index), self._tick_index)
        stored_value = float(entry.get("value", 0.0) or 0.0)
        fatigue = self._decay_attention_fatigue_value(stored_value, from_tick=stored_tick, to_tick=self._tick_index)
        if fatigue <= 0.0001:
            self._attention_fatigue.pop(clean_label, None)
            self._attention_fatigue_value_cache[clean_label] = 0.0
            return 0.0
        entry["value"] = fatigue
        entry["tick_index"] = int(self._tick_index)
        self._attention_fatigue_value_cache[clean_label] = fatigue
        return fatigue

    def _attention_fatigue_multiplier(self, label: str) -> float:
        fatigue = self._attention_fatigue_value(label)
        if fatigue <= 0.0 or self.attention_object_fatigue_gain <= 0.0:
            return 1.0
        return _clamp(1.0 - fatigue * self.attention_object_fatigue_gain, self.attention_object_min_multiplier, 1.0)

    def _commit_attention_focus(self, selected_labels: list[str]) -> None:
        if self._last_committed_focus_tick == self._tick_index:
            return
        unique_labels: list[str] = []
        for raw_label in selected_labels:
            clean_label = str(raw_label or "")
            if clean_label and clean_label not in unique_labels:
                unique_labels.append(clean_label)
        if not unique_labels or self.attention_object_fatigue_step <= 0.0 or self.attention_object_fatigue_max <= 0.0:
            self._last_committed_focus_tick = int(self._tick_index)
            return
        for label in unique_labels:
            fatigue = self._attention_fatigue_value(label)
            updated = min(self.attention_object_fatigue_max, fatigue + self.attention_object_fatigue_step)
            self._attention_fatigue[label] = {
                "value": _round_energy(updated),
                "tick_index": int(self._tick_index),
            }
        self._last_committed_focus_tick = int(self._tick_index)
        self._invalidate_attention_fatigue_cache()
        self._invalidate_view_caches()

    def _attention_fatigue_items(self, *, limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for label in list(self._attention_fatigue.keys()):
            fatigue = self._attention_fatigue_value(label)
            if fatigue <= 0.0:
                continue
            entry = self._entries.get(label, {})
            rows.append(
                {
                    "sa_label": label,
                    "display_text": str(entry.get("display_text", "") or label),
                    "fatigue": _round_energy(fatigue),
                    "multiplier": _round_energy(self._attention_fatigue_multiplier(label)),
                    "channel": str(entry.get("channel", "") or ""),
                    "source_type": str(entry.get("source_type", "") or ""),
                }
            )
        rows.sort(
            key=lambda item: (
                -float(item.get("fatigue", 0.0) or 0.0),
                str(item.get("sa_label", "")),
            )
        )
        return rows[: max(1, int(limit))]

    def _entry_energy_at_tick(self, entry: dict[str, Any], *, tick_index: int | None = None) -> float:
        target_tick = self._tick_index if tick_index is None else int(tick_index)
        stored_tick = _int_or_default(entry.get("energy_tick", entry.get("last_seen_tick", target_tick)), target_tick)
        stored_energy = float(entry.get("energy", 0.0) or 0.0)
        return self._decay_energy_value(stored_energy, from_tick=stored_tick, to_tick=target_tick)

    def _copy_live_entry(self, label: str, *, tick_index: int | None = None) -> dict[str, Any] | None:
        entry = self._entries.get(str(label or ""))
        if entry is None:
            return None
        target_tick = self._tick_index if tick_index is None else int(tick_index)
        energy = self._entry_energy_at_tick(entry, tick_index=target_tick)
        if energy < self.prune_threshold:
            self._entries.pop(str(label or ""), None)
            self._invalidate_view_caches()
            return None
        entry["energy"] = energy
        entry["energy_tick"] = target_tick
        row = dict(entry)
        if isinstance(entry.get("coords"), dict):
            row["coords"] = dict(entry.get("coords", {}) or {})
        if isinstance(entry.get("attributes"), dict):
            row["attributes"] = dict(entry.get("attributes", {}) or {})
        row["energy"] = energy
        return row

    def _live_entry_rows(self) -> list[dict[str, Any]]:
        if self._view_cache_tick == self._tick_index and self._live_entry_rows_cache is not None:
            return self._live_entry_rows_cache

        rows: list[dict[str, Any]] = []
        dead_labels: list[str] = []
        for label, entry in self._entries.items():
            energy = self._entry_energy_at_tick(entry)
            if energy < self.prune_threshold:
                dead_labels.append(label)
                continue
            entry["energy"] = energy
            entry["energy_tick"] = self._tick_index
            row = dict(entry)
            if isinstance(entry.get("coords"), dict):
                row["coords"] = dict(entry.get("coords", {}) or {})
            if isinstance(entry.get("attributes"), dict):
                row["attributes"] = dict(entry.get("attributes", {}) or {})
            row["energy"] = energy
            rows.append(row)

        for label in dead_labels:
            self._entries.pop(label, None)

        self._view_cache_tick = self._tick_index
        self._live_entry_rows_cache = rows
        self._top_items_cache = {}
        self._anchor_items_cache = {}
        self._focus_ranked_cache = {}
        return rows

    def apply_text_packet(
        self,
        packet: dict[str, Any],
        *,
        tick_index: int,
        predicted_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._begin_tick(tick_index)
        self._decay_residual_bucket()

        normalized_text = str(packet.get("normalized_text", "") or "")
        full_units = [str(item or "") for item in ((packet.get("full_stream") or {}).get("units", []) or []) if str(item or "")]
        selected_units = [str(item.get("display_text", "") or "") for item in packet.get("sa_items", []) if isinstance(item, dict)]
        pool_input_items = [item for item in (packet.get("state_pool_sa_items", packet.get("sa_items", [])) or []) if isinstance(item, dict)]

        added_labels: list[str] = []
        total_added_energy = 0.0
        for item in pool_input_items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            entry = self._entries.get(label)
            if entry is None:
                entry = {
                    "sa_label": label,
                    "display_text": str(item.get("display_text", "") or ""),
                    "energy": 0.0,
                    "energy_tick": tick_index,
                    "first_seen_tick": tick_index,
                    "last_seen_tick": tick_index,
                    "hit_count": 0,
                    "source_type": str(item.get("source_type", "external_text") or "external_text"),
                    "sa_kind": str(item.get("sa_kind", "") or ""),
                    "channel": str(item.get("channel", "") or label.split("::", 1)[0] if "::" in label else "generic"),
                    "coords": dict(item.get("coords", {}) or {}),
                    "attributes": dict(item.get("attributes", {}) or {}),
                }
                self._entries[label] = entry
            else:
                entry["energy"] = self._entry_energy_at_tick(entry, tick_index=tick_index)
                entry["energy_tick"] = int(tick_index)
            energy = float(item.get("energy", 0.0) or 0.0)
            entry["energy"] = _round_energy(float(entry.get("energy", 0.0) or 0.0) + energy)
            entry["energy_tick"] = int(tick_index)
            entry["last_seen_tick"] = int(tick_index)
            entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
            entry["display_text"] = str(item.get("display_text", entry.get("display_text", "")) or entry.get("display_text", ""))
            entry["source_type"] = str(item.get("source_type", entry.get("source_type", "external_text")) or entry.get("source_type", "external_text"))
            if item.get("sa_kind"):
                entry["sa_kind"] = str(item.get("sa_kind", "") or "")
            if item.get("channel"):
                entry["channel"] = str(item.get("channel", "") or entry.get("channel", "generic"))
            if isinstance(item.get("coords"), dict) and item.get("coords"):
                entry["coords"] = dict(item.get("coords", {}) or {})
            if isinstance(item.get("attributes"), dict) and item.get("attributes"):
                entry["attributes"] = dict(item.get("attributes", {}) or {})
            total_added_energy += energy
            added_labels.append(label)

        self._recent_external.append(
            {
                "tick_index": int(tick_index),
                "normalized_text": normalized_text,
                "sa_labels": added_labels,
                "selected_units": selected_units,
                "raw_units": list(full_units),
                "total_units": len(full_units),
                "truncated_count": max(0, len(full_units) - len(selected_units)),
                "pool_input_count": len(pool_input_items),
                "total_added_energy": _round_energy(total_added_energy),
            }
        )
        self._append_verbatim(normalized_text)
        self._invalidate_view_caches()

        prediction_rows = [dict(item) for item in (predicted_items if predicted_items is not None else self._pending_prediction_rows) if isinstance(item, dict)]
        prediction_trace = self._record_prediction_trace(
            packet,
            full_units=full_units,
            selected_units=selected_units,
            pool_input_items=pool_input_items,
            predicted_rows=prediction_rows,
        )
        residual_update = self._ingest_residual(packet, full_units=full_units, selected_units=selected_units, pool_input_items=pool_input_items)
        handle = self._record_handle(
            normalized_text=normalized_text,
            added_labels=added_labels,
            selected_units=selected_units,
            residual_labels=residual_update["updated_labels"],
        )

        result = {
            "tick_index": int(tick_index),
            "added_label_count": len(added_labels),
            "total_added_energy": _round_energy(total_added_energy),
            "state_pool_size": len(self._entries),
            "verbatim_preview": self.verbatim_preview(),
            "latest_input_preview": join_text_units(selected_units[: self.items_per_head]),
            "residual_updates": residual_update["updated_count"],
            "residual_truncated_count": residual_update["truncated_count"],
            "pool_input_count": len(pool_input_items),
            "prediction_match_count": int(prediction_trace.get("match_count", 0) or 0),
            "prediction_unexpected_count": int(prediction_trace.get("unexpected_count", 0) or 0),
            "prediction_missed_count": int(prediction_trace.get("missed_count", 0) or 0),
            "prediction_overprediction_mass": _round_energy(float(prediction_trace.get("overprediction_mass", 0.0) or 0.0)),
            "prediction_underprediction_mass": _round_energy(float(prediction_trace.get("underprediction_mass", 0.0) or 0.0)),
            "prediction_mismatch_mass": _round_energy(float(prediction_trace.get("mismatch_mass", 0.0) or 0.0)),
            "anchor_count": len(self._hot_anchor_rows(limit=self.anchor_cache_limit)),
            "handle_id": handle.get("handle_id", ""),
        }
        self._last_pool_result = copy.deepcopy(result)
        return result

    def set_pending_prediction_items(self, items: list[dict[str, Any]] | None) -> None:
        self._pending_prediction_rows = [dict(item) for item in (items or []) if isinstance(item, dict)]

    def refresh_prediction_trace(
        self,
        packet: dict[str, Any],
        *,
        predicted_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        full_units = [str(item or "") for item in ((packet.get("full_stream") or {}).get("units", []) or []) if str(item or "")]
        selected_units = [str(item.get("display_text", "") or "") for item in packet.get("sa_items", []) if isinstance(item, dict)]
        pool_input_items = [item for item in (packet.get("state_pool_sa_items", packet.get("sa_items", [])) or []) if isinstance(item, dict)]
        prediction_rows = [dict(item) for item in (predicted_items if predicted_items is not None else self._pending_prediction_rows) if isinstance(item, dict)]
        return self._record_prediction_trace(
            packet,
            full_units=full_units,
            selected_units=selected_units,
            pool_input_items=pool_input_items,
            predicted_rows=prediction_rows,
        )

    def inject_runtime_items(
        self,
        items: list[dict[str, Any]],
        *,
        tick_index: int,
        source_type: str,
        channel: str,
        record_handle: bool = True,
    ) -> dict[str, Any]:
        self._begin_tick(tick_index)
        added_labels: list[str] = []
        total_added_energy = 0.0
        for raw in items:
            if not isinstance(raw, dict):
                continue
            label = str(raw.get("sa_label", "") or "")
            if not label:
                continue
            display_text = str(raw.get("display_text", "") or label.replace("text::", ""))
            energy = max(0.0, float(raw.get("energy", 0.0) or 0.0))
            entry = self._entries.get(label)
            if entry is None:
                entry = {
                    "sa_label": label,
                    "display_text": display_text,
                    "energy": 0.0,
                    "energy_tick": tick_index,
                    "first_seen_tick": tick_index,
                    "last_seen_tick": tick_index,
                    "hit_count": 0,
                    "source_type": source_type,
                    "sa_kind": str(raw.get("sa_kind", "") or ""),
                    "channel": str(raw.get("channel", "") or label.split("::", 1)[0] if "::" in label else "generic"),
                    "coords": copy.deepcopy(dict(raw.get("coords", {}) or {})),
                    "attributes": copy.deepcopy(dict(raw.get("attributes", {}) or {})),
                }
                self._entries[label] = entry
            else:
                entry["energy"] = self._entry_energy_at_tick(entry, tick_index=tick_index)
                entry["energy_tick"] = int(tick_index)
            entry["display_text"] = display_text
            entry["energy"] = _round_energy(float(entry.get("energy", 0.0) or 0.0) + energy)
            entry["energy_tick"] = int(tick_index)
            entry["last_seen_tick"] = int(tick_index)
            entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
            entry["source_type"] = source_type or str(entry.get("source_type", "runtime") or "runtime")
            if raw.get("sa_kind"):
                entry["sa_kind"] = str(raw.get("sa_kind", "") or "")
            if raw.get("channel"):
                entry["channel"] = str(raw.get("channel", "") or entry.get("channel", "generic"))
            if isinstance(raw.get("coords"), dict) and raw.get("coords"):
                entry["coords"] = copy.deepcopy(dict(raw.get("coords", {}) or {}))
            if isinstance(raw.get("attributes"), dict) and raw.get("attributes"):
                entry["attributes"] = copy.deepcopy(dict(raw.get("attributes", {}) or {}))
            total_added_energy += energy
            added_labels.append(label)

        self._invalidate_view_caches()
        handle_id = ""
        if record_handle:
            handle = self._record_handle(
                normalized_text=f"[{channel}]",
                added_labels=added_labels,
                selected_units=[str(item.get("display_text", "") or "") for item in items if isinstance(item, dict)],
                residual_labels=[],
            )
            handle_id = str(handle.get("handle_id", "") or "")
        return {
            "tick_index": int(tick_index),
            "channel": channel,
            "source_type": source_type,
            "added_label_count": len(added_labels),
            "total_added_energy": _round_energy(total_added_energy),
            "state_pool_size": len(self._entries),
            "handle_id": handle_id,
        }

    def read_r_state(self) -> dict[str, Any]:
        candidate_heads = [
            {"head_id": "head_global", "items": self._attention_items(limit=self.items_per_head)},
            {"head_id": "head_recent", "items": self._recent_items(limit=self.items_per_head)},
            {"head_id": "head_anchor", "items": self._anchor_items(limit=self.items_per_head)},
            {"head_id": "head_verbatim", "items": self._verbatim_items(limit=self.items_per_head)},
            {"head_id": "head_residual", "items": self._residual_items(limit=self.items_per_head)},
        ]
        heads = candidate_heads[: self.head_limit]

        merged: list[str] = []
        seen: set[str] = set()
        for head in heads:
            for item in head["items"]:
                label = str(item.get("display_text", "") or "")
                if label and label not in seen:
                    seen.add(label)
                    merged.append(label)

        return {
            "schema_id": "r_state_snapshot/v1",
            "schema_version": "1.0",
            "tick_index": self._tick_index,
            "head_count": len(heads),
            "heads": heads,
            "merged_preview": merged[: self.items_per_head * self.head_limit],
            "total_pool_size": len(self._entries),
            "available_head_ids": [head["head_id"] for head in candidate_heads],
            "residual_candidate_count": len(self._residual_bucket),
        }

    def read_a_focus(self, limit: int = 4) -> dict[str, Any]:
        return self.read_a_focus_with_bias(
            limit=limit,
            focus_gain=1.0,
            anchor_bias_gain=1.0,
            current_input_gain=1.0,
            history_suppression_gain=1.0,
            prediction_suppression_gain=1.0,
            surprise_focus_gain=1.0,
            commit=False,
        )

    def read_a_focus_with_bias(
        self,
        *,
        limit: int = 4,
        focus_gain: float = 1.0,
        anchor_bias_gain: float = 1.0,
        current_input_gain: float = 1.0,
        history_suppression_gain: float = 1.0,
        prediction_suppression_gain: float = 1.0,
        surprise_focus_gain: float = 1.0,
        commit: bool = False,
    ) -> dict[str, Any]:
        selected_labels: list[str] = []
        scored = self._focus_ranked_labels(
            focus_gain=focus_gain,
            anchor_bias_gain=anchor_bias_gain,
            current_input_gain=current_input_gain,
            history_suppression_gain=history_suppression_gain,
            prediction_suppression_gain=prediction_suppression_gain,
            surprise_focus_gain=surprise_focus_gain,
        )
        for label in scored:
            if label not in selected_labels:
                selected_labels.append(label)
            if len(selected_labels) >= limit:
                break

        if commit:
            self._commit_attention_focus(selected_labels)

        focus_items = []
        for label in selected_labels:
            entry = self._copy_live_entry(label) or {}
            if not entry:
                continue
            focus_items.append(
                {
                    "sa_label": str(label),
                    "display_text": str(entry.get("display_text", "") or ""),
                    "energy": _round_energy(float(entry.get("energy", 0.0) or 0.0)),
                    "position": int(entry.get("position", len(focus_items)) or len(focus_items)),
                    "source_type": str(entry.get("source_type", "") or ""),
                    "sa_kind": str(entry.get("sa_kind", "") or ""),
                    "channel": str(entry.get("channel", "") or ""),
                    "coords": dict(entry.get("coords", {}) or {}),
                    "attributes": dict(entry.get("attributes", {}) or {}),
                    "attention_fatigue": _round_energy(self._attention_fatigue_value(label)),
                    "attention_fatigue_multiplier": _round_energy(self._attention_fatigue_multiplier(label)),
                }
            )
        units = [str(item.get("display_text", "") or "") for item in focus_items]
        return {
            "focus_units": units,
            "focus_text": join_text_units(units),
            "focus_items": focus_items,
        }

    def snapshot_top(self, limit: int = 10) -> list[dict[str, Any]]:
        return self._top_energy_items(limit=limit)

    def snapshot_summary(self) -> dict[str, Any]:
        cached = self._snapshot_summary_cache
        if cached is not None:
            return copy.deepcopy(cached)
        recent_frames = list(self._recent_external)[-min(3, self.recent_queue_limit) :]
        anchor_rows = self._hot_anchor_rows(limit=min(6, self.anchor_cache_limit))
        payload = {
            "tick_index": self._tick_index,
            "state_pool_size": len(self._entries),
            "recent_external_count": len(self._recent_external),
            "verbatim_chars": self._verbatim_chars,
            "top": self.snapshot_top(limit=10),
            "anchor_summary": {
                "count": len(self._hot_anchor_rows(limit=self.anchor_cache_limit)),
                "top": [_clone_item_light(item) for item in anchor_rows],
            },
            "residual_summary": {
                "count": len(self._residual_bucket),
                "top": self._residual_items(limit=min(6, self.residual_limit)),
                "total_unresolved_mass": _round_energy(
                    sum(float(item.get("unresolved_mass", 0.0) or 0.0) for item in self._residual_bucket.values())
                ),
            },
            "attention_fatigue_summary": {
                "count": len(self._attention_fatigue),
                "top": self._attention_fatigue_items(limit=min(6, self.items_per_head)),
                "params": {
                    "decay": _round_energy(self.attention_object_fatigue_decay),
                    "step": _round_energy(self.attention_object_fatigue_step),
                    "gain": _round_energy(self.attention_object_fatigue_gain),
                    "max": _round_energy(self.attention_object_fatigue_max),
                    "min_multiplier": _round_energy(self.attention_object_min_multiplier),
                },
            },
            "prediction_trace": dict(self._prediction_trace),
            "handle_summary": {
                "count": len(self._handle_ring),
                "latest": dict(self._handle_ring[-1]) if self._handle_ring else {},
            },
            "recent_external_summary": [
                {
                    "tick_index": int(frame.get("tick_index", -1)),
                    "preview": _shrink_text(str(frame.get("normalized_text", "") or "")),
                    "sa_count": len(frame.get("sa_labels", []) or []),
                    "truncated_count": int(frame.get("truncated_count", 0) or 0),
                }
                for frame in recent_frames
            ],
            "size_bounds": {
                "recent_queue_limit": self.recent_queue_limit,
                "anchor_cache_limit": self.anchor_cache_limit,
                "residual_limit": self.residual_limit,
                "handle_limit": self.handle_limit,
                "verbatim_window_chars": self.verbatim_window_chars,
                "residual_unit_limit_per_tick": self.residual_unit_limit,
            },
        }
        self._snapshot_summary_cache = copy.deepcopy(payload)
        return payload

    def snapshot_sidecar(self) -> dict[str, Any]:
        cached = self._snapshot_sidecar_cache
        if cached is not None:
            return copy.deepcopy(cached)
        payload = {
            "tick_index": self._tick_index,
            "state_pool_summary": self.snapshot_summary(),
            "hot_anchor_cache": [_clone_item_light(item) for item in self._hot_anchor_rows(limit=self.anchor_cache_limit)],
            "residual_bucket": self._residual_items(limit=self.residual_limit),
            "prediction_trace": dict(self._prediction_trace),
            "handle_ring": [dict(item) for item in list(self._handle_ring)[-min(8, self.handle_limit) :]],
            "last_pool_result": dict(self._last_pool_result),
        }
        self._snapshot_sidecar_cache = copy.deepcopy(payload)
        return payload

    def export_payload(self) -> dict[str, Any]:
        self._ensure_hot_anchor_cache()
        return {
            "tick_index": self._tick_index,
            "entries": copy.deepcopy(self._entries),
            "recent_external": list(self._recent_external),
            "verbatim_window": list(self._verbatim_window),
            "verbatim_chars": self._verbatim_chars,
            "attention_fatigue": copy.deepcopy(self._attention_fatigue),
            "last_committed_focus_tick": self._last_committed_focus_tick,
            "hot_anchor_cache": copy.deepcopy(self._hot_anchor_cache),
            "residual_bucket": copy.deepcopy(self._residual_bucket),
            "handle_ring": list(self._handle_ring),
            "last_pool_result": copy.deepcopy(self._last_pool_result),
            "prediction_trace": copy.deepcopy(self._prediction_trace),
            "pending_prediction_rows": copy.deepcopy(self._pending_prediction_rows),
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        raw_tick_index = payload.get("tick_index", -1)
        self._tick_index = int(-1 if raw_tick_index is None else raw_tick_index)
        raw_entries = copy.deepcopy(payload.get("entries", {}) or {})
        self._entries = {}
        for label, entry in raw_entries.items():
            if not isinstance(entry, dict):
                continue
            row = copy.deepcopy(entry)
            row["energy"] = _round_energy(float(row.get("energy", 0.0) or 0.0))
            row["energy_tick"] = _int_or_default(row.get("energy_tick", row.get("last_seen_tick", self._tick_index)), self._tick_index)
            self._entries[str(label or "")] = row
        self._recent_external = deque(list(payload.get("recent_external", []) or []), maxlen=self.recent_queue_limit)
        self._verbatim_window = deque([str(item or "") for item in (payload.get("verbatim_window", []) or [])])
        self._verbatim_chars = int(payload.get("verbatim_chars", 0) or 0)
        raw_attention_fatigue = copy.deepcopy(payload.get("attention_fatigue", {}) or {})
        self._attention_fatigue = {}
        for label, entry in raw_attention_fatigue.items():
            if not isinstance(entry, dict):
                continue
            self._attention_fatigue[str(label or "")] = {
                "value": _round_energy(float(entry.get("value", 0.0) or 0.0)),
                "tick_index": _int_or_default(entry.get("tick_index", self._tick_index), self._tick_index),
            }
        self._last_committed_focus_tick = _int_or_default(payload.get("last_committed_focus_tick", -1), -1)
        self._hot_anchor_cache = copy.deepcopy(payload.get("hot_anchor_cache", []) or [])
        self._hot_anchor_cache_dirty = False
        self._residual_bucket = copy.deepcopy(payload.get("residual_bucket", {}) or {})
        self._handle_ring = deque(list(payload.get("handle_ring", []) or []), maxlen=self.handle_limit)
        self._last_pool_result = copy.deepcopy(payload.get("last_pool_result", {}) or {})
        self._prediction_trace = copy.deepcopy(payload.get("prediction_trace", {}) or self._prediction_trace)
        self._pending_prediction_rows = [dict(item) for item in (payload.get("pending_prediction_rows", []) or []) if isinstance(item, dict)]
        self._invalidate_view_caches()

    def verbatim_preview(self) -> str:
        return "".join(self._verbatim_window)

    def _decay_residual_bucket(self) -> None:
        if not self._residual_bucket:
            return
        decay_factor = min(0.98, max(0.0, self.decay))
        to_delete: list[str] = []
        for label, entry in self._residual_bucket.items():
            entry["unresolved_mass"] = _round_energy(float(entry.get("unresolved_mass", 0.0) or 0.0) * decay_factor)
            age = max(0, self._tick_index - int(entry.get("last_tick", self._tick_index) or self._tick_index))
            if float(entry.get("unresolved_mass", 0.0) or 0.0) < self.prune_threshold or (age > self.recent_queue_limit * 4 and float(entry.get("unresolved_mass", 0.0) or 0.0) <= self.prune_threshold * 2.0):
                to_delete.append(label)
        for label in to_delete:
            self._residual_bucket.pop(label, None)

    def _append_verbatim(self, text: str) -> None:
        if not text:
            return
        self._verbatim_window.append(text)
        self._verbatim_chars += len(text)
        while self._verbatim_window and self._verbatim_chars > self.verbatim_window_chars:
            removed = self._verbatim_window.popleft()
            self._verbatim_chars -= len(removed)

    def _ingest_residual(
        self,
        packet: dict[str, Any],
        *,
        full_units: list[str],
        selected_units: list[str],
        pool_input_items: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        updated_labels: list[str] = []
        updated_count = 0
        truncated_units = full_units[len(selected_units) :]
        for unit in truncated_units[: self.residual_unit_limit]:
            label = f"text::{unit}"
            self._upsert_residual(label=label, display_text=unit, mass=1.0, reason="truncated")
            updated_labels.append(label)
            updated_count += 1

        remaining_budget = max(0, self.residual_unit_limit - updated_count)
        if remaining_budget > 0:
            for item in (pool_input_items or []):
                suppression = float(item.get("fatigue_suppression", 0.0) or 0.0)
                if suppression <= 0.0:
                    continue
                label = str(item.get("sa_label", "") or "")
                if not label:
                    continue
                self._upsert_residual(
                    label=label,
                    display_text=str(item.get("display_text", "") or ""),
                    mass=max(0.05, suppression),
                    reason="fatigue",
                )
                updated_labels.append(label)
                updated_count += 1
                if updated_count >= self.residual_unit_limit:
                    break

        prediction_trace = dict(self._prediction_trace or {})
        remaining_budget = max(0, self.residual_unit_limit - updated_count)
        if remaining_budget > 0:
            for label in prediction_trace.get("missed_predicted_labels", []) or []:
                if updated_count >= self.residual_unit_limit:
                    break
                display_text = str(label).replace("text::", "").replace("phrase::", "")
                self._upsert_residual(label=str(label), display_text=display_text, mass=0.7, reason="prediction_miss")
                updated_labels.append(str(label))
                updated_count += 1
        remaining_budget = max(0, self.residual_unit_limit - updated_count)
        if remaining_budget > 0:
            for label in prediction_trace.get("unexpected_labels", []) or []:
                if updated_count >= self.residual_unit_limit:
                    break
                if not self._is_cognitively_comparable_label(str(label)):
                    continue
                display_text = str(label).replace("text::", "").replace("phrase::", "")
                self._upsert_residual(label=str(label), display_text=display_text, mass=0.55, reason="prediction_unexpected")
                updated_labels.append(str(label))
                updated_count += 1

        self._prune_residual_bucket_to_limit()
        return {
            "updated_labels": updated_labels[: self.residual_unit_limit],
            "updated_count": updated_count,
            "truncated_count": max(0, len(truncated_units)),
        }

    def _record_prediction_trace(
        self,
        packet: dict[str, Any],
        *,
        full_units: list[str],
        selected_units: list[str],
        pool_input_items: list[dict[str, Any]] | None = None,
        predicted_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        predicted_rows = [dict(item) for item in (predicted_rows or []) if isinstance(item, dict)]
        actual_items = [item for item in (pool_input_items or packet.get("state_pool_sa_items", packet.get("sa_items", [])) or []) if isinstance(item, dict)]
        actual_items = [item for item in actual_items if self._is_cognitively_comparable_item(item)]
        predicted_rows = [item for item in predicted_rows if self._is_cognitively_comparable_item(item)]
        if not predicted_rows:
            actual_mass = _round_energy(
                sum(float(item.get("energy", 0.0) or 0.0) for item in actual_items)
            )
            actual_labels = [str(item.get("sa_label", "") or "") for item in actual_items if str(item.get("sa_label", "") or "")]
            trace = {
                "tick_index": int(self._tick_index),
                "predicted_labels": [],
                "predicted_texts": [],
                "actual_labels": actual_labels[:24],
                "actual_texts": selected_units[:24] if selected_units else full_units[:24],
                "matched_labels": [],
                "unexpected_labels": actual_labels[:24],
                "missed_predicted_labels": [],
                "match_count": 0,
                "unexpected_count": len(actual_labels),
                "missed_count": 0,
                "predicted_mass": 0.0,
                "actual_mass": actual_mass,
                "overprediction_mass": 0.0,
                "underprediction_mass": actual_mass,
                "missed_expected_mass": 0.0,
                "unexpected_novelty_mass": actual_mass,
                "mismatch_mass": actual_mass,
                "match_mass": 0.0,
                "committed_match_mass": 0.0,
                "committed_overprediction_mass": 0.0,
                "committed_underprediction_mass": 0.0,
                "committed_mismatch_mass": 0.0,
                "predicted_commitment_mass": 0.0,
                "committed_labels": [],
                "halo_labels": [],
            }
            self._prediction_trace = trace
            return trace
        predicted_labels = [str(item.get("sa_label", "") or "") for item in predicted_rows if str(item.get("sa_label", "") or "")]
        actual_labels = [str(item.get("sa_label", "") or "") for item in actual_items if str(item.get("sa_label", "") or "")]
        predicted_set = set(predicted_labels)
        actual_set = set(actual_labels)
        matched = sorted(predicted_set & actual_set)
        unexpected = sorted(actual_set - predicted_set)
        missed = sorted(predicted_set - actual_set)
        predicted_energy = {
            str(item.get("sa_label", "") or ""): float(item.get("energy", 0.0) or 0.0)
            for item in predicted_rows
            if str(item.get("sa_label", "") or "")
        }
        predicted_commitment = {
            str(item.get("sa_label", "") or ""): _clamp(float(item.get("commitment", 0.0) or 0.0), 0.0, 1.0)
            for item in predicted_rows
            if str(item.get("sa_label", "") or "")
        }
        actual_energy: dict[str, float] = {}
        for item in actual_items:
            label = str(item.get("sa_label", "") or "")
            if not label:
                continue
            actual_energy[label] = float(actual_energy.get(label, 0.0) or 0.0) + float(item.get("energy", 0.0) or 0.0)
        predicted_mass = sum(float(value or 0.0) for value in predicted_energy.values())
        predicted_commitment_mass = sum(
            float(predicted_energy.get(label, 0.0) or 0.0) * float(predicted_commitment.get(label, 0.0) or 0.0)
            for label in predicted_energy.keys()
        )
        actual_mass = sum(float(value or 0.0) for value in actual_energy.values())
        match_mass = sum(min(float(predicted_energy.get(label, 0.0) or 0.0), float(actual_energy.get(label, 0.0) or 0.0)) for label in matched)
        committed_match_mass = sum(
            min(float(predicted_energy.get(label, 0.0) or 0.0), float(actual_energy.get(label, 0.0) or 0.0))
            * float(predicted_commitment.get(label, 0.0) or 0.0)
            for label in matched
        )
        top_predicted_energy = max((float(value or 0.0) for value in predicted_energy.values()), default=0.0)
        missed_mass = sum(
            float(predicted_energy.get(label, 0.0) or 0.0)
            * (
                max(0.12, min(1.0, float(predicted_energy.get(label, 0.0) or 0.0) / max(1e-6, top_predicted_energy)))
                if top_predicted_energy > 0.0
                else 1.0
            )
            for label in missed
        )
        committed_missed_mass = sum(
            float(predicted_energy.get(label, 0.0) or 0.0)
            * (
                max(0.12, min(1.0, float(predicted_energy.get(label, 0.0) or 0.0) / max(1e-6, top_predicted_energy)))
                if top_predicted_energy > 0.0
                else 1.0
            )
            * float(predicted_commitment.get(label, 0.0) or 0.0)
            for label in missed
        )
        unexpected_mass = sum(float(actual_energy.get(label, 0.0) or 0.0) for label in unexpected)
        shared_labels = sorted(predicted_set & actual_set)
        shared_overprediction_mass = sum(
            max(0.0, float(predicted_energy.get(label, 0.0) or 0.0) - float(actual_energy.get(label, 0.0) or 0.0))
            for label in shared_labels
        )
        committed_shared_overprediction_mass = sum(
            max(0.0, float(predicted_energy.get(label, 0.0) or 0.0) - float(actual_energy.get(label, 0.0) or 0.0))
            * float(predicted_commitment.get(label, 0.0) or 0.0)
            for label in shared_labels
        )
        shared_underprediction_mass = sum(
            max(0.0, float(actual_energy.get(label, 0.0) or 0.0) - float(predicted_energy.get(label, 0.0) or 0.0))
            for label in shared_labels
        )
        overprediction_mass = missed_mass + shared_overprediction_mass
        underprediction_mass = unexpected_mass + shared_underprediction_mass
        committed_overprediction_mass = committed_missed_mass + committed_shared_overprediction_mass
        committed_underprediction_mass = sum(
            max(0.0, float(actual_energy.get(label, 0.0) or 0.0) - float(predicted_energy.get(label, 0.0) or 0.0))
            * float(predicted_commitment.get(label, 0.0) or 0.0)
            for label in shared_labels
        )
        committed_mismatch_mass = committed_overprediction_mass + committed_underprediction_mass
        committed_labels = [
            str(item.get("sa_label", "") or "")
            for item in predicted_rows
            if float(item.get("commitment", 0.0) or 0.0) >= 0.58 and str(item.get("sa_label", "") or "")
        ]
        halo_labels = [
            str(item.get("sa_label", "") or "")
            for item in predicted_rows
            if str(item.get("prediction_role", "") or "") == "halo" and str(item.get("sa_label", "") or "")
        ]
        trace = {
            "tick_index": int(self._tick_index),
            "predicted_labels": predicted_labels[:24],
            "predicted_texts": [str(item.get("display_text", "") or "") for item in predicted_rows[:24]],
            "actual_labels": actual_labels[:24],
            "actual_texts": selected_units[:24] if selected_units else full_units[:24],
            "matched_labels": matched[:24],
            "unexpected_labels": unexpected[:24],
            "missed_predicted_labels": missed[:24],
            "match_count": len(matched),
            "unexpected_count": len(unexpected),
            "missed_count": len(missed),
            "predicted_mass": _round_energy(predicted_mass),
            "actual_mass": _round_energy(actual_mass),
            "overprediction_mass": _round_energy(overprediction_mass),
            "underprediction_mass": _round_energy(underprediction_mass),
            "missed_expected_mass": _round_energy(missed_mass),
            "unexpected_novelty_mass": _round_energy(unexpected_mass),
            "mismatch_mass": _round_energy(overprediction_mass + underprediction_mass),
            "match_mass": _round_energy(match_mass),
            "committed_match_mass": _round_energy(committed_match_mass),
            "committed_overprediction_mass": _round_energy(committed_overprediction_mass),
            "committed_underprediction_mass": _round_energy(committed_underprediction_mass),
            "committed_mismatch_mass": _round_energy(committed_mismatch_mass),
            "predicted_commitment_mass": _round_energy(predicted_commitment_mass),
            "committed_labels": committed_labels[:24],
            "halo_labels": halo_labels[:24],
        }
        self._prediction_trace = trace
        return trace

    def _is_cognitively_comparable_label(self, label: str) -> bool:
        clean = str(label or "")
        if not clean:
            return False
        if clean.startswith(("text::", "phrase::", "audio::")):
            return True
        if clean.startswith("vision_mem::"):
            return True
        return False

    def _is_cognitively_comparable_item(self, item: dict[str, Any]) -> bool:
        label = str(item.get("sa_label", "") or "")
        if self._is_cognitively_comparable_label(label):
            return True
        sa_kind = str(item.get("sa_kind", "") or "")
        if sa_kind in {"visual_focus_feature_unit", "audio_window_unit"}:
            return True
        sample_role = str(((item.get("attributes", {}) or {}).get("sample_role", "") or ""))
        if sample_role == "memory_feature":
            return True
        return False

    def _upsert_residual(self, *, label: str, display_text: str, mass: float, reason: str) -> None:
        entry = self._residual_bucket.get(label)
        if entry is None:
            entry = {
                "sa_label": label,
                "display_text": display_text,
                "unresolved_mass": 0.0,
                "hit_count": 0,
                "first_tick": self._tick_index,
                "last_tick": self._tick_index,
                "last_reason": reason,
            }
            self._residual_bucket[label] = entry
        entry["display_text"] = display_text or str(entry.get("display_text", "") or "")
        entry["unresolved_mass"] = _round_energy(float(entry.get("unresolved_mass", 0.0) or 0.0) + float(mass or 0.0))
        entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
        entry["last_tick"] = self._tick_index
        entry["last_reason"] = reason

    def _prune_residual_bucket_to_limit(self) -> None:
        if len(self._residual_bucket) <= self.residual_limit:
            return
        rows = sorted(
            self._residual_bucket.values(),
            key=lambda item: (
                -float(item.get("unresolved_mass", 0.0) or 0.0),
                -int(-1 if item.get("last_tick", -1) is None else item.get("last_tick", -1)),
                str(item.get("sa_label", "")),
            ),
        )
        keep = {str(item.get("sa_label", "") or "") for item in rows[: self.residual_limit]}
        for label in list(self._residual_bucket.keys()):
            if label not in keep:
                self._residual_bucket.pop(label, None)

    def _ensure_hot_anchor_cache(self) -> None:
        if not self._hot_anchor_cache_dirty:
            return
        rows = self._anchor_items(limit=self.anchor_cache_limit)
        self._hot_anchor_cache = [
            {
                "sa_label": str(item.get("sa_label", "") or ""),
                "display_text": str(item.get("display_text", "") or ""),
                "energy": _round_energy(float(item.get("energy", 0.0) or 0.0)),
                "anchor_score": _round_energy(float(item.get("anchor_score", 0.0) or 0.0)),
                "last_seen_tick": int(-1 if item.get("last_seen_tick", -1) is None else item.get("last_seen_tick", -1)),
            }
            for item in rows[: self.anchor_cache_limit]
        ]
        self._hot_anchor_cache_dirty = False

    def _hot_anchor_rows(self, *, limit: int) -> list[dict[str, Any]]:
        self._ensure_hot_anchor_cache()
        return [_clone_item_light(item) for item in self._hot_anchor_cache[: max(1, int(limit))]]

    def _record_handle(
        self,
        *,
        normalized_text: str,
        added_labels: list[str],
        selected_units: list[str],
        residual_labels: list[str],
    ) -> dict[str, Any]:
        handle = {
            "handle_id": f"sp_handle_{self._tick_index:06d}",
            "tick_index": self._tick_index,
            "input_preview": _shrink_text(normalized_text, limit=64),
            "selected_preview": join_text_units(selected_units[: self.items_per_head]),
            "added_labels": added_labels[: self.items_per_head],
            "anchor_labels": [row.get("sa_label", "") for row in self._anchor_items(limit=self.items_per_head)],
            "residual_labels": residual_labels[: self.items_per_head],
            "state_pool_size": len(self._entries),
            "residual_count": len(self._residual_bucket),
        }
        self._handle_ring.append(handle)
        return handle

    def _top_energy_items(self, *, limit: int) -> list[dict[str, Any]]:
        cache_key = max(1, int(limit))
        cached = self._top_items_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = sorted(
            self._live_entry_rows(),
            key=lambda item: (
                -float(item.get("energy", 0.0) or 0.0),
                -int(-1 if item.get("last_seen_tick", -1) is None else item.get("last_seen_tick", -1)),
                str(item.get("sa_label", "")),
            ),
        )
        selected = self._select_diverse_top_rows(rows, limit=cache_key)
        self._top_items_cache[cache_key] = list(selected)
        return selected

    def _recent_items(self, *, limit: int) -> list[dict[str, Any]]:
        labels: list[str] = []
        seen: set[str] = set()
        for frame in reversed(self._recent_external):
            for label in reversed(frame.get("sa_labels", []) or []):
                if not label or label in seen:
                    continue
                live_entry = self._copy_live_entry(str(label))
                if live_entry is None:
                    continue
                if label not in seen:
                    seen.add(label)
                    labels.append(label)
                if len(labels) >= limit:
                    break
            if len(labels) >= limit:
                break
        rows: list[dict[str, Any]] = []
        for label in labels[:limit]:
            live_entry = self._copy_live_entry(label)
            if live_entry is not None:
                rows.append(live_entry)
        return rows

    def _anchor_ranked_rows(self) -> list[dict[str, Any]]:
        if self._anchor_ranked_rows_cache is not None:
            return self._anchor_ranked_rows_cache
        rows: list[dict[str, Any]] = []
        for entry in self._live_entry_rows():
            age = max(0, self._tick_index - int(entry.get("last_seen_tick", self._tick_index) or self._tick_index))
            recency_bonus = max(0.0, 1.0 - age * 0.2)
            hit_bonus = min(1.5, int(entry.get("hit_count", 0) or 0) * 0.05)
            anchor_score = float(entry.get("energy", 0.0) or 0.0) * (1.0 + recency_bonus + hit_bonus)
            row = dict(entry)
            row["anchor_score"] = _round_energy(anchor_score)
            rows.append(row)
        rows.sort(
            key=lambda item: (
                -float(item.get("anchor_score", 0.0) or 0.0),
                -float(item.get("energy", 0.0) or 0.0),
                str(item.get("sa_label", "")),
            )
        )
        self._anchor_ranked_rows_cache = rows
        return self._anchor_ranked_rows_cache

    def _anchor_items(self, *, limit: int) -> list[dict[str, Any]]:
        cache_key = max(1, int(limit))
        cached = self._anchor_items_cache.get(cache_key)
        if cached is not None:
            return cached
        rows = self._anchor_ranked_rows()
        selected = self._select_diverse_top_rows(rows, limit=cache_key)
        self._anchor_items_cache[cache_key] = list(selected)
        return selected

    def _attention_competition_mass(self, row: dict[str, Any]) -> float:
        energy = max(0.0, float(row.get("energy", 0.0) or 0.0))
        source_type = str(row.get("source_type", "") or "")
        label = str(row.get("sa_label", "") or "")
        channel = str(row.get("channel", "") or "")
        weight = 1.0
        if source_type == "prediction":
            weight = 0.58
        elif source_type == "rules":
            if channel == "attr" or label.startswith("attr::"):
                weight = 0.18
            else:
                weight = 0.32
        return energy * weight

    def _attention_context(self) -> dict[str, Any]:
        if self._attention_context_cache is not None:
            return self._attention_context_cache
        latest_frame = dict(self._recent_external[-1]) if self._recent_external else {}
        latest_labels = {
            str(label or "")
            for label in (latest_frame.get("sa_labels", []) or [])
            if str(label or "")
        }
        latest_mass = float(latest_frame.get("total_added_energy", 0.0) or 0.0)
        live_rows = self._live_entry_rows()
        total_live_mass = sum(self._attention_competition_mass(row) for row in live_rows)
        background_mass = max(0.0, total_live_mass - latest_mass)
        current_pull = latest_mass / max(0.1, latest_mass + background_mass * 0.45) if latest_mass > 0.0 else 0.0
        surprise_mass = float(self._prediction_trace.get("unexpected_novelty_mass", 0.0) or 0.0)
        dissonance_mass = float(self._prediction_trace.get("overprediction_mass", 0.0) or 0.0)
        baseline_mass = latest_mass if latest_mass > 0.0 else total_live_mass
        surprise_pull = min(1.0, surprise_mass / max(0.25, baseline_mass)) if baseline_mass > 0.0 else 0.0
        dissonance_pull = min(1.0, dissonance_mass / max(0.25, total_live_mass)) if total_live_mass > 0.0 else 0.0
        ctx = {
            "latest_frame": latest_frame,
            "latest_labels": latest_labels,
            "latest_mass": _round_energy(latest_mass),
            "total_live_mass": _round_energy(total_live_mass),
            "background_mass": _round_energy(background_mass),
            "current_pull": _round_energy(current_pull),
            "surprise_pull": _round_energy(surprise_pull),
            "dissonance_pull": _round_energy(dissonance_pull),
            "verbatim_tail": {f"text::{unit}" for unit in split_text_units(self.verbatim_preview())[-self.items_per_head :]},
            "anchor_labels": {str(item.get("sa_label", "") or "") for item in self._hot_anchor_rows(limit=self.anchor_cache_limit)},
        }
        self._attention_context_cache = ctx
        return ctx

    def _attention_rank_score(
        self,
        entry: dict[str, Any],
        *,
        focus_gain: float = 1.0,
        anchor_bias_gain: float = 1.0,
        current_input_gain: float = 1.0,
        history_suppression_gain: float = 1.0,
        prediction_suppression_gain: float = 1.0,
        surprise_focus_gain: float = 1.0,
        attention_context: dict[str, Any] | None = None,
        fatigue_multiplier: float | None = None,
    ) -> float:
        label = str(entry.get("sa_label", "") or "")
        if not label:
            return 0.0
        ctx = attention_context or self._attention_context()
        latest_labels = ctx.get("latest_labels") or ()
        current_pull = float(ctx.get("current_pull", 0.0) or 0.0)
        surprise_pull = float(ctx.get("surprise_pull", 0.0) or 0.0)
        dissonance_pull = float(ctx.get("dissonance_pull", 0.0) or 0.0)
        age = max(0, self._tick_index - int(entry.get("last_seen_tick", self._tick_index) or self._tick_index))
        energy = float(entry.get("energy", 0.0) or 0.0)
        source_type = str(entry.get("source_type", "") or "")
        raw_attributes = entry.get("attributes", {}) or {}
        attributes = raw_attributes if isinstance(raw_attributes, dict) else {}
        commitment = _clamp(float(attributes.get("prediction_commitment", 0.0) or 0.0), 0.0, 1.0)
        grasp_hint = _clamp(float(attributes.get("grasp_hint", 0.0) or 0.0), 0.0, 1.0)
        prediction_role = str(attributes.get("prediction_role", "") or "")
        recency_component = max(0.0, 1.0 - age * 0.22)
        score = energy * (0.85 + 0.55 * recency_component) * max(0.5, float(focus_gain))

        if label in latest_labels:
            score *= 1.0 + (
                0.90 * current_pull * max(0.0, float(current_input_gain))
                + 0.60 * surprise_pull * max(0.0, float(surprise_focus_gain))
            ) * max(0.6, float(focus_gain))
        else:
            age_factor = 0.35 + 0.65 * min(1.0, age / max(1.0, float(self.recent_queue_limit)))
            suppression = min(
                0.92,
                current_pull * 0.68 * max(0.0, float(history_suppression_gain))
                + surprise_pull * 0.42 * max(0.0, float(surprise_focus_gain)),
            )
            if source_type in {"prediction", "rules"}:
                suppression = min(
                    0.97,
                    suppression
                    + 0.28 * max(0.0, float(prediction_suppression_gain))
                    + dissonance_pull * 0.16 * max(0.0, float(prediction_suppression_gain)),
                )
            score *= max(0.12, 1.0 - suppression * age_factor)

        verbatim_tail = ctx.get("verbatim_tail") or ()
        if label in verbatim_tail:
            score += 0.45 * max(0.5, float(focus_gain))

        anchor_labels = ctx.get("anchor_labels") or ()
        if label in anchor_labels:
            score *= 1.0 + 0.12 * max(0.0, float(anchor_bias_gain)) * max(0.5, 1.0 - current_pull)

        if source_type in {"prediction", "rules"}:
            low_grasp_pull = 1.0 - grasp_hint
            score *= 0.92 + 0.24 * low_grasp_pull
            if prediction_role == "halo":
                score *= max(0.58, 0.9 - commitment * 0.22)
            else:
                score *= 0.96 + 0.18 * low_grasp_pull + 0.08 * commitment

        score *= self._attention_fatigue_multiplier(label) if fatigue_multiplier is None else float(fatigue_multiplier)
        return max(0.0, score)

    def _attention_items(
        self,
        *,
        limit: int,
        focus_gain: float = 1.0,
        anchor_bias_gain: float = 1.0,
        current_input_gain: float = 1.0,
        history_suppression_gain: float = 1.0,
        prediction_suppression_gain: float = 1.0,
        surprise_focus_gain: float = 1.0,
    ) -> list[dict[str, Any]]:
        cache_key = (
            _round_energy(focus_gain),
            _round_energy(anchor_bias_gain),
            _round_energy(current_input_gain),
            _round_energy(history_suppression_gain),
            _round_energy(prediction_suppression_gain),
            _round_energy(surprise_focus_gain),
            max(1, int(limit)),
        )
        cached = self._attention_items_cache.get(cache_key)
        if cached is not None:
            return [dict(item) for item in cached]
        rows = self._attention_ranked_rows(
            focus_gain=focus_gain,
            anchor_bias_gain=anchor_bias_gain,
            current_input_gain=current_input_gain,
            history_suppression_gain=history_suppression_gain,
            prediction_suppression_gain=prediction_suppression_gain,
            surprise_focus_gain=surprise_focus_gain,
        )
        selected = self._select_diverse_top_rows(rows, limit=max(1, int(limit)))
        self._attention_items_cache[cache_key] = [dict(item) for item in selected]
        return [dict(item) for item in selected]

    def read_query_items(
        self,
        *,
        limit: int = 48,
        focus_gain: float = 1.0,
        anchor_bias_gain: float = 1.0,
        current_input_gain: float = 1.0,
        history_suppression_gain: float = 1.0,
        prediction_suppression_gain: float = 1.0,
        surprise_focus_gain: float = 1.0,
    ) -> list[dict[str, Any]]:
        take = max(8, int(limit))
        merged: dict[str, dict[str, Any]] = {}
        ctx = self._attention_context()
        current_pull = float(ctx.get("current_pull", 0.0) or 0.0)
        surprise_pull = float(ctx.get("surprise_pull", 0.0) or 0.0)
        latest_labels = set(ctx.get("latest_labels", set()) or set())
        attention_ranked_rows = self._attention_ranked_rows(
            focus_gain=focus_gain,
            anchor_bias_gain=anchor_bias_gain,
            current_input_gain=current_input_gain,
            history_suppression_gain=history_suppression_gain,
            prediction_suppression_gain=prediction_suppression_gain,
            surprise_focus_gain=surprise_focus_gain,
        )
        attention_score_by_label = {
            str(row.get("sa_label", "") or ""): float(row.get("attention_score", 0.0) or 0.0)
            for row in attention_ranked_rows
            if str(row.get("sa_label", "") or "")
        }
        anchor_rows = self._anchor_items(limit=max(4, take // 3))

        def put(bucket: str, row: dict[str, Any], *, bucket_gain: float = 1.0) -> None:
            label = str(row.get("sa_label", "") or "")
            if not label:
                return
            cloned = _clone_item_light(row)
            attention_score = float(cloned.get("attention_score", 0.0) or 0.0)
            if attention_score <= 0.0:
                attention_score = float(attention_score_by_label.get(label, 0.0) or 0.0)
            if attention_score <= 0.0 and bucket == "residual":
                attention_score = self._attention_rank_score(
                    cloned,
                    focus_gain=focus_gain,
                    anchor_bias_gain=anchor_bias_gain,
                    current_input_gain=current_input_gain,
                    history_suppression_gain=history_suppression_gain,
                    prediction_suppression_gain=prediction_suppression_gain,
                    surprise_focus_gain=surprise_focus_gain,
                    attention_context=ctx,
                )
            source_type = str(cloned.get("source_type", "") or "")
            energy = max(0.04, float(cloned.get("energy", 0.0) or 0.0))
            bucket_bonus_map = {
                "attention": 0.22,
                "recent": 0.48,
                "anchor": 0.14,
                "residual": 0.10,
            }
            bucket_bonus = float(bucket_bonus_map.get(bucket, 0.0))
            latest_bonus = 0.34 if label in latest_labels else 0.0
            normalized_attention = attention_score / max(0.25, attention_score + 1.0)
            multiplier = 0.92 + 0.88 * normalized_attention + bucket_bonus + latest_bonus * max(0.25, current_pull)
            if source_type in {"prediction", "rules"}:
                multiplier *= max(
                    0.08,
                    1.0
                    - (
                        0.42 * current_pull * max(0.0, float(prediction_suppression_gain))
                        + 0.32 * surprise_pull * max(0.0, float(prediction_suppression_gain))
                    ),
                )
            max_cap = 3.6
            if label in latest_labels:
                max_cap += 0.85 * current_pull + 0.65 * surprise_pull
            if source_type == "prediction":
                max_cap *= max(0.4, 1.0 - (0.45 * current_pull + 0.30 * surprise_pull))
            elif source_type == "rules":
                rule_suppression = 0.58 * current_pull + 0.42 * surprise_pull
                if str(cloned.get("channel", "") or "") == "attr" or label.startswith("attr::"):
                    rule_suppression += 0.18
                max_cap *= max(0.18, 1.0 - rule_suppression)
            query_weight = _round_energy(_clamp(energy * multiplier, 0.04, max_cap))
            cloned["attention_score"] = _round_energy(attention_score)
            cloned["query_weight"] = query_weight
            cloned["query_bucket"] = bucket
            existing = merged.get(label)
            if existing is None or float(cloned.get("query_weight", 0.0) or 0.0) > float(existing.get("query_weight", 0.0) or 0.0):
                merged[label] = cloned

        for row in self._select_diverse_top_rows(attention_ranked_rows, limit=take):
            put("attention", row, bucket_gain=1.0)
        for row in self._recent_items(limit=max(4, take // 2)):
            put("recent", row, bucket_gain=1.15)
        for row in anchor_rows:
            put("anchor", row, bucket_gain=0.92)
        for row in self._residual_items(limit=max(3, take // 4)):
            put("residual", row, bucket_gain=0.85)

        rows = list(merged.values())
        rows.sort(
            key=lambda item: (
                -float(item.get("query_weight", 0.0) or 0.0),
                -float(item.get("attention_score", 0.0) or 0.0),
                str(item.get("sa_label", "")),
            )
        )
        latest_rows = [row for row in rows if str(row.get("sa_label", "") or "") in latest_labels]
        recent_rows = [row for row in rows if str(row.get("query_bucket", "") or "") == "recent" and str(row.get("sa_label", "") or "") not in latest_labels]
        anchor_rows_ranked = [row for row in rows if str(row.get("query_bucket", "") or "") == "anchor"]
        residual_rows_ranked = [row for row in rows if str(row.get("query_bucket", "") or "") == "residual"]
        other_rows = [
            row
            for row in rows
            if row not in latest_rows and row not in recent_rows and row not in anchor_rows_ranked and row not in residual_rows_ranked
        ]
        prioritized: list[dict[str, Any]] = []
        for bucket_rows, bucket_limit in (
            (latest_rows, max(6, min(take, int(round(take * (0.38 + 0.32 * current_pull + 0.18 * surprise_pull)))))),
            (recent_rows, max(3, min(take // 3, int(round(take * 0.18))))),
            (other_rows, max(6, take)),
            (anchor_rows_ranked, max(2, take // 6)),
            (residual_rows_ranked, max(2, take // 8)),
        ):
            used = 0
            for row in bucket_rows:
                if row in prioritized:
                    continue
                prioritized.append(row)
                used += 1
                if len(prioritized) >= take or used >= bucket_limit:
                    break
            if len(prioritized) >= take:
                break
        if len(prioritized) < take:
            for row in rows:
                if row in prioritized:
                    continue
                prioritized.append(row)
                if len(prioritized) >= take:
                    break
        return prioritized[:take]

    def _residual_items(self, *, limit: int) -> list[dict[str, Any]]:
        rows = sorted(
            self._residual_bucket.values(),
            key=lambda item: (
                -float(item.get("unresolved_mass", 0.0) or 0.0),
                -int(-1 if item.get("last_tick", -1) is None else item.get("last_tick", -1)),
                str(item.get("sa_label", "")),
            ),
        )
        return [_clone_item_light(item) for item in rows[: max(1, int(limit))]]

    def _verbatim_items(self, *, limit: int) -> list[dict[str, Any]]:
        units = split_text_units(self.verbatim_preview())
        tail = units[-max(1, int(limit)) :]
        rows: list[dict[str, Any]] = []
        for position, unit in enumerate(tail):
            label = f"text::{unit}"
            live_entry = self._copy_live_entry(label)
            energy = float((live_entry or {}).get("energy", 0.0) or 0.0)
            rows.append(
                {
                    "sa_label": label,
                    "display_text": unit,
                    "energy": _round_energy(energy),
                    "position": position,
                }
            )
        return rows

    def _focus_ranked_labels(
        self,
        *,
        focus_gain: float = 1.0,
        anchor_bias_gain: float = 1.0,
        current_input_gain: float = 1.0,
        history_suppression_gain: float = 1.0,
        prediction_suppression_gain: float = 1.0,
        surprise_focus_gain: float = 1.0,
    ) -> list[str]:
        cache_key = (
            _round_energy(focus_gain),
            _round_energy(anchor_bias_gain),
            _round_energy(current_input_gain),
            _round_energy(history_suppression_gain),
            _round_energy(prediction_suppression_gain),
            _round_energy(surprise_focus_gain),
        )
        cached = self._focus_ranked_cache.get(cache_key)
        if cached is not None:
            return list(cached)
        ranked_rows = self._attention_ranked_rows(
            focus_gain=focus_gain,
            anchor_bias_gain=anchor_bias_gain,
            current_input_gain=current_input_gain,
            history_suppression_gain=history_suppression_gain,
            prediction_suppression_gain=prediction_suppression_gain,
            surprise_focus_gain=surprise_focus_gain,
        )
        result = [str(row.get("sa_label", "") or "") for row in ranked_rows if str(row.get("sa_label", "") or "")]
        self._focus_ranked_cache[cache_key] = list(result)
        return result

    def _attention_ranked_rows(
        self,
        *,
        focus_gain: float = 1.0,
        anchor_bias_gain: float = 1.0,
        current_input_gain: float = 1.0,
        history_suppression_gain: float = 1.0,
        prediction_suppression_gain: float = 1.0,
        surprise_focus_gain: float = 1.0,
    ) -> list[dict[str, Any]]:
        cache_key = (
            _round_energy(focus_gain),
            _round_energy(anchor_bias_gain),
            _round_energy(current_input_gain),
            _round_energy(history_suppression_gain),
            _round_energy(prediction_suppression_gain),
            _round_energy(surprise_focus_gain),
        )
        cached = self._attention_ranked_rows_cache.get(cache_key)
        if cached is not None:
            return cached
        ctx = self._attention_context()
        rows: list[dict[str, Any]] = []
        for entry in self._live_entry_rows():
            row = dict(entry)
            label = str(row.get("sa_label", "") or "")
            fatigue_value = self._attention_fatigue_value(label)
            fatigue_multiplier = self._attention_fatigue_multiplier(label)
            row["attention_score"] = _round_energy(
                self._attention_rank_score(
                    row,
                    focus_gain=focus_gain,
                    anchor_bias_gain=anchor_bias_gain,
                    current_input_gain=current_input_gain,
                    history_suppression_gain=history_suppression_gain,
                    prediction_suppression_gain=prediction_suppression_gain,
                    surprise_focus_gain=surprise_focus_gain,
                    attention_context=ctx,
                    fatigue_multiplier=fatigue_multiplier,
                )
            )
            row["attention_fatigue"] = _round_energy(fatigue_value)
            row["attention_fatigue_multiplier"] = _round_energy(fatigue_multiplier)
            rows.append(row)
        rows.sort(
            key=lambda item: (
                -float(item.get("attention_score", 0.0) or 0.0),
                -float(item.get("energy", 0.0) or 0.0),
                str(item.get("sa_label", "")),
            )
        )
        self._attention_ranked_rows_cache[cache_key] = rows
        return rows

    def _select_diverse_top_rows(self, rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
        max_limit = max(1, int(limit))
        visual_cap = max(1, max_limit // 2)
        selected: list[dict[str, Any]] = []
        visual_count = 0
        deferred_visual: list[dict[str, Any]] = []
        for row in rows:
            channel = str(row.get("channel", "") or "")
            if channel == "vision" and visual_count >= visual_cap:
                deferred_visual.append(row)
                continue
            selected.append(row)
            if channel == "vision":
                visual_count += 1
            if len(selected) >= max_limit:
                return selected
        for row in deferred_visual:
            selected.append(row)
            if len(selected) >= max_limit:
                break
        return selected[:max_limit]
