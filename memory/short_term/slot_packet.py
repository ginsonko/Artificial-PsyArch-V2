from __future__ import annotations

from collections import Counter
from math import exp


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class ShortTermSlotPacketBuilder:
    """
    Build a tick-level narrative packet from the short-term buffer layers.

    The packet is not a hidden answer table. It is the internal, narrative
    inner-sense packet that AP can later read as state-field evidence.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        capacity: int = 32,
        base_virtual_budget: float = 0.72,
        item_real_fraction: float = 0.06,
        item_min_virtual: float = 0.02,
        item_max_virtual: float = 0.14,
        item_rank_decay: float = 0.86,
        item_order_decay: float = 0.92,
        summary_ratio: float = 0.18,
        order_ratio: float = 0.16,
        continuity_ratio: float = 0.14,
        rhythm_ratio: float = 0.10,
        load_floor: float = 0.25,
        continuity_gain: float = 0.35,
        order_gain: float = 0.28,
        rhythm_gain: float = 0.22,
        working_memory_fill_limit: int = 8,
        focus_merge_limit: int = 32,
    ) -> None:
        self.enabled = bool(enabled)
        self.capacity = max(1, int(capacity))
        self.base_virtual_budget = max(0.0, float(base_virtual_budget))
        self.item_real_fraction = max(0.0, min(0.25, float(item_real_fraction)))
        self.item_min_virtual = max(0.0, float(item_min_virtual))
        self.item_max_virtual = max(0.0, float(item_max_virtual))
        self.item_rank_decay = _clamp(float(item_rank_decay), 0.5, 0.98)
        self.item_order_decay = _clamp(float(item_order_decay), 0.5, 0.99)
        self.summary_ratio = _clamp(float(summary_ratio), 0.0, 1.0)
        self.order_ratio = _clamp(float(order_ratio), 0.0, 1.0)
        self.continuity_ratio = _clamp(float(continuity_ratio), 0.0, 1.0)
        self.rhythm_ratio = _clamp(float(rhythm_ratio), 0.0, 1.0)
        self.load_floor = _clamp(float(load_floor), 0.05, 1.0)
        self.continuity_gain = max(0.0, float(continuity_gain))
        self.order_gain = max(0.0, float(order_gain))
        self.rhythm_gain = max(0.0, float(rhythm_gain))
        self.working_memory_fill_limit = max(1, int(working_memory_fill_limit))
        self.focus_merge_limit = max(1, int(focus_merge_limit))

    def build(
        self,
        *,
        tick_index: int,
        focus_items: list[dict],
        focus_continuation_trace: dict | None = None,
        short_term_memory_trace: dict | None = None,
        rhythm_trace: dict | None = None,
        runtime_load_trace: dict | None = None,
        state_rows: list[dict] | None = None,
    ) -> dict:
        now_tick = int(tick_index)
        if not self.enabled:
            return {
                "schema_id": "short_term_slot_packet/v1",
                "tick_index": now_tick,
                "enabled": False,
                "items": [],
                "slot_summary": {},
                "policy": "disabled",
            }

        focus_rows = self._clean_rows(focus_items)[: self.capacity]
        focus_labels = [str(row.get("sa_label", "") or "") for row in focus_rows if str(row.get("sa_label", "") or "")]
        slot_context = self._slot_context(
            focus_continuation_trace=focus_continuation_trace or {},
            short_term_memory_trace=short_term_memory_trace or {},
            rhythm_trace=rhythm_trace or {},
            runtime_load_trace=runtime_load_trace or {},
        )
        base_budget = self.base_virtual_budget * slot_context["continuity"] * slot_context["continuity_quality"] * slot_context["load_gate"]
        slot_virtual_budget = _clamp(base_budget, 0.05, max(0.1, self.base_virtual_budget * 1.6))
        items = []
        order_labels = []
        for index, row in enumerate(focus_rows):
            label = str(row.get("sa_label", "") or "")
            if not label:
                continue
            weight = self._normalized_weight(row, index=index, focus_rows=focus_rows)
            rank_coeff = pow(self.item_rank_decay, max(0, index))
            order_coeff = self._order_coeff(index=index, total=len(focus_rows))
            modality_coeff = self._modality_coeff(row)
            virtual_energy = slot_virtual_budget * weight * rank_coeff * order_coeff * modality_coeff
            virtual_energy = _clamp(virtual_energy, self.item_min_virtual, self.item_max_virtual)
            real_energy = max(0.0, float(row.get("real_energy", 0.0) or 0.0) * self.item_real_fraction)
            order_labels.append(label)
            packet_row = {
                "sa_label": f"short_term_slot::item::{label}",
                "display_text": str(row.get("display_text", label) or label),
                "family": "short_term_slot",
                "source_type": "short_term_slot",
                "real_energy": _round4(real_energy),
                "virtual_energy": _round4(virtual_energy),
                "attention_gain": _round4(virtual_energy * 0.34),
                "cognitive_pressure": _round4(real_energy - virtual_energy),
                "anchor_meta": {
                    "schema_id": "short_term_slot_item/v1",
                    "slot_index": now_tick,
                    "slot_rank": int(index),
                    "relative_order": int(index),
                    "slot_mass": _round4(virtual_energy),
                    "family_mass": _round4(float(row.get("real_energy", 0.0) or 0.0) + float(row.get("virtual_energy", 0.0) or 0.0)),
                    "continuity_score": _round4(slot_context["continuity"]),
                    "continuity_quality": _round4(slot_context["continuity_quality"]),
                    "slot_order_coeff": _round4(order_coeff),
                    "slot_rank_coeff": _round4(rank_coeff),
                    "modality_coeff": _round4(modality_coeff),
                    "source_sa_label": label,
                    "source_family": str(row.get("family", "") or ""),
                    "source_type": str(row.get("source_type", "") or ""),
                },
            }
            if "position" in row:
                packet_row["anchor_meta"]["position"] = row.get("position")
            if isinstance(row.get("numeric_features"), dict):
                packet_row["numeric_features"] = dict(row.get("numeric_features", {}) or {})
            items.append(packet_row)

        order_items = self._build_order_items(
            order_labels,
            tick_index=now_tick,
            slot_context=slot_context,
        )
        summary_item = self._build_summary_item(
            focus_rows,
            tick_index=now_tick,
            slot_context=slot_context,
            slot_virtual_budget=slot_virtual_budget,
            items=items,
        )
        continuity_item = self._build_continuity_item(
            tick_index=now_tick,
            slot_context=slot_context,
            focus_continuation_trace=focus_continuation_trace or {},
            short_term_memory_trace=short_term_memory_trace or {},
        )
        rhythm_items = self._build_rhythm_items(
            tick_index=now_tick,
            slot_context=slot_context,
            rhythm_trace=rhythm_trace or {},
        )
        packet_items = self._compose_packet_items(
            summary_item=summary_item,
            items=items,
            order_items=order_items,
            continuity_item=continuity_item,
            rhythm_items=rhythm_items,
        )
        slot_summary = {
            "schema_id": "short_term_slot_summary/v1",
            "tick_index": now_tick,
            "slot_virtual_budget": _round4(slot_virtual_budget),
            "continuity": _round4(slot_context["continuity"]),
            "continuity_quality": _round4(slot_context["continuity_quality"]),
            "load_gate": _round4(slot_context["load_gate"]),
            "rhythm_gate": _round4(slot_context["rhythm_gate"]),
            "focus_count": len(focus_rows),
            "item_count": len(items),
            "summary_ratio": _round4(self.summary_ratio),
            "order_ratio": _round4(self.order_ratio),
            "continuity_ratio": _round4(self.continuity_ratio),
            "rhythm_ratio": _round4(self.rhythm_ratio),
        }
        return {
            "schema_id": "short_term_slot_packet/v1",
            "tick_index": now_tick,
            "enabled": True,
            "items": packet_items,
            "slot_summary": slot_summary,
            "focus_labels": focus_labels[: self.focus_merge_limit],
            "policy": "tick_level_narrative_packet_for_internal_state_field_competition",
            "source_views": {
                "focus_continuation": dict(focus_continuation_trace or {}),
                "short_term_memory": dict(short_term_memory_trace or {}),
                "rhythm": dict(rhythm_trace or {}),
                "runtime_load": dict(runtime_load_trace or {}),
            },
        }

    def _clean_rows(self, items: list[dict]) -> list[dict]:
        rows = []
        seen = set()
        for item in items or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            rows.append(dict(item))
        return rows

    def _normalized_weight(self, row: dict, *, index: int, focus_rows: list[dict]) -> float:
        energy = max(
            0.0,
            float(row.get("real_energy", 0.0) or 0.0)
            + float(row.get("virtual_energy", 0.0) or 0.0) * 0.65
            + float(row.get("attention_gain", row.get("focus_score", 0.0)) or 0.0) * 0.35
            + abs(float(row.get("cognitive_pressure", 0.0) or 0.0)) * 0.2,
        )
        total = 0.0
        weights = []
        for candidate in focus_rows:
            candidate_energy = max(
                0.0,
                float(candidate.get("real_energy", 0.0) or 0.0)
                + float(candidate.get("virtual_energy", 0.0) or 0.0) * 0.65
                + float(candidate.get("attention_gain", candidate.get("focus_score", 0.0)) or 0.0) * 0.35
                + abs(float(candidate.get("cognitive_pressure", 0.0) or 0.0)) * 0.2,
            )
            weights.append(candidate_energy)
            total += candidate_energy
        normalized = energy / max(1e-6, total)
        if not weights:
            normalized = 1.0 / max(1, len(focus_rows) or 1)
        return max(0.0, normalized)

    def _order_coeff(self, *, index: int, total: int) -> float:
        if total <= 1:
            return 1.0
        if index <= 0:
            return 1.0
        return pow(self.item_order_decay, float(index))

    def _modality_coeff(self, row: dict) -> float:
        family = str(row.get("family", "") or "").lower()
        source_type = str(row.get("source_type", "") or "").lower()
        if family in {"text", "learned_text_phrase"} or source_type == "external_text":
            return 1.0
        if family in {"vision", "vision_object", "vision_channel"} or source_type.startswith("vision"):
            return 0.96
        if family in {"audio", "audio_event", "audio_channel"} or source_type.startswith("audio"):
            return 0.94
        if family in {"cognitive_feeling", "emotion"}:
            return 0.9
        return 0.88

    def _build_order_items(self, labels: list[str], *, tick_index: int, slot_context: dict) -> list[dict]:
        rows = []
        total = max(1, len(labels))
        for index, label in enumerate(labels[: self.capacity]):
            order_strength = pow(self.item_order_decay, float(index))
            virtual_energy = _clamp(
                self.base_virtual_budget * self.order_ratio * order_strength * slot_context["continuity"],
                self.item_min_virtual * 0.5,
                self.item_max_virtual,
            )
            rows.append(
            {
                "sa_label": f"short_term_slot::order::{index}::{label}",
                "display_text": f"order {index + 1} of {total}",
                "family": "short_term_slot",
                "source_type": "short_term_slot",
                "real_energy": 0.0,
                "virtual_energy": _round4(virtual_energy),
                "attention_gain": _round4(virtual_energy * 0.28),
                "cognitive_pressure": _round4(-virtual_energy),
                    "anchor_meta": {
                        "schema_id": "short_term_slot_order/v1",
                        "slot_index": int(tick_index),
                        "relative_order": int(index),
                        "slot_order_coeff": _round4(order_strength),
                        "slot_mass": _round4(virtual_energy),
                        "source_sa_label": label,
                    },
                }
            )
        return rows

    def _build_summary_item(self, focus_rows: list[dict], *, tick_index: int, slot_context: dict, slot_virtual_budget: float, items: list[dict]) -> dict:
        total_item_mass = sum(float(row.get("virtual_energy", 0.0) or 0.0) for row in items)
        virtual_energy = _clamp(
            total_item_mass * self.summary_ratio + slot_virtual_budget * 0.25,
            self.item_min_virtual,
            max(self.item_max_virtual, total_item_mass + slot_virtual_budget),
        )
        summary_text = " ".join(str(row.get("display_text", row.get("sa_label", "")) or "") for row in focus_rows[:4]).strip()
        if not summary_text:
            summary_text = "short term narrative summary"
        return {
            "sa_label": "short_term_slot::summary",
            "display_text": summary_text[:120],
            "family": "short_term_slot",
            "source_type": "short_term_slot",
            "real_energy": _round4(slot_virtual_budget * self.item_real_fraction),
            "virtual_energy": _round4(virtual_energy),
            "attention_gain": _round4(virtual_energy * 0.30),
            "cognitive_pressure": _round4(slot_virtual_budget * self.item_real_fraction - virtual_energy),
            "anchor_meta": {
                "schema_id": "short_term_slot_summary/v1",
                "slot_index": int(tick_index),
                "slot_mass": _round4(total_item_mass),
                "summary_ratio": _round4(self.summary_ratio),
                "item_count": len(items),
                "focus_count": len(focus_rows),
            },
        }

    def _build_continuity_item(
        self,
        *,
        tick_index: int,
        slot_context: dict,
        focus_continuation_trace: dict,
        short_term_memory_trace: dict,
    ) -> dict:
        continuity = float(slot_context.get("continuity", 0.0) or 0.0)
        continuity_quality = float(slot_context.get("continuity_quality", 0.0) or 0.0)
        strength = max(
            0.0,
            continuity * self.continuity_ratio
            + continuity_quality * self.continuity_gain * 0.5
            + float((focus_continuation_trace or {}).get("continuation_strength", 0.0) or 0.0) * 0.25
            + float((short_term_memory_trace or {}).get("last_recall", {}).get("score", 0.0) or 0.0) * 0.08,
        )
        return {
            "sa_label": "short_term_slot::continuity",
            "display_text": "continuity",
            "family": "short_term_slot",
            "source_type": "short_term_slot",
            "real_energy": 0.0,
            "virtual_energy": _round4(strength),
            "attention_gain": _round4(strength * 0.30),
            "cognitive_pressure": _round4(-strength),
            "anchor_meta": {
                "schema_id": "short_term_slot_continuity/v1",
                "slot_index": int(tick_index),
                "continuity": _round4(continuity),
                "continuity_quality": _round4(continuity_quality),
                "source_episode_id": int((focus_continuation_trace or {}).get("active_episode_id", -1) or -1),
                "readback_available": bool((short_term_memory_trace or {}).get("last_recall", {}).get("available", False)),
            },
        }

    def _build_rhythm_items(self, *, tick_index: int, slot_context: dict, rhythm_trace: dict) -> list[dict]:
        rhythm_items = []
        dominant = dict((rhythm_trace or {}).get("dominant_peak", {}) or {})
        family = dict((rhythm_trace or {}).get("family", {}) or {})
        channels = dict((rhythm_trace or {}).get("channels", {}) or {})
        period = int(
            dominant.get("center_delta_t", 0)
            or dominant.get("period_ticks", 0)
            or family.get("period_ticks", 0)
            or channels.get("period_ticks", 0)
            or 0
        )
        if period <= 0:
            return rhythm_items
        rhythm_items_source = list((rhythm_trace or {}).get("items", []) or [])
        groove = max(
            0.0,
            float((rhythm_items_source[0] if rhythm_items_source else {}).get("real_energy", 0.0) or 0.0),
            float(dominant.get("groove", 0.0) or 0.0),
            float(family.get("groove", 0.0) or 0.0),
            float(channels.get("groove", 0.0) or 0.0),
            float(family.get("phase_expectation", 0.0) or 0.0),
            float(channels.get("phase_expectation", 0.0) or 0.0),
        )
        phase = str(
            dominant.get("phase_label", "")
            or family.get("phase_label", "")
            or channels.get("phase_label", "")
            or family.get("family_key", "")
            or channels.get("family_key", "")
            or ""
        )
        virtual_energy = _clamp(
            self.base_virtual_budget * self.rhythm_ratio * (0.4 + groove) * slot_context["rhythm_gate"],
            self.item_min_virtual * 0.5,
            self.item_max_virtual,
        )
        rhythm_items.append(
            {
                "sa_label": "short_term_slot::rhythm",
                "display_text": phase or "rhythm",
                "family": "short_term_slot",
                "source_type": "short_term_slot",
                "real_energy": 0.0,
                "virtual_energy": _round4(virtual_energy),
                "attention_gain": _round4(virtual_energy * 0.30),
                "cognitive_pressure": _round4(-virtual_energy),
                "anchor_meta": {
                    "schema_id": "short_term_slot_rhythm/v1",
                    "slot_index": int(tick_index),
                    "period_ticks": int(period),
                    "phase": phase,
                    "groove": _round4(groove),
                    "time_to_next": family.get("time_to_next", channels.get("time_to_next")),
                    "phase_expectation": _round4(float(family.get("phase_expectation", channels.get("phase_expectation", 0.0)) or 0.0)),
                },
            }
        )
        return rhythm_items

    def _slot_context(self, *, focus_continuation_trace: dict, short_term_memory_trace: dict, rhythm_trace: dict, runtime_load_trace: dict) -> dict:
        continuity = float((focus_continuation_trace or {}).get("continuation_strength", 0.0) or 0.0)
        history = list((focus_continuation_trace or {}).get("recent_entries", []) or [])
        continuity_quality = 1.0
        if history:
            continuity_quality = min(1.0, sum(float(row.get("continuity_score", 0.0) or 0.0) for row in history) / max(1.0, len(history)))
        readback_available = bool((short_term_memory_trace or {}).get("last_recall", {}).get("available", False))
        memory_strength = float((short_term_memory_trace or {}).get("last_recall", {}).get("score", 0.0) or 0.0)
        rhythm_family = dict((rhythm_trace or {}).get("family", {}) or {})
        rhythm_channels = dict((rhythm_trace or {}).get("channels", {}) or {})
        rhythm_dominant = dict((rhythm_trace or {}).get("dominant_peak", {}) or {})
        rhythm_strength = max(
            float(rhythm_dominant.get("confidence", 0.0) or 0.0),
            float(rhythm_family.get("confidence", 0.0) or 0.0),
            float(rhythm_family.get("phase_expectation", 0.0) or 0.0),
            float(rhythm_channels.get("phase_expectation", 0.0) or 0.0),
            float(rhythm_channels.get("groove", 0.0) or 0.0),
        )
        load_ratio = float((runtime_load_trace or {}).get("channels", {}).get("load_ratio", 0.0) or 0.0)
        load_gate = max(self.load_floor, 1.0 - min(0.8, load_ratio * 0.45))
        rhythm_gate = 0.7 + min(0.3, rhythm_strength * 0.3)
        if readback_available:
            continuity_quality = min(1.0, continuity_quality + min(0.18, memory_strength * 0.12))
        return {
            "continuity": _clamp(continuity, 0.05, 1.0),
            "continuity_quality": _clamp(continuity_quality, 0.05, 1.0),
            "load_gate": _clamp(load_gate, self.load_floor, 1.0),
            "rhythm_gate": _clamp(rhythm_gate, 0.4, 1.0),
        }

    def _compose_packet_items(
        self,
        *,
        summary_item: dict,
        items: list[dict],
        order_items: list[dict],
        continuity_item: dict,
        rhythm_items: list[dict],
    ) -> list[dict]:
        cap = max(1, int(self.capacity))
        rows: list[dict] = []
        seen: set[str] = set()

        def add(row: dict | None) -> bool:
            if not isinstance(row, dict):
                return False
            label = str(row.get("sa_label", "") or "")
            if not label or label in seen or len(rows) >= cap:
                return False
            seen.add(label)
            rows.append(row)
            return True

        add(summary_item)
        if len(rows) >= cap:
            return rows

        tail: list[dict] = []
        if cap >= 2:
            tail.append(continuity_item)
        if rhythm_items and cap >= 6:
            tail.extend(list(rhythm_items)[: max(1, min(len(rhythm_items), cap // 8))])

        remaining = max(0, cap - len(rows) - len(tail))
        if remaining > 0:
            item_quota = min(len(items), max(1, int(remaining * 0.65))) if items else 0
            order_quota = min(len(order_items), max(0, remaining - item_quota))
            if order_items and order_quota <= 0 and remaining >= 2 and item_quota > 0:
                item_quota -= 1
                order_quota = 1
            for row in list(items)[:item_quota]:
                add(row)
            for row in list(order_items)[:order_quota]:
                add(row)

        for row in tail:
            add(row)
        fill_index = 0
        fill_pool = list(items) + list(order_items) + list(rhythm_items)
        while len(rows) < cap and fill_index < len(fill_pool):
            add(fill_pool[fill_index])
            fill_index += 1
        return rows[:cap]

    def _dedupe(self, rows: list[dict]) -> list[dict]:
        seen = set()
        deduped = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            label = str(row.get("sa_label", "") or "")
            if not label or label in seen:
                continue
            seen.add(label)
            deduped.append(row)
        return deduped
