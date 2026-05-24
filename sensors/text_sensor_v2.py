# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import deque
from typing import Any

from observatory_v2.schema_tools import load_schema, validate_or_raise


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


def normalize_text(text: str) -> str:
    raw = str(text or "")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t\f\v]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def split_text_units(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return _TOKEN_RE.findall(normalized)


def join_text_units(units: list[str]) -> str:
    if not units:
        return ""
    out: list[str] = []
    prev_is_word = False
    for unit in units:
        is_word = bool(re.fullmatch(r"[A-Za-z0-9_]+", unit))
        if out and is_word and prev_is_word:
            out.append(" ")
        out.append(unit)
        prev_is_word = is_word
    return "".join(out)


class TextSensorV2:
    def __init__(
        self,
        *,
        budget_limit: int,
        fatigue_window: int,
        fatigue_threshold: int,
        max_suppression: float,
    ) -> None:
        self.budget_limit = max(1, int(budget_limit))
        self.fatigue_window = max(1, int(fatigue_window))
        self.fatigue_threshold = max(0, int(fatigue_threshold))
        self.max_suppression = max(0.0, min(1.0, float(max_suppression)))
        self._recent_unit_frames: deque[set[str]] = deque(maxlen=self.fatigue_window)
        self._sensor_tick = 0

    def ingest(self, text: str, *, tick_index: int, source_type: str = "external_text") -> dict[str, Any]:
        envelope = {
            "schema_id": "text_input_envelope/v1",
            "schema_version": "1.0",
            "text": str(text or ""),
            "source_type": source_type,
        }
        validate_or_raise(envelope, load_schema("text_input_envelope.schema.json"), label="text_input_envelope")

        normalized_text = normalize_text(envelope["text"])
        units = split_text_units(normalized_text)
        self._sensor_tick += 1

        prior_occurrences: dict[str, int] = {}
        for frame in self._recent_unit_frames:
            for unit in frame:
                prior_occurrences[unit] = prior_occurrences.get(unit, 0) + 1

        selected_units = units[: self.budget_limit]
        sa_items: list[dict[str, Any]] = []
        suppressed_count = 0
        total_energy = 0.0
        for position, unit in enumerate(selected_units):
            prior = prior_occurrences.get(unit, 0)
            suppression = 0.0
            if prior >= self.fatigue_threshold:
                suppression = min(self.max_suppression, prior / max(1, self.fatigue_window))
            if suppression > 0:
                suppressed_count += 1
            energy = round(max(0.05, 1.0 - suppression), 4)
            total_energy += energy
            sa_items.append(
                {
                    "sa_label": f"text::{unit}",
                    "display_text": unit,
                    "energy": energy,
                    "position": position,
                    "source_type": source_type,
                    "fatigue_suppression": round(suppression, 4),
                    "prior_occurrence": prior,
                }
            )

        packet = {
            "schema_id": "text_sensor_packet/v1",
            "schema_version": "1.0",
            "sensor_name": "text_sensor_v2",
            "tick_index": int(tick_index),
            "sensor_tick": self._sensor_tick,
            "input_text": envelope["text"],
            "normalized_text": normalized_text,
            "source_type": source_type,
            "budget_limit": self.budget_limit,
            "budget_used": len(sa_items),
            "total_units": len(units),
            "sa_items": sa_items,
            "sampling_summary": {
                "budget_limit": self.budget_limit,
                "budget_used": len(sa_items),
                "total_units": len(units),
                "truncated": len(units) > self.budget_limit,
                "unit_preview": selected_units[:8],
            },
            "fatigue_summary": {
                "window": self.fatigue_window,
                "threshold": self.fatigue_threshold,
                "suppressed_count": suppressed_count,
                "distinct_recent_units": sum(len(frame) for frame in self._recent_unit_frames),
                "total_energy": round(total_energy, 4),
            },
            "full_stream": {
                "normalized_text": normalized_text,
                "units": units,
            },
            "sa_flow": [item["sa_label"] for item in sa_items],
        }
        validate_or_raise(packet, load_schema("text_sensor_packet.schema.json"), label="text_sensor_packet")
        self._recent_unit_frames.append(set(units))
        return packet

    def export_payload(self) -> dict[str, Any]:
        return {
            "sensor_tick": self._sensor_tick,
            "recent_unit_frames": [sorted(frame) for frame in self._recent_unit_frames],
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._sensor_tick = int(payload.get("sensor_tick", 0) or 0)
        frames = payload.get("recent_unit_frames", []) or []
        self._recent_unit_frames = deque(
            [set(str(unit or "") for unit in (frame or []) if str(unit or "")) for frame in frames],
            maxlen=self.fatigue_window,
        )
