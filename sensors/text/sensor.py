from __future__ import annotations

import re
from collections import OrderedDict

"""
PHASE1_MINIMAL:
This sensor is intentionally a phase-1 bootstrap implementation.
It keeps explicit text-unit ingress, but it does not yet model fatigue,
hierarchical phrase packaging, or richer replay-oriented sequence metadata.
"""


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]|[^\s]", re.UNICODE)


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
    return TOKEN_RE.findall(normalized)


class TextSensor:
    def __init__(self, *, budget_limit: int) -> None:
        self.budget_limit = max(1, int(budget_limit))
        self._cache_limit = 16
        self._packet_cache: OrderedDict[tuple[str, str, int], dict] = OrderedDict()
        self._preview_limit = 32

    def ingest(self, text: str, *, tick_index: int, source_type: str = "external_text") -> dict:
        key = (str(text or ""), str(source_type or ""), int(self.budget_limit))
        cached = self._packet_cache.get(key)
        if cached is None:
            normalized = normalize_text(text)
            # Avoid normalizing twice on the hot path. `split_text_units()` remains
            # the public helper; ingest already has the normalized string.
            units = TOKEN_RE.findall(normalized) if normalized else []
            limited = units[: self.budget_limit]
            preview_units = limited[: self._preview_limit]
            sa_items_preview = [
                {
                    "sa_label": f"text::{unit}",
                    "display_text": unit,
                    "source_type": source_type,
                    "position": idx,
                    "real_energy": 1.0,
                }
                for idx, unit in enumerate(preview_units)
            ]
            cached = {
                "normalized_text": normalized,
                "units": tuple(units),
                "sa_item_count": len(limited),
                "sa_items_preview": tuple(tuple(item.items()) for item in sa_items_preview),
            }
            self._packet_cache[key] = cached
            if len(self._packet_cache) > self._cache_limit:
                self._packet_cache.popitem(last=False)
        else:
            self._packet_cache.move_to_end(key)
        units = list(cached.get("units", ()) or ())
        sa_items_preview = [dict(item) for item in (cached.get("sa_items_preview", ()) or ())]
        return {
            "tick_index": int(tick_index),
            "source_type": source_type,
            "input_text": str(text or ""),
            "normalized_text": str(cached.get("normalized_text", "") or ""),
            "units": units,
            "sa_item_count": int(cached.get("sa_item_count", len(units[: self.budget_limit])) or 0),
            "sa_items": sa_items_preview,
        }
