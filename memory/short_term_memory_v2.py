# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from typing import Any


class ShortTermMemoryV2:
    def __init__(self, *, max_items: int = 64, successor_tail_limit: int = 8) -> None:
        self.max_items = max(4, int(max_items))
        self.successor_tail_limit = max(2, int(successor_tail_limit))
        self._items: deque[dict[str, Any]] = deque(maxlen=self.max_items)

    def append(self, item: dict[str, Any]) -> None:
        self._items.append(dict(item))

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self._items)

    def recent_focus_units(self, limit: int | None = None) -> list[str]:
        take = int(limit or self.successor_tail_limit)
        out: list[str] = []
        for item in list(self._items)[-take:]:
            units = item.get("focus_units", []) or []
            out.extend(str(unit or "") for unit in units if str(unit or ""))
        return out[-take:]

    def export_payload(self) -> dict[str, Any]:
        return {"items": list(self._items), "max_items": self.max_items, "successor_tail_limit": self.successor_tail_limit}

    def import_payload(self, payload: dict[str, Any]) -> None:
        self._items = deque(list(payload.get("items", []) or []), maxlen=self.max_items)
