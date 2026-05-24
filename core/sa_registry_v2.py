# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SAPrototype:
    sa_id: str
    kind: str
    pattern_units: tuple[str, ...]
    boost: float = 1.0
    family: str = "text"


class SARegistryV2:
    def __init__(self) -> None:
        self._prototypes: list[SAPrototype] = []
        self._pattern_counts: dict[tuple[str, ...], int] = {}
        self._dynamic_phrase_min_observations = 2
        self._dynamic_phrase_max_len = 4
        self._install_builtin_prototypes()

    def _install_builtin_prototypes(self) -> None:
        builtins = [
            SAPrototype("phrase::今天_天气", "text_phrase", ("今", "天", "天", "气"), boost=1.12),
            SAPrototype("phrase::天气_不错", "text_phrase", ("天", "气", "不", "错"), boost=1.15),
            SAPrototype("phrase::我_想", "text_phrase", ("我", "想"), boost=1.08),
            SAPrototype("phrase::想_出门", "text_phrase", ("想", "出", "门"), boost=1.12),
            SAPrototype("phrase::有点", "text_phrase", ("有", "点"), boost=1.08),
            SAPrototype("phrase::算了", "text_phrase", ("算", "了"), boost=1.08),
            SAPrototype("phrase::不说了", "text_phrase", ("不", "说", "了"), boost=1.12),
        ]
        self._prototypes.extend(builtins)

    def register_phrase(self, units: list[str], *, boost: float = 1.1, family: str = "text_phrase") -> SAPrototype | None:
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        if len(clean_units) < 2:
            return None
        sa_id = f"phrase::{'_'.join(clean_units)}"
        existing = self.lookup(sa_id)
        if existing:
            return existing
        prototype = SAPrototype(sa_id=sa_id, kind="text_phrase", pattern_units=tuple(clean_units), boost=float(boost), family=family)
        self._prototypes.append(prototype)
        return prototype

    def lookup(self, sa_id: str) -> SAPrototype | None:
        for prototype in self._prototypes:
            if prototype.sa_id == sa_id:
                return prototype
        return None

    def all_prototypes(self) -> list[SAPrototype]:
        return list(self._prototypes)

    def observe_sequence(self, units: list[str]) -> list[SAPrototype]:
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        if len(clean_units) < 2:
            return []
        created: list[SAPrototype] = []
        max_len = min(self._dynamic_phrase_max_len, len(clean_units))
        for size in range(2, max_len + 1):
            for start in range(0, len(clean_units) - size + 1):
                pattern = tuple(clean_units[start : start + size])
                if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for unit in pattern for ch in unit):
                    continue
                next_count = int(self._pattern_counts.get(pattern, 0) or 0) + 1
                self._pattern_counts[pattern] = next_count
                if next_count < self._dynamic_phrase_min_observations:
                    continue
                boost = 1.04 + min(0.18, 0.03 * float(size))
                prototype = self.register_phrase(list(pattern), boost=boost, family="learned_text_phrase")
                if prototype is not None and prototype not in created:
                    created.append(prototype)
        return created

    def export_payload(self) -> dict[str, Any]:
        return {
            "prototypes": [
                {
                    "sa_id": item.sa_id,
                    "kind": item.kind,
                    "pattern_units": list(item.pattern_units),
                    "boost": float(item.boost),
                    "family": item.family,
                }
                for item in self._prototypes
            ],
            "pattern_counts": [
                {
                    "pattern_units": list(pattern),
                    "count": int(count),
                }
                for pattern, count in sorted(self._pattern_counts.items(), key=lambda item: (-int(item[1]), item[0]))
            ],
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        rows = payload.get("prototypes", []) or []
        loaded: list[SAPrototype] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            sa_id = str(row.get("sa_id", "") or "")
            kind = str(row.get("kind", "") or "")
            pattern_units = tuple(str(unit or "") for unit in (row.get("pattern_units", []) or []) if str(unit or ""))
            if not sa_id or not kind or not pattern_units:
                continue
            loaded.append(
                SAPrototype(
                    sa_id=sa_id,
                    kind=kind,
                    pattern_units=pattern_units,
                    boost=float(row.get("boost", 1.0) or 1.0),
                    family=str(row.get("family", "text") or "text"),
                )
            )
        if loaded:
            self._prototypes = loaded
        raw_pattern_counts = payload.get("pattern_counts", []) or []
        restored_counts: dict[tuple[str, ...], int] = {}
        if isinstance(raw_pattern_counts, list):
            for row in raw_pattern_counts:
                if not isinstance(row, dict):
                    continue
                pattern_units = tuple(str(unit or "") for unit in (row.get("pattern_units", []) or []) if str(unit or ""))
                if len(pattern_units) < 2:
                    continue
                restored_counts[pattern_units] = max(0, int(row.get("count", 0) or 0))
        self._pattern_counts = restored_counts

    def compete(self, units: list[str], *, source_type: str, max_items: int) -> dict[str, Any]:
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        unit_items: list[dict[str, Any]] = []
        for position, unit in enumerate(clean_units):
            unit_items.append(
                {
                    "sa_label": f"text::{unit}",
                    "display_text": unit,
                    "energy": 1.0,
                    "position": position,
                    "source_type": source_type,
                    "sa_kind": "text_unit",
                    "family": "text",
                    "competition_source": "base_unit",
                }
            )

        phrase_hits: list[dict[str, Any]] = []
        prototypes = sorted(self._prototypes, key=lambda item: (-len(item.pattern_units), item.sa_id))
        for prototype in prototypes:
            pattern = list(prototype.pattern_units)
            plen = len(pattern)
            if plen <= 1 or plen > len(clean_units):
                continue
            for start in range(0, len(clean_units) - plen + 1):
                if clean_units[start : start + plen] != pattern:
                    continue
                phrase_hits.append(
                    {
                        "sa_label": prototype.sa_id,
                        "display_text": "".join(pattern),
                        "energy": round(float(prototype.boost), 4),
                        "position": start,
                        "source_type": source_type,
                        "sa_kind": prototype.kind,
                        "family": prototype.family,
                        "competition_source": "registry_phrase",
                        "pattern_units": pattern,
                    }
                )
                break

        selected: list[dict[str, Any]] = []
        for hit in sorted(phrase_hits, key=lambda item: (item["position"], -len(item.get("pattern_units", [])))):
            if len(selected) >= max_items:
                break
            selected.append(hit)
        for item in unit_items:
            if len(selected) >= max_items:
                break
            selected.append(item)

        selected.sort(
            key=lambda item: (
                int(item.get("position", 0) or 0),
                0 if str(item.get("sa_kind", "") or "").endswith("unit") else -1,
                str(item.get("sa_label", "")),
            )
        )
        return {
            "selected_items": selected[:max_items],
            "phrase_hits": phrase_hits[:max_items],
            "prototype_count": len(self._prototypes),
        }
