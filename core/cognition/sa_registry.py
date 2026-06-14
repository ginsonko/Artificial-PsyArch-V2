from __future__ import annotations

from dataclasses import dataclass

"""
PHASE1_MINIMAL:
The registry currently promotes repeated local phrases only.
It does not yet implement the stronger APV2.1 split between raw stimulus
units, competition results, higher-order objects, and learnable handles.
"""


@dataclass(frozen=True)
class SAPrototype:
    sa_id: str
    kind: str
    pattern_units: tuple[str, ...]
    boost: float = 1.0
    family: str = "text"


class SARegistry:
    def __init__(
        self,
        *,
        dynamic_phrase_min_observations: int = 2,
        dynamic_phrase_max_len: int = 3,
        dynamic_phrase_scan_budget: int = 256,
        dynamic_phrase_emit_budget: int = 32,
    ) -> None:
        self._dynamic_phrase_min_observations = max(2, int(dynamic_phrase_min_observations))
        self._dynamic_phrase_max_len = max(2, int(dynamic_phrase_max_len))
        self._dynamic_phrase_scan_budget = max(0, int(dynamic_phrase_scan_budget))
        self._dynamic_phrase_emit_budget = max(0, int(dynamic_phrase_emit_budget))
        self._prototypes: list[SAPrototype] = []
        self._prototype_by_id: dict[str, SAPrototype] = {}
        self._prototype_by_pattern: dict[tuple[str, ...], SAPrototype] = {}
        self._pattern_counts: dict[tuple[str, ...], int] = {}

    def observe_sequence(self, units: list[str]) -> list[SAPrototype]:
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        created: list[SAPrototype] = []
        if len(clean_units) < 2:
            return created
        if self._dynamic_phrase_scan_budget <= 0 or self._dynamic_phrase_emit_budget <= 0:
            return created
        # Phrase promotion is a learnable auxiliary path and must be fixed-budget:
        # wide 1024-token ticks should not rescan every possible n-gram.
        upper = min(len(clean_units), self._dynamic_phrase_max_len)
        scanned = 0
        for size in range(2, upper + 1):
            for start in range(0, len(clean_units) - size + 1):
                if scanned >= self._dynamic_phrase_scan_budget or len(created) >= self._dynamic_phrase_emit_budget:
                    return [item for item in created if item is not None]
                scanned += 1
                pattern = tuple(clean_units[start : start + size])
                next_count = int(self._pattern_counts.get(pattern, 0) or 0) + 1
                self._pattern_counts[pattern] = next_count
                if next_count < self._dynamic_phrase_min_observations:
                    continue
                if self.lookup(f"phrase::{'_'.join(pattern)}") is not None:
                    continue
                created.append(
                    self.register_phrase(
                        list(pattern),
                        boost=1.05 + min(0.15, 0.03 * float(size)),
                        family="learned_text_phrase",
                    )
                )
        return [item for item in created if item is not None]

    def register_phrase(self, units: list[str], *, boost: float = 1.1, family: str = "text_phrase") -> SAPrototype | None:
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        if len(clean_units) < 2:
            return None
        pattern = tuple(clean_units)
        existing = self._prototype_by_pattern.get(pattern)
        if existing is not None:
            return existing
        prototype = SAPrototype(
            sa_id=f"phrase::{'_'.join(clean_units)}",
            kind="text_phrase",
            pattern_units=pattern,
            boost=float(boost),
            family=family,
        )
        self._prototypes.append(prototype)
        self._prototype_by_id[prototype.sa_id] = prototype
        self._prototype_by_pattern[prototype.pattern_units] = prototype
        return prototype

    def lookup(self, sa_id: str) -> SAPrototype | None:
        return self._prototype_by_id.get(str(sa_id or ""))

    def compete(self, units: list[str], *, source_type: str, max_items: int) -> dict:
        clean_units = [str(unit or "") for unit in units if str(unit or "")]
        selected: list[dict] = []
        # Competition is phrase-first then unit passthrough, but it must not scan
        # every learned prototype for every input position. Use exact pattern
        # lookup over the current sequence instead.
        seen_phrase_ids: set[str] = set()
        upper = min(len(clean_units), self._dynamic_phrase_max_len)
        remaining_phrase_slots = max(0, int(max_items) - len(clean_units))
        for start in range(0, len(clean_units)):
            if remaining_phrase_slots <= 0:
                break
            max_size_here = min(upper, len(clean_units) - start)
            for size in range(max_size_here, 1, -1):
                prototype = self._prototype_by_pattern.get(tuple(clean_units[start : start + size]))
                if prototype is None or prototype.sa_id in seen_phrase_ids:
                    continue
                seen_phrase_ids.add(prototype.sa_id)
                selected.append(
                    {
                        "sa_label": prototype.sa_id,
                        "display_text": "".join(prototype.pattern_units),
                        "source_type": source_type,
                        "position": start,
                        "real_energy": round(float(prototype.boost), 4),
                        "sa_kind": prototype.kind,
                        "family": prototype.family,
                        "pattern_units": list(prototype.pattern_units),
                    }
                )
                remaining_phrase_slots -= 1
                if remaining_phrase_slots <= 0:
                    break
        for idx, unit in enumerate(clean_units):
            selected.append(
                {
                    "sa_label": f"text::{unit}",
                    "display_text": unit,
                    "source_type": source_type,
                    "position": idx,
                    "real_energy": 1.0,
                    "sa_kind": "text_unit",
                    "family": "text",
                }
            )
        selected.sort(
            key=lambda item: (
                int(item.get("position", 0)),
                0 if str(item.get("sa_kind", "")).endswith("phrase") else 1,
                str(item.get("sa_label", "")),
            )
        )
        return {"selected_items": selected[:max_items]}
