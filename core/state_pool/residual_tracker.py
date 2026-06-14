from __future__ import annotations

from collections import defaultdict


def _round4(value: float) -> float:
    return round(float(value), 4)


def _int_or_default(value, default: int = -1) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


class ResidualTracker:
    """
    Bounded unresolved-mass tracker for prediction mismatch and suppressed evidence.

    This is not a second state pool. It is a fixed-size side structure used by
    R_state head_residual and white-box explanation.
    """

    def __init__(
        self,
        *,
        limit: int,
        unit_limit_per_tick: int,
        decay: float,
        prune_threshold: float,
    ) -> None:
        self.limit = max(1, int(limit))
        self.unit_limit_per_tick = max(1, int(unit_limit_per_tick))
        self.decay = max(0.0, min(0.98, float(decay)))
        self.prune_threshold = max(0.0, float(prune_threshold))
        self._bucket: dict[str, dict] = {}
        self._tick_index = -1

    def begin_tick(self, tick_index: int) -> None:
        self._tick_index = int(tick_index)
        if not self._bucket:
            return
        to_delete: list[str] = []
        for label, entry in self._bucket.items():
            entry["unresolved_mass"] = _round4(float(entry.get("unresolved_mass", 0.0) or 0.0) * self.decay)
            age = max(0, self._tick_index - int(entry.get("last_tick", self._tick_index) or self._tick_index))
            if float(entry.get("unresolved_mass", 0.0) or 0.0) < self.prune_threshold and age > 2:
                to_delete.append(label)
        for label in to_delete:
            self._bucket.pop(label, None)

    def remove(self, label: str) -> None:
        clean = str(label or "")
        if clean:
            self._bucket.pop(clean, None)

    def ingest_prediction_trace(self, trace: dict) -> dict:
        self._tick_index = int(trace.get("tick_index", self._tick_index) if trace else self._tick_index)
        updated_labels: list[str] = []
        updated_count = 0

        for label in trace.get("missed_predicted_labels", []) or []:
            if updated_count >= self.unit_limit_per_tick:
                break
            display = _display_for_label(label)
            mass = float((trace.get("predicted_energy_by_label", {}) or {}).get(label, 0.0) or 0.0)
            self.upsert(label=str(label), display_text=display, mass=max(0.18, mass), reason="prediction_miss")
            updated_labels.append(str(label))
            updated_count += 1

        for label in trace.get("unexpected_labels", []) or []:
            if updated_count >= self.unit_limit_per_tick:
                break
            display = _display_for_label(label)
            mass = float((trace.get("actual_energy_by_label", {}) or {}).get(label, 0.0) or 0.0)
            self.upsert(label=str(label), display_text=display, mass=max(0.14, mass * 0.75), reason="prediction_unexpected")
            updated_labels.append(str(label))
            updated_count += 1

        self._prune_to_limit()
        return {
            "updated_labels": updated_labels,
            "updated_count": updated_count,
            "bucket_count": len(self._bucket),
        }

    def upsert(self, *, label: str, display_text: str, mass: float, reason: str) -> None:
        clean = str(label or "")
        if not clean:
            return
        entry = self._bucket.get(clean)
        if entry is None:
            entry = {
                "sa_label": clean,
                "display_text": str(display_text or _display_for_label(clean)),
                "unresolved_mass": 0.0,
                "residual_boost": 0.0,
                "hit_count": 0,
                "first_tick": int(self._tick_index),
                "last_tick": int(self._tick_index),
                "last_reason": str(reason or "unknown"),
                "reason_counts": defaultdict(int),
            }
            self._bucket[clean] = entry
        entry["display_text"] = str(display_text or entry.get("display_text", "") or _display_for_label(clean))
        entry["unresolved_mass"] = _round4(float(entry.get("unresolved_mass", 0.0) or 0.0) + max(0.0, float(mass or 0.0)))
        entry["residual_boost"] = _round4(min(0.85, 0.08 + float(entry.get("unresolved_mass", 0.0) or 0.0) * 0.18))
        entry["hit_count"] = int(entry.get("hit_count", 0) or 0) + 1
        entry["last_tick"] = int(self._tick_index)
        entry["last_reason"] = str(reason or "unknown")
        reason_counts = entry.get("reason_counts")
        if not isinstance(reason_counts, defaultdict):
            reason_counts = defaultdict(int, dict(reason_counts or {}))
            entry["reason_counts"] = reason_counts
        reason_counts[str(reason or "unknown")] += 1

    def items(self, *, limit: int) -> list[dict]:
        rows = sorted(
            self._bucket.values(),
            key=lambda item: (
                -float(item.get("unresolved_mass", 0.0) or 0.0),
                -int(item.get("last_tick", -1) or -1),
                str(item.get("sa_label", "") or ""),
            ),
        )
        return [self._public_row(item) for item in rows[: max(1, int(limit))]]

    def snapshot(self, *, limit: int | None = None) -> dict:
        rows = self.items(limit=self.limit if limit is None else int(limit))
        return {
            "schema_id": "state_pool_residual_bucket/v1",
            "tick_index": int(self._tick_index),
            "count": len(self._bucket),
            "limit": int(self.limit),
            "unit_limit_per_tick": int(self.unit_limit_per_tick),
            "total_unresolved_mass": _round4(sum(float(item.get("unresolved_mass", 0.0) or 0.0) for item in self._bucket.values())),
            "top": rows,
        }

    def _prune_to_limit(self) -> None:
        if len(self._bucket) <= self.limit:
            return
        keep = {str(item.get("sa_label", "") or "") for item in self.items(limit=self.limit)}
        for label in list(self._bucket.keys()):
            if label not in keep:
                self._bucket.pop(label, None)

    def _public_row(self, item: dict) -> dict:
        reason_counts = item.get("reason_counts", {}) or {}
        return {
            "sa_label": str(item.get("sa_label", "") or ""),
            "display_text": str(item.get("display_text", "") or ""),
            "unresolved_mass": _round4(float(item.get("unresolved_mass", 0.0) or 0.0)),
            "residual_boost": _round4(float(item.get("residual_boost", 0.0) or 0.0)),
            "hit_count": int(item.get("hit_count", 0) or 0),
            "first_tick": _int_or_default(item.get("first_tick", -1), -1),
            "last_tick": _int_or_default(item.get("last_tick", -1), -1),
            "last_reason": str(item.get("last_reason", "") or ""),
            "reason_counts": {str(key): int(value) for key, value in dict(reason_counts).items()},
        }


def _display_for_label(label: str) -> str:
    clean = str(label or "")
    for prefix in ("text::", "phrase::", "audio::", "vision::", "vision_mem::"):
        if clean.startswith(prefix):
            return clean.removeprefix(prefix).replace("_", "")
    return clean
