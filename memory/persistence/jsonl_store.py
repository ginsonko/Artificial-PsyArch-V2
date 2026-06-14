from __future__ import annotations

import json
from pathlib import Path

from .base import PersistenceWriteResult


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class JsonlMemoryPersistence:
    """
    Local-file adapter for the MemoryPersistenceAdapter contract.

    This adapter is intentionally simple: it writes one authoritative snapshot
    payload per JSONL line and can reload recent snapshots into a fresh
    MemoryStore. It is a paper-material persistence boundary, not a production
    database replacement.
    """

    backend_name = "jsonl"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.write_count = 0
        self.error_count = 0
        self.last_error = ""
        self._runtime_state_path = self.path.with_suffix(self.path.suffix + ".runtime.json")

    def write_snapshot(
        self,
        *,
        snapshot: dict,
        features: dict,
        vector: list[float],
        energy_profile: dict[str, float],
        energy_mass: float,
        numeric_features: dict[str, list[float]],
        relation_features: dict,
        previous_memory_id: str,
        learned_vector: list[float] | None = None,
    ) -> PersistenceWriteResult:
        payload = {
            "schema_id": "apv2_jsonl_persistence_record/v1",
            "snapshot": _jsonable(dict(snapshot or {})),
            "features": _jsonable(dict(features or {})),
            "vector": _jsonable(list(vector or [])),
            "vector_spaces": _jsonable(
                {
                    "hash_vector": list(vector or []),
                    "online_learned_vector": list(learned_vector or []),
                }
            ),
            "energy_profile": _jsonable(dict(energy_profile or {})),
            "energy_mass": float(energy_mass or 0.0),
            "numeric_features": _jsonable({str(key): list(value or []) for key, value in dict(numeric_features or {}).items()}),
            "relation_features": _jsonable(dict(relation_features or {})),
            "previous_memory_id": str(previous_memory_id or ""),
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            self.write_count += 1
            rows = 1 + len(payload["snapshot"].get("items", []) or []) + len(payload["features"].get("labels", []) or [])
            return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_snapshot", rows_written=rows)
        except OSError as exc:
            self.error_count += 1
            self.last_error = str(exc)
            return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_snapshot", rows_written=0, error=self.last_error)

    def summary(self) -> dict:
        exists = self.path.exists()
        return {
            "backend": self.backend_name,
            "enabled": True,
            "path": str(self.path),
            "exists": exists,
            "bytes": int(self.path.stat().st_size) if exists else 0,
            "write_count": int(self.write_count),
            "error_count": int(self.error_count),
            "last_error": self.last_error,
            "meaning": "local_jsonl_authoritative_snapshot_boundary",
        }

    def load_recent_snapshots(self, *, memory_kind: str | None = None, limit_per_kind: int = 128) -> list[dict]:
        if not self.path.exists():
            return []
        cap = max(1, int(limit_per_kind))
        grouped: dict[str, list[dict]] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if not clean:
                    continue
                try:
                    payload = json.loads(clean)
                except json.JSONDecodeError:
                    continue
                snapshot = dict(payload.get("snapshot", {}) or {})
                kind = str(snapshot.get("memory_kind", "") or "")
                if not kind:
                    continue
                if memory_kind is not None and kind != str(memory_kind or ""):
                    continue
                grouped.setdefault(kind, []).append(snapshot)
        rows: list[dict] = []
        for kind, snapshots in sorted(grouped.items()):
            ordered = sorted(snapshots, key=lambda row: int(row.get("tick_index", -1) or -1))
            rows.extend(ordered[-cap:])
        rows.sort(key=lambda row: (str(row.get("memory_kind", "") or ""), int(row.get("tick_index", -1) or -1), str(row.get("memory_id", "") or "")))
        return rows

    def snapshot_by_id(self, memory_id: str) -> dict | None:
        clean_id = str(memory_id or "")
        if not clean_id or not self.path.exists():
            return None
        found = None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                clean = line.strip()
                if not clean:
                    continue
                try:
                    payload = json.loads(clean)
                except json.JSONDecodeError:
                    continue
                snapshot = dict(payload.get("snapshot", {}) or {})
                if str(snapshot.get("memory_id", "") or "") == clean_id:
                    found = snapshot
        return found

    def write_runtime_state(self, *, state: dict) -> PersistenceWriteResult:
        payload = {
            "schema_id": "apv21_runtime_state_record/v1",
            "state": _jsonable(dict(state or {})),
        }
        try:
            with self._runtime_state_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_runtime_state", rows_written=1)
        except OSError as exc:
            self.error_count += 1
            self.last_error = str(exc)
            return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_runtime_state", rows_written=0, error=self.last_error)

    def load_runtime_state(self) -> dict | None:
        if not self._runtime_state_path.exists():
            return None
        try:
            payload = json.loads(self._runtime_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        state = payload.get("state", {}) if isinstance(payload, dict) else {}
        return dict(state) if isinstance(state, dict) else None
