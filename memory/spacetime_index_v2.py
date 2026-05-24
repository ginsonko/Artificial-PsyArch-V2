# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import defaultdict
from bisect import bisect_left, bisect_right, insort
import json
from pathlib import Path
from typing import Any


def _round4(value: float) -> float:
    return round(float(value), 4)


class SpacetimeIndexV2:
    def __init__(
        self,
        *,
        backend: str = "bucket_grid",
        time_bucket_size: int = 8,
        space_bucket_size: float = 0.25,
        default_time_radius: int = 24,
        default_space_radius: float = 0.45,
    ) -> None:
        self.backend = self._normalize_backend(backend)
        self.time_bucket_size = max(1, int(time_bucket_size))
        self.space_bucket_size = max(0.05, float(space_bucket_size))
        self.default_time_radius = max(1, int(default_time_radius))
        self.default_space_radius = max(0.05, float(default_space_radius))
        self._records: dict[str, dict[str, Any]] = {}
        self._time_buckets: dict[int, set[str]] = defaultdict(set)
        self._space_buckets: dict[tuple[int, int, int], set[str]] = defaultdict(set)
        self._time_sorted: list[tuple[int, str]] = []

    def add(self, memory: dict[str, Any]) -> None:
        memory_id = str(memory.get("memory_id", "") or "")
        if not memory_id:
            return
        existing = self._records.get(memory_id)
        if isinstance(existing, dict):
            self._remove_indexes(memory_id, existing)
        spacetime = dict(memory.get("spacetime", {}) or {})
        record = {
            "memory_id": memory_id,
            "tick_index": int(-1 if memory.get("tick_index", -1) is None else memory.get("tick_index", -1)),
            "has_space": bool(spacetime.get("has_space", False)),
            "x": float(spacetime.get("x", 0.0) or 0.0),
            "y": float(spacetime.get("y", 0.0) or 0.0),
            "z": float(spacetime.get("z", 0.0) or 0.0),
            "local_order_span": int(spacetime.get("local_order_span", 0) or 0),
            "modalities": list(memory.get("modalities", []) or []),
        }
        self._records[memory_id] = record
        self._time_buckets[self._time_bucket(record["tick_index"])].add(memory_id)
        insort(self._time_sorted, (int(record["tick_index"]), memory_id))
        if record["has_space"]:
            self._space_buckets[self._space_bucket(record["x"], record["y"], record["z"])].add(memory_id)

    def neighbors_for_memory(
        self,
        memory: dict[str, Any],
        *,
        limit: int = 12,
        time_radius: int | None = None,
        space_radius: float | None = None,
    ) -> list[dict[str, Any]]:
        memory_id = str(memory.get("memory_id", "") or "")
        if not memory_id:
            return []
        spacetime = dict(memory.get("spacetime", {}) or {})
        center_tick = int(-1 if memory.get("tick_index", -1) is None else memory.get("tick_index", -1))
        radius_t = max(1, int(time_radius or self.default_time_radius))
        radius_s = max(0.05, float(space_radius or self.default_space_radius))
        candidates: set[str] = set()
        if self._time_sorted:
            left = bisect_left(self._time_sorted, (center_tick - radius_t, ""))
            right = bisect_right(self._time_sorted, (center_tick + radius_t, "\uffff"))
            for _, candidate_memory_id in self._time_sorted[left:right]:
                candidates.add(candidate_memory_id)
        else:
            start_bucket = self._time_bucket(center_tick - radius_t)
            end_bucket = self._time_bucket(center_tick + radius_t)
            for bucket in range(start_bucket, end_bucket + 1):
                candidates.update(self._time_buckets.get(bucket, set()))
        if bool(spacetime.get("has_space", False)):
            cx = float(spacetime.get("x", 0.0) or 0.0)
            cy = float(spacetime.get("y", 0.0) or 0.0)
            cz = float(spacetime.get("z", 0.0) or 0.0)
            bx, by, bz = self._space_bucket(cx, cy, cz)
            bucket_radius = max(1, int(math.ceil(radius_s / self.space_bucket_size)))
            for dx in range(-bucket_radius, bucket_radius + 1):
                for dy in range(-bucket_radius, bucket_radius + 1):
                    for dz in range(-bucket_radius, bucket_radius + 1):
                        candidates.update(self._space_buckets.get((bx + dx, by + dy, bz + dz), set()))
        candidates.discard(memory_id)
        rows: list[dict[str, Any]] = []
        source_modalities = set(str(item or "") for item in (memory.get("modalities", []) or []) if str(item or ""))
        for other_id in candidates:
            other = self._records.get(other_id)
            if not other:
                continue
            time_gap = abs(int(other.get("tick_index", center_tick) or center_tick) - center_tick)
            if time_gap > radius_t * 3:
                continue
            temporal_bonus = 1.0 / (1.0 + float(time_gap))
            has_space = bool(spacetime.get("has_space", False)) and bool(other.get("has_space", False))
            space_bonus = 0.0
            if has_space:
                dist = math.sqrt(
                    (float(other.get("x", 0.0) or 0.0) - float(spacetime.get("x", 0.0) or 0.0)) ** 2
                    + (float(other.get("y", 0.0) or 0.0) - float(spacetime.get("y", 0.0) or 0.0)) ** 2
                    + (float(other.get("z", 0.0) or 0.0) - float(spacetime.get("z", 0.0) or 0.0)) ** 2
                )
                space_bonus = max(0.0, 1.0 - dist / max(radius_s, 1e-6))
            modality_overlap = len(source_modalities & set(str(item or "") for item in (other.get("modalities", []) or []) if str(item or "")))
            modality_bonus = min(1.0, modality_overlap * 0.25)
            order_gap = abs(int(other.get("local_order_span", 0) or 0) - int(spacetime.get("local_order_span", 0) or 0))
            order_bonus = 1.0 / (1.0 + float(order_gap))
            score = 0.48 * temporal_bonus + 0.28 * space_bonus + 0.14 * modality_bonus + 0.10 * order_bonus
            rows.append(
                {
                    "memory_id": other_id,
                    "spacetime_score": _round4(score),
                    "distance_time": int(time_gap),
                    "space_bonus": _round4(space_bonus),
                    "temporal_bonus": _round4(temporal_bonus),
                    "modality_bonus": _round4(modality_bonus),
                    "order_bonus": _round4(order_bonus),
                }
            )
        rows.sort(key=lambda item: (-float(item.get("spacetime_score", 0.0) or 0.0), int(item.get("distance_time", 0) or 0), item["memory_id"]))
        return rows[: max(1, int(limit))]

    def summary(self) -> dict[str, Any]:
        return {
            "requested_backend": self.backend,
            "effective_backend": self.backend,
            "engine": self.backend,
            "record_count": len(self._records),
            "time_bucket_size": self.time_bucket_size,
            "space_bucket_size": self.space_bucket_size,
            "default_time_radius": self.default_time_radius,
            "default_space_radius": self.default_space_radius,
            "time_bucket_count": len(self._time_buckets),
            "space_bucket_count": len(self._space_buckets),
            "time_sorted_count": len(self._time_sorted),
            "bundle_format": "layered_v2",
        }

    def export_payload(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "time_bucket_size": self.time_bucket_size,
            "space_bucket_size": self.space_bucket_size,
            "default_time_radius": self.default_time_radius,
            "default_space_radius": self.default_space_radius,
            "records": self._records,
            "time_sorted": self._time_sorted,
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self.backend = self._normalize_backend(str(payload.get("backend", self.backend) or self.backend))
        self.time_bucket_size = max(1, int(payload.get("time_bucket_size", self.time_bucket_size) or self.time_bucket_size))
        self.space_bucket_size = max(0.05, float(payload.get("space_bucket_size", self.space_bucket_size) or self.space_bucket_size))
        self.default_time_radius = max(1, int(payload.get("default_time_radius", self.default_time_radius) or self.default_time_radius))
        self.default_space_radius = max(0.05, float(payload.get("default_space_radius", self.default_space_radius) or self.default_space_radius))
        self._records = {str(key): dict(value) for key, value in (payload.get("records", {}) or {}).items() if str(key)}
        self._time_buckets = defaultdict(set)
        self._space_buckets = defaultdict(set)
        self._time_sorted = []
        for record in self._records.values():
            memory_id = str(record.get("memory_id", "") or "")
            if not memory_id:
                continue
            self._time_buckets[self._time_bucket(int(record.get("tick_index", -1) or -1))].add(memory_id)
            insort(self._time_sorted, (int(record.get("tick_index", -1) or -1), memory_id))
            if bool(record.get("has_space", False)):
                self._space_buckets[self._space_bucket(float(record.get("x", 0.0) or 0.0), float(record.get("y", 0.0) or 0.0), float(record.get("z", 0.0) or 0.0))].add(memory_id)

    def save_bundle(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.export_payload()
        legacy_path = directory / "spacetime_index_v2.json"
        legacy_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        records_path = directory / "spacetime_records.jsonl"
        lines = [json.dumps(record, ensure_ascii=False) for _, record in sorted(self._records.items(), key=lambda item: item[0])]
        records_path.write_text("\n".join(lines), encoding="utf-8")

        meta = {
            "schema_id": "spacetime_index_bundle/v2",
            "schema_version": "2.0",
            "backend": self.backend,
            "time_bucket_size": self.time_bucket_size,
            "space_bucket_size": self.space_bucket_size,
            "default_time_radius": self.default_time_radius,
            "default_space_radius": self.default_space_radius,
            "record_count": len(self._records),
            "files": {
                "legacy_json": legacy_path.name,
                "records_jsonl": records_path.name,
            },
        }
        meta_path = directory / "spacetime_index_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(meta_path), "record_count": len(self._records)}

    def load_bundle(self, directory: Path) -> dict[str, Any]:
        meta_path = directory / "spacetime_index_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.backend = self._normalize_backend(str(meta.get("backend", self.backend) or self.backend))
            self.time_bucket_size = max(1, int(meta.get("time_bucket_size", self.time_bucket_size) or self.time_bucket_size))
            self.space_bucket_size = max(0.05, float(meta.get("space_bucket_size", self.space_bucket_size) or self.space_bucket_size))
            self.default_time_radius = max(1, int(meta.get("default_time_radius", self.default_time_radius) or self.default_time_radius))
            self.default_space_radius = max(0.05, float(meta.get("default_space_radius", self.default_space_radius) or self.default_space_radius))
            files = dict(meta.get("files", {}) or {})
            records_path = directory / str(files.get("records_jsonl", "spacetime_records.jsonl") or "spacetime_records.jsonl")
            if records_path.exists():
                self._records = {}
                self._time_buckets = defaultdict(set)
                self._space_buckets = defaultdict(set)
                self._time_sorted = []
                for line in records_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(record, dict):
                        continue
                    memory_id = str(record.get("memory_id", "") or "")
                    if not memory_id:
                        continue
                    self._records[memory_id] = dict(record)
                    self._time_buckets[self._time_bucket(int(record.get("tick_index", -1) or -1))].add(memory_id)
                    insort(self._time_sorted, (int(record.get("tick_index", -1) or -1), memory_id))
                    if bool(record.get("has_space", False)):
                        self._space_buckets[self._space_bucket(float(record.get("x", 0.0) or 0.0), float(record.get("y", 0.0) or 0.0), float(record.get("z", 0.0) or 0.0))].add(memory_id)
                return {"ok": True, "path": str(meta_path), "record_count": len(self._records), "loaded_via": "layered_v2"}
            legacy_path = directory / str(files.get("legacy_json", "spacetime_index_v2.json") or "spacetime_index_v2.json")
            if legacy_path.exists():
                payload = json.loads(legacy_path.read_text(encoding="utf-8"))
                self.import_payload(payload)
                return {"ok": True, "path": str(legacy_path), "record_count": len(self._records), "loaded_via": "legacy_json"}
            return {"ok": False, "error": "spacetime bundle files missing", "path": str(meta_path)}
        path = directory / "spacetime_index_v2.json"
        if not path.exists():
            return {"ok": False, "error": "spacetime bundle not found", "path": str(path)}
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.import_payload(payload)
        return {"ok": True, "path": str(path), "record_count": len(self._records), "loaded_via": "legacy_json"}

    def _time_bucket(self, tick_index: int) -> int:
        return int(tick_index) // self.time_bucket_size

    def _space_bucket(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        return (
            int(math.floor(float(x) / self.space_bucket_size)),
            int(math.floor(float(y) / self.space_bucket_size)),
            int(math.floor(float(z) / self.space_bucket_size)),
        )

    def _normalize_backend(self, backend: str) -> str:
        clean = str(backend or "bucket_grid").strip().lower()
        if clean not in {"bucket_grid", "bundle_only"}:
            clean = "bucket_grid"
        return clean

    def _remove_indexes(self, memory_id: str, record: dict[str, Any]) -> None:
        bucket = self._time_bucket(int(record.get("tick_index", -1) or -1))
        self._time_buckets.get(bucket, set()).discard(memory_id)
        if not self._time_buckets.get(bucket):
            self._time_buckets.pop(bucket, None)
        if bool(record.get("has_space", False)):
            space_key = self._space_bucket(
                float(record.get("x", 0.0) or 0.0),
                float(record.get("y", 0.0) or 0.0),
                float(record.get("z", 0.0) or 0.0),
            )
            self._space_buckets.get(space_key, set()).discard(memory_id)
            if not self._space_buckets.get(space_key):
                self._space_buckets.pop(space_key, None)
        target = (int(record.get("tick_index", -1) or -1), memory_id)
        left = bisect_left(self._time_sorted, target)
        right = bisect_right(self._time_sorted, target)
        for idx in range(left, right):
            if self._time_sorted[idx] == target:
                self._time_sorted.pop(idx)
                break
