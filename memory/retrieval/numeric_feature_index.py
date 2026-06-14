from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from memory.retrieval.faiss_index import FaissHnswConfig, FaissHnswIndex


def _round4(value: float) -> float:
    return round(float(value), 4)


def _coerce_vector(values: object, *, dim: int) -> list[float]:
    cap = max(1, int(dim))
    if isinstance(values, dict):
        raw = [values[key] for key in sorted(values)]
    elif isinstance(values, (list, tuple)):
        raw = list(values)
    else:
        raw = [values]
    vector: list[float] = []
    for value in raw[:cap]:
        try:
            vector.append(float(value))
        except (TypeError, ValueError):
            vector.append(0.0)
    if len(vector) < cap:
        vector.extend([0.0] * (cap - len(vector)))
    return vector[:cap]


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    usable = min(len(left), len(right))
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for idx in range(usable):
        a = float(left[idx] or 0.0)
        b = float(right[idx] or 0.0)
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        return 0.0
    return max(0.0, dot / math.sqrt(left_norm * right_norm))


@dataclass(frozen=True)
class NumericFeatureIndexConfig:
    dim: int = 64
    hnsw_m: int = 16
    ef_search: int = 48
    ef_construction: int = 64


class NumericFeatureIndex:
    """
    Channel-separated numeric feature recall for APV2.1 memory.

    The index is intentionally narrow:
    - state/focus are separated by memory_kind.
    - each feature channel has its own ANN index.
    - if FAISS/HNSW is unavailable, search() returns no global candidates and
      callers can still do bounded rerank over already selected candidates.
    """

    def __init__(self, *, config: NumericFeatureIndexConfig | None = None) -> None:
        self.config = config or NumericFeatureIndexConfig()
        self.dim = max(4, int(self.config.dim))
        self._ann_by_bucket: dict[tuple[str, str], FaissHnswIndex] = {}
        self._features_by_kind_id: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(dict)
        self._ids_by_bucket: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._tombstones_by_bucket: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(self, memory_kind: str, memory_id: str, channel_vectors: dict[str, object]) -> None:
        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not kind or not clean_id or not isinstance(channel_vectors, dict):
            return
        normalized: dict[str, list[float]] = {}
        for channel, values in channel_vectors.items():
            clean_channel = str(channel or "").strip()
            if not clean_channel:
                continue
            vector = _coerce_vector(values, dim=self.dim)
            if not any(abs(float(value or 0.0)) > 1e-12 for value in vector):
                continue
            normalized[clean_channel] = vector
            bucket = (kind, clean_channel)
            self._ids_by_bucket[bucket].add(clean_id)
            self._tombstones_by_bucket[bucket].discard(clean_id)
            ann = self._ann_for_bucket(bucket)
            if ann.enabled():
                ann.add(clean_id, vector)
        if normalized:
            self._features_by_kind_id[kind][clean_id] = normalized

    def remove(self, memory_kind: str, memory_id: str) -> None:
        kind = str(memory_kind or "")
        clean_id = str(memory_id or "")
        if not kind or not clean_id:
            return
        features = self._features_by_kind_id.get(kind, {}).pop(clean_id, None)
        if not isinstance(features, dict):
            return
        for channel in features:
            bucket = (kind, str(channel or ""))
            self._ids_by_bucket.get(bucket, set()).discard(clean_id)
            ann = self._ann_by_bucket.get(bucket)
            if ann is not None and ann.remove(clean_id):
                self._tombstones_by_bucket[bucket].add(clean_id)

    def search(
        self,
        memory_kind: str,
        query_features: dict[str, object],
        *,
        top_k_per_channel: int,
        overall_limit: int,
    ) -> list[dict]:
        kind = str(memory_kind or "")
        if not kind or not isinstance(query_features, dict):
            return []
        cap = max(1, int(overall_limit))
        per_channel = max(1, int(top_k_per_channel))
        rows: dict[str, dict] = {}
        for channel, values in query_features.items():
            clean_channel = str(channel or "").strip()
            if not clean_channel:
                continue
            bucket = (kind, clean_channel)
            ann = self._ann_by_bucket.get(bucket)
            if ann is None or not ann.enabled():
                continue
            query_vector = _coerce_vector(values, dim=self.dim)
            for hit in ann.search(query_vector, top_k=per_channel):
                memory_id = str(hit.get("memory_id", "") or "")
                if not memory_id:
                    continue
                if memory_id in self._tombstones_by_bucket.get(bucket, set()):
                    continue
                if memory_id not in self._features_by_kind_id.get(kind, {}):
                    continue
                score = max(0.0, float(hit.get("vector_score", 0.0) or 0.0))
                if score <= 0.0:
                    continue
                row = rows.setdefault(
                    memory_id,
                    {
                        "memory_id": memory_id,
                        "numeric_score": 0.0,
                        "numeric_score_breakdown": {},
                        "candidate_sources": [],
                    },
                )
                row["numeric_score"] = float(row["numeric_score"]) + score
                row["numeric_score_breakdown"][clean_channel] = max(
                    float(row["numeric_score_breakdown"].get(clean_channel, 0.0) or 0.0),
                    score,
                )
                source = f"numeric:{clean_channel}"
                if source not in row["candidate_sources"]:
                    row["candidate_sources"].append(source)
        return self._ordered_rows(rows, limit=cap)

    def rerank_candidates(
        self,
        memory_kind: str,
        query_features: dict[str, object],
        candidate_ids: list[str],
    ) -> dict[str, dict]:
        kind = str(memory_kind or "")
        if not kind or not isinstance(query_features, dict):
            return {}
        rows: dict[str, dict] = {}
        candidate_set = [str(memory_id or "") for memory_id in candidate_ids if str(memory_id or "")]
        by_id = self._features_by_kind_id.get(kind, {})
        for memory_id in candidate_set:
            snapshot_features = by_id.get(memory_id)
            if not isinstance(snapshot_features, dict):
                continue
            breakdown: dict[str, float] = {}
            total = 0.0
            matched = 0
            for channel, query_values in query_features.items():
                clean_channel = str(channel or "").strip()
                if not clean_channel or clean_channel not in snapshot_features:
                    continue
                score = _cosine(
                    _coerce_vector(query_values, dim=self.dim),
                    snapshot_features[clean_channel],
                )
                if score <= 0.0:
                    continue
                breakdown[clean_channel] = _round4(score)
                total += score
                matched += 1
            if matched <= 0:
                continue
            rows[memory_id] = {
                "memory_id": memory_id,
                "numeric_score": _round4(total / float(matched)),
                "numeric_score_breakdown": breakdown,
                "candidate_sources": [f"numeric_rerank:{channel}" for channel in sorted(breakdown)],
            }
        return rows

    def summary(self) -> dict:
        buckets = {}
        for bucket, ids in sorted(self._ids_by_bucket.items()):
            kind, channel = bucket
            ann = self._ann_by_bucket.get(bucket)
            buckets[f"{kind}:{channel}"] = {
                "count": len(ids),
                "engine": ann.engine_name() if ann is not None else "none",
                "faiss_available": bool(ann.enabled()) if ann is not None else False,
                "tombstones": len(self._tombstones_by_bucket.get(bucket, set())),
            }
        return {"schema_id": "numeric_feature_index/v1", "dim": self.dim, "buckets": buckets}

    def _ann_for_bucket(self, bucket: tuple[str, str]) -> FaissHnswIndex:
        ann = self._ann_by_bucket.get(bucket)
        if ann is None:
            ann = FaissHnswIndex(
                config=FaissHnswConfig(
                    dim=self.dim,
                    m=max(8, int(self.config.hnsw_m)),
                    ef_search=max(8, int(self.config.ef_search)),
                    ef_construction=max(8, int(self.config.ef_construction)),
                )
            )
            self._ann_by_bucket[bucket] = ann
        return ann

    def _ordered_rows(self, rows: dict[str, dict], *, limit: int) -> list[dict]:
        ordered = list(rows.values())
        for row in ordered:
            breakdown = dict(row.get("numeric_score_breakdown", {}) or {})
            matched = max(1, len(breakdown))
            row["numeric_score"] = _round4(float(row.get("numeric_score", 0.0) or 0.0) / float(matched))
            row["numeric_score_breakdown"] = {key: _round4(value) for key, value in sorted(breakdown.items())}
        ordered.sort(key=lambda item: (-float(item.get("numeric_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return ordered[: max(1, int(limit))]
