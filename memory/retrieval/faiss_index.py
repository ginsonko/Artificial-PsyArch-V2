from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def _round4(value: float) -> float:
    return round(float(value), 4)


@dataclass
class FaissHnswConfig:
    dim: int = 64
    m: int = 24
    ef_search: int = 64
    ef_construction: int = 80


class FaissHnswIndex:
    """
    FAISS HNSW inner-product index with explicit id mapping.

    Notes:
    - We store vectors normalized (L2) so inner product ~= cosine similarity.
    - This is designed for online incremental add() and fast search().
    """

    def __init__(self, *, config: FaissHnswConfig) -> None:
        self.config = config
        self.dim = max(16, int(config.dim))
        self._memory_id_to_int: dict[str, int] = {}
        self._int_to_memory_id: dict[int, str] = {}
        self._next_id = 1
        # Keep a copy of normalized vectors so we can rebuild the index if FAISS
        # raises, and so we can keep behavior stable across backend changes.
        self._vectors: dict[str, np.ndarray] = {}
        self._index = self._build_index()

    def _build_index(self) -> Any | None:
        if faiss is None:
            return None
        # IMPORTANT:
        # - IndexHNSWFlat itself does NOT implement add_with_ids in the Python wheel
        #   we use. We must wrap it with IndexIDMap2 to enable stable external ids.
        # - This mirrors the mature legacy VectorIndexV2 behavior.
        base = faiss.IndexHNSWFlat(self.dim, max(8, int(self.config.m)), faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efSearch = max(8, int(self.config.ef_search))
        base.hnsw.efConstruction = max(8, int(self.config.ef_construction))
        return faiss.IndexIDMap2(base)

    def enabled(self) -> bool:
        return self._index is not None and faiss is not None

    def count(self) -> int:
        return len(self._vectors)

    def engine_name(self) -> str:
        if not self.enabled():
            return "numpy_flat"
        return "faiss_hnsw_ip"

    def summary(self) -> dict:
        return {
            "engine": self.engine_name(),
            "faiss_available": bool(faiss is not None),
            "vector_dim": self.dim,
            "vector_count": self.count(),
            "hnsw_m": int(self.config.m),
            "hnsw_ef_search": int(self.config.ef_search),
            "hnsw_ef_construction": int(self.config.ef_construction),
        }

    def _normalize(self, vector: list[float] | np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size != self.dim:
            fixed = np.zeros((self.dim,), dtype=np.float32)
            usable = min(self.dim, int(arr.size))
            if usable > 0:
                fixed[:usable] = arr[:usable]
            arr = fixed
        norm = float(np.linalg.norm(arr))
        if norm > 1e-9:
            arr = arr / norm
        return arr.astype(np.float32, copy=False)

    def add(self, memory_id: str, vector: list[float]) -> None:
        clean = str(memory_id or "")
        if not clean:
            return
        vec = self._normalize(vector)
        self._vectors[clean] = vec
        if clean not in self._memory_id_to_int:
            int_id = self._next_id
            self._next_id += 1
            self._memory_id_to_int[clean] = int_id
            self._int_to_memory_id[int_id] = clean
        if self._index is None:
            return
        int_id = int(self._memory_id_to_int[clean])
        try:
            self._index.add_with_ids(vec.reshape(1, -1), np.asarray([int_id], dtype=np.int64))
        except Exception:
            # HNSW cannot remove ids in our wheel; rebuild is the safe way to recover
            # from accidental duplicate ids / dimension mismatches / backend quirks.
            self.rebuild()

    def remove(self, memory_id: str) -> bool:
        """
        Best-effort removal.

        IMPORTANT:
        The FAISS wheel used in this workspace does NOT implement `remove_ids()` for
        HNSW-based indices (even when wrapped by IndexIDMap2). See the local probe
        in the repo history notes.

        Therefore removal here only updates the stored vector copy and id maps.
        The live index will still contain the vector until a rebuild() happens.

        Return value:
        - True: vector existed and was removed from the stored copy.
        - False: vector id did not exist.
        """

        clean = str(memory_id or "")
        if not clean:
            return False
        if clean not in self._vectors:
            return False
        self._vectors.pop(clean, None)
        # Keep id mapping stable; we don't recycle ids in this implementation.
        return True

    def search(self, query_vector: list[float], *, top_k: int) -> list[dict]:
        k = max(1, int(top_k))
        query = self._normalize(query_vector)
        if self._index is None or self.count() == 0:
            return self._fallback_search(query, top_k=k)
        try:
            scores, ids = self._index.search(query.reshape(1, -1), k)
        except Exception:
            return self._fallback_search(query, top_k=k)
        rows: list[dict] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if int(idx) < 0:
                continue
            memory_id = self._int_to_memory_id.get(int(idx), "")
            if not memory_id:
                continue
            rows.append({"memory_id": memory_id, "vector_score": _round4(score), "candidate_sources": ["faiss_hnsw_ip"]})
        return rows

    def rebuild(self) -> None:
        self._index = self._build_index()
        if self._index is None:
            return
        if not self._vectors:
            return
        rows: list[tuple[str, np.ndarray, int]] = []
        for memory_id, vec in self._vectors.items():
            if memory_id not in self._memory_id_to_int:
                int_id = self._next_id
                self._next_id += 1
                self._memory_id_to_int[memory_id] = int_id
                self._int_to_memory_id[int_id] = memory_id
            rows.append((memory_id, vec, int(self._memory_id_to_int[memory_id])))
        rows.sort(key=lambda item: (int(item[2]), str(item[0])))
        matrix = np.stack([vec for _, vec, _ in rows]).astype(np.float32, copy=False)
        ids = np.asarray([int_id for _, _, int_id in rows], dtype=np.int64)
        try:
            self._index.add_with_ids(matrix, ids)
        except Exception:
            # If rebuild still fails, disable FAISS and fall back to numpy.
            self._index = None

    def _fallback_search(self, query: np.ndarray, *, top_k: int) -> list[dict]:
        rows: list[dict] = []
        for memory_id, vec in self._vectors.items():
            score = float(np.dot(query, vec))
            if score <= 0.0:
                continue
            rows.append({"memory_id": memory_id, "vector_score": _round4(score), "candidate_sources": ["numpy_flat"]})
        rows.sort(key=lambda item: (-float(item.get("vector_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return rows[: max(1, int(top_k))]
