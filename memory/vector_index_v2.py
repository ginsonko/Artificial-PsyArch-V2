# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from typing import Any

import numpy as np
from pathlib import Path

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - fallback path
    faiss = None


def _round4(value: float) -> float:
    return round(float(value), 4)


class VectorIndexV2:
    def __init__(
        self,
        *,
        dim: int,
        backend: str = "auto",
        ann_enabled: bool = True,
        ann_top_k: int = 48,
        hnsw_m: int = 24,
        hnsw_ef_search: int = 64,
        hnsw_ef_construction: int = 80,
    ) -> None:
        self.dim = max(32, int(dim))
        self.backend = self._normalize_backend(backend)
        self.ann_enabled = bool(ann_enabled)
        self.ann_top_k = max(1, int(ann_top_k))
        self.hnsw_m = max(8, int(hnsw_m))
        self.hnsw_ef_search = max(8, int(hnsw_ef_search))
        self.hnsw_ef_construction = max(8, int(hnsw_ef_construction))
        self._memory_id_to_int: dict[str, int] = {}
        self._int_to_memory_id: dict[int, str] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._next_id = 1
        self._index = self._build_index()

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        clean_id = str(memory_id or "")
        if not clean_id:
            return
        vec = self._normalize(vector)
        self._vectors[clean_id] = vec
        if clean_id not in self._memory_id_to_int:
            int_id = self._next_id
            self._next_id += 1
            self._memory_id_to_int[clean_id] = int_id
            self._int_to_memory_id[int_id] = clean_id
        if self._index is None:
            return
        int_id = self._memory_id_to_int[clean_id]
        try:
            self._index.add_with_ids(vec.reshape(1, -1), np.asarray([int_id], dtype=np.int64))
        except Exception:
            self.rebuild()

    def add_batch(self, rows: list[tuple[str, np.ndarray]]) -> None:
        clean_rows: list[tuple[str, np.ndarray, int]] = []
        for memory_id, vector in rows:
            clean_id = str(memory_id or "")
            if not clean_id:
                continue
            vec = self._normalize(vector)
            self._vectors[clean_id] = vec
            if clean_id not in self._memory_id_to_int:
                int_id = self._next_id
                self._next_id += 1
                self._memory_id_to_int[clean_id] = int_id
                self._int_to_memory_id[int_id] = clean_id
            else:
                int_id = self._memory_id_to_int[clean_id]
            clean_rows.append((clean_id, vec, int_id))
        if not clean_rows or self._index is None:
            return
        try:
            matrix = np.stack([vec for _, vec, _ in clean_rows]).astype(np.float32, copy=False)
            ids = np.asarray([int_id for _, _, int_id in clean_rows], dtype=np.int64)
            self._index.add_with_ids(matrix, ids)
        except Exception:
            self.rebuild()

    def search(self, query_vector: np.ndarray, *, top_k: int | None = None) -> list[dict[str, Any]]:
        query = self._normalize(query_vector)
        limit = max(1, int(top_k or self.ann_top_k))
        if self._index is None or self.count() == 0:
            return self._fallback_search(query, top_k=limit)
        try:
            scores, ids = self._index.search(query.reshape(1, -1), limit)
        except Exception:
            return self._fallback_search(query, top_k=limit)
        rows: list[dict[str, Any]] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if int(idx) < 0:
                continue
            memory_id = self._int_to_memory_id.get(int(idx), "")
            if not memory_id:
                continue
            rows.append(
                {
                    "memory_id": memory_id,
                    "vector_score": _round4(score),
                    "engine": self.engine_name(),
                }
            )
        return rows

    def get_vector(self, memory_id: str) -> np.ndarray | None:
        vec = self._vectors.get(str(memory_id or ""))
        if vec is None:
            return None
        return vec.copy()

    def rebuild(self) -> None:
        self._index = self._build_index()
        if self._index is None:
            return
        if not self._vectors:
            return
        ids: list[int] = []
        vectors: list[np.ndarray] = []
        for memory_id, vector in self._vectors.items():
            int_id = self._memory_id_to_int.get(memory_id)
            if int_id is None:
                int_id = self._next_id
                self._next_id += 1
                self._memory_id_to_int[memory_id] = int_id
                self._int_to_memory_id[int_id] = memory_id
            ids.append(int_id)
            vectors.append(self._normalize(vector))
        if not vectors:
            return
        matrix = np.stack(vectors).astype(np.float32)
        self._index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))

    def count(self) -> int:
        return len(self._vectors)

    def engine_name(self) -> str:
        if self._index is None:
            if self.backend == "bundle_only":
                return "bundle_only_scan"
            return "numpy_flat"
        return "faiss_hnsw_ip"

    def summary(self) -> dict[str, Any]:
        return {
            "vector_dim": self.dim,
            "requested_backend": self.backend,
            "effective_backend": self.engine_name(),
            "ann_enabled": self.ann_enabled,
            "engine": self.engine_name(),
            "vector_count": self.count(),
            "ann_top_k": self.ann_top_k,
            "faiss_available": bool(faiss is not None),
            "bundle_format": "layered_v2",
        }

    def export_payload(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "backend": self.backend,
            "ann_enabled": self.ann_enabled,
            "ann_top_k": self.ann_top_k,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_search": self.hnsw_ef_search,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "memory_id_to_int": dict(self._memory_id_to_int),
            "int_to_memory_id": {str(key): value for key, value in self._int_to_memory_id.items()},
            "vectors": {key: value.tolist() for key, value in self._vectors.items()},
            "next_id": self._next_id,
        }

    def import_payload(self, payload: dict[str, Any]) -> None:
        self.dim = max(32, int(payload.get("dim", self.dim) or self.dim))
        self.backend = self._normalize_backend(str(payload.get("backend", self.backend) or self.backend))
        self.ann_enabled = bool(payload.get("ann_enabled", self.ann_enabled))
        self.ann_top_k = max(1, int(payload.get("ann_top_k", self.ann_top_k) or self.ann_top_k))
        self.hnsw_m = max(8, int(payload.get("hnsw_m", self.hnsw_m) or self.hnsw_m))
        self.hnsw_ef_search = max(8, int(payload.get("hnsw_ef_search", self.hnsw_ef_search) or self.hnsw_ef_search))
        self.hnsw_ef_construction = max(8, int(payload.get("hnsw_ef_construction", self.hnsw_ef_construction) or self.hnsw_ef_construction))
        self._memory_id_to_int = {str(key): int(value) for key, value in (payload.get("memory_id_to_int", {}) or {}).items() if str(key)}
        self._int_to_memory_id = {int(key): str(value) for key, value in (payload.get("int_to_memory_id", {}) or {}).items() if str(value)}
        self._vectors = {
            str(key): self._normalize(np.asarray(value, dtype=np.float32))
            for key, value in (payload.get("vectors", {}) or {}).items()
            if str(key)
        }
        self._next_id = int(payload.get("next_id", max([0, *self._memory_id_to_int.values()]) + 1) or 1)
        self.rebuild()

    def save_bundle(self, directory: Path) -> dict[str, Any]:
        directory.mkdir(parents=True, exist_ok=True)
        payload = self.export_payload()
        legacy_path = directory / "vector_index_v2.json"
        legacy_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        ordered_ids = sorted(self._memory_id_to_int.items(), key=lambda item: (int(item[1]), item[0]))
        matrix = np.zeros((0, self.dim), dtype=np.float32)
        if ordered_ids:
            matrix = np.stack([self._normalize(self._vectors.get(memory_id, np.zeros((self.dim,), dtype=np.float32))) for memory_id, _ in ordered_ids]).astype(np.float32)
        vectors_path = directory / "vectors.npy"
        np.save(vectors_path, matrix, allow_pickle=False)

        id_rows = [{"memory_id": memory_id, "int_id": int(int_id)} for memory_id, int_id in ordered_ids]
        ids_path = directory / "vector_ids.json"
        ids_path.write_text(json.dumps(id_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        faiss_path = directory / "vector_index.faiss"
        faiss_written = False
        if self._index is not None and faiss is not None:
            try:
                faiss.write_index(self._index, str(faiss_path))
                faiss_written = True
            except Exception:
                if faiss_path.exists():
                    try:
                        faiss_path.unlink()
                    except Exception:
                        pass

        meta = {
            "schema_id": "vector_index_bundle/v2",
            "schema_version": "2.0",
            "vector_dim": self.dim,
            "backend": self.backend,
            "ann_enabled": self.ann_enabled,
            "ann_top_k": self.ann_top_k,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_search": self.hnsw_ef_search,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "engine": self.engine_name(),
            "faiss_available": bool(faiss is not None),
            "faiss_index_written": faiss_written,
            "vector_count": self.count(),
            "next_id": self._next_id,
            "files": {
                "legacy_json": legacy_path.name,
                "vectors_npy": vectors_path.name,
                "id_map_json": ids_path.name,
                "faiss_index": faiss_path.name if faiss_written else "",
            },
        }
        meta_path = directory / "vector_index_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "ok": True,
            "path": str(meta_path),
            "vector_count": self.count(),
            "engine": self.engine_name(),
            "faiss_index_written": faiss_written,
        }

    def load_bundle(self, directory: Path) -> dict[str, Any]:
        meta_path = directory / "vector_index_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self.dim = max(32, int(meta.get("vector_dim", self.dim) or self.dim))
            self.backend = self._normalize_backend(str(meta.get("backend", self.backend) or self.backend))
            self.ann_enabled = bool(meta.get("ann_enabled", self.ann_enabled))
            self.ann_top_k = max(1, int(meta.get("ann_top_k", self.ann_top_k) or self.ann_top_k))
            self.hnsw_m = max(8, int(meta.get("hnsw_m", self.hnsw_m) or self.hnsw_m))
            self.hnsw_ef_search = max(8, int(meta.get("hnsw_ef_search", self.hnsw_ef_search) or self.hnsw_ef_search))
            self.hnsw_ef_construction = max(8, int(meta.get("hnsw_ef_construction", self.hnsw_ef_construction) or self.hnsw_ef_construction))
            files = dict(meta.get("files", {}) or {})
            vectors_path = directory / str(files.get("vectors_npy", "vectors.npy") or "vectors.npy")
            ids_path = directory / str(files.get("id_map_json", "vector_ids.json") or "vector_ids.json")
            if not vectors_path.exists() or not ids_path.exists():
                legacy_path = directory / str(files.get("legacy_json", "vector_index_v2.json") or "vector_index_v2.json")
                if legacy_path.exists():
                    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
                    self.import_payload(payload)
                    return {"ok": True, "path": str(legacy_path), "vector_count": self.count(), "loaded_via": "legacy_json"}
                return {"ok": False, "error": "vector bundle files missing", "path": str(meta_path)}
            matrix = np.load(vectors_path, allow_pickle=False)
            id_rows = json.loads(ids_path.read_text(encoding="utf-8"))
            self._memory_id_to_int = {}
            self._int_to_memory_id = {}
            self._vectors = {}
            for index, row in enumerate(id_rows if isinstance(id_rows, list) else []):
                if not isinstance(row, dict):
                    continue
                memory_id = str(row.get("memory_id", "") or "")
                int_id = int(row.get("int_id", 0) or 0)
                if not memory_id or int_id <= 0:
                    continue
                vec = matrix[index] if index < len(matrix) else np.zeros((self.dim,), dtype=np.float32)
                self._memory_id_to_int[memory_id] = int_id
                self._int_to_memory_id[int_id] = memory_id
                self._vectors[memory_id] = self._normalize(vec)
            self._next_id = int(meta.get("next_id", max([0, *self._memory_id_to_int.values()]) + 1) or 1)
            self._index = None
            faiss_name = str(files.get("faiss_index", "") or "")
            faiss_path = directory / faiss_name if faiss_name else None
            if bool(meta.get("faiss_index_written", False)) and faiss_path is not None and faiss_path.exists() and self.ann_enabled and faiss is not None:
                try:
                    self._index = faiss.read_index(str(faiss_path))
                except Exception:
                    self.rebuild()
            else:
                self.rebuild()
            return {"ok": True, "path": str(meta_path), "vector_count": self.count(), "loaded_via": "layered_v2"}
        path = directory / "vector_index_v2.json"
        if not path.exists():
            return {"ok": False, "error": "vector bundle not found", "path": str(path)}
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.import_payload(payload)
        return {"ok": True, "path": str(path), "vector_count": self.count(), "loaded_via": "legacy_json"}

    def _build_index(self) -> Any | None:
        if self.backend == "bundle_only":
            return None
        if self.backend == "numpy_flat":
            return None
        if not self.ann_enabled or faiss is None:
            return None
        if self.backend not in ("auto", "faiss_hnsw"):
            return None
        try:
            base = faiss.IndexHNSWFlat(self.dim, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efSearch = self.hnsw_ef_search
            base.hnsw.efConstruction = self.hnsw_ef_construction
            index = faiss.IndexIDMap2(base)
            return index
        except Exception:
            return None

    def _fallback_search(self, query: np.ndarray, *, top_k: int) -> list[dict[str, Any]]:
        rows = []
        for memory_id, vector in self._vectors.items():
            score = float(np.dot(query, self._normalize(vector)))
            rows.append(
                {
                    "memory_id": memory_id,
                    "vector_score": _round4(score),
                    "engine": self.engine_name(),
                }
            )
        rows.sort(key=lambda item: (-float(item.get("vector_score", 0.0) or 0.0), item["memory_id"]))
        return rows[: max(1, int(top_k))]

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float32).reshape(-1)
        if arr.size != self.dim:
            fixed = np.zeros((self.dim,), dtype=np.float32)
            usable = min(self.dim, arr.size)
            if usable > 0:
                fixed[:usable] = arr[:usable]
            arr = fixed
        norm = float(np.linalg.norm(arr))
        if norm > 0.0:
            arr = arr / norm
        return arr.astype(np.float32)

    def _normalize_backend(self, backend: str) -> str:
        clean = str(backend or "auto").strip().lower()
        if clean in {"faiss", "faiss_hnsw_ip"}:
            clean = "faiss_hnsw"
        if clean in {"flat", "np", "numpy"}:
            clean = "numpy_flat"
        if clean not in {"auto", "faiss_hnsw", "numpy_flat", "bundle_only"}:
            clean = "auto"
        return clean
