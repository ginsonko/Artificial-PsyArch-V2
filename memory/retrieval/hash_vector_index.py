from __future__ import annotations

import hashlib
import math
from collections import OrderedDict


def _round4(value: float) -> float:
    return round(float(value), 4)


class HashVectorIndex:
    def __init__(self, *, dim: int = 64) -> None:
        self.dim = max(16, int(dim))
        self._vectors: dict[str, list[float]] = {}
        self._token_projection_cache: OrderedDict[str, list[tuple[int, float]]] = OrderedDict()
        self._token_projection_cache_limit = 32768

    def add(self, memory_id: str, tokens: list[str]) -> list[float]:
        clean_id = str(memory_id or "")
        if not clean_id:
            return [0.0] * self.dim
        vector = self.embed(tokens)
        self._vectors[clean_id] = vector
        return vector

    def add_vector(self, memory_id: str, vector: list[float]) -> list[float]:
        clean_id = str(memory_id or "")
        if not clean_id:
            return [0.0] * self.dim
        fixed = list(vector or [])[: self.dim]
        if len(fixed) < self.dim:
            fixed.extend([0.0] * (self.dim - len(fixed)))
        self._vectors[clean_id] = fixed
        return fixed

    def remove(self, memory_id: str) -> None:
        """
        Remove a stored vector by id.

        NOTE:
        In APV2.1, HashVectorIndex is retained as a deterministic embedder.
        We still remove vectors to keep memory bounded when snapshots are evicted.
        """
        clean_id = str(memory_id or "")
        if not clean_id:
            return
        self._vectors.pop(clean_id, None)

    def embed(self, tokens: list[str]) -> list[float]:
        vector = [0.0] * self.dim
        seen_any = False
        for token in tokens:
            clean = str(token or "").strip()
            if not clean:
                continue
            seen_any = True
            for bucket, sign in self._token_projection(clean):
                vector[bucket] += sign
        if not seen_any:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-9:
            return vector
        return [_round4(value / norm) for value in vector]

    def _token_projection(self, token: str) -> list[tuple[int, float]]:
        cached = self._token_projection_cache.get(token)
        if cached is not None:
            self._token_projection_cache.move_to_end(token)
            return cached
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        projection = []
        for offset in range(0, len(digest), 2):
            bucket = int.from_bytes(digest[offset : offset + 2], "little") % self.dim
            sign = 1.0 if digest[offset] % 2 == 0 else -1.0
            projection.append((bucket, sign))
        self._token_projection_cache[token] = projection
        while len(self._token_projection_cache) > self._token_projection_cache_limit:
            self._token_projection_cache.popitem(last=False)
        return projection

    def search(self, query_tokens: list[str], *, top_k: int) -> list[dict]:
        query = self.embed(query_tokens)
        rows = []
        for memory_id, vector in self._vectors.items():
            score = sum(a * b for a, b in zip(query, vector))
            if score <= 0.0:
                continue
            rows.append(
                {
                    "memory_id": memory_id,
                    "vector_score": _round4(score),
                    "candidate_sources": ["hash_vector"],
                }
            )
        rows.sort(key=lambda item: (-float(item.get("vector_score", 0.0) or 0.0), str(item.get("memory_id", "") or "")))
        return rows[: max(1, int(top_k))]
