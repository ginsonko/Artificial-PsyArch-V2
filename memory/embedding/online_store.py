from __future__ import annotations

import hashlib
import math
from collections import OrderedDict


def _round4(value: float) -> float:
    return round(float(value), 4)


class OnlineEmbeddingStore:
    def __init__(
        self,
        *,
        dim: int,
        token_limit: int,
        min_support_to_promote: int,
        per_tick_update_limit: int,
    ) -> None:
        self.dim = max(8, int(dim))
        self.token_limit = max(32, int(token_limit))
        self.min_support_to_promote = max(1, int(min_support_to_promote))
        self.per_tick_update_limit = max(1, int(per_tick_update_limit))
        self._entries: OrderedDict[str, dict] = OrderedDict()
        self._updates_this_tick = 0
        self._current_tick = -1
        self.vector_learning_rate = 0.16

    def export_state(self) -> dict:
        entries = []
        for token, entry in self._entries.items():
            if not str(token or "").strip():
                continue
            entries.append(
                {
                    "token": str(token),
                    "vector": list(entry.get("vector", []) or [])[: self.dim],
                    "support": _round4(float(entry.get("support", 0.0) or 0.0)),
                    "negative_support": _round4(float(entry.get("negative_support", 0.0) or 0.0)),
                    "anchor_support": _round4(float(entry.get("anchor_support", 0.0) or 0.0)),
                    "transition_support": _round4(float(entry.get("transition_support", 0.0) or 0.0)),
                    "co_counts": {str(key): _round4(float(value or 0.0)) for key, value in dict(entry.get("co_counts", {}) or {}).items()},
                    "negative_counts": {str(key): _round4(float(value or 0.0)) for key, value in dict(entry.get("negative_counts", {}) or {}).items()},
                    "transition_counts": {str(key): _round4(float(value or 0.0)) for key, value in dict(entry.get("transition_counts", {}) or {}).items()},
                }
            )
        return {
            "schema_id": "apv21_online_embedding_state/v1",
            "dim": int(self.dim),
            "token_limit": int(self.token_limit),
            "min_support_to_promote": int(self.min_support_to_promote),
            "per_tick_update_limit": int(self.per_tick_update_limit),
            "vector_learning_rate": _round4(self.vector_learning_rate),
            "current_tick": int(self._current_tick),
            "entries": entries,
            "entry_count": len(entries),
            "promoted_count": sum(1 for token, _entry in self._entries.items() if self._is_promoted(token)),
        }

    def import_state(self, state: dict | None) -> dict:
        payload = dict(state or {})
        if not payload:
            self._entries = OrderedDict()
            self._updates_this_tick = 0
            self._current_tick = -1
            return {
                "schema_id": "apv21_online_embedding_state/v1",
                "restored": False,
                "entry_count": 0,
                "promoted_count": 0,
            }

        self.dim = max(8, int(payload.get("dim", self.dim) or self.dim))
        self.token_limit = max(32, int(payload.get("token_limit", self.token_limit) or self.token_limit))
        self.min_support_to_promote = max(1, int(payload.get("min_support_to_promote", self.min_support_to_promote) or self.min_support_to_promote))
        self.per_tick_update_limit = max(1, int(payload.get("per_tick_update_limit", self.per_tick_update_limit) or self.per_tick_update_limit))
        try:
            self.vector_learning_rate = max(0.0, float(payload.get("vector_learning_rate", self.vector_learning_rate) or self.vector_learning_rate))
        except (TypeError, ValueError):
            pass

        entries = list(payload.get("entries", []) or [])
        if len(entries) > self.token_limit:
            entries = entries[-self.token_limit :]
        rebuilt: OrderedDict[str, dict] = OrderedDict()
        for row in entries:
            if not isinstance(row, dict):
                continue
            token = str(row.get("token", "") or "").strip()
            if not token:
                continue
            vector = list(row.get("vector", []) or [])
            if not vector:
                vector = self._seed_vector(token)
            if len(vector) < self.dim:
                vector = list(vector) + [0.0] * (self.dim - len(vector))
            vector = self._normalize([float(value or 0.0) for value in vector[: self.dim]])
            rebuilt[token] = {
                "vector": vector,
                "support": float(row.get("support", 0.0) or 0.0),
                "negative_support": float(row.get("negative_support", 0.0) or 0.0),
                "anchor_support": float(row.get("anchor_support", 0.0) or 0.0),
                "transition_support": float(row.get("transition_support", 0.0) or 0.0),
                "co_counts": {str(key): float(value or 0.0) for key, value in dict(row.get("co_counts", {}) or {}).items()},
                "negative_counts": {str(key): float(value or 0.0) for key, value in dict(row.get("negative_counts", {}) or {}).items()},
                "transition_counts": {str(key): float(value or 0.0) for key, value in dict(row.get("transition_counts", {}) or {}).items()},
            }
        self._entries = rebuilt
        self._current_tick = int(payload.get("current_tick", -1) or -1)
        self._updates_this_tick = 0
        return {
            "schema_id": "apv21_online_embedding_state/v1",
            "restored": True,
            "entry_count": len(self._entries),
            "promoted_count": sum(1 for token in self._entries if self._is_promoted(token)),
            "current_tick": int(self._current_tick),
            "token_limit": int(self.token_limit),
            "min_support_to_promote": int(self.min_support_to_promote),
        }

    def begin_tick(self, tick_index: int) -> None:
        if int(tick_index) != self._current_tick:
            self._current_tick = int(tick_index)
            self._updates_this_tick = 0

    def observe_positive_pair(self, token_a: str, token_b: str, *, weight: float = 1.0) -> None:
        if self._updates_this_tick >= self.per_tick_update_limit:
            return
        clean_a = str(token_a or "").strip()
        clean_b = str(token_b or "").strip()
        if not clean_a or not clean_b or clean_a == clean_b:
            return
        amount = self._event_amount(weight)
        if amount <= 0:
            return
        self._touch(clean_a)
        self._touch(clean_b)
        self._entries[clean_a]["co_counts"][clean_b] = float(self._entries[clean_a]["co_counts"].get(clean_b, 0.0) or 0.0) + amount
        self._entries[clean_b]["co_counts"][clean_a] = float(self._entries[clean_b]["co_counts"].get(clean_a, 0.0) or 0.0) + amount
        self._entries[clean_a]["support"] = float(self._entries[clean_a]["support"] or 0.0) + amount
        self._entries[clean_b]["support"] = float(self._entries[clean_b]["support"] or 0.0) + amount
        self._pull_together(clean_a, clean_b, weight=amount)
        self._pull_together(clean_b, clean_a, weight=amount)
        self._updates_this_tick += 1

    def observe_positive_anchor(self, token: str, anchor_token: str, *, weight: float = 1.0) -> None:
        self._observe_directed(token, anchor_token, weight=weight, relation="positive")

    def observe_negative_pair(self, token_a: str, token_b: str, *, weight: float = 1.0) -> None:
        if self._updates_this_tick >= self.per_tick_update_limit:
            return
        clean_a = str(token_a or "").strip()
        clean_b = str(token_b or "").strip()
        if not clean_a or not clean_b or clean_a == clean_b:
            return
        amount = self._event_amount(weight)
        if amount <= 0:
            return
        self._touch(clean_a)
        self._touch(clean_b)
        self._entries[clean_a]["negative_counts"][clean_b] = float(self._entries[clean_a]["negative_counts"].get(clean_b, 0.0) or 0.0) + amount
        self._entries[clean_b]["negative_counts"][clean_a] = float(self._entries[clean_b]["negative_counts"].get(clean_a, 0.0) or 0.0) + amount
        self._entries[clean_a]["negative_support"] = float(self._entries[clean_a]["negative_support"] or 0.0) + amount
        self._entries[clean_b]["negative_support"] = float(self._entries[clean_b]["negative_support"] or 0.0) + amount
        self._push_apart(clean_a, clean_b, weight=amount)
        self._push_apart(clean_b, clean_a, weight=amount)
        self._updates_this_tick += 1

    def observe_negative_anchor(self, token: str, anchor_token: str, *, weight: float = 1.0) -> None:
        self._observe_directed(token, anchor_token, weight=weight, relation="negative")

    def observe_transition_pair(self, token_a: str, token_b: str, *, weight: float = 1.0) -> None:
        if self._updates_this_tick >= self.per_tick_update_limit:
            return
        clean_a = str(token_a or "").strip()
        clean_b = str(token_b or "").strip()
        if not clean_a or not clean_b:
            return
        amount = self._event_amount(weight)
        if amount <= 0:
            return
        self._touch(clean_a)
        self._touch(clean_b)
        self._entries[clean_a]["transition_counts"][clean_b] = float(self._entries[clean_a]["transition_counts"].get(clean_b, 0.0) or 0.0) + amount
        self._entries[clean_a]["transition_support"] = float(self._entries[clean_a].get("transition_support", 0.0) or 0.0) + amount
        self._entries[clean_b]["transition_support"] = float(self._entries[clean_b].get("transition_support", 0.0) or 0.0) + amount
        self._updates_this_tick += 1

    def learned_similarity(self, query_tokens: list[str], candidate_tokens: list[str], *, limit: int | None = None) -> dict:
        token_limit = self._score_limit(limit)
        promoted_query = [token for token in self._unique(query_tokens, limit=token_limit) if self._is_promoted(token)]
        promoted_candidate = [token for token in self._unique(candidate_tokens, limit=token_limit) if self._is_promoted(token)]
        if not promoted_query or not promoted_candidate:
            return {"score": 0.0, "contributions": [], "negative_contributions": []}
        contributions = []
        negative_contributions = []
        total = 0.0
        for q in promoted_query:
            q_entry = self._entries.get(q, {})
            q_counts = dict(q_entry.get("co_counts", {}) or {})
            q_negative_counts = dict(q_entry.get("negative_counts", {}) or {})
            q_support = max(1.0, float(q_entry.get("support", 0.0) or 0.0) + float(q_entry.get("negative_support", 0.0) or 0.0) * 0.65)
            for c in promoted_candidate:
                positive_raw = float(q_counts.get(c, 0.0) or 0.0)
                negative_raw = float(q_negative_counts.get(c, 0.0) or 0.0)
                if positive_raw <= 0.0 and negative_raw <= 0.0:
                    continue
                positive_score = positive_raw / q_support
                negative_score = negative_raw / q_support
                score = positive_score - negative_score
                if positive_score > 0.0:
                    contributions.append({"query_token": q, "candidate_token": c, "score": _round4(positive_score)})
                if negative_score > 0.0:
                    negative_contributions.append({"query_token": q, "candidate_token": c, "score": _round4(-negative_score)})
                total += score
        evidence_count = len(contributions) + len(negative_contributions)
        score = 0.0 if evidence_count <= 0 else total / max(1, evidence_count)
        contributions.sort(key=lambda item: (-float(item["score"]), item["query_token"], item["candidate_token"]))
        negative_contributions.sort(key=lambda item: (float(item["score"]), item["query_token"], item["candidate_token"]))
        return {
            "score": _round4(score),
            "contributions": contributions[:8],
            "negative_contributions": negative_contributions[:8],
        }

    def learned_transition(self, source_tokens: list[str], candidate_tokens: list[str], *, limit: int | None = None) -> dict:
        token_limit = self._score_limit(limit)
        promoted_source = [token for token in self._unique(source_tokens, limit=token_limit) if self._is_promoted(token)]
        promoted_candidate = [token for token in self._unique(candidate_tokens, limit=token_limit) if self._is_promoted(token)]
        if not promoted_source or not promoted_candidate:
            return {"score": 0.0, "contributions": []}
        contributions = []
        total = 0.0
        for source in promoted_source:
            source_entry = self._entries.get(source, {})
            transition_counts = dict(source_entry.get("transition_counts", {}) or {})
            source_support = max(
                1.0,
                float(source_entry.get("support", 0.0) or 0.0)
                + float(source_entry.get("transition_support", 0.0) or 0.0),
            )
            for candidate in promoted_candidate:
                raw = float(transition_counts.get(candidate, 0.0) or 0.0)
                if raw <= 0:
                    continue
                score = raw / source_support
                contributions.append({"source_token": source, "candidate_token": candidate, "score": _round4(score)})
                total += score
        score = 0.0 if not contributions else total / max(1, len(contributions))
        contributions.sort(key=lambda item: (-float(item["score"]), item["source_token"], item["candidate_token"]))
        return {"score": _round4(score), "contributions": contributions[:8]}

    def token_vector(self, token: str) -> list[float]:
        clean = str(token or "").strip()
        if not clean:
            return [0.0] * self.dim
        self._touch(clean)
        return list(self._entries[clean].get("vector", []) or [0.0] * self.dim)

    def learned_vector(self, tokens: list[str], *, limit: int | None = None) -> list[float]:
        selected = self._unique(tokens, limit=self._score_limit(limit))
        if not selected:
            return [0.0] * self.dim
        weighted: list[tuple[list[float], float]] = []
        for token in selected:
            self._touch(token)
            entry = self._entries.get(token, {})
            vector = list(entry.get("vector", []) or [])
            if not vector:
                continue
            support = (
                float(entry.get("support", 0.0) or 0.0)
                + float(entry.get("negative_support", 0.0) or 0.0) * 0.65
                + float(entry.get("anchor_support", 0.0) or 0.0) * 0.35
            )
            weight = 0.35 + min(2.0, support) ** 0.5
            weighted.append((vector, weight))
        if not weighted:
            return [0.0] * self.dim
        acc = [0.0] * self.dim
        for vector, weight in weighted:
            for idx in range(0, min(self.dim, len(vector))):
                acc[idx] += float(vector[idx] or 0.0) * float(weight or 0.0)
        return self._normalize(acc)

    def learned_vector_similarity(self, query_tokens: list[str], candidate_tokens: list[str], *, limit: int | None = None) -> dict:
        query_vector = self.learned_vector(query_tokens, limit=limit)
        candidate_vector = self.learned_vector(candidate_tokens, limit=limit)
        score = sum(a * b for a, b in zip(query_vector, candidate_vector))
        return {
            "score": _round4(score),
            "query_norm": _round4(math.sqrt(sum(value * value for value in query_vector))),
            "candidate_norm": _round4(math.sqrt(sum(value * value for value in candidate_vector))),
        }

    def pair_evidence(self, source_token: str, target_token: str) -> dict:
        """
        White-box readout for tests and observability.

        `learned_similarity` returns a normalized relation score. That score can
        plateau once the positive/negative ratio is stable, while real AP
        learning is still accumulating support. This method exposes the raw
        support behind one directed pair so acceptance tests can check both
        confidence direction and evidence growth.
        """

        source = str(source_token or "").strip()
        target = str(target_token or "").strip()
        entry = self._entries.get(source)
        target_entry = self._entries.get(target)
        if not source or not target or not entry:
            return {
                "source": source,
                "target": target,
                "positive_raw": 0.0,
                "negative_raw": 0.0,
                "transition_raw": 0.0,
                "source_support": 0.0,
                "source_negative_support": 0.0,
                "source_anchor_support": 0.0,
                "source_transition_support": 0.0,
                "source_promoted": False,
                "target_promoted": bool(target_entry and self._is_promoted(target)),
            }
        return {
            "source": source,
            "target": target,
            "positive_raw": _round4(float((entry.get("co_counts", {}) or {}).get(target, 0.0) or 0.0)),
            "negative_raw": _round4(float((entry.get("negative_counts", {}) or {}).get(target, 0.0) or 0.0)),
            "transition_raw": _round4(float((entry.get("transition_counts", {}) or {}).get(target, 0.0) or 0.0)),
            "source_support": _round4(float(entry.get("support", 0.0) or 0.0)),
            "source_negative_support": _round4(float(entry.get("negative_support", 0.0) or 0.0)),
            "source_anchor_support": _round4(float(entry.get("anchor_support", 0.0) or 0.0)),
            "source_transition_support": _round4(float(entry.get("transition_support", 0.0) or 0.0)),
            "source_promoted": self._is_promoted(source),
            "target_promoted": bool(target_entry and self._is_promoted(target)),
            "vector_similarity": _round4(self._cosine(
                list(entry.get("vector", []) or []),
                list((target_entry or {}).get("vector", []) or []),
            )),
        }

    def summary(self) -> dict:
        promoted = [token for token in self._entries if self._is_promoted(token)]
        return {
            "token_count": len(self._entries),
            "promoted_count": len(promoted),
            "token_limit": self.token_limit,
            "min_support_to_promote": self.min_support_to_promote,
            "per_tick_update_limit": self.per_tick_update_limit,
            "vector_learning_rate": _round4(self.vector_learning_rate),
            "vector_semantics": "online_learned_token_vectors_move_by_pressure_pairs;transition_pairs_do_not_move_symmetric_vectors",
        }

    def _score_limit(self, limit: int | None) -> int:
        if limit is None:
            return self.token_limit
        return max(1, min(self.token_limit, int(limit)))

    def _touch(self, token: str) -> None:
        clean = str(token or "").strip()
        if not clean:
            return
        if clean not in self._entries and len(self._entries) >= self.token_limit:
            self._entries.popitem(last=False)
        entry = self._entries.setdefault(
            clean,
            {
                "vector": self._seed_vector(clean),
                "support": 0.0,
                "negative_support": 0.0,
                "anchor_support": 0.0,
                "transition_support": 0.0,
                "co_counts": {},
                "negative_counts": {},
                "transition_counts": {},
            },
        )
        self._entries.move_to_end(clean)
        entry["support"] = float(entry.get("support", 0.0) or 0.0)
        entry["negative_support"] = float(entry.get("negative_support", 0.0) or 0.0)
        entry["anchor_support"] = float(entry.get("anchor_support", 0.0) or 0.0)
        entry["transition_support"] = float(entry.get("transition_support", 0.0) or 0.0)

    def _observe_directed(self, token: str, anchor_token: str, *, weight: float, relation: str) -> None:
        if self._updates_this_tick >= self.per_tick_update_limit:
            return
        clean_token = str(token or "").strip()
        clean_anchor = str(anchor_token or "").strip()
        if not clean_token or not clean_anchor or clean_token == clean_anchor:
            return
        amount = self._event_amount(weight)
        if amount <= 0:
            return
        self._touch(clean_token)
        self._touch(clean_anchor)
        self._entries[clean_anchor]["anchor_support"] = float(self._entries[clean_anchor]["anchor_support"] or 0.0) + amount
        if str(relation or "") == "negative":
            self._entries[clean_token]["negative_counts"][clean_anchor] = float(self._entries[clean_token]["negative_counts"].get(clean_anchor, 0.0) or 0.0) + amount
            self._entries[clean_token]["negative_support"] = float(self._entries[clean_token]["negative_support"] or 0.0) + amount
            self._push_apart(clean_token, clean_anchor, weight=amount)
        else:
            self._entries[clean_token]["co_counts"][clean_anchor] = float(self._entries[clean_token]["co_counts"].get(clean_anchor, 0.0) or 0.0) + amount
            self._entries[clean_token]["support"] = float(self._entries[clean_token]["support"] or 0.0) + amount
            self._pull_together(clean_token, clean_anchor, weight=amount)
        self._updates_this_tick += 1

    def _pull_together(self, source: str, target: str, *, weight: float) -> None:
        self._move_vector(source, target, weight=weight, direction="pull")

    def _push_apart(self, source: str, target: str, *, weight: float) -> None:
        self._move_vector(source, target, weight=weight, direction="push")

    def _move_vector(self, source: str, target: str, *, weight: float, direction: str) -> None:
        source_entry = self._entries.get(str(source or ""))
        target_entry = self._entries.get(str(target or ""))
        if not source_entry or not target_entry:
            return
        source_vector = list(source_entry.get("vector", []) or [])
        target_vector = list(target_entry.get("vector", []) or [])
        if len(source_vector) < self.dim or len(target_vector) < self.dim:
            return
        amount = min(0.28, self.vector_learning_rate * self._learning_signal(weight))
        if amount <= 0.0:
            return
        updated = []
        sign = 1.0 if str(direction or "") == "pull" else -1.0
        for src, tgt in zip(source_vector, target_vector):
            updated.append(float(src or 0.0) + sign * amount * float(tgt or 0.0))
        source_entry["vector"] = self._normalize(updated)

    def _learning_signal(self, weight: float) -> float:
        try:
            value = float(weight)
        except (TypeError, ValueError):
            value = 1.0
        value = max(0.0, value)
        return value / (value + 1.0) if value > 0.0 else 0.0

    def _normalize(self, vector: list[float]) -> list[float]:
        fixed = list(vector or [])[: self.dim]
        if len(fixed) < self.dim:
            fixed.extend([0.0] * (self.dim - len(fixed)))
        norm = math.sqrt(sum(float(value or 0.0) * float(value or 0.0) for value in fixed))
        if norm <= 1e-9:
            return fixed
        return [_round4(float(value or 0.0) / norm) for value in fixed]

    def _cosine(self, left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        return sum(float(a or 0.0) * float(b or 0.0) for a, b in zip(left, right))

    def _seed_vector(self, token: str) -> list[float]:
        digest = hashlib.blake2b(str(token).encode("utf-8"), digest_size=16).digest()
        vec = [0.0] * self.dim
        for idx, byte in enumerate(digest):
            vec[idx % self.dim] += 1.0 if byte % 2 == 0 else -1.0
        return self._normalize(vec)

    def _is_promoted(self, token: str) -> bool:
        entry = self._entries.get(str(token or "").strip())
        if not entry:
            return False
        evidence = (
            float(entry.get("support", 0.0) or 0.0)
            + float(entry.get("negative_support", 0.0) or 0.0)
            + float(entry.get("anchor_support", 0.0) or 0.0)
            + float(entry.get("transition_support", 0.0) or 0.0)
        )
        return evidence >= self.min_support_to_promote

    def _unique(self, tokens: list[str], *, limit: int | None = None) -> list[str]:
        seen = set()
        rows = []
        cap = self._score_limit(limit)
        for token in tokens:
            clean = str(token or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
            if len(rows) >= cap:
                break
        return rows

    def _event_amount(self, weight: float) -> float:
        try:
            value = float(weight)
        except (TypeError, ValueError):
            value = 1.0
        if value <= 0.0:
            return 0.0
        return max(0.05, min(4.0, value))
